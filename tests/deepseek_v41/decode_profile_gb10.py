"""
Plain-decode profile with synced per-module timing, for comparing weight placement on unified-memory
boxes (GB10 / DGX Spark): weights aliased from page cache (EXL3_ATS_MMAP=1) against weights copied
into CUDA memory (EXL3_ATS_MMAP=0, or a subset via EXL3_ATS_COPY=<regex>).

    MODEL_DIR=/path/to/model python tests/deepseek_v41/decode_profile_gb10.py

Env: MODEL_DIR (required), CTX (6144), TOKENS (96 profiled), PROMPT, SKIP_PREWARM=1 (skip the page
cache prewarm, which is pointless when the weights are in CUDA memory), HUGE_PREFAULT=1 (evict each
weight file and read its text-tensor ranges back through a MADV_HUGEPAGE mapping, so the cache
refills in 2 MB folios), plus any exllamav3 variables.

Prints: load time, aliased/copied bytes, CUDA allocated, host memory; warm tok/s unprofiled; then
ms per emitted token per module, grouped by layer half.
"""
import collections, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

M = os.environ["MODEL_DIR"]
CTX = int(os.environ.get("CTX", "6144"))
TOKENS = int(os.environ.get("TOKENS", "96"))
PROMPT = os.environ.get("PROMPT", "Explain, step by step, why the sky appears blue during the day.")


def mem():
    d = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {k: round(d.get(k, 0) / 2**20, 1) for k in
            ("MemFree", "MemAvailable", "Cached", "AnonPages", "FileHugePages", "FilePmdMapped")}


config = Config.from_directory(M)

if os.environ.get("HUGE_PREFAULT") == "1":
    # Whole files, not tensor ranges: small scale tensors between the multi-MB trellis tensors
    # otherwise keep 4 KB pages in nearly every 2 MB block and no PMD folio can form
    import mmap
    stc0 = config.stc
    t0, before, touched, blk = time.time(), mem(), 0, 1 << 21
    per_file = {}
    for key, fn in stc0.tensor_file_map.items():
        h = stc0.file_headers[fn]
        b, e = h[key]["data_offsets"]
        text = not (".engram.embed." in key or key.startswith("mtp."))
        per_file.setdefault(fn, []).append((h["_header_offset"] + b, h["_header_offset"] + e, text))
    for fn, items in per_file.items():
        if not any(t for _, _, t in items):
            continue
        size = os.path.getsize(fn)
        merged = []
        for b, e in sorted((b // blk * blk, min(-(-e // blk) * blk, size)) for b, e, t in items if t and e > b):
            if merged and b <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([b, e])
        fd = os.open(fn, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        mmf = mmap.mmap(fd, 0, flags = mmap.MAP_SHARED, prot = mmap.PROT_READ)
        mmf.madvise(mmap.MADV_HUGEPAGE)
        s = 0
        for b, e in merged:
            for off in range(b, e, 4096):
                s += mmf[off]
            touched += e - b
        mmf.close()
        os.close(fd)
    print(json.dumps({"huge_prefault_gib": round(touched / 2**30, 1), "seconds": round(time.time() - t0, 1),
                      "mem_before_gib": before, "mem_after_gib": mem()}), flush = True)

model = Model.from_config(config)
cache = Cache(model, max_num_tokens = CTX, max_history = 1, max_batch_size = 1)
t0 = time.time()
model.load("cuda:0", progressbar = False, verbose = False)
load_s = time.time() - t0
tokenizer = Tokenizer.from_config(config)
stc = config.stc
aliased, copied = getattr(stc, "ats_bytes", (0, 0))
print(json.dumps({"copy_regex": os.environ.get("EXL3_ATS_COPY"), "ats_mmap": os.environ.get("EXL3_ATS_MMAP"),
                  "load_s": round(load_s, 1), "aliased_gib": round(aliased / 2**30, 1),
                  "copied_offgrid_gib": round(copied / 2**30, 1),
                  "copied_policy_gib": round(getattr(stc, "ats_copied_policy", 0) / 2**30, 1),
                  "torch_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 1), "mem_gib": mem()}), flush = True)

# Page-cache prewarm of the text weights (not the Engram tables)
fds = {} if os.environ.get("SKIP_PREWARM") != "1" else None
for key, fn in (stc.tensor_file_map.items() if fds is not None else []):
    if ".engram.embed." in key or key.startswith("mtp."):
        continue
    b, e = stc.file_headers[fn][key]["data_offsets"]
    fds.setdefault(fn, os.open(fn, os.O_RDONLY))
    os.posix_fadvise(fds[fn], stc.file_headers[fn]["_header_offset"] + b, e - b, os.POSIX_FADV_WILLNEED)
for fd in (fds.values() if fds is not None else []):
    os.close(fd)
if fds is not None:
    prev, waited = mem()["Cached"], 0
    while waited < 180:
        time.sleep(5)
        waited += 5
        cur = mem()["Cached"]
        if cur - prev < 0.25:
            break
        prev = cur
    print(json.dumps({"prewarm_settle_s": waited, "mem_gib": mem()}), flush = True)

PROF = {"on": False}
ms = collections.defaultdict(float)


def wrap(m):
    f = m.forward
    def w(*a, **k):
        if not PROF["on"]:
            return f(*a, **k)
        torch.cuda.synchronize()
        t = time.perf_counter()
        y = f(*a, **k)
        torch.cuda.synchronize()
        ms[m.key] += (time.perf_counter() - t) * 1000
        return y
    m.forward = w


for m in model.modules:
    wrap(m)

generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 2048)


def run(n, on_first = None):
    ids = tokenizer.encode(PROMPT, add_bos = True)
    generator.enqueue(Job(input_ids = ids, max_new_tokens = n, stop_conditions = [], sampler = GreedySampler()))
    first = last = None
    k = 0
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "error":
                raise RuntimeError(str({kk: v for kk, v in r.items() if not torch.is_tensor(v) and kk != "job"}))
            t = r.get("token_ids")
            if r.get("stage") != "streaming" or t is None or not t.numel():
                continue
            now = time.perf_counter()
            if first is None:
                first = last = now
                if on_first:
                    on_first()
                continue
            k += t.numel()
            last = now
    return k, last - first


run(128)
tps = []
for _ in range(3):
    k, dt = run(128)
    tps.append(round(k / dt, 2))
print(json.dumps({"warm_unprofiled_tps": tps, "mem_gib": mem()}), flush = True)


def on():
    ms.clear()
    PROF["on"] = True


k, dt = run(TOKENS, on_first = on)
PROF["on"] = False
groups = collections.defaultdict(float)
per_layer = [0.0] * 256
for key, v in ms.items():
    v /= k
    mm = re.match(r"layers\.(\d+)(?:\.(\w+))?", key)
    if mm:
        li = int(mm.group(1))
        per_layer[li] += v
        groups[f"{'L_first_half' if li < 20 else 'L_second_half'}_{mm.group(2) or 'block'}"] += v
    else:
        groups[f"other_{key}"] += v
n_layers = max((i + 1 for i, v in enumerate(per_layer) if v), default = 0)
print(json.dumps({"synced_tps": round(k / dt, 2), "tokens": k,
                  "ms_per_token_groups": {g: round(v, 2) for g, v in sorted(groups.items())},
                  "ms_per_token_per_layer": [round(v, 2) for v in per_layer[:n_layers]]}), flush = True)
print("DECODE_PROFILE_OK", flush = True)
