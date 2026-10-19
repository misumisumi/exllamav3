"""
Prefill throughput: time to first token for fresh random-token prompts, two distinct prompts per
length so the generator's prefix reuse never skips work.

    MODEL_DIR=/path/to/model CHUNK=4096 python tests/deepseek_v41/prefill_bench_gb10.py

Env: MODEL_DIR (required), CTX (6144), CHUNK (2048), LENGTHS (512,1024,2048,4096,6000),
SKIP_PREWARM=1 (skip the page-cache prewarm; use it when the weights are in CUDA memory).

Prefill numbers are only comparable once the weights are actually resident: with aliased weights,
check that Cached has stopped growing (the prewarm waits for that) before trusting a run.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

M = os.environ["MODEL_DIR"]
CTX = int(os.environ.get("CTX", "6144"))
CHUNK = int(os.environ.get("CHUNK", "2048"))
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,1024,2048,4096,6000").split(",")]


def mem():
    d = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {k: round(d[k] / 2**20, 1) for k in ("MemFree", "MemAvailable", "Cached")}


config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = CTX, max_history = 1, max_batch_size = 1)
model.load("cuda:0", progressbar = False, verbose = False)
tokenizer = Tokenizer.from_config(config)
stc = config.stc

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
    print(json.dumps({"prewarm_settle_s": waited}), flush = True)

print(json.dumps({"chunk": CHUNK, "ctx": CTX, "ats_mmap": os.environ.get("EXL3_ATS_MMAP"),
                  "copy_regex": os.environ.get("EXL3_ATS_COPY"),
                  "torch_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 1), "mem_gib": mem()}), flush = True)

generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = CHUNK)
gen = torch.Generator().manual_seed(1234)


def ttft(ids):
    generator.enqueue(Job(input_ids = ids, max_new_tokens = 1, stop_conditions = [], sampler = GreedySampler()))
    t0 = time.perf_counter()
    first = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "error":
                raise RuntimeError(str({k: v for k, v in r.items() if not torch.is_tensor(v) and k != "job"}))
            t = r.get("token_ids")
            if first is None and t is not None and t.numel():
                first = time.perf_counter()
    return (first or time.perf_counter()) - t0


ttft(torch.randint(1000, 60000, (1, min(CHUNK + 64, CTX - 8)), generator = gen))   # warm kernels/autotune

for n in LENGTHS:
    if n + 1 > CTX:
        continue
    times = [ttft(torch.randint(1000, 60000, (1, n), generator = gen)) for _ in range(2)]
    print(json.dumps({"tokens": n, "ttft_s": [round(t, 2) for t in times],
                      "prefill_tps": [round(n / t, 1) for t in times], "mem_gib": mem()}), flush = True)
print("PREFILL_OK", flush = True)
