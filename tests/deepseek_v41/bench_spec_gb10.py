#!/usr/bin/env python3
"""DeepSeek-V4.1 DSpark speculative decoding benchmark on GB10 loads.

Benchmark model with and without drafter, tracking acceptance rate and decode tokens/s.
Set DRAFT=1 to enable speculative decoding (loads the mtp component as draft_model).
Tokens are counted per streamed result as r["token_ids"].numel(), so speculative steps
emit several tokens per result.

    DRAFT=1 MODEL_DIR=... python tests/deepseek_v41/bench_spec_gb10.py [--tokens N]
"""

import os
import sys
import time
import json
import re
import statistics

sys.path.insert(0, "/home/markus/work/exl3_v41")

import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

M = os.environ["MODEL_DIR"]
CTX = 6144
LAZY = os.environ.get("EXL3_FP8_LAZY") == "1"
PREWARM = os.environ.get("PREWARM") == "1"
TOKENS = int(os.environ.get("TOKENS", "64"))
DRAFT = os.environ.get("DRAFT") == "1"

def mem():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemAvailable", "Cached", "AnonPages"):
            d[k] = round(int(v.split()[0]) / 1048576, 2)
    return d

def io():
    s = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k == "pgmajfault":
            s[k] = int(v)
    s["disk"] = sum(int(p[5]) * 512 for p in (l.split() for l in open("/proc/diskstats")) if re.fullmatch(r"nvme\d+n\d+", p[2]))
    return s

print("model", M, "lazy", LAZY, "prewarm", PREWARM, "tokens", TOKENS, "draft", DRAFT, "mem", mem(), flush = True)
config = Config.from_directory(M)
model = Model.from_config(config)
draft = None
draft_cache = None
max_history = 1
if DRAFT:
    draft = Model.from_config(config, component = "mtp")
    max_history = draft.caps.get("default_draft_size", 4)

# Caches must exist before their models load
cache = Cache(model, max_num_tokens = CTX, max_history = max_history, max_batch_size = 1)
if DRAFT:
    draft_cache = Cache(draft, max_num_tokens = CTX)
t0 = time.time()
model.load("cuda:0", progressbar = False, verbose = False)
if DRAFT:
    draft.load("cuda:0", progressbar = False, verbose = False)

load_s = time.time() - t0
print(f"loaded in {load_s:.1f}s aliased/copied GiB {[round(b / 2**30, 2) for b in config.stc.ats_bytes]} torch_alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB mem {mem()}", flush = True)

if PREWARM:
    # Ask the kernel to read ahead exactly the byte ranges of text-model weights (not the Engram
    # tables, drafter or vision tower), then wait for the page cache to stop growing
    stc = config.stc
    t0 = time.time()
    total = 0
    fds = {}
    for key, fn in stc.tensor_file_map.items():
        if ".engram.embed." in key or key.startswith(("mtp.", "vision.")):
            continue
        h = stc.file_headers[fn][key]
        b, e = h["data_offsets"]
        if fn not in fds:
            fds[fn] = os.open(fn, os.O_RDONLY)
        os.posix_fadvise(fds[fn], stc.file_headers[fn]["_header_offset"] + b, e - b, os.POSIX_FADV_WILLNEED)
        total += e - b
    last, stable = -1, 0
    while time.time() - t0 < 300 and stable < 3:
        time.sleep(2)
        c = mem()["Cached"]
        stable = stable + 1 if abs(c - last) < 0.2 else 0
        last = c
    for fd in fds.values():
        os.close(fd)
    print(f"prewarm {total / 2**30:.1f} GiB requested, settled in {time.time() - t0:.0f}s, mem {mem()}", flush = True)

tokenizer = Tokenizer.from_config(config)

PROMPTS = [
    "The history of the printing press begins",
    "def merge_sort(items):\n    \"\"\"Return a sorted copy of items.\"\"\"\n",
    "Solve for x and show each step: 3x + 7 = 2(x - 4) + 15.",
    "Write a short poem about autumn rain on a city street.",
    "Explain how mRNA vaccines teach the immune system to recognize a virus.",
    "A simple recipe for weeknight vegetable curry:",
    "Summarize the main differences between a contract and a memorandum of understanding.",
    "The 1986 World Cup quarter-final between Argentina and England is remembered for",
    "Plan a three-day itinerary for a first visit to Kyoto.",
    "What are the tradeoffs between index funds and actively managed funds?",
    "Describe the life cycle of a monarch butterfly.",
    "Why does a minor chord sound sadder than a major chord to many listeners?",
]

def run(name, prompt, n, generator):
    ids = tokenizer.encode(prompt, add_bos = True)
    generator.enqueue(Job(input_ids = ids, max_new_tokens = n, stop_conditions = [], sampler = GreedySampler()))
    steps, first, last, c0, text = [], None, None, None, []
    accepted_tokens = 0
    rejected_tokens = 0
    total_tokens = 0
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "error":
                raise RuntimeError(str({k: v for k, v in r.items() if not torch.is_tensor(v) and k != "job"}))
            if r.get("stage") != "streaming":
                continue
            if r.get("text"):
                text.append(r["text"])
            # Draft counters are running totals for the job (final values ride on the eos result)
            if "accepted_draft_tokens" in r:
                accepted_tokens = r["accepted_draft_tokens"]
                rejected_tokens = r.get("rejected_draft_tokens", 0)
            t = r.get("token_ids")
            if t is None or not t.numel():
                continue
            now = time.perf_counter()
            if first is None:
                c0 = io()
                first = last = time.perf_counter()
                continue
            # Tokens streamed after the first result, over the time since it
            total_tokens += t.numel()
            steps.append(now - last)
            last = now
    c1 = io()
    k = len(steps)
    rep = {
        "job": name, "decode_tps": round(total_tokens / (last - first), 2) if steps else 0.0,
        "p50_ms": round(sorted(steps)[k // 2] * 1000) if steps else 0,
        "nvme_mib_per_tok": round((c1["disk"] - c0["disk"]) / total_tokens / 2**20, 1) if total_tokens else 0.0,
        "majflt_per_tok": round((c1["pgmajfault"] - c0["pgmajfault"]) / total_tokens, 1) if total_tokens else 0.0,
        "mem": mem(), "text": "".join(text)[:80],
    }
    if DRAFT and (accepted_tokens + rejected_tokens) > 0:
        rep["accepted_draft_tokens"] = accepted_tokens
        rep["rejected_draft_tokens"] = rejected_tokens
        rep["acceptance"] = round(accepted_tokens / (accepted_tokens + rejected_tokens), 3)
    print(json.dumps(rep), flush = True)
    return rep

# One generator per run: a plain pass ahead of the speculative pass would warm the page cache and
# bias the comparison, so the baseline is a separate DRAFT=0 run
if DRAFT:
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft,
                    draft_cache = draft_cache, max_chunk_size = 2048)
else:
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 2048)
fresh = [run(f"p{i:02d}", p, TOKENS, gen) for i, p in enumerate(PROMPTS)]
again = [run(f"p{i:02d}_again", p, TOKENS, gen) for i, p in enumerate(PROMPTS[:4])]
tail = [r["decode_tps"] for r in fresh[4:]]
summary = {
    "draft": DRAFT, "lazy": LAZY, "prewarm": PREWARM,
    "fresh_first4_tps": [r["decode_tps"] for r in fresh[:4]],
    "fresh_5to12_median_tps": statistics.median(tail), "fresh_5to12_mean_tps": round(statistics.mean(tail), 2),
    "fresh_5to12_nvme_mib_per_tok": round(statistics.mean(r["nvme_mib_per_tok"] for r in fresh[4:]), 1),
    "again_tps": [r["decode_tps"] for r in again],
}
if DRAFT:
    summary["mean_acceptance"] = round(statistics.mean(r.get("acceptance", 0.0) for r in fresh), 3)
print(json.dumps(summary), flush = True)
print("BENCH_SPEC_GB10_OK", flush = True)
