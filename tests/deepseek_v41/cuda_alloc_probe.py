"""
How many bytes CUDA can actually obtain on a unified-memory box (GB10 / DGX Spark), and whether the
kernel reclaims page cache to satisfy it. Allocates in CHUNK_GIB steps until an allocation fails or
MemAvailable reaches STOP_AVAIL_GIB.

    STOP_AVAIL_GIB=6 python tests/deepseek_v41/cuda_alloc_probe.py

Env: CHUNK_GIB (1), MAX_GIB (140), STOP_AVAIL_GIB (12), MODE (cached | drop), MODEL_DIR (required
for MODE=drop, which first evicts that model's shards from page cache).

Set STOP_AVAIL_GIB only as a safety floor for the box's OOM killer (earlyoom defaults to 3% of RAM).
A high floor makes the probe stop before CUDA does, which measures the floor, not the ceiling.
Measured on one GB10 (127.6 GiB RAM): 106 GiB allocated in both modes, stopped by the floor rather
than any failure, with 78.7 GiB of page cache reclaimed on the way; cudaMemGetInfo's free value
tracked MemFree exactly while MemAvailable still read 92 GiB.
"""
import glob, json, os, sys, time
import torch

STOP_AVAIL_GIB = float(os.environ.get("STOP_AVAIL_GIB", "12"))
CHUNK_GIB = float(os.environ.get("CHUNK_GIB", "1"))
MAX_GIB = float(os.environ.get("MAX_GIB", "140"))
MODE = os.environ.get("MODE", "cached")


def mem():
    d = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {k: round(d.get(k, 0) / 2**20, 1) for k in
            ("MemFree", "MemAvailable", "Cached", "AnonPages", "Shmem", "Committed_AS")}


def cuda_info():
    free, total = torch.cuda.mem_get_info()
    return {"cuda_free_gib": round(free / 2**30, 1), "cuda_total_gib": round(total / 2**30, 1),
            "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 1)}


if MODE == "drop":
    n = 0
    for p in sorted(glob.glob(os.environ["MODEL_DIR"] + "/**/*.safetensors", recursive = True)):
        fd = os.open(os.path.realpath(p), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        n += 1
    time.sleep(2)
    print(json.dumps({"dropped_shards": n, "mem": mem()}), flush = True)

torch.zeros(1, device = "cuda")
start = {"mode": MODE, "mem": mem(), "cuda": cuda_info()}
print(json.dumps({"start": start}), flush = True)

blocks = []
reason = "max_reached"
n = int(CHUNK_GIB * 2**30)
while len(blocks) * CHUNK_GIB < MAX_GIB:
    if mem()["MemAvailable"] < STOP_AVAIL_GIB:
        reason = "stop_threshold (raise STOP_AVAIL_GIB only as an OOM-killer floor)"
        break
    try:
        blocks.append(torch.empty(n, dtype = torch.uint8, device = "cuda"))
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
        break
    if len(blocks) % 5 == 0:
        print(json.dumps({"allocated_gib": round(len(blocks) * CHUNK_GIB, 1), "mem": mem(), "cuda": cuda_info()}),
              flush = True)

print(json.dumps({"stopped": reason, "allocated_gib": round(len(blocks) * CHUNK_GIB, 1), "mem": mem(),
                  "cuda": cuda_info(), "cached_drop_gib": round(start["mem"]["Cached"] - mem()["Cached"], 1)}),
      flush = True)
del blocks
torch.cuda.empty_cache()
time.sleep(3)
print(json.dumps({"after_free": {"mem": mem(), "cuda": cuda_info()}}), flush = True)
print("ALLOC_PROBE_OK", flush = True)
