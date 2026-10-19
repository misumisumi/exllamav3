#!/usr/bin/env python3
"""Stage 3: load DeepseekV41 TP1 with MoE CPU split and run a short nc forward."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
MODEL = os.path.expanduser("~/tp1/v41port/model")
OUT = os.path.expanduser("~/tp1/v41port/stage3_load.json")

os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "256")
os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, EXL3)

def log(msg):
    print(msg, flush=True)

def mem():
    import torch
    import shutil
    vm = shutil.disk_usage("/")
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    # rss
    rss = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1e6
                    break
    except OSError:
        pass
    return {"gpu_alloc_gb": round(gpu, 2), "gpu_reserved_gb": round(reserved, 2),
            "rss_gb": round(rss, 2), "disk_free_gb": round(vm.free / 1e9, 1)}

def main():
    t_all = time.time()
    result = {"split": os.environ["EXL3_MOE_CPU_SPLIT"], "ok": False}
    try:
        log(f"sys.path[0]={EXL3}")
        import torch
        from exllamav3 import Config, Model, Tokenizer
        import exllamav3
        log(f"exllamav3={exllamav3.__file__}")
        t0 = time.time()
        cfg = Config.from_directory(MODEL)
        log(f"config {time.time()-t0:.1f}s arch={cfg.arch_string} split={cfg.infer_params.moe_cpu_split} {mem()}")
        result["arch"] = cfg.arch_string
        result["moe_cpu_split"] = cfg.infer_params.moe_cpu_split

        t0 = time.time()
        model = Model.from_config(cfg)
        result["construct_s"] = round(time.time() - t0, 1)
        result["n_modules"] = len(model.modules)
        log(f"from_config {result['construct_s']}s modules={result['n_modules']} {mem()}")

        t0 = time.time()
        model.load("cuda:0", progressbar=False, verbose=True)
        result["load_s"] = round(time.time() - t0, 1)
        result["mem_after_load"] = mem()
        log(f"load {result['load_s']}s {result['mem_after_load']}")

        tok = Tokenizer.from_config(cfg)
        ids = tok.encode("The capital of France is")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        log(f"ids {tuple(ids.shape)} {ids.tolist()[0][:16]}")
        params = {"attn_mode": "flash_attn_nc", "position": 0}
        t0 = time.time()
        with torch.inference_mode():
            logits = model.forward(ids, params)
        torch.cuda.synchronize()
        result["fwd_s"] = round(time.time() - t0, 1)
        result["logits_shape"] = list(logits.shape)
        result["logits_mean"] = float(logits.float().abs().mean())
        result["logits_finite"] = bool(torch.isfinite(logits).all())
        top = logits[0, -1].float().argmax().item()
        result["next_id"] = int(top)
        try:
            result["next_token"] = tok.decode(torch.tensor([top]))
        except Exception:
            result["next_token"] = str(top)
        log(f"forward {result['fwd_s']}s shape={result['logits_shape']} finite={result['logits_finite']} "
            f"mean={result['logits_mean']:.4f} next={result['next_token']!r}")
        result["ok"] = bool(result["logits_finite"])
        result["mem_after_fwd"] = mem()
    except Exception:
        traceback.print_exc()
        result["error"] = traceback.format_exc()[-2000:]
    result["seconds"] = round(time.time() - t_all, 1)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result, indent=2)[:3000])
    log("STAGE3_LOAD_OK" if result.get("ok") else "STAGE3_LOAD_FAIL")
    return 0 if result.get("ok") else 1

if __name__ == "__main__":
    sys.exit(main())
