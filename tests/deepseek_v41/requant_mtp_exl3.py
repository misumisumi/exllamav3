#!/usr/bin/env python3
"""
Requantize the fp16-loaded Linears of a DeepSeek-V4.1 DSpark drafter (mtp component,
source format: packed FP4 routed experts, FP8 attention/shared/main_proj, all with E8M0
scale grids) to EXL3 with UNCALIBRATED quantization (meta Hessian).

Speculative decoding verifies every token, so drafter quality only affects acceptance rate.

    EXL3_FP8_LAZY=1 EXL3_ATS_MMAP=1 python tests/deepseek_v41/requant_mtp_exl3.py --model DIR --out DIR [--K 4]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Set environment variables before importing exllamav3 so FP4/FP8 weights stay compact
os.environ["EXL3_FP8_LAZY"] = "1"
os.environ.setdefault("EXL3_ATS_MMAP", "1")

import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.linear import Linear

# Target all "mtp." Linears with quant_type "fp16", multiples of 16
TARGET = re.compile(r"^mtp\.")
# Exclude routers, small heads, embeddings, norms
# Routers (ffn.gate: routing scores read by the MoE kernels), small heads, embeddings and norms stay as loaded
EXCLUDE = re.compile(r"(markov|confidence|embed|norm|\.gate$)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--K", type = int, default = 4)
    ap.add_argument("--flush-gib", type = float, default = 2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)
    dev = torch.device("cuda:0")

    # Record every Linear as it loads
    linears = []
    orig_load = Linear.load
    def load_rec(self, device, **kwargs):
        linears.append(self)
        return orig_load(self, device, **kwargs)
    Linear.load = load_rec

    config = Config.from_directory(args.model)
    draft = Model.from_config(config, component = "mtp")
    t0 = time.time()
    draft.load(dev, progressbar = False, verbose = False)
    Linear.load = orig_load
    print(f"loaded in {time.time() - t0:.1f}s, {len(linears)} linears", flush = True)

    # Filter targets: mtp.* with quant_type fp16, 16-aligned, not excluded
    targets = []
    excluded_keys = []
    for lin in linears:
        if not TARGET.match(lin.key):
            continue
        if EXCLUDE.search(lin.key):
            excluded_keys.append(lin.key)
            continue
        if lin.quant_type != "fp16":
            continue
        if lin.in_features % 16 or lin.out_features % 16:
            print(f" -- skip {lin.key}: {lin.in_features}x{lin.out_features} not multiples of 16", flush = True)
            continue
        targets.append(lin)

    print(f"excluded {len(excluded_keys)} routers/heads/norms:", flush = True)
    for k in excluded_keys:
        print(f"  {k}", flush = True)
    print(f"{len(targets)} target linears", flush = True)

    # Process all targets, flushing to disk every flush_gib
    stats = []
    tensors = {}
    cur_bytes = 0
    part_idx = 0

    for i, lin in enumerate(targets):
        # Uncalibrated: meta Hessian only
        H_data = lin.init_H_data(False)
        qa = {
            "seed": i, "K": args.K, "devices": [0], "device_ratios": None,
            "apply_out_scales": None, "debug_dir": os.path.join(args.out, "debug"), "mul1": True,
        }
        t1 = time.time()
        proxy = lin.convert_exl3(H_data, qa)
        lin_tensors = lin.get_tensors()
        for k, v in lin_tensors.items():
            tensors[k] = v.detach().to("cpu").contiguous()

        rec = {
            "key": lin.key, "K": args.K, "in": lin.in_features, "out": lin.out_features,
            "proxy_err": round(float(proxy), 6), "fallback": bool(qa.get("q_fallback")),
            "s": round(time.time() - t1, 1),
        }
        stats.append(rec)
        print(json.dumps(rec), flush = True)

        # Track bytes and flush if threshold exceeded
        for v in lin_tensors.values():
            cur_bytes += v.numel() * v.itemsize
        if cur_bytes > args.flush_gib * 2**30:
            path = os.path.join(args.out, f"requant_mtp_part{part_idx:02d}.safetensors")
            save_file(tensors, path, metadata = {"format": "pt"})
            print(f"flushed part {part_idx} with {len(tensors)} tensors to {path}", flush = True)
            tensors = {}
            cur_bytes = 0
            part_idx += 1

        # Periodically clear cache
        if (i + 1) % 64 == 0:
            torch.cuda.empty_cache()

    # Final flush
    if tensors:
        path = os.path.join(args.out, f"requant_mtp_part{part_idx:02d}.safetensors")
        save_file(tensors, path, metadata = {"format": "pt"})
        print(f"flushed part {part_idx} with {len(tensors)} tensors to {path}", flush = True)

    # Save stats
    with open(os.path.join(args.out, "requant_mtp_stats.json"), "w") as f:
        json.dump(stats, f, indent = 1)

    elapsed = time.time() - t0
    print(f"REQUANT_MTP_OK {len(stats)} linears {elapsed:.0f}s", flush = True)


if __name__ == "__main__":
    main()
