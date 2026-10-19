#!/usr/bin/env python3
"""A/B timing of grouped mixed-K MoE dispatch against the per-expert loop, in one process.

Loads the model once, then alternates modes by toggling BlockSparseMLP.grouped_state on every
module (None = per-expert loop). Times prefill forwards of the reference sequences and 1-token
forwards (the MoE path at bsz 1), and compares logits between modes. Run order is
grouped, loop, grouped so warm-up effects show up as a difference between the two grouped runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

TREE = os.environ.get("EXL3_TREE")
if TREE:
    sys.path.insert(0, TREE)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = os.path.expanduser("~/tp1/v41port/model"))
    p.add_argument("--reference", default = os.path.expanduser("~/tp1/v41port/ref_logprobs_4x512.safetensors"))
    p.add_argument("--engram-rows", default = os.path.expanduser("~/tp1/e2e/engram-rows-4k.safetensors"))
    p.add_argument("--split", type = int, default = 256)
    p.add_argument("--seqs", type = int, default = 2)
    p.add_argument("--decode-steps", type = int, default = 24)
    p.add_argument("--output", default = None)
    args = p.parse_args()

    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", str(args.split))
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")
    os.environ["EXL3_ENGRAM_ROWS"] = args.engram_rows

    import torch
    from safetensors import safe_open
    from exllamav3 import Config, Model
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

    with safe_open(args.reference, "pt") as f:
        tokens = f.get_tensor("tokens").long()[: args.seqs]

    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    model.load("cuda:0", progressbar = False, verbose = False)
    dev = torch.device("cuda:0")

    mlps, seen, stack = [], set(), list(model.modules)
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if isinstance(m, BlockSparseMLP):
            mlps.append(m)
        stack.extend(getattr(m, "modules", []) or [])
    saved = {id(m): m.grouped_state for m in mlps}
    n_grouped = sum(1 for m in mlps if m.grouped_state is not None)
    print(f"MoE modules: {len(mlps)}, with grouped state: {n_grouped}", flush = True)

    def fwd(ids, pos):
        with torch.inference_mode():
            out = model.forward(ids.to(dev), {"attn_mode": "flash_attn_nc", "position": pos})
        torch.cuda.synchronize()
        return out

    def run(mode):
        for m in mlps:
            m.grouped_state = saved[id(m)] if mode == "grouped" else None
        prefill, decode, logits0 = [], [], None
        for s in range(tokens.shape[0]):
            t0 = time.perf_counter()
            out = fwd(tokens[s : s + 1], 0)
            prefill.append(time.perf_counter() - t0)
            if s == 0:
                logits0 = out[0, :, :].float().cpu()
        for i in range(args.decode_steps):
            t0 = time.perf_counter()
            fwd(tokens[0:1, i : i + 1], 0)
            decode.append(time.perf_counter() - t0)
        dec = sorted(decode[2:]) if len(decode) > 4 else sorted(decode)
        return {
            "prefill_s": [round(v, 3) for v in prefill],
            "decode_median_ms": round(1000 * dec[len(dec) // 2], 2),
        }, logits0

    results = {}
    results["grouped_1"], lg = run("grouped")
    results["loop"], ll = run("loop")
    results["grouped_2"], _ = run("grouped")
    diff = (lg - ll).abs()
    results["logits_max_abs_diff_seq0"] = round(diff.max().item(), 5)
    results["top1_agree_seq0"] = round((lg.argmax(-1) == ll.argmax(-1)).float().mean().item(), 5)
    results["moe_modules"] = len(mlps)
    results["grouped_modules"] = n_grouped
    print(json.dumps(results, indent = 2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent = 2)
    print("BENCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
