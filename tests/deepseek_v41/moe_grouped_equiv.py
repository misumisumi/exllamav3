#!/usr/bin/env python3
"""Module-level equivalence of grouped mixed-K MoE dispatch against the per-expert loop.

One no-cache forward captures the real inputs of a few BlockSparseMLP modules; each captured input is
then run through the same module twice (grouped_state set, then None) and the outputs compared per
token. fp16 kernel differences give relative errors around 1e-3; a wrong expert slice or count gives
errors of order 1 on the tokens routed to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TREE = os.environ.get("EXL3_TREE")
if TREE:
    sys.path.insert(0, TREE)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = os.path.expanduser("~/tp1/v41port/model"))
    p.add_argument("--reference", default = os.path.expanduser("~/tp1/v41port/ref_logprobs_4x512.safetensors"))
    p.add_argument("--engram-rows", default = os.path.expanduser("~/tp1/e2e/engram-rows-4k.safetensors"))
    p.add_argument("--split", type = int, default = 256)
    p.add_argument("--layers", default = "0,1,8,20,39")
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
        tokens = f.get_tensor("tokens").long()[:1]

    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    model.load("cuda:0", progressbar = False, verbose = False)
    dev = torch.device("cuda:0")

    wanted = {f"layers.{int(l)}.ffn" for l in args.layers.split(",")}
    captured = {}
    original = BlockSparseMLP.forward

    def hooked(self, x, params, out_dtype = None):
        if self.key in wanted and self.key not in captured:
            captured[self.key] = (self, x.detach().clone())
        return original(self, x, params, out_dtype)

    BlockSparseMLP.forward = hooked
    with torch.inference_mode():
        model.forward(tokens.to(dev), {"attn_mode": "flash_attn_nc", "position": 0})
    BlockSparseMLP.forward = original

    rows = []
    for key in sorted(captured, key = lambda k: int(k.split(".")[1])):
        module, x = captured[key]
        state = module.grouped_state
        with torch.inference_mode():
            module.grouped_state = state
            yg = module.forward(x.clone(), {}).float()
            module.grouped_state = None
            yl = module.forward(x.clone(), {}).float()
            module.grouped_state = state
            yg2 = module.forward(x.clone(), {}).float()
        torch.cuda.synchronize()
        yg, yl, yg2 = (t.reshape(-1, t.shape[-1]) for t in (yg, yl, yg2))
        rel = (yg - yl).norm(dim = -1) / yl.norm(dim = -1).clamp_min(1e-6)
        rel_self = (yg - yg2).norm(dim = -1) / yg.norm(dim = -1).clamp_min(1e-6)
        row = {
            "key": key,
            "grouped": state is not None,
            "groups": len(state["groups"]) if state is not None else 0,
            "rel_err_mean": round(rel.mean().item(), 6),
            "rel_err_max": round(rel.max().item(), 6),
            "tokens_rel_err_over_1e-2": int((rel > 1e-2).sum().item()),
            "grouped_repeat_rel_err_max": round(rel_self.max().item(), 6),
        }
        rows.append(row)
        print(json.dumps(row), flush = True)

    ok = bool(rows) and all(r["rel_err_max"] < 5e-2 for r in rows)
    result = {"ok": ok, "layers": rows}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent = 2)
    print("EQUIV_OK" if ok else "EQUIV_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
