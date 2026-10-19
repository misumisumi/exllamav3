#!/usr/bin/env python3
"""Localize a DeepSeek-V4.1 cached-decode mismatch to the first diverging module.

Run A: one cached chunk over the first N tokens (bit-identical to the nc path).
Run B: cached prefill of N-1 tokens, then one single-token decode step.
Both capture every top-level module's output for the LAST token (hidden streams after embedding, each
Engram layer and each transformer block, then the head), and the per-module relative error is printed in
order. The first module well above the noise level is where single-token decode breaks.
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
    p.add_argument("--split", type = int, default = 128)
    p.add_argument("--seq", type = int, default = 0)
    p.add_argument("--length", type = int, default = 480)
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
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.recurrent_util import prepare_for_recurrence
    from exllamav3.modules.dsv41_hc import PRE_MIX_KEY

    with safe_open(args.reference, "pt") as f:
        ids = f.get_tensor("tokens").long()[args.seq : args.seq + 1, : args.length]
    N = ids.shape[1]

    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    model.load("cuda:0", progressbar = False, verbose = False)
    ids = ids.to("cuda:0")

    def last_token(y):
        if not torch.is_tensor(y):
            return None
        if y.dim() >= 3:
            return y[0, -1].detach().float().cpu().clone()
        if y.dim() == 2:
            return y[-1].detach().float().cpu().clone()
        return None

    def forward_chunk(state, xc, capture):
        params = {"attn_mode": "flash_attn", "recurrent_states": [state],
                  "batch_shape": (1, xc.shape[1]), "past_len": state.position, "input_ids": xc}
        params.pop(PRE_MIX_KEY, None)
        prepare_for_recurrence(xc, params, model)
        outs = []
        y = xc
        with torch.inference_mode():
            for m in model.modules:
                y = m.prepare_for_device(y, params)
                y = m.forward(y, params)
                if capture:
                    outs.append((m.key, last_token(y)))
        state.position += xc.shape[1]
        state.post_advance()
        return outs

    with torch.inference_mode():
        sa = cache.get_new_state()
    run_a = forward_chunk(sa, ids, True)
    sa.free()

    with torch.inference_mode():
        sb = cache.get_new_state()
    forward_chunk(sb, ids[:, : N - 1], False)
    run_b = forward_chunk(sb, ids[:, N - 1 :], True)
    sb.free()

    rows, first_bad = [], None
    for i, ((ka, ya), (kb, yb)) in enumerate(zip(run_a, run_b)):
        if ya is None or yb is None or ya.shape != yb.shape:
            rows.append({"i": i, "key": ka, "note": "no comparable tensor"})
            continue
        rel = ((ya - yb).norm() / ya.norm().clamp_min(1e-6)).item()
        row = {"i": i, "key": ka, "rel_err": round(rel, 6)}
        rows.append(row)
        if first_bad is None and rel > 1e-2:
            first_bad = row
        print(json.dumps(row), flush = True)

    print("FIRST_DIVERGENT:", json.dumps(first_bad))
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"length": N, "modules": rows, "first_divergent": first_bad}, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
