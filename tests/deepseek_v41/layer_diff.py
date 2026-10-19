#!/usr/bin/env python3
"""DeepSeek-V4.1: compare exllamav3 MoE inputs with the reference harness, layer by layer.

The reference dumps come from `sage_exl3 v41 e2e-run --dump-moe-input`. Each layer file holds
`hidden` [S*N, D] (the normed FFN input), `topk_ids` [S*N, 6] and `topk_weights` [S*N, 6], rows
ordered by sequence then position, for the sequences whose tokens are in the teacher log-prob dump.

Routing is recomputed from both inputs with the reference gate: scores = sqrt(softplus(x @ W.T)),
experts = topk(scores + bias), weights = scores[experts] normalized and scaled by route_scale.
The formula check (reference input vs dumped ids) should be ~1.0; the first layer whose relative
error jumps is where the port diverges.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
if os.path.isdir(EXL3):
    sys.path.insert(0, EXL3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/tp1/v41port/model"))
    p.add_argument("--reference-logprobs", default=os.path.expanduser("~/tp1/v41port/ref_logprobs_4x512.safetensors"))
    p.add_argument("--dump-dir", default=os.path.expanduser("~/tp1/v41port/ref_moe_in"))
    p.add_argument("--layers", default="0,1,2,3,8,14,20,21,24,39")
    p.add_argument("--engram-rows", default=os.path.expanduser("~/tp1/e2e/engram-rows-4k.safetensors"))
    p.add_argument("--split", type=int, default=256)
    p.add_argument("--max-seqs", type=int, default=None)
    p.add_argument("--route-scale", type=float, default=1.5)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", str(args.split))
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not os.path.isfile(args.engram_rows):
        raise SystemExit(f"engram rows file not found: {args.engram_rows}")
    os.environ["EXL3_ENGRAM_ROWS"] = args.engram_rows

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from exllamav3 import Config, Model
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

    layers = [int(v) for v in args.layers.split(",")]
    with safe_open(args.reference_logprobs, "pt") as f:
        tokens = f.get_tensor("tokens").long()
    S, N = tokens.shape
    if args.max_seqs:
        S = min(S, args.max_seqs)

    key_to_layer = {f"layers.{l}.ffn": l for l in layers}
    captured: dict[int, list] = {l: [] for l in layers}
    original_forward = BlockSparseMLP.forward

    def hooked_forward(self, x, params, out_dtype = None):
        layer = key_to_layer.get(self.key)
        if layer is not None:
            captured[layer].append(x.detach().reshape(-1, x.shape[-1]).float().cpu())
        return original_forward(self, x, params, out_dtype)

    BlockSparseMLP.forward = hooked_forward

    t0 = time.time()
    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    model.load("cuda:0", progressbar = False, verbose = False)
    load_s = time.time() - t0
    dev = torch.device("cuda:0")

    fwd_s = []
    for s in range(S):
        t1 = time.time()
        with torch.inference_mode():
            model.forward(tokens[s:s + 1].to(dev), {"attn_mode": "flash_attn_nc", "position": 0})
        torch.cuda.synchronize()
        fwd_s.append(round(time.time() - t1, 2))

    def route(h, weight, bias):
        scores = F.softplus(F.linear(h, weight)).sqrt()
        ids = (scores + bias).topk(6, dim = -1).indices
        w = scores.gather(1, ids)
        w = w / w.sum(dim = -1, keepdim = True) * args.route_scale
        return ids, w

    def overlap(a, b):
        return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1) / a.shape[-1]

    rows = []
    for layer in layers:
        path = Path(args.dump_dir) / f"moe-input-L{layer:02d}.safetensors"
        entry = {"layer": layer}
        if not path.exists():
            entry["error"] = "no reference dump"
            rows.append(entry)
            continue
        with safe_open(str(path), "pt") as f:
            ref_h = f.get_tensor("hidden")[: S * N].float()
            ref_ids = f.get_tensor("topk_ids")[: S * N].long()
            ref_w = f.get_tensor("topk_weights")[: S * N].float()
        if len(captured[layer]) != S:
            entry["error"] = f"captured {len(captured[layer])} calls, expected {S}"
            rows.append(entry)
            continue
        exl_h = torch.cat(captured[layer], 0)
        if exl_h.shape != ref_h.shape:
            entry["error"] = f"shape exl {tuple(exl_h.shape)} vs ref {tuple(ref_h.shape)}"
            rows.append(entry)
            continue

        rel = (exl_h - ref_h).norm(dim = -1) / ref_h.norm(dim = -1).clamp_min(1e-6)
        cos = F.cosine_similarity(exl_h, ref_h, dim = -1)
        per_seq = rel.view(S, N).mean(-1)

        weight = cfg.stc.get_tensor(f"layers.{layer}.ffn.gate.weight", dev).float()
        bias = cfg.stc.get_tensor(f"layers.{layer}.ffn.gate.bias", dev).float()
        exl_ids, _ = route(exl_h.to(dev), weight, bias)
        form_ids, form_w = route(ref_h.to(dev), weight, bias)
        ref_ids_d = ref_ids.to(dev)
        entry.update({
            "rel_err_mean": round(rel.mean().item(), 5),
            "rel_err_median": round(rel.median().item(), 5),
            "rel_err_p90": round(torch.quantile(rel, 0.9).item(), 5),
            "rel_err_pos0": round(rel.view(S, N)[:, 0].mean().item(), 5),
            "rel_err_per_seq": [round(v, 5) for v in per_seq.tolist()],
            "rel_err_by_pos_bucket": {
                f"{a}-{b - 1}": round(rel.view(S, N)[:, a:min(b, N)].mean().item(), 5)
                for a, b in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128),
                             (128, 129), (129, 256), (256, 512)) if a < N
            },
            "cosine_mean": round(cos.mean().item(), 5),
            "ref_norm_mean": round(ref_h.norm(dim = -1).mean().item(), 3),
            "exl_norm_mean": round(exl_h.norm(dim = -1).mean().item(), 3),
            "routing_overlap": round(overlap(exl_ids, ref_ids_d).mean().item(), 5),
            "formula_check": round(overlap(form_ids, ref_ids_d).mean().item(), 5),
            "formula_weight_err": round((form_w.sort(-1).values - ref_w.to(dev).sort(-1).values).abs().max().item(), 6),
        })
        rows.append(entry)
        del weight, bias

    result = {"ok": all("error" not in r for r in rows), "load_s": round(load_s, 1), "forward_s": fwd_s,
              "seqs": S, "positions": N, "split": args.split, "layers": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent = 2))

    print(f"{'layer':>5} {'rel_mean':>9} {'rel_med':>9} {'rel_p90':>9} {'pos0':>8} {'cos':>8} "
          f"{'|ref|':>9} {'|exl|':>9} {'route':>7} {'formula':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['layer']:>5} ERROR {r['error']}")
            continue
        print(f"{r['layer']:>5} {r['rel_err_mean']:>9.5f} {r['rel_err_median']:>9.5f} {r['rel_err_p90']:>9.5f} "
              f"{r['rel_err_pos0']:>8.5f} {r['cosine_mean']:>8.5f} {r['ref_norm_mean']:>9.3f} {r['exl_norm_mean']:>9.3f} "
              f"{r['routing_overlap']:>7.4f} {r['formula_check']:>7.4f}")
    print("LAYER_DIFF_OK" if result["ok"] else "LAYER_DIFF_FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
