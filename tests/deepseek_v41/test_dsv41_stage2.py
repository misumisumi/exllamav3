"""Stage-2 checks: DSV41Attention construct/load, compressor vs ref math, nc forward."""
from __future__ import annotations

import json
import os
import sys
import time

import torch

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tp1/v41port/model")
sys.path.insert(0, EXL3)

from exllamav3 import Config
from exllamav3.modules.dsv41 import DSV41Attention

torch.manual_seed(0)


def make_attn(cfg, idx):
    cand = None
    if idx == cfg.candidate_source_layer_id:
        cand = "source"
    elif cfg.candidate_source_layer_id >= 0 and idx > cfg.candidate_source_layer_id \
            and idx in cfg.index_source_layer_ids:
        cand = "consumer"
    return DSV41Attention(
        config=cfg, key=f"layers.{idx}.attn", layer_idx=idx,
        compress_rate=cfg.compress_ratios[idx],
        is_kv_source=idx in cfg.kv_source_layer_ids,
        is_index_source=idx in cfg.index_source_layer_ids,
        kv_source_layer=cfg.kv_source_of[idx],
        index_source_layer=cfg.index_source_of[idx],
        candidate_role=cand,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        hidden_size=cfg.hidden_size, num_q_heads=cfg.num_q_heads, head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim, q_lora_rank=cfg.q_lora_rank,
        o_groups=cfg.o_groups, o_lora_rank=cfg.o_lora_rank, sliding_window=cfg.sliding_window,
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim, index_topk=cfg.index_topk,
        rope_theta=cfg.rope_theta, compress_rope_theta=cfg.compress_rope_theta,
        rope_scaling=cfg.rope_scaling, rms_norm_eps=cfg.rms_norm_eps,
        qmap="block.attn", out_dtype=torch.float,
    )


def dequant_linear(stc, key, device):
    w = stc.get_tensor(f"{key}.weight", device, no_defer=True)
    s = stc.get_tensor(f"{key}.scale", device, optional=True, no_defer=True)
    if s is None:
        return w.float()
    wf, sf = w.float(), s
    rows, cols = wf.shape
    sr, sc = sf.shape
    scv = (sf.float() - 127.0).exp2() if sf.dtype == torch.uint8 else sf.float()
    return (wf.view(sr, rows // sr, sc, cols // sc) * scv.view(sr, 1, sc, 1)).view(rows, cols)


def main():
    t0 = time.time()
    cfg = Config.from_directory(MODEL)
    dev = torch.device("cuda:0")
    layers = [2, 3, 20, 24]
    attns = {}
    for idx in layers:
        a = make_attn(cfg, idx)
        a.load(dev)
        attns[idx] = a
        print(f"loaded L{idx} kv={a.is_kv_source} idx={a.is_index_source} owns_k={a.owns_index_keys} "
              f"rate={a.compress_rate} cand={a.candidate_role} q_a={a.q_a.quant_type}")
    print(f"load {time.time()-t0:.1f}s")

    # ---- compressor vs torch math (layer 2 ratio-2, layer 20 ratio-1)
    report = {}
    x = torch.randn(1, 16, cfg.hidden_size, device=dev, dtype=torch.half) * 0.2
    params = {}
    for idx, ntok in ((2, 16), (20, 16)):
        a = attns[idx]
        m = a.compress_rate
        n = ntok // m
        kv_rows, gate_rows = a.compressor.project(x[:, :n * m], params)
        got = a.compressor.pool(kv_rows, gate_rows, params).float()
        W = dequant_linear(cfg.stc, f"layers.{idx}.attn.compressor.wkv", dev)
        # LinearFP16: weight is (in, out) after transposed_load
        # project() is Linear.forward(x) -> x @ W
        xf = x[:, :n * m].float()
        # recover via the module outputs vs reconstructed GEMM
        w_mod = a.compressor.wkv.inner.weight.float()
        kv_ref = xf @ w_mod
        if a.compressor.gated:
            g_mod = a.compressor.wgate.inner.weight.float()
            gate_ref = xf @ g_mod
            kv_g = kv_ref.view(1, n, m, a.head_dim)
            gate_g = gate_ref.view(1, n, m, a.head_dim).softmax(2)
            pooled = (kv_g * gate_g).sum(2)
        else:
            pooled = kv_ref.view(1, n, m, a.head_dim).sum(2)
        # RMSNorm
        nw = a.compressor.norm.weight.float()
        eps = cfg.rms_norm_eps
        rstd = torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps)
        ref = (pooled * rstd * nw).half().float()
        d = (got - ref).abs().max().item()
        report[f"compressor_L{idx}_max_abs"] = d
        assert d < 2e-2, f"compressor L{idx} drift {d}"

    # ---- nc forward: source then consumer (must not crash; finite output)
    x2 = torch.randn(1, 32, cfg.hidden_size, device=dev, dtype=torch.half) * 0.2
    p = {"attn_mode": "flash_attn_nc", "position": 0}
    with torch.inference_mode():
        y2 = attns[2].forward(x2, p)
        y3 = attns[3].forward(x2, p)  # reuses L2 nc pack
        y20 = attns[20].forward(x2, p)
        y24 = attns[24].forward(x2, p)
    for name, y in ("L2", y2), ("L3", y3), ("L20", y20), ("L24", y24):
        assert torch.isfinite(y).all(), f"{name} nc produced non-finite"
        report[f"{name}_nc_abs_mean"] = y.float().abs().mean().item()
        report[f"{name}_nc_shape"] = list(y.shape)

    report["seconds"] = time.time() - t0
    print(json.dumps(report, indent=1))
    print("STAGE2_OK")
    for a in attns.values():
        a.unload()


if __name__ == "__main__":
    main()
