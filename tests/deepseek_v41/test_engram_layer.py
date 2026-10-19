"""Stage-1 check: exllamav3 EngramLayer vs the reference Engram math with the real layer weights.

Tier 1 ("wiring"): same gathered rows, plain fp32 torch math for wkv and the gate. Should agree
to fp16-GEMM precision. Tier 2 ("reference numerics"): the reference quantizes the wkv input
per 32 elements with power-of-two e8m0 scales before its fp8 GEMM; measured as a second delta.
"""
import json, os, sys, time
import torch

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
MODEL = sys.argv[1]
sys.path.insert(0, EXL3)
from exllamav3 import Config
from exllamav3.modules.engram import EngramHasher, EngramLayer, DEAD

torch.manual_seed(0)
dev = torch.device("cuda:0")
t0 = time.time()
cfg = Config.from_directory(MODEL)
print("config:", type(cfg).__name__, "arch", cfg.arch_string, "engram layers", cfg.engram_layer_ids,
      "ratios", cfg.compress_ratios[:3], "...", "kv_source_of[21]", cfg.kv_source_of[21], "index_source_of[25]", cfg.index_source_of[25])
hasher = EngramHasher(cfg, tokenizer_json = os.path.expanduser("~/tp1/metadata/source/tokenizer.json"),
                      cache_dir = os.path.expanduser("~/tp1/v41port/cache"))
layer = EngramLayer(cfg, key = "layers.1.engram", layer_idx = -2, table_index = 0, hasher = hasher,
                    hidden_size = cfg.hidden_size, hc_mult = cfg.hc_mult, rms_norm_eps = cfg.rms_norm_eps, gather_threads = 48)
layer.load(dev)
print(f"loaded wkv ({layer.wkv.quant_type}) + tables in {time.time() - t0:.1f} s; wkv weight dtype/shape:",
      getattr(layer.wkv.inner, 'weight', torch.empty(0)).dtype, tuple(getattr(layer.wkv.inner, 'weight', torch.empty(0)).shape))

bsz, seq = 2, 12
ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
x = torch.randn(bsz, seq, cfg.hc_mult, cfg.hidden_size, device = dev, dtype = torch.float) * 0.5
params = {"input_ids": ids, "position": 0}
t1 = time.time()
with torch.inference_mode():
    out = layer.forward(x, params)
torch.cuda.synchronize()
print(f"forward {bsz}x{seq} tokens ({bsz * seq * hasher.n_cols} rows) in {time.time() - t1:.1f} s")

# ---- reference math on the same rows
with torch.inference_mode():
    window = torch.cat((torch.full((bsz, hasher.context_len), DEAD, dtype = torch.long), hasher.compress(ids)), 1)
    hash_ids = hasher.hash_window(window, 0)[:, :, 0, :]
    rows = layer.table.rows(hash_ids, dev).view(bsz, seq, -1)                       # (bsz, seq, 6144) fp32
    stc = cfg.stc
    w8 = stc.get_tensor("layers.1.engram.wkv.weight", dev, no_defer = True)            # fp8 (25600, 6144)
    s8 = stc.get_tensor("layers.1.engram.wkv.scale", dev, no_defer = True)             # e8m0 (800, 192)
    print("wkv raw:", w8.dtype, tuple(w8.shape), s8.dtype, tuple(s8.shape))
    s = torch.exp2(s8.float() - 127.0) if s8.dtype == torch.uint8 else s8.float()
    W = (w8.float().view(800, 32, 192, 32) * s.view(800, 1, 192, 1)).view(25600, 6144)
    q = stc.get_tensor("layers.1.engram.q_weight", dev, allow_bf16 = True, no_defer = True).float()
    k = stc.get_tensor("layers.1.engram.k_weight", dev, allow_bf16 = True, no_defer = True).float()
    H, D = cfg.hc_mult, cfg.hidden_size

    def ref(kv_in):
        kv = kv_in @ W.t()
        key, value = kv[..., :H * D].view(bsz, seq, H, D), kv[..., H * D:]
        weight = q * k
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + cfg.rms_norm_eps) * torch.rsqrt(key.square().mean(-1) + cfg.rms_norm_eps)
        dot = (h * weight * key).sum(-1) * rstd * D ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.unsqueeze(-1) * value.unsqueeze(-2), gate

    ref1, gate1 = ref(rows.to(torch.bfloat16).float())        # tier 1: rows as the reference's bf16 embedding output
    # tier 2: reference fp8 activation quant (per 32, power-of-two e8m0 scale, amax >= 1e-4)
    a = rows.to(torch.bfloat16).float().view(bsz, seq, -1, 32)
    amax = a.abs().amax(-1, keepdim = True).clamp_min(1e-4)
    sc = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    aq = ((a / sc).to(torch.float8_e4m3fn).float() * sc).view(bsz, seq, -1)
    ref2, gate2 = ref(aq)

    delta_layer = out - x
    d1 = (out - ref1).abs().max().item(); rel1 = d1 / (delta_layer.abs().max().item() + 1e-9)
    d2 = (out - ref2).abs().max().item(); rel2 = d2 / (delta_layer.abs().max().item() + 1e-9)
    print(json.dumps({"delta_max": delta_layer.abs().max().item(), "delta_mean": delta_layer.abs().mean().item(),
                      "tier1_max_abs": d1, "tier1_rel_to_delta": rel1, "tier2_max_abs": d2, "tier2_rel_to_delta": rel2,
                      "gate_mean": gate1.mean().item(), "gate_min": gate1.min().item(), "gate_max": gate1.max().item(),
                      "hash_ids_max": int(hash_ids.max()), "rows_abs_mean": rows.abs().mean().item()}, indent = 1))
layer.unload()
