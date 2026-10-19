"""Stage-1 Engram checks: hasher vs DeepSeek NgramHashState, prefill/decode, layer vs ref math."""
from __future__ import annotations

import json
import os
import sys
import time
import types

import torch

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
REF = os.path.expanduser("~/tp1/metadata/source/inference")
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tp1/v41port/model")
TOK = os.path.expanduser("~/tp1/metadata/source/tokenizer.json")
CACHE = os.path.expanduser("~/tp1/v41port/cache")

sys.path.insert(0, EXL3)
from exllamav3 import Config
from exllamav3.modules.engram import DEAD, EngramHasher, EngramLayer

torch.manual_seed(0)


def _ref_hasher(cfg):
    sys.path.insert(0, REF)
    from engram import EngramLayout, NgramHashState  # noqa: E402
    from transformers import PreTrainedTokenizerFast  # noqa: E402

    tok = PreTrainedTokenizerFast(tokenizer_file=TOK)
    args = types.SimpleNamespace(
        engram_layer_ids=tuple(cfg.engram_layer_ids),
        engram_num_embeddings=tuple(cfg.engram_num_embeddings),
        engram_max_ngram_size=cfg.engram_max_ngram_size,
        engram_vocab_size=cfg.engram_vocab_size,
        engram_n_heads=cfg.engram_n_heads,
        engram_head_dim=cfg.engram_head_dim,
        engram_pad_id=cfg.engram_pad_token_id,
        engram_compressed_vocab_size=cfg.engram_compressed_vocab_size,
        max_batch_size=4,
        max_seq_len=64,
    )
    layout = EngramLayout.from_args(args)
    return NgramHashState(args, layout, tok)


def main():
    t0 = time.time()
    cfg = Config.from_directory(MODEL)
    hasher = EngramHasher(cfg, tokenizer_json=TOK, cache_dir=CACHE)
    print(f"config {type(cfg).__name__} hasher in {time.time()-t0:.1f}s compressed_vocab ok")

    bsz, seq = 2, 16
    ids = torch.randint(0, min(cfg.vocab_size, 4000), (bsz, seq))
    # sprinkle a dead image span in the middle of row 0
    mask = torch.ones((bsz, seq), dtype=torch.bool)
    mask[0, 5:8] = False

    # --- hasher vs reference NgramHashState (prefill + dead spans)
    ref = _ref_hasher(cfg)
    with torch.inference_mode():
        ours = hasher.hash_window(
            torch.cat((torch.full((bsz, hasher.context_len), DEAD, dtype=torch.long),
                       hasher.compress(ids, token_mask=mask)), 1),
            0,
        )
        theirs = ref(ids, 0, mask)
    n_mismatch = int((ours != theirs).sum())
    print(json.dumps({"hasher_vs_ref_mismatches": n_mismatch, "ours_shape": list(ours.shape),
                      "theirs_shape": list(theirs.shape), "hash_max": int(ours.max())}))
    assert n_mismatch == 0, f"hasher != NgramHashState at {n_mismatch} cells"

    # --- prefill hashes vs token-by-token decode
    with torch.inference_mode():
        full = hasher.hash_window(
            torch.cat((torch.full((bsz, hasher.context_len), DEAD, dtype=torch.long),
                       hasher.compress(ids)), 1),
            0,
        )
        step_chunks = []
        ctx = torch.full((bsz, hasher.context_len), DEAD, dtype=torch.long)
        for t in range(seq):
            tok = hasher.compress(ids[:, t:t + 1])
            window = torch.cat((ctx, tok), 1)
            step_chunks.append(hasher.hash_window(window, t))
            ctx = torch.cat((ctx, tok), 1)[:, -hasher.context_len:]
        stepped = torch.cat(step_chunks, dim=1)
    n_step = int((full != stepped).sum())
    print(json.dumps({"prefill_vs_decode_hash_mismatches": n_step}))
    assert n_step == 0

    # --- layer vs reference math + dead-token gate shutoff
    dev = torch.device("cuda:0")
    layer = EngramLayer(cfg, key="layers.1.engram", layer_idx=-2, table_index=0, hasher=hasher,
                        hidden_size=cfg.hidden_size, hc_mult=cfg.hc_mult, rms_norm_eps=cfg.rms_norm_eps,
                        gather_threads=48)
    t1 = time.time()
    layer.load(dev)
    print(f"loaded layer in {time.time()-t1:.1f}s quant={layer.wkv.quant_type}")

    x = torch.randn(bsz, seq, cfg.hc_mult, cfg.hidden_size, device=dev, dtype=torch.float) * 0.5
    params = {"input_ids": ids, "position": 0, "token_mask": mask}
    with torch.inference_mode():
        out = layer.forward(x, params)
        torch.cuda.synchronize()
        # dead positions must be pass-through
        dead_delta = (out - x)[0, 5:8].abs().max().item()
        live_delta = (out - x)[1].abs().max().item()

        window = torch.cat((torch.full((bsz, hasher.context_len), DEAD, dtype=torch.long),
                            hasher.compress(ids, token_mask=mask)), 1)
        hash_ids = hasher.hash_window(window, 0)[:, :, 0, :]
        rows = layer.table.rows(hash_ids, dev).view(bsz, seq, -1)
        stc = cfg.stc
        w8 = stc.get_tensor("layers.1.engram.wkv.weight", dev, no_defer=True)
        s8 = stc.get_tensor("layers.1.engram.wkv.scale", dev, no_defer=True)
        s = torch.exp2(s8.float() - 127.0) if s8.dtype == torch.uint8 else s8.float()
        W = (w8.float().view(800, 32, 192, 32) * s.view(800, 1, 192, 1)).view(25600, 6144)
        q = stc.get_tensor("layers.1.engram.q_weight", dev, allow_bf16=True, no_defer=True).float()
        k = stc.get_tensor("layers.1.engram.k_weight", dev, allow_bf16=True, no_defer=True).float()
        H, D = cfg.hc_mult, cfg.hidden_size
        kv = rows.float() @ W.t()
        key, value = kv[..., :H * D].view(bsz, seq, H, D), kv[..., H * D:]
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + cfg.rms_norm_eps) * torch.rsqrt(key.square().mean(-1) + cfg.rms_norm_eps)
        dot = (h * (q * k) * key).sum(-1) * rstd * D ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~mask.to(dev).unsqueeze(-1), 0)
        ref_out = h + gate.unsqueeze(-1) * value.unsqueeze(-2)
        d1 = (out - ref_out).abs().max().item()
        rel = d1 / ((out - x).abs().max().item() + 1e-9)

        # decode: last token of a 1-token step at pos=seq-1 vs prefill at that index
        last = ids[:, -1:]
        ctx = hasher.compress(ids[:, :-1])[:, -hasher.context_len:]
        if ctx.shape[1] < hasher.context_len:
            ctx = torch.cat((torch.full((bsz, hasher.context_len - ctx.shape[1]), DEAD, dtype=torch.long), ctx), 1)
        window_d = torch.cat((ctx, hasher.compress(last)), 1)
        hid = hasher.hash_window(window_d, seq - 1)[:, :, 0, :]
        x_last = x[:, -1:]
        delta_d = layer.forward_streams(x_last, hid, {})
        d_dec = (delta_d - (out - x)[:, -1:]).abs().max().item()

    print(json.dumps({
        "dead_delta_max": dead_delta, "live_delta_max": live_delta,
        "tier1_max_abs": d1, "tier1_rel_to_delta": rel,
        "decode_vs_prefill_last_max_abs": d_dec,
        "seconds": time.time() - t0,
    }, indent=1))
    assert dead_delta < 1e-6, f"token_mask did not shut the gate: {dead_delta}"
    assert rel < 1e-2, f"layer drifted from reference math: rel={rel}"
    # batched prefill GEMM vs a 1-token decode GEMM in fp16; ~1e-3 abs is expected
    assert d_dec < 1e-2, f"decode step != prefill last token: {d_dec}"
    layer.unload()
    print("STAGE1_OK")


if __name__ == "__main__":
    main()
