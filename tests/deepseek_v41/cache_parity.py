#!/usr/bin/env python3
"""DeepSeek-V4.1 cached path (attn_mode flash_attn: rings, pools, DSA kernels) vs the stateless nc
path, following tests/test_dsv4_cached.py: the recurrent state owns its pages (params carry only
attn_mode and recurrent_states), the state is advanced manually after each chunk, and only the
LAST 32 positions of every run are compared, because nc drops trailing sub-window compressor rows
per chunk so only the final positions see identical entry sets. Tolerances come from a noise floor
(fp16-scale perturbation of the embedding output through the same nc path). Also checks
rewind-and-replay exactness and reports KL of the cached tail against the FP4 reference log-probs.
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
    p.add_argument("--tail", type = int, default = 32)
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

    with safe_open(args.reference, "pt") as f:
        ids = f.get_tensor("tokens").long()[args.seq : args.seq + 1]
        ref_lp = f.get_tensor("logprobs")[args.seq].float()          # [N-1, V], row t predicts t+1
    N, T = ids.shape[1], args.tail

    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    model.load("cuda:0", progressbar = False, verbose = False)
    dev = torch.device("cuda:0")
    ids = ids.to(dev)

    def fwd(x, params):
        with torch.inference_mode():
            return model.forward(x, params)[0].float().cpu()

    def kl_top1(got, ref):
        lp_r = torch.log_softmax(ref.double(), -1)
        lp_g = torch.log_softmax(got.double(), -1)
        kl = (lp_r.exp() * (lp_r - lp_g)).sum(-1)
        return kl.mean().item(), kl.max().item(), (got.argmax(-1) == ref.argmax(-1)).float().mean().item()

    ref = fwd(ids, {"attn_mode": "flash_attn_nc", "position": 0})

    # Noise floor: perturb the first module's output at fp16 tiling scale, same nc path
    first = model.modules[0]
    orig_forward = first.forward
    def noisy(x, params, out_dtype = None):
        y = orig_forward(x, params, out_dtype) if out_dtype is not None else orig_forward(x, params)
        return y + torch.randn_like(y) * 2e-4
    first.forward = noisy
    torch.manual_seed(21)
    floor = fwd(ids, {"attn_mode": "flash_attn_nc", "position": 0})
    first.forward = orig_forward
    floor_kl, _, floor_am = kl_top1(floor[-T:], ref[-T:])
    kl_tol = max(5e-4, 1.5 * floor_kl)
    arg_tol = min(0.99, floor_am - 0.05)
    print(f"noise floor: KL {floor_kl:.6f} argmax {floor_am * 100:.1f}% -> tolerances KL {kl_tol:.6f} "
          f"argmax {arg_tol * 100:.1f}%", flush = True)

    from exllamav3.cache.recurrent_util import prepare_for_recurrence
    from exllamav3.modules.dsv41_hc import PRE_MIX_KEY

    def fwd_modules(xc, params):
        # tests/test_dsv4_cached.py drives the module list directly; V4.1 additionally needs the
        # recurrence slot index (Engram history) that prepare_for_recurrence derives from
        # batch_shape/past_len, and a fresh pre-mix per forward
        params["input_ids"] = xc
        params.pop(PRE_MIX_KEY, None)
        prepare_for_recurrence(xc, params, model)
        y = xc
        with torch.inference_mode():
            for m in model.modules:
                y = m.prepare_for_device(y, params)
                y = m.forward(y, params)
        return y[0].float().cpu()

    def run_cached(state, chunks, x):
        outs, a = [], 0
        for size in chunks:
            b = min(a + size, x.shape[1])
            if b <= a:
                break
            outs.append(fwd_modules(x[:, a:b], {"attn_mode": "flash_attn", "recurrent_states": [state],
                                                "batch_shape": (1, b - a), "past_len": state.position}))
            state.position += b - a
            state.post_advance()
            a = b
        return torch.cat(outs, dim = 0)

    rows, ok = [], True
    schedules = [
        ([N], "single chunk"),
        ([N // 2, N - N // 2], "two halves"),
        ([300, 100, N - 400], "uneven chunks"),
        ([N - T] + [1] * T, "prefill + decode steps"),
        ([N - T] + [16] * (T // 16), "prefill + 16-token chunks"),
    ]
    for chunks, tag in schedules:
        with torch.inference_mode():
            state = cache.get_new_state()
        got = run_cached(state, chunks, ids)
        state.free()
        kl_mean, kl_max, am = kl_top1(got[-T:], ref[-T:])
        # cached tail vs FP4 reference (rows N-1-T .. N-2 predict the last T tokens except the final one)
        lp_g = torch.log_softmax(got[-T - 1 : -1].double(), -1)
        lp_f = ref_lp[-T:].double()
        kl_fp4 = (lp_f.exp() * (lp_f - lp_g)).sum(-1).mean().item()
        lp_n = torch.log_softmax(ref[-T - 1 : -1].double(), -1)
        kl_fp4_nc = (lp_f.exp() * (lp_f - lp_n)).sum(-1).mean().item()
        # Gate on KL: over the last 32 positions argmax moves in 3.1-point steps, so it is reported
        # but not gated (arg_tol stays in the result for reference). The cached tail must also track
        # the FP4 reference about as well as nc does (0.05: over 32 positions a run with KL vs nc of
        # 0.003 still moved its FP4 KL by 0.024)
        passed = kl_mean < kl_tol and kl_fp4 <= kl_fp4_nc + 0.05
        ok &= passed
        row = {"schedule": tag, "pass": passed, "kl_vs_nc_mean": round(kl_mean, 6), "kl_vs_nc_max": round(kl_max, 6),
               "argmax_vs_nc": round(am, 4), "kl_vs_fp4_cached": round(kl_fp4, 5), "kl_vs_fp4_nc": round(kl_fp4_nc, 5)}
        rows.append(row)
        print(json.dumps(row), flush = True)

    with torch.inference_mode():
        state = cache.get_new_state()
    run_cached(state, [N - T], ids[:, : N - T])
    first_pass = run_cached(state, [1] * 12, ids[:, N - T : N - T + 12])
    state.rewind(12)
    second_pass = run_cached(state, [1] * 12, ids[:, N - T : N - T + 12])
    state.free()
    rewind_err = (first_pass - second_pass).abs().max().item()
    ok &= rewind_err == 0.0
    print(f"rewind-and-replay maxdiff {rewind_err:.3e}", flush = True)

    result = {"ok": ok, "noise_floor_kl": floor_kl, "noise_floor_argmax": floor_am, "kl_tol": kl_tol,
              "arg_tol": arg_tol, "schedules": rows, "rewind_maxdiff": rewind_err, "split": args.split}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent = 2)
    print("CACHE_PARITY_OK" if ok else "CACHE_PARITY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
