#!/usr/bin/env python3
"""KL divergence parity check for DeepSeek-V4.1 with reference logits."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
sys.path.insert(0, EXL3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/tp1/v41port/model"), help="Model directory")
    parser.add_argument("--reference", type=str, required=True, help="Reference safetensors file with logprobs")
    parser.add_argument("--engram-rows", type=str, default=os.path.expanduser("~/tp1/e2e/engram-rows-4k.safetensors"))
    parser.add_argument("--split", type=int, default=256, help="EXL3_MOE_CPU_SPLIT")
    parser.add_argument("--max-seqs", type=int, help="Max sequences to process")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    # Set environment variables before importing exllamav3
    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", str(args.split))
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.engram_rows and os.path.isfile(args.engram_rows):
        os.environ["EXL3_ENGRAM_ROWS"] = args.engram_rows

    result = {"ok": False, "sequences": []}

    try:
        import torch
        from exllamav3 import Config, Model, Tokenizer
        from safetensors import safe_open
        from torch.nn.functional import log_softmax

        # Load reference data
        with safe_open(args.reference, "pt") as f:
            ref_tokens = f.get_tensor("tokens").long()
            ref_logprobs = f.get_tensor("logprobs").half()
            ref_seq_index = f.get_tensor("seq_index").long()

        S, N = ref_tokens.shape
        V = ref_logprobs.shape[-1]
        max_seqs = args.max_seqs or S

        # Load model
        t0 = time.time()
        cfg = Config.from_directory(args.model)
        model = Model.from_config(cfg)
        model.load("cuda:0", progressbar=False, verbose=False)
        tok = Tokenizer.from_config(cfg)
        device = torch.device("cuda:0")

        # Process sequences
        total_kl = 0.0
        total_top1 = 0
        total_nll_exl3 = 0.0
        total_nll_ref = 0.0
        num_tokens = 0

        for s in range(min(max_seqs, S)):
            ids = ref_tokens[s:s+1].long().to(device)
            seq_len = N

            t_fwd = time.time()
            with torch.inference_mode():
                logits = model.forward(ids, {"attn_mode": "flash_attn_nc", "position": 0})
            fwd_time = time.time() - t_fwd

            # Extract logits for positions 0 to N-2 (predicting tokens 1 to N-1)
            lp = log_softmax(logits[0, :N-1].float(), dim=-1)

            # Reference logprobs for this sequence
            ref = ref_logprobs[s, :N-1].to(device).float()

            # Per-token metrics
            kl = (ref.exp() * (ref - lp)).sum(dim=-1)
            top1 = (lp.argmax(dim=-1) == ref.argmax(dim=-1))

            # NLL for actual next tokens
            next_token_ids = ids[0, 1:N]
            nll_exl3 = -lp.gather(1, next_token_ids.unsqueeze(-1)).squeeze(-1)
            nll_ref = -ref.gather(1, next_token_ids.unsqueeze(-1)).squeeze(-1)

            seq_result = {
                "seq": s,
                "mean_kl": float(kl.mean()),
                "top1_agreement": float(top1.float().mean()),
                "nll_exl3": float(nll_exl3.mean()),
                "nll_ref": float(nll_ref.mean()),
                "fwd_s": round(fwd_time, 2),
            }
            result["sequences"].append(seq_result)

            total_kl += kl.sum()
            total_top1 += top1.sum()
            total_nll_exl3 += nll_exl3.sum()
            total_nll_ref += nll_ref.sum()
            num_tokens += N - 1

            torch.cuda.synchronize()

        # Overall metrics
        result["overall_mean_kl"] = float(total_kl / num_tokens) if num_tokens > 0 else 0.0
        result["overall_top1_agreement"] = float(total_top1 / num_tokens) if num_tokens > 0 else 0.0
        result["overall_nll_exl3"] = float(total_nll_exl3 / num_tokens) if num_tokens > 0 else 0.0
        result["overall_nll_ref"] = float(total_nll_ref / num_tokens) if num_tokens > 0 else 0.0
        result["num_tokens"] = num_tokens
        result["total_time_s"] = round(time.time() - t0, 1)

        # Check if all values are finite
        all_finite = bool(torch.isfinite(torch.tensor([
            result["overall_mean_kl"],
            result["overall_top1_agreement"],
            result["overall_nll_exl3"],
            result["overall_nll_ref"],
        ])).all())
        result["ok"] = all_finite

    except Exception:
        result["error"] = traceback.format_exc()[-2000:]
        traceback.print_exc()

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2)[:3000])
    print("PARITY_OK" if result.get("ok") else "PARITY_FAIL")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
