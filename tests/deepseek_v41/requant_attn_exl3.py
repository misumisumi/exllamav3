#!/usr/bin/env python3
"""
Requantize the fp16-loaded non-expert Linears of a DeepSeek-V4.1 EXL3 pack (attention projections kept
in FP8 source format, shared experts, Engram wkv) to EXL3 with calibrated Hessians.

The pack is loaded as usual; Hessians are captured on the bundled calibration corpus through generator
prefill, layer chunk by layer chunk, so each chunk calibrates on the already-quantized chunks before it
(the converter's ordering). Each chunk's tensors ({key}.trellis/suh/svh/mul1) go to
<out>/requant_partNN.safetensors; the loader prefers them over the source .weight/.scale when the files
sit in the model directory.

indexer.weights_proj stays fp16 (the DSA decode graph requires it). Linears whose input is provably
shared (shared expert w1/w3, compressor wkv/wgate) share one Hessian; everything else gets its own.

    EXL3_ATS_MMAP=1 EXL3_BC_DSA=0 python tests/deepseek_v41/requant_attn_exl3.py --model DIR --out DIR [--K 5]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules.linear import Linear
from exllamav3.conversion.calibration_data import get_default_calibration

TARGET = re.compile(r"^layers\.(\d+)\.(attn\..+|ffn\.shared_experts\.w[123]|engram\.wkv)$")
EXCLUDE = re.compile(r"\.indexer\.weights_proj$")


def site(key: str) -> str:
    m = re.match(r"^(layers\.\d+\.ffn\.shared_experts)\.w[13]$", key)
    if m:
        return m.group(1) + ".in"
    m = re.match(r"^(.+\.compressor)\.w(kv|gate)$", key)
    if m:
        return m.group(1) + ".in"
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--K", type = int, default = 5)
    ap.add_argument("--cal-rows", type = int, default = 20)
    ap.add_argument("--cal-cols", type = int, default = 2048)
    ap.add_argument("--h-budget-gib", type = float, default = 12.0)
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

    # Capture through normal inference: the generator's params carry no "capture" key
    cap = {"on": False, "H": {}}
    orig_fwd = Linear.forward
    def fwd(self, x, params, *a, **k):
        if cap["on"] and self.qmap:
            xc = x if x.shape[-1] >= self.in_features else torch.nn.functional.pad(x, (0, self.in_features - x.shape[-1]))
            self.capture_H(xc, {"capture": cap["H"]})
        return orig_fwd(self, x, params, *a, **k)
    Linear.forward = fwd

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cal_cols + 512, max_history = 1, max_batch_size = 1)
    t0 = time.time()
    model.load(dev, progressbar = False, verbose = False)
    Linear.load = orig_load
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = args.cal_cols)
    print(f"loaded in {time.time() - t0:.1f}s, {len(linears)} linears", flush = True)

    by_layer = {}
    for lin in linears:
        lin.qmap = None
        if not TARGET.match(lin.key) or EXCLUDE.search(lin.key) or lin.quant_type != "fp16":
            continue
        if lin.in_features % 16 or lin.out_features % 16:
            print(f" -- skip {lin.key}: {lin.in_features}x{lin.out_features} not multiples of 16", flush = True)
            continue
        by_layer.setdefault(int(TARGET.match(lin.key).group(1)), []).append(lin)

    chunks, cur, cur_bytes = [], [], 0
    for layer in sorted(by_layer):
        sites = {site(lin.key): lin.in_features for lin in by_layer[layer]}
        lb = sum(4 * d * d for d in sites.values())
        if cur and cur_bytes + lb > args.h_budget_gib * 2**30:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(layer)
        cur_bytes += lb
    if cur:
        chunks.append(cur)
    n_targets = sum(len(v) for v in by_layer.values())
    print(f"{n_targets} target linears in {len(by_layer)} layers, {len(chunks)} chunks: {[(c[0], c[-1]) for c in chunks]}", flush = True)
    print("example targets:", [lin.key for lin in by_layer[min(by_layer)]], flush = True)

    rows = get_default_calibration({"cal_cols": args.cal_cols, "cal_rows": args.cal_rows}, tokenizer)
    stats = []
    for ci, layers in enumerate(chunks):
        chunk_lins = [lin for layer in layers for lin in by_layer[layer]]
        for lin in chunk_lins:
            lin.qmap = site(lin.key)
        cap["H"] = {}
        cap["on"] = True
        t1 = time.time()
        for ri, row in enumerate(rows):
            ids = row.view(1, -1).clone()
            ids[0, 0] = 1000 + ci * 97 + ri   # unique first token: no prefix reuse between passes
            tr = time.time()
            generator.enqueue(Job(input_ids = ids, max_new_tokens = 1, stop_conditions = [], sampler = GreedySampler()))
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r.get("stage") == "error":
                        raise RuntimeError(str({k: v for k, v in r.items() if not torch.is_tensor(v) and k != "job"}))
            avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1048576
            print(f"  chunk {ci} row {ri + 1}/{len(rows)}: {time.time() - tr:.1f}s, MemAvailable {avail:.1f} GiB", flush = True)
        cap["on"] = False
        print(f"chunk {ci} layers {layers[0]}-{layers[-1]}: {len(cap['H'])} Hessians from {len(rows)} rows in {time.time() - t1:.0f}s", flush = True)

        # The generator runs under inference mode, so the captured tensors are inference tensors; the
        # quantizer updates H in place, which torch only allows on normal tensors
        for H_data in cap["H"].values():
            H_data["H"] = H_data["H"].clone()
            H_data["inf_nan"] = H_data["inf_nan"].clone()

        tensors = {}
        for lin in chunk_lins:
            H_data = cap["H"].get(lin.qmap)
            if H_data is None:
                H_data = lin.init_H_data(False)   # never reached the Python forward: uncalibrated fallback
            qa = {
                "seed": len(stats), "K": args.K, "devices": [0], "device_ratios": None,
                "apply_out_scales": None, "debug_dir": os.path.join(args.out, "debug"), "mul1": True,
            }
            t2 = time.time()
            if getattr(lin.inner, "lazy", False):
                # EXL3_FP8_LAZY=1: store the dequantized weight once, contiguous, for the quantizer
                lin.inner.weight = lin.inner.weight.contiguous()
            proxy = lin.convert_exl3(H_data, qa)
            for k, v in lin.get_tensors().items():
                tensors[k] = v.detach().to("cpu").contiguous()
            rec = {
                "key": lin.key, "K": args.K, "in": lin.in_features, "out": lin.out_features,
                "proxy_err": round(float(proxy), 6), "fallback": bool(qa.get("q_fallback")),
                "rows": int(H_data.get("count", 0)), "s": round(time.time() - t2, 1),
            }
            stats.append(rec)
            print(json.dumps(rec), flush = True)
            lin.qmap = None
        cap["H"] = {}
        torch.cuda.empty_cache()
        path = os.path.join(args.out, f"requant_part{ci:02d}.safetensors")
        save_file(tensors, path, metadata = {"format": "pt"})
        with open(os.path.join(args.out, "requant_stats.json"), "w") as f:
            json.dump(stats, f, indent = 1)
        print(f"chunk {ci} saved {len(tensors)} tensors to {path}", flush = True)

    print(f"REQUANT_OK {len(stats)} linears, fallback {sum(r['fallback'] for r in stats)}", flush = True)


if __name__ == "__main__":
    main()
