"""
Mixed-K MoE group graphs (EXL3_MOE_GROUP_GRAPH=1) against the per-expert Python loop, per layer at
bsz 1..MAX_BSZN, through the eager, record and replay calls of each graph. Run with EXL3_INT8_GEMV=0
so the loop uses the exact GEMV kernel too.

    EXL3_MOE_GROUP_GRAPH=1 EXL3_INT8_GEMV=0 MODEL_DIR=/path/to/model \
        python tests/deepseek_v41/moe_group_graph_check.py

Prints one JSON line per layer and GROUP_GRAPH_OK or GROUP_GRAPH_FAIL (exit code 1).
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, MAX_BSZN, _GROUPED_STATS

assert os.environ.get("EXL3_MOE_GROUP_GRAPH") in ("1", "2"), "set EXL3_MOE_GROUP_GRAPH=1 or 2"
M = os.environ["MODEL_DIR"]
TOL = float(os.environ.get("TOL", "5e-3"))
LAYERS = int(os.environ.get("LAYERS", "8"))

config = Config.from_directory(M)
model = Model.from_config(config)
model.load("cuda:0", progressbar = False, verbose = False)

def n_groups(m):
    return len(m.bc_groups) if m.bc_groups is not None else m.bc_mixedk_groups


mods = sorted((m for m in gc.get_objects() if isinstance(m, BlockSparseMLP) and
               (m.bc_groups is not None or m.bc_mixedk is not None)), key = lambda m: m.key)
print(json.dumps({"group_graph_layers": len(mods), "mode": os.environ["EXL3_MOE_GROUP_GRAPH"],
                  "groups_per_layer": sorted({n_groups(m) for m in mods})}), flush = True)
if not mods:
    print("no mixed-K layer built group graphs GROUP_GRAPH_FAIL", flush = True)
    sys.exit(1)
if len(mods) > LAYERS:
    mods = [mods[i] for i in sorted({round(j * (len(mods) - 1) / (LAYERS - 1)) for j in range(LAYERS)})]


def forward(m, x, ids):
    err = None
    for dev in (m.device, "cpu"):
        try:
            return m.forward(x, {"input_ids": ids.to(dev)}).float().clone()
        except Exception as e:
            err = e
    raise err


torch.manual_seed(0)
ok = True
worst = 0.0
with torch.inference_mode():
    for m in mods:
        H = m.hidden_size
        rec = {"layer": m.key, "groups": n_groups(m), "max_rel": 0.0, "not_taken": 0, "nondet": 0, "nonfinite": 0}
        for bsz in sorted({1, 2, 3, 5, MAX_BSZN}):
            for rep in range(3):
                x = torch.randn((1, bsz, H), dtype = torch.half, device = m.device)
                ids = torch.randint(0, 1000, (1, bsz), dtype = torch.long)
                for o in m.bc_group_outs:
                    o.fill_(float("nan"))
                calls = _GROUPED_STATS["group_graph_calls"]
                yg = forward(m, x, ids)
                if _GROUPED_STATS["group_graph_calls"] == calls:
                    rec["not_taken"] += 1
                if rep == 2 and not torch.equal(yg, forward(m, x, ids)):
                    rec["nondet"] += 1
                saved = (m.bc_groups, m.bc_mixedk)
                m.bc_groups = m.bc_mixedk = None
                try:
                    yr = forward(m, x, ids)
                finally:
                    m.bc_groups, m.bc_mixedk = saved
                if not bool(torch.isfinite(yg).all()):
                    rec["nonfinite"] += 1
                rel = float((yg - yr).norm() / (yr.norm() + 1e-9))
                if rel > rec["max_rel"]:
                    rec["worst_case"] = {"bsz": bsz, "rep": rep, "ref_norm": round(float(yr.norm()), 3),
                                         "max_abs_diff": round(float((yg - yr).abs().max()), 4),
                                         "ref_max_abs": round(float(yr.abs().max()), 3)}
                rec["max_rel"] = max(rec["max_rel"], rel)
        rec["max_rel"] = float(f"{rec['max_rel']:.3e}")
        worst = max(worst, rec["max_rel"])
        if rec["max_rel"] > TOL or rec["not_taken"] or rec["nondet"] or rec["nonfinite"]:
            ok = False
        print(json.dumps(rec), flush = True)

print(json.dumps({"worst_rel": worst, "tol": TOL}), "GROUP_GRAPH_OK" if ok else "GROUP_GRAPH_FAIL", flush = True)
sys.exit(0 if ok else 1)
