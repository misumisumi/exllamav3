from __future__ import annotations
from typing_extensions import override
import math
import torch
from ..model.config import Config
from .module import Module
from .linear import Linear
from .rmsnorm import RMSNorm
from .dsv4 import DSV4Attention, _ext_rope
from ..util.rope import yarn_inv_freq
from ..cache.dsa import DSV4LayerState, CacheLayer_dsa
from .attention_fn.dsa_triton import dsa_attn
from ..constants import PAGE_SIZE

"""
DeepSeek-V4.1 attention for compress_ratio > 0 layers (reference: inference/model.py
Attention / Compressor / Indexer / SharedAttentionRuntime).

Every layer keeps its own 128-token sliding window (ring) exactly like the V4 module. The
compressed path is shared: only the kv_source layers project their tokens into latents
(ratio 2: two tokens pooled with a softmax gate; ratio 1: one projection per token),
rope them at the group's first position with the compress rope table, and write them into
their paged pool; every later layer with the same ratio attends over that source's pool.
Only the index_source layers score entries (queries from their own indexer.wq_b and
weights_proj; keys come from the most recent key owner's pool_idx, written by the kv
sources from the pre-rope latent through indexer.wk + k_norm) and select the top
index_topk entries; consumers reuse the selection. Layer candidate_source additionally
picks candidate_topk_blocks blocks of candidate_block_size entries that later indexers
must select from. Hand-off between layers goes through the forward's params dict, which is
shared by all modules of one forward:

    ("v41_kl", src, inst)  -> (CacheLayer_dsa, block-table row)
    ("v41_idx", src, inst) -> (indices (seq, K_pad) int32 -1 padded, k_len)
    ("v41_cand", inst)     -> candidate mask (seq, ec) bool

The indexer and candidate selection run in plain torch (no fused kernels yet); the
attention itself is the shared dsa_attn kernel over [ring window ++ gathered entries].
"""


def fp8_qdq(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Reference act_quant(..., scale_fmt='ue8m0') quant-dequant: fp8 e4m3 values with one
    power-of-two scale per `block` elements (amax floor 1e-4)."""
    shape = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(-1, keepdim = True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return ((xf / s).to(torch.float8_e4m3fn).float() * s).reshape(shape).to(x.dtype)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_qdq(x: torch.Tensor, block: int, scale_e4m3: bool) -> torch.Tensor:
    """Reference fp4_act_quant quant-dequant: e2m1 values with per-`block` scales, either
    power-of-two (e8m0, indexer) or e4m3 (compressed KV)."""
    shape = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(-1, keepdim = True)
    if scale_e4m3:
        s = (amax.clamp_min(6.0 * 2.0 ** -9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 6.0)))
    grid = _E2M1.to(x.device)
    v = (xf / s)
    idx = (v.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    q = grid[idx] * v.sign()
    return (q * s).reshape(shape).to(x.dtype)


class DSV41Compressor:
    """Non-overlapping V4.1 compressor: ratio 2 pools token pairs with a per-channel softmax
    gate (wkv + wgate), ratio 1 is a plain projection (wkv only). No position bias."""

    def __init__(self, attn, key, head_dim, compress_rate, qmap, select_hq_bits):
        cfg = attn.config
        self.key = key
        self.head_dim = head_dim
        self.compress_rate = compress_rate
        self.gated = compress_rate > 1
        self.wkv = Linear(cfg, f"{key}.wkv", attn.hidden_size, head_dim, qmap = qmap, out_dtype = torch.half,
                          trim_padded_out = True, select_hq_bits = select_hq_bits)
        self.wgate = Linear(cfg, f"{key}.wgate", attn.hidden_size, head_dim, qmap = qmap, out_dtype = torch.half,
                            trim_padded_out = True, select_hq_bits = select_hq_bits) if self.gated else None
        self.norm = RMSNorm(cfg, f"{key}.norm", attn.rms_norm_eps)

    def modules(self):
        return [m for m in (self.wkv, self.wgate, self.norm) if m is not None]

    def project(self, x, params):
        kv = self.wkv.forward(x, params).float()
        gate = self.wgate.forward(x, params).float() if self.gated else None
        return kv, gate

    def pool(self, kv, gate, params):
        """kv/gate: (bsz, n * m, head_dim) fp32 complete groups -> (bsz, n, head_dim) fp16
        normed latents (pre-rope)."""
        bsz = kv.shape[0]
        m = self.compress_rate
        kv = kv.view(bsz, -1, m, self.head_dim)
        if self.gated:
            comp = (kv * gate.view(bsz, -1, m, self.head_dim).softmax(dim = 2)).sum(dim = 2)
        else:
            comp = kv.sum(dim = 2)
        return self.norm.forward(comp.half(), params)

    # ---- tensor-parallel support -------------------------------------------------------
    # Headless shared pool entries, fully replicated per rank, same as DSV4Compressor.
    # Note wgate is None when compress_rate == 1 (ratio-1 layers are a plain projection).

    def make_bc(self, inv_freq):
        # No-op. This compressor applies no position bias (see the class docstring),
        # so there is no rope-derived state to precompute; project() and pool() read
        # none. DSV4Compressor needs this hook, V4.1 does not.
        return

    def tp_export(self, plan, producer):
        return {
            "cls": DSV41Compressor,
            "args": {
                "key": self.key,
                "head_dim": self.head_dim,
                "compress_rate": self.compress_rate,
            },
            "wkv": self.wkv.tp_export(plan, producer),
            "wgate": self.wgate.tp_export(plan, producer) if self.wgate is not None else None,
            "norm": self.norm.tp_export(plan, producer),
        }

    @staticmethod
    def tp_import(local_context, exported, plan, attn):
        def _imp(name):
            e = exported.get(name)
            return e["cls"].tp_import(local_context, e, plan) if e is not None else None
        a = exported["args"]
        comp = DSV41Compressor(
            attn, a["key"], a["head_dim"], a["compress_rate"], None, 0)
        comp.wkv = _imp("wkv")
        comp.wgate = _imp("wgate")
        comp.norm = _imp("norm")
        return comp


class DSV41LayerState(DSV4LayerState):
    """Ring state of the V4 module plus, for gated kv sources, the projected rows of the
    trailing incomplete group (position-derived count: pos % m rows pending)."""

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        super().__init__(module, max_batch_size, max_history, cache_id)
        self.pend_kv = self.pend_gate = None
        if module.is_kv_source and module.compress_rate > 1:
            m, hd = module.compress_rate, module.head_dim
            self.pend_kv = torch.zeros((max_batch_size, m - 1, hd), dtype = torch.float, device = "meta")
            self.pend_gate = torch.zeros((max_batch_size, m - 1, hd), dtype = torch.float, device = "meta")

    def _tensors(self):
        return super()._tensors() + [t for t in (self.pend_kv, self.pend_gate) if t is not None]

    def _set_tensors(self, ts):
        n = len(super()._tensors())
        super()._set_tensors(ts[:n])
        it = iter(ts[n:])
        for name in ("pend_kv", "pend_gate"):
            if getattr(self, name) is not None:
                setattr(self, name, next(it))

    def stash(self, slot, position):
        out = super().stash(slot, position)
        if self.pend_kv is not None:
            out += [self.pend_kv[slot].cpu(), self.pend_gate[slot].cpu()]
        return out

    def unstash(self, slot, stashed, position):
        n = len(stashed) - (2 if self.pend_kv is not None else 0)
        super().unstash(slot, stashed[:n], position)
        if self.pend_kv is not None:
            self.pend_kv[slot].copy_(stashed[n])
            self.pend_gate[slot].copy_(stashed[n + 1])


class DSV41Attention(DSV4Attention):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        compress_rate: int,
        is_kv_source: bool,
        is_index_source: bool,
        kv_source_layer: int,
        index_source_layer: int | None,
        candidate_role: str | None,          # "source" | "consumer" | None
        candidate_topk_blocks: int = 0,
        candidate_block_size: int = 8,
        ref_quant: bool = False,             # reference cache numerics (fp8 window, fp4 latents/keys)
        select_hq_bits: int = 0,
        qmap: str | None = None,
        **kwargs,
    ):
        assert compress_rate >= 1
        super().__init__(config, key, layer_idx, "v41", compress_rate = compress_rate, qmap = qmap,
                         select_hq_bits = select_hq_bits, **kwargs)
        self.is_kv_source = is_kv_source
        self.is_index_source = is_index_source
        self.owns_index_keys = is_kv_source and is_index_source
        self.kv_source_layer = kv_source_layer
        self.index_source_layer = index_source_layer
        self.candidate_role = candidate_role
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.ref_quant = ref_quant
        self.index_pool = self.owns_index_keys           # CacheLayer_dsa allocates pool_idx
        self.idx_wk = self.idx_k_norm = None
        self.layer_state_cls = DSV41LayerState
        if self.num_q_heads == 0:
            return
        if is_kv_source:
            self.compressor = DSV41Compressor(self, f"{key}.compressor", self.head_dim, compress_rate, qmap, select_hq_bits)
            for m in self.compressor.modules():
                self.register_submodule(m)
            self.caps.update({"kv_cache": True})
        # `and self.idx_wq_b is None`: under TP the parent has already assigned the
        # imported projections (dsv4.py:858-859 -> :505-506), and rebuilding them here
        # with config = None would replace them with empty shells (inner = None) that
        # only fail later, inside the indexer's forward. Same guard the parent uses for
        # the identical construction at dsv4.py:527.
        if is_index_source and self.idx_wq_b is None:
            self.idx_wq_b = Linear(config, f"{key}.indexer.wq_b", self.q_lora_rank_, self.index_n_heads * self.index_head_dim,
                                   qmap = f"{key}.q_b", out_dtype = torch.half, trim_padded_out = True, select_hq_bits = select_hq_bits)
            self.idx_weights = Linear(config, f"{key}.indexer.weights_proj", self.hidden_size, self.index_n_heads,
                                      qmap = None, out_dtype = torch.half, pad_to = 1)
            self.register_submodule(self.idx_wq_b)
            self.register_submodule(self.idx_weights)
        if self.owns_index_keys:
            self.idx_wk = Linear(config, f"{key}.indexer.wk", self.head_dim, self.index_head_dim, qmap = None,
                                 out_dtype = torch.half, pad_to = 1)
            self.idx_k_norm = RMSNorm(config, f"{key}.indexer.k_norm", self.rms_norm_eps)
            self.register_submodule(self.idx_wk)
            self.register_submodule(self.idx_k_norm)

    @property
    def q_lora_rank_(self):
        return self.q_a.out_features_unpadded if hasattr(self.q_a, "out_features_unpadded") else self.q_a.out_features

    def cache_layer_type(self, default, kwargs: dict):
        return CacheLayer_dsa, {}        # fp16 pools (no k_bits packing on V4.1 yet)

    @override
    def load(self, device: torch.device, **kwargs):
        Module.load(self, device, **kwargs)
        stc = self.config.stc
        self.sinks = stc.get_tensor(f"{self.key}.attn_sink", device, no_defer = True).float().contiguous()
        self.inv_freq_main = yarn_inv_freq(self.rope_head_dim, self.rope_theta, device)
        self.inv_freq_compress = yarn_inv_freq(self.rope_head_dim, self.compress_rope_theta, device, rope_scaling = self.rope_scaling)
        self.inv_freq_main_neg = -self.inv_freq_main
        self.inv_freq_compress_neg = -self.inv_freq_compress
        self.kv_norm_w = self.kv_norm.weight.data
        self.q_ones = torch.ones(self.head_dim, dtype = self.kv_norm_w.dtype, device = device)
        for rl in self.recurrent_layers:
            rl.alloc(device)
        for cl in self.cache_layers:
            cl.alloc(device)

    @override
    def unload(self):
        Module.unload(self)
        for rl in self.recurrent_layers:
            rl.free()
        for cl in self.cache_layers:
            cl.free()
        self.sinks = None

    def _forward_nc(self, x, params, out_dtype):
        """Stateless pass: complete compressor groups only (remainder dropped), no ring history."""
        bsz, seq, _ = x.shape
        device = x.device
        pos0 = int(params.get("position", 0))
        inst = params.get("layer_instance", 0)
        m = self.compress_rate
        w = self.sliding_window
        hpg = self.num_q_heads // self.o_groups
        hd = self.head_dim
        D_c, D_r = hd - self.rope_head_dim, self.rope_head_dim
        q_res, q, kv = self._project_qkv(x, params, pos0)
        if self.ref_quant:
            kv = fp8_qdq(kv, 32)
        outs = []
        nc_pack = [] if self.is_kv_source else params[("v41_nc", self.kv_source_layer, inst)]
        idx_pack = [] if self.is_index_source else None
        for b in range(bsz):
            if self.is_kv_source:
                n = seq // m
                pool_c = torch.zeros((max(n, 1), D_c), dtype = torch.half, device = device)
                pool_r = torch.zeros((max(n, 1), D_r), dtype = torch.half, device = device)
                pool_idx = (torch.zeros((max(n, 1), self.index_head_dim), dtype = torch.half, device = device)
                            if self.owns_index_keys else None)
                if n:
                    sl = x[b:b + 1, :n * m]
                    kv_rows, gate_rows = self.compressor.project(sl, params)
                    latent = self.compressor.pool(kv_rows, gate_rows, params)
                    gpos = (torch.arange(n, device = device) * m).to(torch.int).unsqueeze(0).contiguous()
                    if self.owns_index_keys:
                        k = self.idx_k_norm.forward(self.idx_wk.forward(latent, params), params, out_dtype = torch.half)
                        k = k.view(1, n, 1, self.index_head_dim).contiguous()
                        _ext_rope(k[..., -self.rope_head_dim:], self.inv_freq_compress, position_ids = gpos)
                        k = k.view(n, self.index_head_dim)
                        if self.ref_quant:
                            k = fp4_qdq(k, 32, scale_e4m3 = False)
                        pool_idx[:n] = k.half()
                    lat = latent.view(1, n, 1, hd).contiguous()
                    _ext_rope(lat[..., -self.rope_head_dim:], self.inv_freq_compress, position_ids = gpos)
                    lat = lat.view(n, hd)
                    if self.ref_quant:
                        lat = fp4_qdq(lat, 16, scale_e4m3 = True)
                    pool_c[:n] = lat[:, :D_c].half()
                    pool_r[:n] = lat[:, D_c:].half()
                nc_pack.append((pool_c, pool_r, pool_idx, n))
            else:
                pool_c, pool_r, pool_idx, n = nc_pack[b]
            T = n
            cap = max(T, 1)
            bt = torch.arange(-(-cap // PAGE_SIZE), dtype = torch.int32, device = device).unsqueeze(0)
            indices, k_len = None, 0
            if T > 0:
                if self.is_index_source:
                    keys = pool_idx if pool_idx is not None else params[("v41_nc", self.kv_source_layer, inst)][b][2]
                    indices, k_len = self._v41_indexer(x[b:b + 1], params, q_res[b:b + 1], keys[:T], T, pos0, seq, inst)
                    idx_pack.append((indices, k_len))
                elif self.index_source_layer is not None:
                    indices, k_len = params[("v41_idx", self.index_source_layer, inst)][b]
            if indices is None:
                indices = torch.full((seq, 32), -1, dtype = torch.int32, device = device)
                k_len = 0
            out = dsa_attn(
                q[b].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
                kv_chunk = kv[b].contiguous(), win_len = w, win_floor = pos0,
                indices = indices, k_len = k_len, pool_len = T, q_pos0 = pos0,
                compress_rate = m, scale = self.sm_scale,
                derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
                out = torch.empty((self.o_groups, seq, hpg * hd), dtype = torch.half, device = device),
                nc_chunk = bool(params.get("nc_chunk", False)),
            )
            outs.append(self._project_o_grouped(out.unsqueeze(1), params, out_dtype))
        if self.is_kv_source:
            params[("v41_nc", self.layer_idx, inst)] = nc_pack
        if self.is_index_source:
            params[("v41_idx", self.layer_idx, inst)] = idx_pack
        return torch.cat(outs, dim = 0) if bsz > 1 else outs[0]

    def _forward_cached_batch(self, x, params, rsg, layer_instance, out_dtype, kl, bt):
        outs = [
            self._forward_cached_one(x[i:i + 1], params, rsg[i], self._get_rsl(rsg[i], layer_instance), out_dtype,
                                     copy_static = True, kl = kl, bt_row = bt[i:i + 1] if bt is not None else None)
            for i in range(x.shape[0])
        ]
        return torch.cat(outs, dim = 0)

    # ---- compressed path ---------------------------------------------------------------

    def _pool_slots(self, bt_row, first, count, epp):
        e = torch.arange(first, first + count, device = bt_row.device)
        return bt_row[0, e // epp].long(), (e % epp).long()

    def _compress_into_pool(self, x, params, rsl, slot, kl, bt_row, pos0, seq):
        """kv source: project this chunk, complete groups with the pending rows, write the
        latents (and index keys) of the completed groups into the pools."""
        m = self.compress_rate
        comp = self.compressor
        kv_rows, gate_rows = comp.project(x, params)                       # (1, seq, hd) fp32
        pend = pos0 % m
        if pend:
            kv_rows = torch.cat((rsl.pend_kv[slot:slot + 1, :pend], kv_rows), dim = 1)
            if gate_rows is not None:
                gate_rows = torch.cat((rsl.pend_gate[slot:slot + 1, :pend], gate_rows), dim = 1)
        n_rows = pend + seq
        n_groups = n_rows // m
        rem = n_rows - n_groups * m
        if n_groups:
            latent = comp.pool(kv_rows[:, :n_groups * m], None if gate_rows is None else gate_rows[:, :n_groups * m], params)
            g0 = pos0 // m
            epp = kl.epp
            pages, slots = self._pool_slots(bt_row, g0, n_groups, epp)
            gpos = ((torch.arange(n_groups, device = x.device) + g0) * m).to(torch.int).unsqueeze(0).contiguous()
            if self.owns_index_keys:
                k = self.idx_k_norm.forward(self.idx_wk.forward(latent, params), params, out_dtype = torch.half)
                k = k.view(1, n_groups, 1, self.index_head_dim).contiguous()
                _ext_rope(k[..., -self.rope_head_dim:], self.inv_freq_compress, position_ids = gpos)
                k = k.view(n_groups, self.index_head_dim)
                if self.ref_quant:
                    k = fp4_qdq(k, 32, scale_e4m3 = False)
                kl.pool_idx[pages, slots] = k.half()
            lat = latent.view(1, n_groups, 1, self.head_dim).contiguous()
            _ext_rope(lat[..., -self.rope_head_dim:], self.inv_freq_compress, position_ids = gpos)
            lat = lat.view(n_groups, self.head_dim)
            if self.ref_quant:
                lat = fp4_qdq(lat, 16, scale_e4m3 = True)
            kl.pool_c[pages, slots] = lat[:, :kl.D_c].half()
            kl.pool_r[pages, slots] = lat[:, kl.D_c:].half()
        if rsl.pend_kv is not None and rem:
            rsl.pend_kv[slot, :rem].copy_(kv_rows[0, n_groups * m:])
            rsl.pend_gate[slot, :rem].copy_(gate_rows[0, n_groups * m:])

    def _gather_pool_idx(self, kl, bt_row, ec):
        pages, slots = self._pool_slots(bt_row, 0, ec, kl.epp)
        return kl.pool_idx[pages, slots]                                      # (ec, D_i) fp16

    def _v41_indexer(self, x, params, q_res, keys, ec, pos0, seq, inst):
        """Indexer scoring and top-k. keys: (ec, D_i) fp16 from this or a source pool."""
        Hi, Di, m, rd = self.index_n_heads, self.index_head_dim, self.compress_rate, self.rope_head_dim
        dev = x.device
        q_idx = self.idx_wq_b.forward(q_res, params).view(1, seq, Hi, Di).contiguous()
        _ext_rope(q_idx[..., -rd:], self.inv_freq_compress, position = pos0)
        q_idx = q_idx[0]
        if self.ref_quant:
            q_idx = fp4_qdq(q_idx, 32, scale_e4m3 = False)
        wts = self.idx_weights.forward(x, params).float()[0] * (Di ** -0.5 * Hi ** -0.5)   # (seq, Hi)
        keys = keys.float()                                                                # (ec, Di)
        positions = torch.arange(pos0, pos0 + seq, device = dev)
        comp_lens = (positions + 1) // m
        scores = torch.empty((seq, ec), device = dev)
        step = 256
        for s0 in range(0, seq, step):
            qs = q_idx[s0:s0 + step].float()
            sc = torch.relu(torch.einsum("nhd,ed->nhe", qs, keys))
            scores[s0:s0 + step] = torch.einsum("nhe,nh->ne", sc, wts[s0:s0 + step])
        vis = torch.arange(ec, device = dev).unsqueeze(0) < comp_lens.unsqueeze(1)
        scores.masked_fill_(~vis, float("-inf"))
        if self.candidate_role == "source":
            params[("v41_cand", inst)] = self._candidate_mask(scores, comp_lens, ec)
        elif self.candidate_role == "consumer":
            cand = params.get(("v41_cand", inst))
            if cand is not None:
                scores.masked_fill_(~cand, float("-inf"))
        k = min(self.index_topk, ec)
        K_pad = -(-k // 32) * 32
        top = scores.topk(k, dim = -1, sorted = False)
        idx = top.indices
        idx = torch.where(top.values > float("-inf"), idx, torch.full_like(idx, ec))
        idx = idx.sort(dim = -1).values
        idx = torch.where(idx < ec, idx, torch.full_like(idx, -1))
        indices = torch.full((seq, K_pad), -1, dtype = torch.int32, device = dev)
        indices[:, :k] = idx.to(torch.int32)
        return indices, k

    def _candidate_mask(self, scores, comp_lens, ec):
        """Reference select_candidate_blocks: per query, block-max scores over blocks of
        candidate_block_size entries, the block holding the newest visible entry pinned,
        keep the top candidate_topk_blocks blocks."""
        bs = self.candidate_block_size
        seq = scores.shape[0]
        nb = -(-ec // bs)
        padded = torch.full((seq, nb * bs), float("-inf"), device = scores.device)
        padded[:, :ec] = scores
        blk = padded.view(seq, nb, bs).amax(-1)                                # (seq, nb)
        newest = ((comp_lens - 1).clamp_min(0) // bs)
        blk[torch.arange(seq, device = scores.device), newest] = float("inf")
        k = min(self.candidate_topk_blocks, nb)
        top = blk.topk(k, dim = -1)
        keep = torch.zeros_like(blk, dtype = torch.bool)
        keep.scatter_(1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(bs, dim = 1)[:, :ec]

    # ---- per-job forward -----------------------------------------------------------------

    def _forward_cached_one(self, x, params, rs, rsl, out_dtype, copy_static = False, pre = None,
                            return_o = False, kl = None, bt_row = None):
        _, seq, _ = x.shape
        device = x.device
        pos0 = rs.position
        slot = rs.slot
        m = self.compress_rate
        inst = params.get("layer_instance", 0)

        q_res, q, kv = self._project_qkv(x, params, pos0)
        if self.ref_quant:
            kv = fp8_qdq(kv, 32)

        ec = (pos0 + seq) // m
        if self.is_kv_source:
            assert kl is not None and bt_row is not None
            assert ec <= bt_row.shape[1] * kl.epp, f"DSA pool overflow: entry {ec} beyond block table"
            self._compress_into_pool(x, params, rsl, slot, kl, bt_row, pos0, seq)
            params[("v41_kl", self.layer_idx, inst)] = (kl, bt_row)
        else:
            kl, bt_row = params[("v41_kl", self.kv_source_layer, inst)]

        indices, k_len = None, 0
        if ec > 0:
            if self.is_index_source:
                keys = self._gather_pool_idx(kl, bt_row, ec)
                indices, k_len = self._v41_indexer(x, params, q_res, keys, ec, pos0, seq, inst)
                params[("v41_idx", self.layer_idx, inst)] = (indices, k_len)
            elif self.index_source_layer is not None:
                indices, k_len = params[("v41_idx", self.index_source_layer, inst)]
        if indices is None:
            indices = torch.full((seq, 32), -1, dtype = torch.int32, device = device)
            k_len = 0

        # Window sources: this chunk's kv rows plus prior rows read from the ring
        w = self.sliding_window
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        win_floor = pos0 - n_prev
        ring = rsl.ring[slot]
        ring_beg = rs.window_beg

        hpg = self.num_q_heads // self.o_groups
        out = dsa_attn(
            q[0].half().contiguous(), kl.pool_c_view(), kl.pool_r, bt_row, sinks = self.sinks,
            ring = ring, kv_chunk = kv[0], win_len = w, win_floor = win_floor, ring_beg = ring_beg,
            indices = indices, k_len = k_len, pool_len = ec, q_pos0 = pos0, compress_rate = m,
            scale = self.sm_scale, derot_inv_freq = self._rope_type_neg(), groups = self.o_groups,
            group_major = True, page_size = kl.epp, qc = kl.qc(),
            out = torch.empty((self.o_groups, seq, hpg * self.head_dim), dtype = torch.half, device = device),
            nc_chunk = bool(params.get("nc_chunk", False)),
        )

        # Ring update after attention (identical to the V4 module)
        offset = pos0 - rs.window_beg
        pos_end = pos0 + seq
        if offset + seq <= rsl.ring_rows:
            ring[offset: offset + seq].copy_(kv[0])
        elif seq < PAGE_SIZE:
            need = offset + seq - rsl.ring_rows
            shift = -(-need // PAGE_SIZE) * PAGE_SIZE
            ring[:rsl.ring_rows - shift].copy_(ring[shift:].clone())
            rs.wshift = shift
            ring[offset - shift: offset - shift + seq].copy_(kv[0])
        else:
            new_beg = max(pos_end - (w - 1), 0) // PAGE_SIZE * PAGE_SIZE
            n_keep = pos_end - new_beg
            n_from_kv = min(n_keep, seq)
            n_from_ring = n_keep - n_from_kv
            assert 0 < n_keep <= rsl.ring_rows
            if n_from_ring > 0:
                src0 = pos0 - n_from_ring - ring_beg
                ring[:n_from_ring].copy_(ring[src0: src0 + n_from_ring].clone())
            ring[n_from_ring: n_keep].copy_(kv[0, seq - n_from_kv:])
            rs.wshift = new_beg - rs.window_beg

        if return_o:
            return out
        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype)
