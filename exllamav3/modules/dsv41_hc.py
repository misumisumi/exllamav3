from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from .module import Module
from .hyperconnections import HyperConnection
from .transformer import TransformerBlock

"""
DeepSeek-V4.1 hyper-connection wiring (reference: inference/model.py Block.forward, Transformer.forward).

The mixing math is the V4 mHC math, but each site's `pre` coefficients collapse the stream stack
for the NEXT site: the attention input is collapsed with the previous block's FFN-site pre (an
identity on copy 0 before block 0), the FFN input with this block's attention-site pre, and the
final collapse before the norm/head uses the last block's FFN-site pre. There is no separate head
mixer (no hc_head tensors). The pending pre travels between modules in params["v41_pre_mix"];
DeepseekV41Model.prepare_inputs clears it at the start of every forward.
"""

PRE_MIX_KEY = "v41_pre_mix"


class DSV41HyperConnection(HyperConnection):

    def mix_full(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """streams (b, s, H, D) -> (pre (b,s,H), post (b,s,H), comb (b,s,H,H)), fp32."""
        hc = self.hc_mult
        b, s, H, D = streams.shape
        if hc == 4 and streams.dtype == torch.float and D % 4 == 0 and streams.is_contiguous():
            # post and comb (with the Sinkhorn iterations) from the fused mixer, two launches instead
            # of ~130; its collapsed output uses this site's pre, which V4.1 applies at the next site,
            # so it is ignored here. post/comb live in shared static workspaces and are consumed by
            # apply_() before the next site mixes
            post, comb, _ = self.mix(streams, {})
            xf = streams.view(b, s, H * D)
            pre_w = F.linear(xf, self.fn[:hc]) * torch.rsqrt(xf.square().mean(-1, keepdim = True) + self.rms_eps)
            pre = torch.sigmoid(pre_w * self.scale[0] + self.base[:hc]) + self.hc_eps
            return pre, post, comb
        xf = streams.flatten(2).float()
        mix = F.linear(xf, self.fn) * torch.rsqrt(xf.square().mean(-1, keepdim = True) + self.rms_eps)
        pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim = -1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_s, post_s, comb_s = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim = -1) + self.hc_eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + self.hc_eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        return pre, post.contiguous(), comb.contiguous()


def collapse(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return (pre.unsqueeze(-1) * streams.float()).sum(dim = 2)


def pending_pre_mix(params: dict, streams: torch.Tensor, hc_mult: int) -> torch.Tensor:
    pre = params.get(PRE_MIX_KEY)
    if pre is None or pre.shape[:2] != streams.shape[:2]:
        pre = torch.zeros((*streams.shape[:2], hc_mult), dtype = torch.float, device = streams.device)
        pre[..., 0] = 1.0
    return pre


class DSV41TransformerBlock(TransformerBlock):

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        H = self.attn_hc.hc_mult
        pre_in = pending_pre_mix(params, x, H)

        # DSpark drafter taps (export_state_layers): V4.1 reads the ATTENTION INPUT of its target layers,
        # i.e. the stream mean before the block (reference Transformer.forward appends h.mean(dim=2)
        # before calling the layer), unlike TransformerBlock, which exports its output. Tapping the input
        # also keeps the prefill early return below from truncating the tap of the last KV layer
        export = params.get("export_state_layers")
        if export and self.layer_idx in export and params.get("layer_instance", 0) == 0:
            states = params.get("export_states")
            if not states:
                states = params["export_states"] = []
            tap = x.mean(dim = 2)
            tap = tap.clamp_(-65504.0, 65504.0) if tap.dtype == torch.half else tap.clamp(-65504.0, 65504.0).half()
            states.append(tap)

        attn_pre, attn_post, attn_comb = self.attn_hc.mix_full(x)
        y = self.attn_norm.forward(collapse(x, pre_in).half(), params, out_dtype = torch.half)
        y = self.attn.forward(y, params)
        if params.get("prefill"):
            # cache-fill pass of the last KV layer: nothing downstream reads the streams
            return x
        x = self.attn_hc.apply_(x, y, attn_post, attn_comb, params)

        mlp_pre, mlp_post, mlp_comb = self.mlp_hc.mix_full(x)
        y = self.mlp_norm.forward(collapse(x, attn_pre).half(), params, out_dtype = torch.half)
        y = self.mlp.forward(y, params)
        x = self.mlp_hc.apply_(x, y, mlp_post, mlp_comb, params)
        params[PRE_MIX_KEY] = mlp_pre
        return x


class DSV41HyperHead(Module):
    """Final stream collapse with the last block's FFN-site pre mix. Holds no tensors."""

    def __init__(self, config, key: str, hc_mult: int):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult

    @override
    def get_tensors(self):
        return {}

    @override
    def weights_numel(self):
        return 0

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return collapse(x, pending_pre_mix(params, x, self.hc_mult))

    # ---- tensor-parallel support -------------------------------------------------------
    # This module holds no tensors (see get_tensors/weights_numel above), so both halves are
    # kwargs only and every rank builds an identical copy.

    @override
    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": DSV41HyperHead,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
            },
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        module = DSV41HyperHead(config = None, **exported["kwargs"])
        module.device = local_context["device"]
        return module
