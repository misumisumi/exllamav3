from __future__ import annotations

import os
import torch
from typing_extensions import override

from ..cache.recurrent_util import advance_recurrent_states
from ..modules import Embedding, Linear
from .dflash import DFlashConfig, DFlashModel


class Qwen3DSparkConfig(DFlashConfig):
    """Qwen3 dSparK uses the DFlash backbone with raw target-layer taps."""

    arch_string = "Qwen3DSparkModel"
    tap_shift = 0

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, model_classes={"text": Qwen3DSparkModel}, **kwargs)
        self.markov_rank = self.read_cfg(int, ["dflash_config->markov_rank", "markov_rank"], None)
        self.enable_confidence_head = self.read_cfg(
            bool, ["dflash_config->enable_confidence_head", "enable_confidence_head"], False
        )
        self.confidence_head_with_markov = self.read_cfg(
            bool, ["dflash_config->confidence_head_with_markov", "confidence_head_with_markov"], False
        )
        self.sample_from_anchor = self.read_cfg(
            bool, ["dflash_config->sample_from_anchor", "sample_from_anchor"], True
        )
        self.draft_vocab_size = self.read_cfg(int, "draft_vocab_size", self.vocab_size)
        self.default_draft_size = self.block_size if self.sample_from_anchor else self.block_size - 1
        if not self.markov_rank or self.markov_rank <= 0:
            raise ValueError("Qwen3 dSparK requires a positive markov_rank")
        if self.confidence_head_with_markov and not self.enable_confidence_head:
            raise ValueError("confidence_head_with_markov requires enable_confidence_head")
        if self.sample_from_anchor and self.block_size != 15:
            # The adapter intentionally supports this official fixed-block checkpoint only.
            raise ValueError("Qwen3 dSparK anchor-as-first adapter requires block_size=15")


class Qwen3DSparkModel(DFlashModel):
    """DFlash Qwen3 draft stack plus its sequential Markov/confidence heads."""

    config_class = Qwen3DSparkConfig

    def __init__(self, config: Qwen3DSparkConfig, **kwargs):
        super().__init__(config, **kwargs)
        # The DFlash backbone ends at its final norm. Markov/confidence heads are
        # sampled explicitly and must never enter the normal module forward chain.
        self.fwd_end_idx = len(self.modules)
        self.markov_w1 = Embedding(
            config=config, key="markov_head.markov_w1", vocab_size=config.vocab_size,
            hidden_size=config.markov_rank, allow_table=False,
        )
        # This table is consulted once per drafted position. Keep it on-device to
        # avoid a host synchronization in the sequential Markov loop.
        self.markov_w1.caps["prefer_cpu"] = False
        self.markov_w2 = Linear(
            config=config, key="markov_head.markov_w2", in_features=config.markov_rank,
            out_features=config.draft_vocab_size, qmap=None,
        )
        self.modules += [self.markov_w1, self.markov_w2]
        self.confidence = None
        if config.enable_confidence_head:
            input_size = config.hidden_size + (config.markov_rank if config.confidence_head_with_markov else 0)
            self.confidence = Linear(
                config=config, key="confidence_head.proj", in_features=input_size,
                out_features=1, qmap=None, pad_to=1,
            )
            self.modules.append(self.confidence)
        self.draft_conf_threshold = float(os.environ.get("EXL3_DSPARK_CONF", "0.5"))
        self.caps.update({
            "dflash_sample_from_anchor": config.sample_from_anchor,
            "default_draft_size": config.default_draft_size,
        })
        self.enable_exact_verification()

    def enable_exact_verification(self):
        """Require target-authoritative one-token verification.

        MiniCPM-V 4.7's target is a recurrent quantized MoE. Its multi-row
        verification path is not numerically equivalent to ordinary one-token
        decode after a partial recurrent-state replay, and can change greedy
        argmax. dSparK therefore drafts a block, but the generator verifies and
        commits each proposal through the target's normal one-token path.
        """
        self.caps["exact_serial_verify"] = True

    @classmethod
    def checkpoint_tensor_consumption(cls, names):
        return audit_checkpoint_tensors(names)

    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if self.config.sample_from_anchor:
            if input_ids.shape[-1] != 1:
                raise ValueError("Qwen3 dSparK requires exactly one anchor token per request")
            params["dspark_anchor_ids"] = input_ids[:, -1]
        return super().prepare_inputs(input_ids, params)

    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None) -> torch.Tensor:
        """Run only the DFlash backbone; Markov/confidence heads are sampled separately."""
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        for module in self.modules[:self.fwd_end_idx]:
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        advance_recurrent_states(input_ids, params, self)
        return x

    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        target = self.attached_model()
        lm_head = target.modules[target.logit_layer_idx]
        logits = lm_head.forward(lm_head.prepare_for_device(state, params), params).float()
        if self.config.sample_from_anchor:
            anchor_ids = params["dspark_anchor_ids"]

            def markov_embed(token_ids):
                return self.markov_w1.forward(token_ids, params, out_dtype=state.dtype).to(
                    state.device, dtype=state.dtype
                )

            def markov_bias(markov):
                return self.markov_w2.forward(
                    markov.unsqueeze(1), params, out_dtype=logits.dtype
                ).squeeze(1)

            sampled, confidence = dspark_sequential_sample(
                logits,
                anchor_ids=anchor_ids,
                markov_embed_fn=markov_embed,
                markov_bias_fn=markov_bias,
                confidence_fn=self._confidence_function(params),
                position_hidden=state,
            )
            if confidence is not None:
                params["draft_conf"] = confidence
                threshold = float(params.get("dspark_conf_threshold", self.draft_conf_threshold))
                keep = confidence >= threshold
                lengths = torch.cumprod(keep.to(torch.int32), dim=-1).sum(dim=-1)
                params["draft_confidence_len"] = int(lengths.max().item())
            return sampled
        raise NotImplementedError("Qwen3 dSparK fill-in mode is not implemented")

    def _confidence_function(self, params):
        if self.confidence is None:
            return None

        def project(hidden, markov_embed):
            if self.config.confidence_head_with_markov:
                hidden = torch.cat((hidden, markov_embed), dim=-1)
            logits = self.confidence.forward(hidden, params).squeeze(-1)
            return torch.sigmoid(logits.float())

        return project


def dspark_sequential_sample(
    base_logits: torch.Tensor,
    *,
    anchor_ids: torch.Tensor,
    markov_embed_fn,
    markov_bias_fn,
    confidence_fn=None,
    position_hidden: torch.Tensor | None = None,
):
    """Sample query rows left-to-right using the previously selected token.

    The anchor is the predecessor of row zero, not an emitted output column.
    Embedding and projection are callables so production modules own placement and
    dtype while the autoregressive dependency stays independently testable.
    """
    batch, rows, _ = base_logits.shape
    previous = anchor_ids.reshape(batch).to(base_logits.device)
    output = torch.empty((batch, rows), dtype=torch.long, device=base_logits.device)
    confidence = []
    for row in range(rows):
        markov = markov_embed_fn(previous)
        bias = markov_bias_fn(markov)
        token = torch.argmax(base_logits[:, row, :] + bias.to(base_logits.dtype), dim=-1)
        output[:, row] = token
        if confidence_fn is not None:
            if position_hidden is None:
                raise ValueError("position_hidden is required when confidence_fn is supplied")
            confidence.append(confidence_fn(position_hidden[:, row, :], markov))
        previous = token
    return output, (torch.stack(confidence, dim=1) if confidence else None)


def audit_checkpoint_tensors(names):
    """Classify every official checkpoint tensor; target embedding/head are shared.

    The official checkpoint ships neither target embedding nor lm_head. ExLlama
    borrows them from the attached target model; all present checkpoint tensors
    must belong to one of the draft module keyspaces below.
    """
    names = set(names)
    draft_prefixes = (
        "fc.", "hidden_norm.", "layers.", "norm.",
        "markov_head.markov_w1.", "markov_head.markov_w2.", "confidence_head.proj.",
    )
    consumed = {name for name in names if name.startswith(draft_prefixes)}
    return {
        "consumed": consumed,
        "unconsumed": names - consumed,
        "borrowed_target": {"embed_tokens", "lm_head"},
    }
