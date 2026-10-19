from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, RMSNorm, Linear, GatedMLP, BlockSparseMLP, TransformerBlock, \
    HyperConnection, ExpandStreams, HyperHead
from ..modules.dsv4 import DSV4Attention
from ..modules.dsv41 import DSV41Attention
from ..modules.dsv41_hc import DSV41HyperConnection, DSV41TransformerBlock, DSV41HyperHead, PRE_MIX_KEY
from ..modules.engram import EngramHasher, EngramLayer
from ..modules.attn import prepare_for_attn
from .deepseek_v41_mtp import DeepseekV41MTPModel

# DeepSeek-V4.1: the V4 trunk (mHC streams, sqrt-softplus MoE, DSpark drafter) with
#  - compress_ratios in {0, 1, 2}: 0 = sliding only; r > 0 = sliding + top-k over a compressed
#    KV that only the kv_source_layer_ids compute (ratio-2 sources pool token pairs through a
#    gate, ratio-1 sources project each token) and later layers share, likewise the indexer keys
#    (index_source_layer_ids) and a two-level candidate mask from candidate_source_layer_id;
#  - Engram n-gram embedding layers (engram_layer_ids) added to the residual streams;
#  - no hash-routed MoE layers, DSpark heads with their own expert counts.
# Config keys sit under text_config (HF layout); the checkpoint keeps DeepSeek's namespace.


def _t(key: str) -> list:
    return [f"text_config->{key}", key]


class DeepseekV41Config(Config):
    arch_string = "DeepseekV41ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": DeepseekV41Model, "mtp": DeepseekV41MTPModel}, **kwargs)

        # Attention
        self.hidden_size = self.read_cfg(int, _t("hidden_size"), no_default)
        self.num_q_heads = self.read_cfg(int, _t("num_attention_heads"), no_default)
        self.num_kv_heads = self.read_cfg(int, _t("num_key_value_heads"), 1)
        assert self.num_kv_heads == 1, "DeepseekV4.1: expected shared-KV MQA (num_key_value_heads == 1)"
        self.head_dim = self.read_cfg(int, _t("head_dim"), 512)
        self.qk_rope_head_dim = self.read_cfg(int, _t("qk_rope_head_dim"), 64)
        self.q_lora_rank = self.read_cfg(int, _t("q_lora_rank"), no_default)
        self.o_groups = self.read_cfg(int, _t("o_groups"), 8)
        self.o_lora_rank = self.read_cfg(int, _t("o_lora_rank"), 1024)
        self.sliding_window = self.read_cfg(int, _t("sliding_window"), 128)
        self.index_n_heads = self.read_cfg(int, _t("index_n_heads"), 32)
        self.index_head_dim = self.read_cfg(int, _t("index_head_dim"), 128)
        self.index_topk = self.read_cfg(int, _t("index_topk"), 512)

        # Layer schedule and cross-layer sharing
        self.num_hidden_layers = self.read_cfg(int, _t("num_hidden_layers"), no_default)
        ratios = self.read_cfg(list, _t("compress_ratios"), no_default)
        self.compress_ratios = [int(r) for r in ratios[:self.num_hidden_layers]]
        assert all(r in (0, 1, 2) for r in self.compress_ratios), f"DeepseekV4.1: compress_ratios must be 0/1/2, got {sorted(set(self.compress_ratios))}"
        self.layer_types = ["sliding" if r == 0 else "v41" for r in self.compress_ratios]
        self.kv_source_layer_ids = [int(i) for i in self.read_cfg(list, _t("kv_source_layer_ids"), [])]
        self.index_source_layer_ids = [int(i) for i in self.read_cfg(list, _t("index_source_layer_ids"), [])]
        self.candidate_source_layer_id = self.read_cfg(int, _t("candidate_source_layer_id"), -1)
        self.candidate_topk_blocks = self.read_cfg(int, _t("candidate_topk_blocks"), 0)
        self.candidate_block_size = self.read_cfg(int, _t("candidate_block_size"), 8)
        # Consumers read the most recent source before them (reference SharedAttentionRuntime)
        self.kv_source_of, self.index_source_of = [], []
        kv_src, idx_src = None, None
        for i, r in enumerate(self.compress_ratios):
            if i in self.kv_source_layer_ids:
                assert r > 0, f"kv source layer {i} needs compress_ratio > 0"
                kv_src = i
            if i in self.index_source_layer_ids:
                idx_src = i
            self.kv_source_of.append(kv_src if r > 0 else None)
            self.index_source_of.append(idx_src if r > 0 else None)
            if r > 0:
                assert kv_src is not None and self.compress_ratios[kv_src] == r, \
                    f"layer {i} (ratio {r}) has no kv source with the same ratio before it"

        # mHC
        self.hc_mult = self.read_cfg(int, _t("hc_mult"), 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, _t("hc_sinkhorn_iters"), 20)
        self.hc_eps = self.read_cfg(float, _t("hc_eps"), 1e-6)

        # MoE (no hash-routed layers in V4.1)
        self.assert_cfg(str, _t("scoring_func"), "sqrtsoftplus", optional = True)
        self.assert_cfg(str, _t("topk_method"), "noaux_tc", optional = True)
        self.moe_intermediate_size = self.read_cfg(int, _t("moe_intermediate_size"), no_default)
        self.num_experts = self.read_cfg(int, _t("n_routed_experts"), no_default)
        self.num_experts_per_tok = self.read_cfg(int, _t("num_experts_per_tok"), no_default)
        self.num_shared_experts = self.read_cfg(int, _t("n_shared_experts"), 1)
        self.num_hash_layers = 0
        self.routed_scaling_factor = self.read_cfg(float, _t("routed_scaling_factor"), 1.0)
        self.swiglu_limit = self.read_cfg(float, _t("swiglu_limit"), 10.0)
        self.norm_topk_prob = self.read_cfg(bool, _t("norm_topk_prob"), True)

        # Norms / rope
        self.rms_norm_eps = self.read_cfg(float, _t("rms_norm_eps"), 1e-20)
        self.rope_theta = self.read_cfg(float, _t("rope_theta"), 10000.0)
        self.compress_rope_theta = self.read_cfg(float, _t("compress_rope_theta"), 160000.0)
        self.rope_scaling = self.read_cfg(dict, _t("rope_scaling"), None)
        self.tie_word_embeddings = self.read_cfg(bool, _t("tie_word_embeddings"), False)

        # Engram
        self.engram_layer_ids = [int(i) for i in self.read_cfg(list, _t("engram_layer_ids"), [])]
        self.engram_num_embeddings = [int(i) for i in self.read_cfg(list, _t("engram_num_embeddings"), [])]
        self.engram_max_ngram_size = self.read_cfg(int, _t("engram_max_ngram_size"), 4)
        self.engram_vocab_size = self.read_cfg(int, _t("engram_vocab_size"), 16000000)
        self.engram_n_heads = self.read_cfg(int, _t("engram_n_heads"), 8)
        self.engram_head_dim = self.read_cfg(int, _t("engram_head_dim"), 256)
        self.engram_pad_token_id = self.read_cfg(int, _t("engram_pad_token_id"), 2)
        self.engram_compressed_vocab_size = self.read_cfg(int, _t("engram_compressed_vocab_size"), no_default)

        # DSpark drafter
        self.dspark_block_size = self.read_cfg(int, _t("dspark_block_size"), 0)
        self.dspark_noise_token_id = self.read_cfg(int, _t("dspark_noise_token_id"), 0)
        self.dspark_markov_rank = self.read_cfg(int, _t("dspark_markov_rank"), 256)
        self.dspark_target_layer_ids = self.read_cfg(list, _t("dspark_target_layer_ids"), [])
        self.dspark_num_experts = self.read_cfg(int, _t("dspark_n_routed_experts"), 0) or self.num_experts
        self.dspark_num_experts_per_tok = self.read_cfg(int, _t("dspark_num_experts_per_tok"), 0) or self.num_experts_per_tok
        self.block_size = self.dspark_block_size + 1
        self.num_mtp_layers = max(0, len(ratios) - self.num_hidden_layers)
        if self.num_mtp_layers == 0 or not any(
            self.stc.has_tensor(f"mtp.0.attn.wkv.{t}") for t in ("weight", "trellis")):
            del self.model_classes["mtp"]
        self.vision = None
        # Reference cache numerics (fp8 window rows, fp4 latents and indexer keys) for parity
        # tests against DeepSeek's reference implementation; off for speed
        import os
        self.ref_quant = os.environ.get("EXL3_V41_REF_QUANT", "0") != "0"


class DeepseekV41Model(Model):
    config_class = DeepseekV41Config

    def __init__(self, config: DeepseekV41Config, **kwargs):
        super().__init__(config, **kwargs)
        self.modules += [
            Embedding(config = config, key = "embed", vocab_size = config.vocab_size, hidden_size = config.hidden_size),
            ExpandStreams(config = config, key = "hc_expand", hc_mult = config.hc_mult),
        ]
        self.first_block_idx = len(self.modules)
        self.hasher = EngramHasher(config) if config.engram_layer_ids else None

        for idx in range(config.num_hidden_layers):
            key = f"layers.{idx}"
            if idx in config.engram_layer_ids:
                self.modules += [EngramLayer(
                    config = config,
                    key = f"{key}.engram",
                    layer_idx = -(idx + 1),
                    table_index = config.engram_layer_ids.index(idx),
                    hasher = self.hasher,
                    hidden_size = config.hidden_size,
                    hc_mult = config.hc_mult,
                    rms_norm_eps = config.rms_norm_eps,
                )]
            layer_type = config.layer_types[idx]
            if layer_type == "sliding":
                attn = DSV4Attention(
                    config = config,
                    key = f"{key}.attn",
                    layer_idx = idx,
                    layer_type = "sliding",
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    head_dim = config.head_dim,
                    rope_head_dim = config.qk_rope_head_dim,
                    q_lora_rank = config.q_lora_rank,
                    o_groups = config.o_groups,
                    o_lora_rank = config.o_lora_rank,
                    sliding_window = config.sliding_window,
                    compress_rate = None,
                    q_head_norm = False,
                    index_n_heads = config.index_n_heads,
                    index_head_dim = config.index_head_dim,
                    index_topk = config.index_topk,
                    rope_theta = config.rope_theta,
                    compress_rope_theta = config.compress_rope_theta,
                    rope_scaling = config.rope_scaling,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                )
            else:
                cand_role = None
                if idx == config.candidate_source_layer_id:
                    cand_role = "source"
                elif config.candidate_source_layer_id >= 0 and idx > config.candidate_source_layer_id \
                        and idx in config.index_source_layer_ids:
                    cand_role = "consumer"
                attn = DSV41Attention(
                    config = config,
                    key = f"{key}.attn",
                    layer_idx = idx,
                    compress_rate = config.compress_ratios[idx],
                    is_kv_source = idx in config.kv_source_layer_ids,
                    is_index_source = idx in config.index_source_layer_ids,
                    kv_source_layer = config.kv_source_of[idx],
                    index_source_layer = config.index_source_of[idx],
                    candidate_role = cand_role,
                    candidate_topk_blocks = config.candidate_topk_blocks,
                    candidate_block_size = config.candidate_block_size,
                    ref_quant = config.ref_quant,
                    q_head_norm = False,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    head_dim = config.head_dim,
                    rope_head_dim = config.qk_rope_head_dim,
                    q_lora_rank = config.q_lora_rank,
                    o_groups = config.o_groups,
                    o_lora_rank = config.o_lora_rank,
                    sliding_window = config.sliding_window,
                    index_n_heads = config.index_n_heads,
                    index_head_dim = config.index_head_dim,
                    index_topk = config.index_topk,
                    rope_theta = config.rope_theta,
                    compress_rope_theta = config.compress_rope_theta,
                    rope_scaling = config.rope_scaling,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                )
            mlp = BlockSparseMLP(
                config = config,
                key = f"{key}.ffn",
                hidden_size = config.hidden_size,
                intermediate_size = config.moe_intermediate_size,
                num_experts = config.num_experts,
                num_experts_per_tok = config.num_experts_per_tok,
                key_up = "experts.{expert_idx}.w3",
                key_gate = "experts.{expert_idx}.w1",
                key_down = "experts.{expert_idx}.w2",
                key_routing_gate = "gate",
                key_e_score_bias = "gate.bias",
                key_e_score_bias_vl = "gate.bias_vl",
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                router_type = "sqrtsp",
                activation_fn = "silu",
                act_limit = config.swiglu_limit,
                routed_scaling_factor = config.routed_scaling_factor,
                shared_experts = GatedMLP(
                    config = config,
                    key = f"{key}.ffn.shared_experts",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                    key_up = "w3",
                    key_gate = "w1",
                    key_down = "w2",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
                ),
            )
            attn_hc = DSV41HyperConnection(config = config, key = f"{key}.hc_attn", hidden_size = config.hidden_size,
                                      hc_mult = config.hc_mult, sinkhorn_iters = config.hc_sinkhorn_iters,
                                      hc_eps = config.hc_eps, rms_norm_eps = config.rms_norm_eps)
            mlp_hc = DSV41HyperConnection(config = config, key = f"{key}.hc_ffn", hidden_size = config.hidden_size,
                                     hc_mult = config.hc_mult, sinkhorn_iters = config.hc_sinkhorn_iters,
                                     hc_eps = config.hc_eps, rms_norm_eps = config.rms_norm_eps)
            self.modules += [DSV41TransformerBlock(
                config = config,
                key = key,
                layer_idx = idx,
                attn_norm = RMSNorm(config = config, key = f"{key}.attn_norm", rms_norm_eps = config.rms_norm_eps),
                attn = attn,
                mlp_norm = RMSNorm(config = config, key = f"{key}.ffn_norm", rms_norm_eps = config.rms_norm_eps),
                mlp = mlp,
                attn_hc = attn_hc,
                mlp_hc = mlp_hc,
            )]

        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            DSV41HyperHead(config = config, key = "hc_head", hc_mult = config.hc_mult),
            RMSNorm(config = config, key = "norm", rms_norm_eps = config.rms_norm_eps, out_dtype = torch.half),
            Linear(config = config, key = "head", qbits_key = "head_bits", in_features = config.hidden_size,
                   out_features = config.vocab_size, qmap = "block", caps = {"logits_output": True}),
        ]
        self.logit_layer_idx = len(self.modules) - 1
        self.caps.update({"recurrent_states": True, "default_recurrent_checkpoint_interval": 2048})
        from ..cache.dsa import DSV4State
        self.recurrent_state_cls = DSV4State

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        from ..cache.recurrent_util import prepare_for_recurrence
        params["input_ids"] = input_ids
        params.pop(PRE_MIX_KEY, None)
        prepare_for_recurrence(input_ids, params, self)
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids
