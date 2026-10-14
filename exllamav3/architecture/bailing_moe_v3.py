from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP,
    GatedDeltaNet,
)
from ..modules.gated_delta_net import GDNState
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence

# Ling 3.x (BailingMoeV3ForCausalLM, inclusionAI): hybrid of full-rank Kimi Delta Attention and
# MLA, in groups of layer_group_size (the last layer of each group is MLA, as is any tail past
# the last whole group). MLA layers rotate the 64-wide pe slices (interleaved pairs, DeepSeek-V3
# style) and gate each head's output with sigmoid(g_proj(x)) before the output projection
# ("dense"). KDA layers use single hidden-wide forget/output gate projections (no_kda_lora).
# MoE: DeepSeek-V3-style grouped sigmoid routing (expert_bias for selection only, top-2 group
# scores, routed_scaling_factor) with one shared expert, and per-layer SwiGLU clamps
# (expert_swiglu_limit_list / share_expert_swiglu_limit_list, 0 = none). The trailing MTP layer
# (num_nextn_predict_layers) is not loaded. Reference: bundled modeling_bailing_moe_v3.py.


class BailingMoeV3Config(Config):
    arch_string = "BailingMoeV3ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": BailingMoeV3Model},
            **kwargs
        )

        # MLA
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", None)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.assert_cfg(bool, "rope_interleave", True, True)
        self.assert_cfg(bool, "use_mla_nope", False, True)
        self.assert_cfg(bool, "use_qkv_bias", False, True)
        self.assert_cfg(bool, "use_bias", False, True)
        gate_type = self.read_cfg(str, "gated_attention_proj_granularity_type", None)
        assert gate_type in (None, "head_wise"), \
            f"Unsupported gated_attention_proj_granularity_type: {gate_type}"
        self.mla_head_gate = gate_type == "head_wise"

        # RoPE on the pe slices only (interleaved pairs = GPT-J style), as in DeepSeek-V3
        self.head_dim = self.qk_head_dim
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.GPTJ,
            override_head_dim = self.qk_rope_head_dim,
            yarn_mscale_ratio = True,
        )
        # partial_rotary_factor / rotary_dim in the config describe the KDA-era head_dim; the
        # reference builds its rotary table over the full pe width
        self.rope_settings.partial_rotary_factor = 1.0
        self.rope_settings.rotary_dim = None
        assert self.rope_settings.rope_scaling is None, \
            "rope_scaling is not handled for BailingMoeV3 yet"
        self.sm_scale = self.qk_head_dim ** -0.5

        # Layer schedule
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        group = self.read_cfg(int, "layer_group_size", no_default)
        tail = self.num_hidden_layers // group * group
        self.layer_types = [
            "full_attention" if (idx + 1) % group == 0 or idx >= tail else "linear_attention"
            for idx in range(self.num_hidden_layers)
        ]

        # KDA (full-rank gates)
        self.assert_cfg(bool, "no_kda_lora", True, True)
        self.linear_num_heads = self.num_q_heads
        self.linear_head_dim = self.read_cfg(int, "head_dim", no_default)
        self.linear_conv_kernel_size = self.read_cfg(int, "short_conv_kernel_size", no_default)
        safe_gate = self.read_cfg(bool, "kda_safe_gate", False)
        self.linear_lower_bound = self.read_cfg(float, "kda_lower_bound", None) if safe_gate else None
        assert safe_gate, "Only the safe-gate KDA form is implemented (kda_safe_gate)"

        # MLP
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.shared_intermediate_size = self.read_cfg(int, "moe_shared_expert_intermediate_size", self.moe_intermediate_size)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        self.first_k_dense = self.read_cfg(int, "first_k_dense_replace", 0)
        self.n_group = self.read_cfg(int, "n_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)
        self.assert_cfg(str, "score_function", "sigmoid", True)
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        self.assert_cfg(bool, "moe_router_enable_expert_bias", True, True)
        self.assert_cfg(bool, "scale_router_input", False, True)

        def limits(name):
            lst = self.read_cfg(list, name, None) or []
            return [float(lst[i]) if i < len(lst) and lst[i] else 0.0 for i in range(self.num_hidden_layers)]
        self.expert_act_limit = limits("expert_swiglu_limit_list")
        self.shared_act_limit = limits("share_expert_swiglu_limit_list")

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)


class BailingMoeV3Model(Model):
    config_class = BailingMoeV3Config

    def __init__(
        self,
        config: BailingMoeV3Config,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.word_embeddings",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"

            if config.layer_types[idx] == "linear_attention":
                attn = GatedDeltaNet(
                    config = config,
                    key = f"{key}.attention",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    k_head_dim = config.linear_head_dim,
                    v_head_dim = config.linear_head_dim,
                    num_k_heads = config.linear_num_heads,
                    num_v_heads = config.linear_num_heads,
                    rms_norm_eps = config.rms_norm_eps,
                    conv_kernel_size = config.linear_conv_kernel_size,
                    key_qkv = "qkv_proj",
                    key_qkv_alt = ["q_proj", "k_proj", "v_proj"],
                    key_conv1d = "conv1d",
                    key_conv1d_q = "q_conv1d",
                    key_conv1d_k = "k_conv1d",
                    key_conv1d_v = "v_conv1d",
                    key_b = "b_proj",
                    key_f = "f_proj",
                    key_g = "g_proj",
                    gate_lower_bound = config.linear_lower_bound,
                    key_a_log = "A_log",
                    key_dt_bias = "dt_bias",
                    key_norm = "o_norm",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )
            else:
                attn = MLAttention(
                    config = config,
                    key = f"{key}.attention",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    kv_lora_rank = config.kv_lora_rank,
                    qk_nope_head_dim = config.qk_nope_head_dim,
                    qk_rope_head_dim = config.qk_rope_head_dim,
                    v_head_dim = config.v_head_dim,
                    rope_settings = config.rope_settings,
                    q_lora_rank = config.q_lora_rank,
                    sm_scale = config.sm_scale,
                    rms_norm_eps = config.rms_norm_eps,
                    key_o = "dense",
                    key_g = "g_proj" if config.mla_head_gate else None,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )

            if idx < config.first_k_dense:
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    activation_fn = "silu",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    select_hq_bits = 1,
                )
            else:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    key_e_score_bias = "gate.expert_bias",
                    activation_fn = "silu",
                    act_limit = config.expert_act_limit[idx],
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    router_type = "ds3" if config.n_group > 1 else "dots",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = config.n_group,
                    topk_group = config.topk_group,
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"{key}.mlp.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.shared_intermediate_size * config.num_shared_experts,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        activation_fn = "silu",
                        act_limit = config.shared_act_limit[idx],
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                    ) if config.num_shared_experts else None,
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = mlp,
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.word_embeddings"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.calibration_all_experts = True
        self.caps.update({
            "supports_tp": True,
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # Bundled chat_template.jinja, single turn, thinking on (the template's default)
        p = "<role>SYSTEM</role>"
        if system_prompt:
            p += f"{system_prompt}\n"
        p += "detailed thinking on<|role_end|>"
        p += f"<role>HUMAN</role>{prompt}<|role_end|>"
        p += "<role>ASSISTANT</role>\n<think>"
        return p
