from __future__ import annotations

from .qwen3_5 import Qwen3_5BaseModel, Qwen3_5VLMoeBaseConfig


class MiniCPMV4_7TextModel(Qwen3_5BaseModel):
    """Qwen3.5-MoE language stack used by MiniCPM-V 4.7."""

    def __init__(self, config: MiniCPMV4_7Config, **kwargs):
        super().__init__(
            config=config,
            key_prefix="model.language_model",
            use_moe=True,
            **kwargs,
        )


class MiniCPMV4_7Config(Qwen3_5VLMoeBaseConfig):
    """MiniCPM-V 4.7 config with the known Qwen3.5-MoE text component."""

    arch_string = "MiniCPMV4_7ForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        # This first vertical slice deliberately registers only the verified text tower.
        # Vision is added as a separate component once its tensor layout is implemented.
        super().__init__(
            directory,
            text_cfg="text_config",
            text_model=MiniCPMV4_7TextModel,
            vision_model=None,
            mtp_model=None,
            **kwargs,
        )


MiniCPMV4_7TextModel.config_class = MiniCPMV4_7Config
