"""話速推定のモデル定義（docs/PLAN.md 第5段階）。"""

from spkrate.models.cnn import (
    CnnConfig,
    SpeechRateCNN,
    count_parameters,
    load_config,
    model_summary,
    receptive_field_frames,
    receptive_field_seconds,
)

__all__ = [
    "CnnConfig",
    "SpeechRateCNN",
    "count_parameters",
    "load_config",
    "model_summary",
    "receptive_field_frames",
    "receptive_field_seconds",
]
