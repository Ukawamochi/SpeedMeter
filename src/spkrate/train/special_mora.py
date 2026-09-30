"""窓の特殊拍の割合と、損失の窓ごとの重み（docs/decisions/013-special-mora-loss-weight.md。exp020）。

- 特殊拍: 長音「ー」・撥音「ン」・促音「ッ」の1文字のモーラ（``split_mora`` の単位。
  scripts/special_mora_analysis.py の ``classify_mora`` の ``special`` と同じ）
- 窓の特殊拍の割合 r = 按分後の特殊拍のモーラ数 ÷ 按分後の総モーラ数（窓の正解）。按分は窓の正解と
  同じ ``window_mora``（各モーラの窓内の割合 f を足す）。正解が ``MIN_LABEL_MORA``（1.0モーラ、
  毎秒0.5）未満の窓（正解0の窓を含む）と無音サンプルは r = 0（scripts/special_mora_analysis.py の
  ``--min-true-mora`` の既定と同じ。正解の小さい窓では割合が0か1に偏り、端で切れたモーラ1つで
  重みが上限に届くため）
- 重み: 生の重み w_raw = min(1 + α · r, 上限)。バッチ内で平均が1になるよう w = w_raw ÷ mean(w_raw)
  （無音サンプルを含むバッチの全件で平均する）。α = 0（既定）で無効。無効なら重みを掛けない
  （損失は ``torch.nn.functional.mse_loss`` のまま）
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "SPECIAL_KANA",
    "MIN_LABEL_MORA",
    "RATIO_BIN_COUNT",
    "is_special_mora",
    "ratio_bin",
    "special_mora_flags",
    "special_ratio",
    "special_mora_weights",
    "weighted_mse_loss",
    "validate_weight_settings",
]

SPECIAL_KANA = frozenset("ーンッ")
# 集計の区間: 割合 0.1 刻み（[0, 0.1), …, [0.9, 1.0]。1.0 は最後の区間に入れる）
RATIO_BIN_COUNT = 10
# 割合を求める窓の正解の下限（モーラ）。これ未満の窓は割合0（モジュールの docstring）
MIN_LABEL_MORA = 1.0


def is_special_mora(kana: str) -> bool:
    """``split_mora`` の1単位のモーラが特殊拍（ー・ン・ッ の1文字）か。"""
    return len(kana) == 1 and kana in SPECIAL_KANA


def special_mora_flags(kanas) -> np.ndarray:
    """モーラの列（かな）の特殊拍の印（bool 配列）。"""
    return np.asarray([is_special_mora(str(k)) for k in kanas], dtype=bool)


def special_ratio(special: float, total: float) -> float:
    """窓の特殊拍の割合。正解（総モーラ数）が ``MIN_LABEL_MORA`` 未満なら0。[0, 1] に収める。"""
    total = float(total)
    if not total >= MIN_LABEL_MORA:
        return 0.0
    return float(min(max(float(special) / total, 0.0), 1.0))


def ratio_bin(ratio: float) -> int:
    """割合の 0.1 刻みの区間の番号（0〜9）。"""
    return min(max(int(math.floor(float(ratio) * RATIO_BIN_COUNT)), 0), RATIO_BIN_COUNT - 1)


def validate_weight_settings(alpha: float, cap: float) -> None:
    """α は0以上の有限値、上限は1以上の有限値。"""
    alpha = float(alpha)
    cap = float(cap)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"train.special_mora_weight_alpha は0以上の有限値: {alpha}")
    if not math.isfinite(cap) or cap < 1.0:
        raise ValueError(f"train.special_mora_weight_max は1以上の有限値: {cap}")


def special_mora_weights(ratios, alpha: float, cap: float) -> np.ndarray:
    """バッチの窓ごとの重み（float32、平均1）。

    生の重み min(1 + α · r, 上限) をバッチ内の平均で割る。上限は生の重みに掛ける
    （正規化の後の最大は 上限 ÷ バッチの生の重みの平均 で、上限をわずかに超えうる）。
    """
    validate_weight_settings(alpha, cap)
    r = np.clip(np.asarray(ratios, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    if r.size == 0:
        return np.zeros(0, dtype=np.float32)
    raw = np.minimum(np.float32(1.0) + np.float32(alpha) * r, np.float32(cap)).astype(np.float32)
    return (raw / raw.mean(dtype=np.float32)).astype(np.float32)


def weighted_mse_loss(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """重み付きの二乗誤差 mean(w · (予測 − 正解)²)。重みの平均が1なら重みなしの尺度と同じ。"""
    if weight.shape != prediction.shape:
        raise ValueError(f"重みの形が予測と違う: {tuple(weight.shape)} != {tuple(prediction.shape)}")
    return torch.mean(weight * (prediction - target) ** 2)
