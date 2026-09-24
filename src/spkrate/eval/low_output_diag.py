"""低出力事例の切り分けに使う計算（docs/PLAN.md 第6段階の追加診断）。

誤差上位100件のうち推定毎秒モーラ数が1.0未満の件（対象群）と、誤差の小さい件から
無作為に選んだ比較群について、波形・特徴量・フレームごとの出力の要約統計を出し、
群の差を Mann-Whitney U 検定と Holm 法で調べる。ここには音声の読み込みやモデルの
前向き計算を含めず、純粋な数値計算だけを置く（``tests/test_low_output_diag.py``）。

float64 は使わない（CLAUDE.md「実行環境」）。統計量は Python の float で返す。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "SILENCE_AMPLITUDE_THRESHOLD",
    "FRAME_THRESHOLDS",
    "waveform_stats",
    "feature_stats",
    "normalized_feature_stats",
    "frame_output_stats",
    "select_comparison",
    "holm_adjust",
    "group_summary",
]

#: 無音区間の割合に使う振幅の閾値（|x| がこれ未満の標本を無音とみなす）。-60dBFS。
SILENCE_AMPLITUDE_THRESHOLD = 1e-3

#: フレームごとの softplus 出力（モーラ/フレーム、1フレーム=10ミリ秒）の閾値。
#: 0.01 モーラ/フレーム = 1 モーラ/秒、0.05 モーラ/フレーム = 5 モーラ/秒 に相当する。
FRAME_THRESHOLDS = (0.01, 0.05)


def _f32(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def waveform_stats(
    samples: np.ndarray, *, threshold: float = SILENCE_AMPLITUDE_THRESHOLD
) -> dict[str, float]:
    """波形の実効値・ピーク振幅・無音標本の割合。"""
    wave = _f32(samples)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("1次元の空でない波形を渡すこと")
    absolute = np.abs(wave)
    return {
        "rms": float(np.sqrt(np.mean(np.square(wave), dtype=np.float32))),
        "peak": float(absolute.max()),
        "silence_frac": float(np.mean(absolute < np.float32(threshold), dtype=np.float32)),
    }


def feature_stats(feature: np.ndarray, prefix: str = "logmel") -> dict[str, float]:
    """``(フレーム数, n_mels)`` の特徴量の平均・標準偏差・最大・最小。"""
    array = _f32(feature)
    return {
        f"{prefix}_mean": float(array.mean(dtype=np.float32)),
        f"{prefix}_std": float(array.std(dtype=np.float32)),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_min": float(array.min()),
    }


def normalized_feature_stats(feature: np.ndarray, prefix: str = "norm") -> dict[str, float]:
    """正規化後の特徴量の平均・標準偏差・絶対値の最大。"""
    array = _f32(feature)
    return {
        f"{prefix}_mean": float(array.mean(dtype=np.float32)),
        f"{prefix}_std": float(array.std(dtype=np.float32)),
        f"{prefix}_absmax": float(np.abs(array).max()),
    }


def _threshold_key(value: float) -> str:
    return f"frame_frac_gt_{value:g}".replace(".", "p")


def frame_output_stats(
    frames: np.ndarray, *, thresholds: Sequence[float] = FRAME_THRESHOLDS
) -> dict[str, float]:
    """フレームごとの softplus 出力（詰め物を除いた有効フレームのみ）の要約統計。

    ``frame_sum`` はクリップのモーラ数の推定値そのもの。``frame_frac_gt_*`` は
    閾値を超えるフレームの割合で、全フレームが低いのか一部だけが高いのかを見るためのもの。
    ``frame_top10pct_share`` は値の大きい上位10%のフレームが総和に占める割合。
    """
    values = _f32(frames).ravel()
    if values.size == 0:
        raise ValueError("フレームが1つも無い")
    total = float(values.sum(dtype=np.float32))
    ordered = np.sort(values)[::-1]
    top_n = max(1, int(math.ceil(values.size * 0.1)))
    stats = {
        "n_frames": float(values.size),
        "frame_sum": total,
        "frame_mean": float(values.mean(dtype=np.float32)),
        "frame_median": float(np.median(values)),
        "frame_p90": float(np.quantile(values, 0.90)),
        "frame_p99": float(np.quantile(values, 0.99)),
        "frame_max": float(values.max()),
        "frame_min": float(values.min()),
        "frame_top10pct_share": (
            float(ordered[:top_n].sum(dtype=np.float32)) / total if total > 0 else float("nan")
        ),
    }
    for threshold in thresholds:
        stats[_threshold_key(threshold)] = float(
            np.mean(values > np.float32(threshold), dtype=np.float32)
        )
    return stats


def select_comparison(
    abs_errors: Mapping[str, float],
    n: int,
    *,
    seed: int,
    exclude: Sequence[str] = (),
) -> tuple[list[str], float, int]:
    """絶対誤差が全体の中央値以下のクリップから ``n`` 件を固定シードで無作為抽出する。

    中央値は ``abs_errors`` の全件（除外前）で求める。候補は clip_id 順に並べてから
    ``np.random.default_rng(seed).choice(..., replace=False)`` で選ぶので、辞書の
    順序に依存せず再現できる。

    Returns:
        (選ばれた clip_id の列（clip_id 順）, 中央値, 候補の件数)
    """
    if n < 1:
        raise ValueError(f"n は1以上: {n}")
    ids = sorted(abs_errors)
    errors = np.asarray([abs_errors[i] for i in ids], dtype=np.float32)
    median = float(np.median(errors))
    excluded = set(exclude)
    pool = [i for i, e in zip(ids, errors, strict=True) if e <= median and i not in excluded]
    if len(pool) < n:
        raise ValueError(f"候補 {len(pool)} 件が抽出数 {n} 件に満たない")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pool), size=n, replace=False)
    return sorted(pool[int(k)] for k in chosen), median, len(pool)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm 法で補正した p 値（NaN はそのまま NaN、族の数に数えない）。"""
    indexed = [(p, i) for i, p in enumerate(p_values) if not math.isnan(p)]
    m = len(indexed)
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, (p, index) in enumerate(sorted(indexed)):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def group_summary(values: Sequence[float]) -> dict[str, float]:
    """群ごとの要約（件数・最小・第1四分位・中央値・第3四分位・最大）。"""
    array = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float32)
    if array.size == 0:
        return {"n": 0.0, "min": math.nan, "q1": math.nan, "median": math.nan,
                "q3": math.nan, "max": math.nan}
    return {
        "n": float(array.size),
        "min": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }
