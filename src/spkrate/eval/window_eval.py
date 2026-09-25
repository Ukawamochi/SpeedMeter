"""窓単位の評価（docs/directives/2026-09-25.md タスク5-2）の集計。

dev_window（``spkrate.eval.dev_window``）の各窓について、推定器の出力（窓内のモーラ数）と
正解（窓内のモーラ数、按分）から指標を求める。推論は ``scripts/eval_dev_window.py`` が行い、
窓ごとの予測を保存する。ここには保存した予測から指標を計算する純粋な関数だけを置く
（除外の有無や集計の切り方を再推論なしで変えられるようにするため）。

## 単位

- 窓の毎秒モーラ数 = 窓内のモーラ数 ÷ 窓長（2.0秒）。正解も予測も同じ換算
- 話速帯は**正解の**毎秒モーラ数で分ける（``metrics.band_of`` と同じ4帯。正解0の窓は「4未満」）
- 偏り（bias）= 予測 − 正解 の平均（毎秒モーラ数。正なら過大）

## 集計

- ``rate_summary``: 件数・MAE・偏り・相関係数（全体と帯別）。MAE と相関係数は
  ``metrics.compute_metrics`` と同じ定義（単体テストで一致を確認する）
- ``zero_window_summary``: 正解0の窓での出力の平均・分位点と、窓内0.5・1.0・2.0モーラ以上を
  出した割合
- ``short_clip_window_mask``: 連続発話の窓のうち、2.0秒未満のクリップの区間と重なる窓
  （「含む」= 窓 [s, s + W) とクリップの区間 [offset, offset + 長さ) の重なりが1標本以上）
- ``shifted_window_labels``: すべてのモーラの時刻を一定量ずらして作り直した窓の正解
  （窓の正解の不確かさの見積もり。docs/questions.md 2026-09-26 回答の集計2）。
  連結音声では各クリップ内でずらし、クリップの開始位置（オフセット）は動かさない。
  ずらした時刻はそのクリップの区間 [0, クリップ長] に収める（モーラがクリップの外、
  すなわち前後の無音や隣のクリップへ出ないようにする。収めた件数を返す）
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spkrate.eval.dev_window import KIND_CONCAT, DevWindowSet, window_mora
from spkrate.eval.metrics import BAND_KEYS, SPEED_BANDS

__all__ = [
    "ZERO_THRESHOLDS_MORA",
    "band_codes",
    "rate_summary",
    "shifted_window_labels",
    "short_clip_window_mask",
    "source_window_ranges",
    "zero_window_summary",
]

# 正解0の窓で「出力が大きい」割合を数える閾値（窓内のモーラ数）
ZERO_THRESHOLDS_MORA: tuple[float, ...] = (0.5, 1.0, 2.0)
_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)


def band_codes(true_rate: np.ndarray) -> np.ndarray:
    """正解の毎秒モーラ数から帯の番号（``BAND_KEYS`` の添字）。"""
    rate = np.asarray(true_rate, dtype=np.float32)
    if np.isnan(rate).any():
        raise ValueError("正解に NaN がある")
    codes = np.full(rate.shape, -1, dtype=np.int8)
    for index, (_, low, high) in enumerate(SPEED_BANDS):
        codes[(rate >= low) & (rate < high)] = index
    if (codes < 0).any():  # pragma: no cover - SPEED_BANDS は実数全体を覆う
        raise ValueError("話速帯に入らない値がある")
    return codes


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    vx = float(np.dot(dx, dx))
    vy = float(np.dot(dy, dy))
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(np.dot(dx, dy)) / float(np.sqrt(vx * vy))


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def rate_summary(
    true_mora: np.ndarray, pred_mora: np.ndarray, window_sec: float = 2.0
) -> dict[str, Any]:
    """窓の正解・予測モーラ数から、毎秒モーラ数の指標を求める。

    Returns:
        ``n``・``mae``・``bias``・``correlation``・``true_mean``・``pred_mean`` と、
        帯ごとの ``bands[帯] = {n, mae, bias}``（0件の帯は mae・bias が NaN）。
    """
    t = np.asarray(true_mora, dtype=np.float32) / np.float32(window_sec)
    p = np.asarray(pred_mora, dtype=np.float32) / np.float32(window_sec)
    if t.shape != p.shape:
        raise ValueError(f"正解と予測の件数が違う: {t.shape} != {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("予測に有限でない値がある")
    diff = p - t
    err = np.abs(diff)
    codes = band_codes(t) if t.size else np.zeros(0, dtype=np.int8)
    bands: dict[str, dict[str, float]] = {}
    for index, key in enumerate(BAND_KEYS):
        mask = codes == index
        bands[key] = {"n": int(mask.sum()), "mae": _mean(err[mask]), "bias": _mean(diff[mask])}
    return {
        "n": int(t.size),
        "mae": _mean(err),
        "bias": _mean(diff),
        "correlation": _corr(t, p),
        "true_mean": _mean(t),
        "pred_mean": _mean(p),
        "bands": bands,
    }


def zero_window_summary(
    pred_mora: np.ndarray,
    window_sec: float = 2.0,
    thresholds_mora: Sequence[float] = ZERO_THRESHOLDS_MORA,
) -> dict[str, Any]:
    """正解0の窓での予測の分布。

    ``pred_mora`` は正解0の窓の予測（窓内のモーラ数）だけを渡す。平均・分位点は窓内の
    モーラ数と毎秒モーラ数の両方で返し、``frac_ge[閾値]`` は窓内のモーラ数が閾値以上の割合。
    """
    p = np.asarray(pred_mora, dtype=np.float32)
    out: dict[str, Any] = {"n": int(p.size)}
    if p.size == 0:
        out.update({"mean_mora": float("nan"), "mean_rate": float("nan"), "quantiles_mora": {},
                    "max_mora": float("nan"), "frac_ge": {f"{v:g}": float("nan") for v in thresholds_mora}})
        return out
    out["mean_mora"] = float(p.mean())
    out["mean_rate"] = float(p.mean()) / float(window_sec)
    out["quantiles_mora"] = {f"{q:g}": float(np.quantile(p, q)) for q in _QUANTILES}
    out["max_mora"] = float(p.max())
    out["frac_ge"] = {f"{v:g}": float((p >= np.float32(v)).mean()) for v in thresholds_mora}
    return out


def source_window_ranges(window_set: DevWindowSet) -> np.ndarray:
    """音源ごとの窓の添字の範囲 ``[開始, 終了)``（形 (音源数, 2)）。

    ``source_index`` は音源の順に連続して並んでいる前提（``build_dev_window`` の出力）で、
    そうでなければ ``ValueError``。窓を持たない音源は空の範囲。
    """
    index = np.asarray(window_set.source_index, dtype=np.int64)
    if index.size and np.any(np.diff(index) < 0):
        raise ValueError("窓が音源の順に並んでいない")
    numbers = np.arange(len(window_set.sources), dtype=np.int64)
    starts = np.searchsorted(index, numbers, side="left")
    ends = np.searchsorted(index, numbers, side="right")
    return np.stack([starts, ends], axis=1)


def short_clip_window_mask(window_set: DevWindowSet, window_samples: int) -> np.ndarray:
    """連続発話の窓のうち、``window_samples`` 未満のクリップの区間と重なる窓の真偽配列。

    重なり: 窓 [s, s + W) とクリップ [offset, offset + 長さ) の共通部分が1標本以上。
    単一クリップの窓は常に False（2.0秒未満のクリップは単一クリップの窓を持たない）。
    """
    mask = np.zeros(len(window_set), dtype=bool)
    starts = np.asarray(window_set.start_sample, dtype=np.int64)
    ranges = source_window_ranges(window_set)
    for number, source in enumerate(window_set.sources):
        if source.kind != KIND_CONCAT:
            continue
        lo, hi = ranges[number]
        if lo == hi:
            continue
        s = starts[lo:hi]
        hit = np.zeros(hi - lo, dtype=bool)
        for offset, length in zip(source.offsets, source.clip_samples, strict=True):
            if length >= window_samples:
                continue
            hit |= (s < offset + length) & (offset < s + window_samples)
        mask[lo:hi] = hit
    return mask


def shifted_window_labels(
    window_set: DevWindowSet,
    alignments: Mapping[str, Mapping[str, Any]],
    shift_sec: float,
    *,
    window_sec: float = 2.0,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """すべてのモーラの開始・終了を ``shift_sec`` ずらして作り直した窓の正解。

    各クリップ内でずらし、ずらした時刻はそのクリップの区間 [0, クリップ長] に収める。
    連結音声ではクリップのオフセットと無音長は変えない。按分は ``window_mora`` と同じ。

    Returns:
        (窓ごとの正解モーラ数 float32, 区間に収めるために時刻を動かしたモーラの数)
    """
    labels = np.zeros(len(window_set), dtype=np.float32)
    starts = np.asarray(window_set.start_sample, dtype=np.int64)
    ranges = source_window_ranges(window_set)
    clamped = 0
    shift = float(shift_sec)
    for number, source in enumerate(window_set.sources):
        lo, hi = ranges[number]
        if lo == hi:
            continue
        # 時刻の足し算は source_moras（dev_window）と同じく Python の数で行ってから float32 にする
        # （ずらし0なら元の正解とビット一致する）
        mora_start: list[float] = []
        mora_end: list[float] = []
        for clip_id, offset, length in zip(
            source.clip_ids, source.offsets, source.clip_samples, strict=True
        ):
            clip_sec = length / float(sample_rate)
            base = offset / float(sample_rate)
            for mora in alignments[clip_id]["moras"]:
                s = float(mora["start"]) + shift
                e = float(mora["end"]) + shift
                if s < 0.0 or e < 0.0 or s > clip_sec or e > clip_sec:
                    clamped += 1
                    s = min(max(s, 0.0), clip_sec)
                    e = min(max(e, 0.0), clip_sec)
                mora_start.append(s + base)
                mora_end.append(e + base)
        if not mora_start:
            continue
        labels[lo:hi] = window_mora(
            np.asarray(mora_start, dtype=np.float32),
            np.asarray(mora_end, dtype=np.float32),
            starts[lo:hi].astype(np.float32) / np.float32(sample_rate),
            window_sec,
        )
    return labels, clamped
