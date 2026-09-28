"""高速度域の評価（docs/directives/2026-09-28-fast-speech.md タスク1）。

評価1（正解の話速の区間別）と評価2（速めた音声 dev_fast）の純粋な計算をここに置く。
入口は ``scripts/build_dev_fast.py``（dev_fast の作成）と ``scripts/eval_fast_speech.py``
（推論・集計・metrics.csv への追記）、設定は ``configs/eval/dev_fast.yaml``。

## 区間（評価1・2で共通）

窓の**正解の**毎秒モーラ数（窓内のモーラ数 ÷ 2.0秒）を1刻みの区間に分ける。
``lt6``（6未満）・``6to7``（6以上7未満）・…・``13to14``（13以上14未満）・``ge14``（14以上）。
指示書の区間は 6以上7未満〜14以上であり、``lt6`` は参考として json と md にだけ出す。
区間ごとに窓数・正解の平均・出力の平均・出力の10%点と90%点・偏り（出力 − 正解の平均）・MAE を
求める（``binned_summary``）。窓数が0の区間は値を NaN にしてそのまま残す。

## dev_fast（速めた音声）

- 音源: dev_window（``data/processed/dev_window``）の音源（単一クリップと連続発話の組）から、
  最も速い条件（2倍速）で窓を1つ以上持つもの（元の長さ ≥ 2.0秒 × 2 = 4.0秒）を候補とし、
  ``default_rng(seed).permutation`` の先頭 ``num_sources`` 件を選ぶ（``select_sources``）。
  選んだ音源は全条件で共通。dev_window の除外（アライメント失敗・端の孤立）は音源の段階で
  済んでいる。主指標の除外（known_no_speech・no_speech_suspect）は dev_window と同じく集計で行う
- 速める: 音源の波形全体（連続発話は無音を含む連結音声の全体）を ``speed`` 倍速にする。
  長さの倍率は ``stretch = 1 / speed``、速めた音源の標本数は ``round(元の標本数 × stretch)``
  （``stretched_num_samples``）。方式は WSOLA（audiotsm、``wsola_speed_up``）で、学習の
  時間伸縮（位相ボコーダ、``spkrate.data.augment.time_stretch``）とは別の方式である
- 時刻の変換: 音源内のモーラ時刻 t（連結では各クリップの開始位置を足した時刻）を ``t × stretch``
  に変換する（``stretched_source_moras``。docs/spec.md の方式Bの時間伸縮と同じ規則）
- 窓と正解: 速めた音源に dev_window と同じ規則（2.0秒窓・0.25秒ずらし・窓全体が収まるものだけ・
  按分の正解 ``window_mora``）を適用する（``build_fast_windows``）
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spkrate.eval.dev_window import (
    DevWindowSet,
    WindowSource,
    window_mora,
    window_starts,
)

__all__ = [
    "RATE_BIN_EDGES",
    "RATE_BIN_KEYS",
    "REPORTED_BIN_KEYS",
    "binned_metrics_rows",
    "binned_summary",
    "build_fast_windows",
    "condition_name",
    "rate_bin_codes",
    "select_sources",
    "stretched_num_samples",
    "stretched_source_moras",
    "wsola_speed_up",
]

# 区間の境界（毎秒モーラ数）。区間 i は [edges[i-1], edges[i])、先頭は (-inf, 6)、末尾は [14, inf)
RATE_BIN_EDGES: tuple[float, ...] = tuple(float(v) for v in range(6, 15))
RATE_BIN_KEYS: tuple[str, ...] = (
    "lt6",
    *(f"{int(lo)}to{int(hi)}" for lo, hi in zip(RATE_BIN_EDGES[:-1], RATE_BIN_EDGES[1:])),
    "ge14",
)
# 指示書の区間（6以上7未満〜14以上）。metrics.csv にはこの区間だけを書く
REPORTED_BIN_KEYS: tuple[str, ...] = RATE_BIN_KEYS[1:]


def condition_name(speed: float) -> str:
    """速さの条件名（``1.5`` → ``x1.5``、``2.0`` → ``x2``）。"""
    return f"x{float(speed):g}"


# --------------------------------------------------------------------------------------
# 区間別の集計


def rate_bin_codes(rate: np.ndarray) -> np.ndarray:
    """毎秒モーラ数から区間の番号（``RATE_BIN_KEYS`` の添字）。"""
    values = np.asarray(rate, dtype=np.float32)
    if np.isnan(values).any():
        raise ValueError("正解に NaN がある")
    edges = np.asarray(RATE_BIN_EDGES, dtype=np.float32)
    # side="right": 値がちょうど境界なら上の区間（6.0 → 6to7、14.0 → ge14）
    return np.searchsorted(edges, values, side="right").astype(np.int8)


def _stat(values: np.ndarray, fn: Any) -> float:
    return float(fn(values)) if values.size else float("nan")


def binned_summary(
    true_mora: np.ndarray, pred_mora: np.ndarray, window_sec: float = 2.0
) -> dict[str, dict[str, float]]:
    """窓の正解・出力（窓内のモーラ数）から、正解の話速の区間ごとの集計。

    Returns:
        ``{区間: {n, true_mean, pred_mean, pred_p10, pred_p90, bias, mae}}``（毎秒モーラ数）。
        すべての区間（``RATE_BIN_KEYS``）を含み、0件の区間の値は NaN。
    """
    t = np.asarray(true_mora, dtype=np.float32) / np.float32(window_sec)
    p = np.asarray(pred_mora, dtype=np.float32) / np.float32(window_sec)
    if t.shape != p.shape:
        raise ValueError(f"正解と出力の件数が違う: {t.shape} != {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("出力に有限でない値がある")
    codes = rate_bin_codes(t)
    out: dict[str, dict[str, float]] = {}
    for index, key in enumerate(RATE_BIN_KEYS):
        mask = codes == index
        tt, pp = t[mask], p[mask]
        out[key] = {
            "n": int(mask.sum()),
            "true_mean": _stat(tt, np.mean),
            "pred_mean": _stat(pp, np.mean),
            "pred_p10": _stat(pp, lambda v: np.quantile(v, 0.1)),
            "pred_p90": _stat(pp, lambda v: np.quantile(v, 0.9)),
            "bias": _stat(pp - tt, np.mean),
            "mae": _stat(np.abs(pp - tt), np.mean),
        }
    return out


def binned_metrics_rows(
    split_prefix: str,
    true_mora: np.ndarray,
    pred_mora: np.ndarray,
    *,
    window_sec: float = 2.0,
    include_overall: bool,
) -> list[tuple[str, Any]]:
    """metrics.csv に書く行の (split, ``SpeedRateMetrics``) の列。

    ``include_overall`` なら先頭に全窓の行（split は ``split_prefix``）、続けて
    ``REPORTED_BIN_KEYS`` の区間のうち窓が1つ以上ある区間の行（split は
    ``{split_prefix}_bin_{区間}``。例: ``dev_window_bin_12to13``、``dev_fast_x2_bin_ge14``）。
    値は ``metrics.compute_metrics``（窓長 ``window_sec``）で、既存の列（窓数・MAE・4帯の MAE と
    窓数・相関係数）にそのまま入る。偏り・出力の平均・分位点は列が無いので json と md にだけ出す。
    """
    from spkrate.eval.metrics import compute_metrics

    t = np.asarray(true_mora, dtype=np.float32)
    p = np.asarray(pred_mora, dtype=np.float32)
    rows: list[tuple[str, Any]] = []
    if include_overall:
        rows.append((split_prefix, compute_metrics(t.tolist(), p.tolist(), [window_sec] * t.size)))
    codes = rate_bin_codes(t / np.float32(window_sec))
    for key in REPORTED_BIN_KEYS:
        mask = codes == RATE_BIN_KEYS.index(key)
        if not mask.any():
            continue
        rows.append((f"{split_prefix}_bin_{key}",
                     compute_metrics(t[mask].tolist(), p[mask].tolist(), [window_sec] * int(mask.sum()))))
    return rows


# --------------------------------------------------------------------------------------
# dev_fast の音源の選択・時刻の変換・窓


def stretched_num_samples(num_samples: int, speed: float) -> int:
    """``speed`` 倍速にした音源の標本数 ``round(num_samples / speed)``。"""
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"速さは正の有限値: {speed}")
    return int(round(int(num_samples) / float(speed)))


def select_sources(
    window_set: DevWindowSet,
    *,
    seed: int,
    num_sources: int,
    max_speed: float,
    window_samples: int,
) -> np.ndarray:
    """dev_window の音源から dev_fast に使う音源の番号を選ぶ（昇順）。

    候補は ``max_speed`` 倍速にしても窓を1つ以上持つ音源（``stretched_num_samples`` が
    ``window_samples`` 以上）。候補を音源の番号順に並べ、``default_rng(seed).permutation`` の
    先頭 ``num_sources`` 件を取り、番号順に戻す。候補が足りなければ ``ValueError``。
    """
    eligible = np.array(
        [
            number
            for number, source in enumerate(window_set.sources)
            if stretched_num_samples(source.num_samples, max_speed) >= window_samples
        ],
        dtype=np.int64,
    )
    if num_sources > eligible.size:
        raise ValueError(f"候補の音源が足りない: {eligible.size} < {num_sources}")
    rng = np.random.default_rng(int(seed))
    chosen = eligible[rng.permutation(eligible.size)[:num_sources]]
    return np.sort(chosen)


def stretched_source_moras(
    source: WindowSource,
    alignments: Mapping[str, Mapping[str, Any]],
    sample_rate: int,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``speed`` 倍速にした音源内の全モーラの (開始秒, 終了秒)。

    元の音源内の時刻（``dev_window.source_moras`` と同じ。連結では各クリップの開始位置を足す）に
    ``stretch = 1 / speed`` を掛ける。足し算と掛け算は Python の数で行ってから float32 にする
    （``speed = 1`` なら ``source_moras`` とビット一致する）。
    """
    stretch = 1.0 / float(speed)
    starts: list[float] = []
    ends: list[float] = []
    for clip_id, offset in zip(source.clip_ids, source.offsets, strict=True):
        shift = offset / float(sample_rate)
        for mora in alignments[clip_id]["moras"]:
            starts.append((float(mora["start"]) + shift) * stretch)
            ends.append((float(mora["end"]) + shift) * stretch)
    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


def build_fast_windows(
    window_set: DevWindowSet,
    numbers: Sequence[int],
    alignments: Mapping[str, Mapping[str, Any]],
    *,
    speed: float,
    window_sec: float = 2.0,
    hop_sec: float = 0.25,
    sample_rate: int = 16000,
) -> tuple[DevWindowSet, np.ndarray]:
    """選んだ音源を ``speed`` 倍速にしたときの窓の定義と正解。

    Returns:
        (窓の定義 ``DevWindowSet``（``sources`` は選んだ音源の元の定義、``source_index`` は
        その中の番号、``start_sample`` は速めた音源内の開始標本、``mora`` は按分の正解）,
        速めた音源の標本数の配列)
    """
    window_samples = int(round(window_sec * sample_rate))
    hop_samples = int(round(hop_sec * sample_rate))
    sources = [window_set.sources[int(n)] for n in numbers]
    lengths = np.array([stretched_num_samples(s.num_samples, speed) for s in sources], dtype=np.int64)
    kind_codes = {"single": 0, "concat": 1}
    idx, st, mora, kind, known = [], [], [], [], []
    for number, (source, length) in enumerate(zip(sources, lengths, strict=True)):
        starts = window_starts(int(length), window_samples, hop_samples)
        if starts.size == 0:
            continue
        m_start, m_end = stretched_source_moras(source, alignments, sample_rate, speed)
        labels = window_mora(m_start, m_end, starts.astype(np.float32) / np.float32(sample_rate), window_sec)
        idx.append(np.full(starts.size, number, dtype=np.int32))
        st.append(starts)
        mora.append(labels)
        kind.append(np.full(starts.size, kind_codes[source.kind], dtype=np.int8))
        known.append(np.full(starts.size, source.known_no_speech, dtype=bool))

    def cat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    fast = DevWindowSet(
        sources=sources,
        source_index=cat(idx, np.int32),
        start_sample=cat(st, np.int64),
        mora=cat(mora, np.float32),
        kind=cat(kind, np.int8),
        known_no_speech=cat(known, bool),
    )
    return fast, lengths


# --------------------------------------------------------------------------------------
# 速める処理（WSOLA）


def wsola_speed_up(
    samples: np.ndarray,
    speed: float,
    *,
    frame_length: int = 1024,
    synthesis_hop: int | None = None,
    tolerance: int | None = None,
    pad_samples: int = 4096,
) -> np.ndarray:
    """WSOLA（audiotsm の ``wsola``）で ``speed`` 倍速にした波形。長さは ``round(N / speed)``。

    audiotsm は末尾の数フレームを出力しない（出力が ``N / speed`` より短くなる）ため、
    入力の末尾に ``pad_samples`` 標本の0を足してから速め、先頭から ``round(N / speed)`` 標本を
    取る。足りない場合は ``ValueError``（``pad_samples`` を増やす）。出力の時刻 t' は入力の時刻
    t と t' ≈ t / speed で対応する（WSOLA の窓の位置合わせにより ``tolerance`` 標本以内でずれる）。
    """
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("モノラルの1次元の波形を渡す")
    target = stretched_num_samples(x.size, speed)
    # audiotsm の分析の刻みは int(合成の刻み × speed)。切り捨てで実際の速さがずれないようにする
    hop = int(synthesis_hop) if synthesis_hop is not None else int(frame_length) // 2
    if abs(hop * float(speed) - round(hop * float(speed))) > 1e-9:
        raise ValueError(f"合成の刻み {hop} × 速さ {speed} が整数でない（実際の速さがずれる）")
    padded = np.concatenate([x, np.zeros(int(pad_samples), dtype=np.float32)])
    writer = ArrayWriter(channels=1)
    kwargs: dict[str, Any] = {"speed": float(speed), "frame_length": int(frame_length)}
    if synthesis_hop is not None:
        kwargs["synthesis_hop"] = int(synthesis_hop)
    if tolerance is not None:
        kwargs["tolerance"] = int(tolerance)
    wsola(1, **kwargs).run(ArrayReader(padded[None, :]), writer)
    y = np.asarray(writer.data, dtype=np.float32)[0]
    if y.size < target:
        raise ValueError(f"WSOLA の出力が短い: {y.size} < {target}（pad_samples を増やす）")
    return np.ascontiguousarray(y[:target])
