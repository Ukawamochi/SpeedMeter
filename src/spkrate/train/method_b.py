"""方式B（窓単位の正解）の学習データ（docs/directives/2026-09-26.md タスク3）。

設計は docs/decisions/009-method-b.md の1節・2節。要点:

- 入力は2.0秒（32,000標本）の窓。正解は dev_window と同じ按分（``window_mora``）による窓内の
  モーラ数（float32）。``ClipItem.mora`` は窓のモーラ数、``duration_sec`` は2.0
- 1エポックの入力は (1) 長さ2.0秒以上の各クリップから1窓（単一、件数 N_single）と
  (2) N_single 組の連結から各1窓（1対1）。データセットの長さは 2 × N_single
- 連結の組は ``build_concat_groups`` を種 (seed, epoch, 0xC0CA7, p) で p = 0, 1, … と繰り返し、
  組の総数が N_single 以上になったら種 (seed, epoch, 0xC0CA7, 0x5E1EC7) で N_single 組を
  非復元抽出する（``build_epoch_concat_groups``）。エポックごとに作り直す（``set_epoch``）
- 件ごとの乱数は ``default_rng((seed, epoch, index))``。伸縮の抽選 → 伸縮率 → 窓の位置 →
  残りの拡張 → 周波数マスクの順にこの1つの生成器から引く
- 時間伸縮: 確率 ``time_stretch_prob`` で伸縮し、伸縮率 s は [max(下限, W / N), 上限] の一様
  （``time_stretch_distribution: log_uniform`` なら対数一様。011 4.2節）。
  窓の開始 w は伸縮後の音源長 N' = round(N · s) に対し {0, …, N' − W} の一様整数
- 抜粋: 元の時刻で [w / s − 余白, (w + W) / s + 余白] を音源の範囲に切り詰めた区間
  [a, b)（標本）。窓に重なるクリップだけを読み、間の無音は0で埋める（``assemble_excerpt``）
- 抜粋に「時間伸縮 → 残響 → 雑音 → 帯域制限 → 音量」を掛けてから、抜粋内の開始
  k = w − round(a · s)（[0, 出力長 − W] に収める）から W 標本を切る
- 正解: モーラ時刻 t を t' = (t − a / sr) · s に変換し、窓 [k / sr, k / sr + 2.0) で数える
  （``stretched_window_label``）。残響・雑音・帯域制限・音量・周波数マスクは時刻を変えない
- SNR の基準は抜粋の実効値（009 1.4節の注）
- 速い窓の選び直し（``windows.fast_window_redraws``。既定0で無効。docs/decisions/011-fast-window-sampling.md）:
  伸縮の抽選に当たった窓の正解が毎秒 ``fast_window_min_rate``（既定10）未満なら、別の生成器
  ``default_rng((seed, epoch, index, FAST_REDRAW_TAG))`` で伸縮率だけを最大 K 回引き直し、
  窓の相対位置 w / (N' − W) を保ったまま、閾値以上になった最初の候補を使う（無ければ元の窓）。
  主の生成器は引き直しに使わないので、残りの拡張・周波数マスクの乱数の系列は無効のときと同じ
- 強く縮めた窓に雑音を重ねない（``augment.params.noise_skip_stretch_below``。既定 None で無効。
  docs/decisions/012-no-noise-on-strong-stretch.md。exp019）: 窓の伸縮率（選び直しの後の値）が
  閾値未満なら、雑音重畳の抽選に当たっても雑音を足さない。抽選・SNR・雑音の選択の乱数は同じだけ
  消費するので、他の窓と、同じ窓の後続の拡張（帯域制限・音量・周波数マスク）の系列は無効のときと同じ。
  省いた窓は ``ClipItem.augment_applied`` に ``noise_skipped`` の印を入れ、エポックの集計に数える

モーラ区間は ``data/processed/alignments/train.jsonl``（dev.jsonl と同じ形式）から読み、
クリップ全体を連結した float32 配列と開始位置で持つ（DataLoader のワーカーへの受け渡しを
軽くするため）。クリップの標本数はアライメントの ``duration_sec × 16000``（四捨五入）で、
読み込んだ波形の長さと違えば止める（``DevWindowAudio.source_waveform`` と同じ検査）。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from spkrate.data.augment import (
    AugmentConfig,
    NoiseSource,
    augment_feature_with_status,
    augment_waveform,
    draw_time_stretch,
    time_stretch,
)
from spkrate.eval.dev_window import (
    KIND_CONCAT,
    KIND_SINGLE,
    ConcatSpec,
    WindowSource,
    build_concat_groups,
    window_mora,
)
from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram
from spkrate.train.data import ClipItem, Normalizer

__all__ = [
    "CONCAT_SEED_TAG",
    "CONCAT_PICK_TAG",
    "FAST_REDRAW_TAG",
    "NOISE_SKIPPED",
    "TrainClip",
    "WindowPlan",
    "WindowSettings",
    "WindowTrainDataset",
    "assemble_excerpt",
    "build_epoch_concat_groups",
    "draw_window_plan",
    "plan_for_stretch",
    "redraw_stretch_keep_position",
    "load_train_clips",
    "read_clip_list",
    "WindowEpochStats",
    "stretch_lower_bound",
    "stretched_window_label",
    "window_start_in_excerpt",
]

SAMPLE_RATE = 16000
CONCAT_SEED_TAG = 0xC0CA7  # 連結の組の作成の種に入れる印（009 1.3節）
CONCAT_PICK_TAG = 0x5E1EC7  # 組の非復元抽出の種に入れる印（009 1.3節）
FAST_REDRAW_TAG = 0xFA57  # 速い窓の選び直しの種に入れる印（011-fast-window-sampling）
HIGH_RATE_THRESHOLDS = (10.0, 12.0, 14.0)  # エポックの集計に数える毎秒モーラ数の閾値
# 雑音重畳を規則で省いた窓の印（ClipItem.augment_applied に入れる。拡張の実適用回数の行には出ない）
NOISE_SKIPPED = "noise_skipped"


# --------------------------------------------------------------------------------------
# 設定


@dataclass(frozen=True)
class WindowSettings:
    """方式Bの窓の設定（学習の設定ファイルの ``windows`` 節）。

    既定値は docs/decisions/009-method-b.md。連結の値は configs/eval/dev_window.yaml と同じ。
    """

    window_sec: float = 2.0
    margin_sec: float = 0.5
    concat_group_size: tuple[int, int] = (3, 5)
    concat_gap_sec: tuple[float, float] = (0.3, 1.5)
    concat_gap_round_sec: float = 0.01
    sample_rate: int = SAMPLE_RATE
    # 速い窓の選び直し（docs/decisions/011-fast-window-sampling.md。exp015）。0 で無効（既定）。
    # 伸縮の抽選に当たった窓の正解が毎秒 fast_window_min_rate 未満なら、伸縮率だけを最大この回数
    # 引き直す（窓の相対位置は保つ）。エポックあたりの窓数は変わらない
    fast_window_redraws: int = 0
    fast_window_min_rate: float = 10.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "WindowSettings":
        values = dict(mapping or {})
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(f"WindowSettings に未知の設定項目: {unknown}")
        if "concat_group_size" in values:
            values["concat_group_size"] = tuple(int(v) for v in values["concat_group_size"])
        if "concat_gap_sec" in values:
            values["concat_gap_sec"] = tuple(float(v) for v in values["concat_gap_sec"])
        settings = cls(**values)
        settings.concat_spec  # 値の検査
        if settings.margin_sec < 0:
            raise ValueError(f"windows.margin_sec は0以上: {settings.margin_sec}")
        if abs(settings.window_samples / settings.sample_rate - settings.window_sec) > 1e-9:
            raise ValueError(f"窓長が標本の整数倍でない: {settings.window_sec}")
        redraws = settings.fast_window_redraws
        if isinstance(redraws, bool) or not isinstance(redraws, int) or redraws < 0:
            raise ValueError(f"windows.fast_window_redraws は0以上の整数: {redraws!r}")
        rate = float(settings.fast_window_min_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"windows.fast_window_min_rate は正の有限値: {rate}")
        return settings

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.sample_rate))

    @property
    def margin_samples(self) -> int:
        return int(round(self.margin_sec * self.sample_rate))

    @property
    def concat_spec(self) -> ConcatSpec:
        return ConcatSpec(
            group_size=tuple(self.concat_group_size),  # type: ignore[arg-type]
            gap_sec=tuple(self.concat_gap_sec),  # type: ignore[arg-type]
            gap_round_sec=float(self.concat_gap_round_sec),
        )


# --------------------------------------------------------------------------------------
# 窓の位置と伸縮率（純粋な計算）


def stretch_lower_bound(low: float, num_samples: int, window_samples: int) -> float:
    """伸縮率の下限 max(low, W / N)。伸縮後の音源が窓より短くならないようにする（009 1.4節）。"""
    if num_samples < window_samples:
        raise ValueError(f"音源が窓より短い: {num_samples} < {window_samples}")
    return max(float(low), float(window_samples) / float(num_samples))


@dataclass(frozen=True)
class WindowPlan:
    """1件の窓の取り方（音声を読む前に決まる部分）。

    Attributes:
        stretch: 伸縮率 s（長さの倍率。伸縮しなければ1.0）
        stretch_drawn: 伸縮の抽選に当たったか
        start: 伸縮後の音源上の窓の開始 w（標本）
        stretched_length: 伸縮後の音源長 N' = round(N · s)
        excerpt_start / excerpt_end: 元の音源上の抜粋 [a, b)（標本）
        fast_redraws: 速い窓の選び直しで採った候補の番号（1始まり）。選び直していなければ0
    """

    stretch: float
    stretch_drawn: bool
    start: int
    stretched_length: int
    excerpt_start: int
    excerpt_end: int
    fast_redraws: int = 0


def plan_for_stretch(
    num_samples: int,
    *,
    stretch: float,
    stretch_drawn: bool,
    start: int,
    window_samples: int,
    margin_samples: int,
    fast_redraws: int = 0,
) -> WindowPlan:
    """伸縮率 s と窓の開始 w から抜粋の範囲を決める（乱数は使わない）。"""
    n = int(num_samples)
    stretched_length = n if stretch == 1.0 else int(round(n * stretch))
    stretched_length = max(stretched_length, window_samples)
    a = int(math.floor(start / stretch - margin_samples))
    b = int(math.ceil((start + window_samples) / stretch + margin_samples))
    a = max(0, a)
    b = min(n, b)
    return WindowPlan(
        stretch=float(stretch),
        stretch_drawn=bool(stretch_drawn),
        start=int(start),
        stretched_length=stretched_length,
        excerpt_start=a,
        excerpt_end=b,
        fast_redraws=int(fast_redraws),
    )


def draw_window_plan(
    rng: np.random.Generator,
    num_samples: int,
    *,
    window_samples: int,
    margin_samples: int,
    augment: AugmentConfig | None,
) -> WindowPlan:
    """伸縮の抽選 → 伸縮率 → 窓の位置 の順に乱数を引き、抜粋の範囲を決める。

    伸縮の確率・範囲は ``augment``（``AugmentConfig`` の time_stretch_*）に従う。
    ``augment`` が None か ``time_stretch_enabled`` が偽なら伸縮しない（抽選もしない）。
    """
    n = int(num_samples)
    if n < window_samples:
        raise ValueError(f"音源が窓より短い: {n} < {window_samples}")
    stretch = 1.0
    drawn = False
    if augment is not None and augment.time_stretch_enabled:
        if rng.random() < augment.time_stretch_prob:
            drawn = True
            low = stretch_lower_bound(augment.time_stretch_range[0], n, window_samples)
            high = float(augment.time_stretch_range[1])
            if low > high:
                raise ValueError(f"伸縮率の範囲が空: [{low}, {high}]")
            # 分布は time_stretch_distribution（既定 uniform は rng.uniform(low, high) と同じ値）。
            # どちらの分布も乱数の消費は1回で、後に続く窓の位置・拡張の系列はずれない。
            stretch = draw_time_stretch(
                rng, low, high, getattr(augment, "time_stretch_distribution", "uniform")
            )
    stretched_length = n if stretch == 1.0 else int(round(n * stretch))
    stretched_length = max(stretched_length, window_samples)
    start = int(rng.integers(0, stretched_length - window_samples + 1))
    return plan_for_stretch(
        n,
        stretch=stretch,
        stretch_drawn=drawn,
        start=start,
        window_samples=window_samples,
        margin_samples=margin_samples,
    )


def redraw_stretch_keep_position(
    plan: WindowPlan,
    rng: np.random.Generator,
    num_samples: int,
    *,
    window_samples: int,
    margin_samples: int,
    augment: AugmentConfig,
    attempt: int,
) -> WindowPlan:
    """伸縮率だけを引き直した候補（速い窓の選び直し。011-fast-window-sampling）。

    伸縮率は ``draw_window_plan`` と同じ範囲 [max(下限, W / N), 上限] と分布から ``rng`` で1回引く。
    窓の開始は相対位置 w / (N' − W) を保つ: w' = round(w · (N'' − W) / (N' − W))（N' − W = 0 なら0）。
    伸縮の抽選（確率）は引き直さない（``plan.stretch_drawn`` の窓にだけ使う）。
    """
    n = int(num_samples)
    low = stretch_lower_bound(augment.time_stretch_range[0], n, window_samples)
    high = float(augment.time_stretch_range[1])
    if low > high:
        raise ValueError(f"伸縮率の範囲が空: [{low}, {high}]")
    stretch = draw_time_stretch(
        rng, low, high, getattr(augment, "time_stretch_distribution", "uniform")
    )
    new_length = n if stretch == 1.0 else int(round(n * stretch))
    new_length = max(new_length, window_samples)
    old_span = plan.stretched_length - window_samples
    new_span = new_length - window_samples
    start = 0 if old_span <= 0 else int(round(plan.start * new_span / old_span))
    start = min(max(start, 0), new_span)
    return plan_for_stretch(
        n,
        stretch=stretch,
        stretch_drawn=True,
        start=start,
        window_samples=window_samples,
        margin_samples=margin_samples,
        fast_redraws=attempt,
    )


def window_start_in_excerpt(
    start: int, excerpt_start: int, stretch: float, excerpt_length: int, window_samples: int
) -> int:
    """抜粋（伸縮後）の中の窓の開始 k = w − round(a · s) を [0, 出力長 − W] に収めた値。"""
    k = int(start) - int(round(excerpt_start * stretch))
    upper = int(excerpt_length) - int(window_samples)
    if upper < 0:
        raise ValueError(f"抜粋が窓より短い: {excerpt_length} < {window_samples}")
    return min(max(k, 0), upper)


def stretched_window_label(
    mora_starts: np.ndarray,
    mora_ends: np.ndarray,
    *,
    excerpt_start: int,
    stretch: float,
    window_start: int,
    window_sec: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.float32:
    """伸縮した抜粋内の窓 [k / sr, k / sr + W) の正解モーラ数。

    モーラ時刻 t（音源内の秒）を t' = (t − a / sr) · s に変換し、``window_mora``
    （dev_window と同じ按分）で数える。
    """
    shift = np.float32(excerpt_start / float(sample_rate))
    s = np.float32(stretch)
    starts = (np.asarray(mora_starts, dtype=np.float32) - shift) * s
    ends = (np.asarray(mora_ends, dtype=np.float32) - shift) * s
    t0 = np.asarray([window_start / float(sample_rate)], dtype=np.float32)
    return np.float32(window_mora(starts, ends, t0, window_sec)[0])


# --------------------------------------------------------------------------------------
# 抜粋の組み立て


ClipLoader = Callable[[str], np.ndarray]


def assemble_excerpt(
    source: WindowSource,
    excerpt_start: int,
    excerpt_end: int,
    load_clip: ClipLoader,
) -> np.ndarray:
    """音源（単一または連結）の [a, b) の波形。重なるクリップだけを読み、無音は0で埋める。

    ``load_clip(clip_id)`` は16kHzの波形を返す。長さが ``source.clip_samples`` と違えば止める。
    """
    a, b = int(excerpt_start), int(excerpt_end)
    if not 0 <= a < b <= source.num_samples:
        raise ValueError(f"抜粋が音源の外: [{a}, {b}) / {source.num_samples}")
    out = np.zeros(b - a, dtype=np.float32)
    for clip_id, offset, length in zip(
        source.clip_ids, source.offsets, source.clip_samples, strict=True
    ):
        lo, hi = max(a, offset), min(b, offset + length)
        if lo >= hi:
            continue
        samples = np.asarray(load_clip(clip_id), dtype=np.float32)
        if samples.size != length:
            raise ValueError(
                f"{clip_id} の長さがアライメント時と違う: {samples.size} != {length} 標本"
            )
        out[lo - a : hi - a] = samples[lo - offset : hi - offset]
    return out


# --------------------------------------------------------------------------------------
# 学習クリップの読み込み


@dataclass
class TrainClip:
    """学習に使う1クリップ（アライメント由来の長さとモーラ区間の位置）。"""

    clip_id: str
    client_id: str
    audio_path: str
    num_samples: int
    mora_offset: int  # 全クリップを連結したモーラ配列の中の開始位置
    mora_count: int


def read_clip_list(path: str | Path) -> list[str]:
    """学習に使う clip_id の一覧（1行1件。scripts/summarize_train_alignment.py の出力）。"""
    ids: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    if len(set(ids)) != len(ids):
        raise ValueError(f"一覧に重複した clip_id がある: {path}")
    return ids


def _clip_id_of_line(line: str) -> str | None:
    """jsonl の行の先頭の ``"clip_id": "..."`` を json を解かずに取り出す（速さのため）。"""
    marker = '"clip_id": "'
    start = line.find(marker)
    if start < 0 or start > 16:
        return None
    start += len(marker)
    end = line.find('"', start)
    return line[start:end] if end > start else None


def load_train_clips(
    clip_ids: Sequence[str],
    *,
    clips_jsonl: str | Path,
    train_split: str | Path,
    alignments_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    limit: int | None = None,
) -> tuple[list[TrainClip], np.ndarray, np.ndarray]:
    """一覧の clip_id について、音声のパス・話者・標本数・モーラ区間を読む。

    ``limit`` を与えると、一覧を (client_id, clip_id) の順に並べた先頭 ``limit`` 件だけを使う
    （試走用。同一話者のクリップがまとまるので、少ない件数でも連結の組が作れる）。

    - ``train_split`` の話者に属さない clip_id があれば止める（test.json は ``load_split`` が拒否する）
    - アライメントが無い・``ok=false`` の clip_id があれば止める

    Returns:
        (クリップの列（一覧の順）, 全モーラの開始秒, 全モーラの終了秒)。後の2つは float32。
    """
    from spkrate.data.splits import load_clip_records, load_split

    wanted = list(clip_ids)
    wanted_set = set(wanted)
    speakers = set(load_split(train_split))
    records: dict[str, Any] = {}
    for record in load_clip_records(clips_jsonl):
        if record.clip_id in wanted_set:
            records[record.clip_id] = record
    missing = [c for c in wanted if c not in records]
    if missing:
        raise ValueError(f"clips.jsonl に無い clip_id が一覧にある: {len(missing)}件（例 {missing[:3]}）")
    outside = [c for c in wanted if records[c].client_id not in speakers]
    if outside:
        raise ValueError(
            f"{train_split} の話者に属さない clip_id が一覧にある: {len(outside)}件（例 {outside[:3]}）"
        )
    if limit is not None:
        wanted = sorted(wanted, key=lambda c: (records[c].client_id, c))[: int(limit)]
        wanted_set = set(wanted)

    rows: dict[str, dict] = {}
    with open(alignments_path, encoding="utf-8") as handle:
        for line in handle:
            clip_id = _clip_id_of_line(line)
            if clip_id is None or clip_id not in wanted_set:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # 書き込み中の最終行
                continue
            if row["clip_id"] in rows:
                raise ValueError(f"アライメントに重複した clip_id: {row['clip_id']}")
            rows[row["clip_id"]] = row
    no_align = [c for c in wanted if c not in rows]
    if no_align:
        raise ValueError(
            f"アライメントが無い clip_id が一覧にある: {len(no_align)}件（例 {no_align[:3]}）"
        )
    failed = [c for c in wanted if not rows[c].get("ok", False)]
    if failed:
        raise ValueError(f"アライメントが失敗した clip_id が一覧にある: {len(failed)}件（例 {failed[:3]}）")

    clips: list[TrainClip] = []
    starts: list[float] = []
    ends: list[float] = []
    for clip_id in wanted:
        row = rows[clip_id]
        record = records[clip_id]
        moras = row["moras"]
        clips.append(
            TrainClip(
                clip_id=clip_id,
                client_id=record.client_id,
                audio_path=record.audio_path,
                num_samples=int(round(float(row["duration_sec"]) * sample_rate)),
                mora_offset=len(starts),
                mora_count=len(moras),
            )
        )
        starts.extend(float(m["start"]) for m in moras)
        ends.extend(float(m["end"]) for m in moras)
    return clips, np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


# --------------------------------------------------------------------------------------
# 連結の組（エポックごと）


def build_epoch_concat_groups(
    clips_by_speaker: Mapping[str, Sequence[str]],
    *,
    seed: int,
    epoch: int,
    count: int,
    spec: ConcatSpec,
    sample_rate: int = SAMPLE_RATE,
    max_passes: int = 10000,
) -> tuple[list[tuple[str, tuple[str, ...], tuple[int, ...]]], dict[str, int]]:
    """エポック ``epoch`` の連結の組を ``count`` 組作る（009 1.3節）。

    種 (seed, epoch, CONCAT_SEED_TAG, p) で ``build_concat_groups`` を p = 0, 1, … と繰り返し、
    組の総数が ``count`` 以上になったら、種 (seed, epoch, CONCAT_SEED_TAG, CONCAT_PICK_TAG) の
    生成器で ``count`` 組を非復元抽出する（作成の順に並べた番号を昇順で取る）。

    Returns:
        (組の列, 集計（1回目の作成の集計と作成回数）)
    """
    if count <= 0:
        return [], {"passes": 0}
    pool: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = []
    first_stats: dict[str, int] = {}
    passes = 0
    while len(pool) < count:
        if passes >= max_passes:
            raise RuntimeError(f"連結の組が {max_passes} 回の作成でも {count} 組に届かない")
        groups, stats = build_concat_groups(
            clips_by_speaker,
            seed=(int(seed), int(epoch), CONCAT_SEED_TAG, passes),
            spec=spec,
            sample_rate=sample_rate,
        )
        if passes == 0:
            first_stats = dict(stats)
            if not groups:
                raise ValueError(
                    "連結の組が1つも作れない（同一話者のクリップが "
                    f"{spec.group_size[0]} 件以上ある話者が無い）"
                )
        pool.extend(groups)
        passes += 1
    rng = np.random.default_rng((int(seed), int(epoch), CONCAT_SEED_TAG, CONCAT_PICK_TAG))
    picked = np.sort(rng.choice(len(pool), size=count, replace=False))
    summary = {f"first_pass_{k}": int(v) for k, v in first_stats.items()}
    summary["passes"] = passes
    summary["pool_groups"] = len(pool)
    return [pool[int(i)] for i in picked], summary


# --------------------------------------------------------------------------------------
# データセット


class WindowTrainDataset(Dataset):
    """方式Bの学習データ（単一の窓 N_single 件 → 連結の窓 N_single 件の順）。

    ``set_epoch`` で連結の組を作り直す（主プロセスで呼ぶ。DataLoader のワーカーは非永続なので
    エポックごとに最新の状態が渡る）。構築時にはエポック0の組を作っておく。
    """

    def __init__(
        self,
        clips: Sequence[TrainClip],
        mora_starts: np.ndarray,
        mora_ends: np.ndarray,
        *,
        audio_root: str | Path = ".",
        settings: WindowSettings = WindowSettings(),
        normalizer: Normalizer | None = None,
        augment: AugmentConfig | None = None,
        noise_source: NoiseSource | None = None,
        seed: int = 0,
        mel_config: Any = MEL_DEFAULTS,
        loader: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        if not clips:
            raise ValueError("クリップが1件も無い")
        self.clips = list(clips)
        self.mora_starts = np.asarray(mora_starts, dtype=np.float32)
        self.mora_ends = np.asarray(mora_ends, dtype=np.float32)
        self.audio_root = Path(audio_root)
        self.settings = settings
        self.normalizer = normalizer
        self.augment = augment
        # 残りの拡張（時間伸縮は窓の位置と一緒に自前で掛ける）
        self.rest_augment = (
            None if augment is None else replace(augment, time_stretch_enabled=False)
        )
        self.noise_source = noise_source
        if settings.fast_window_redraws > 0 and (
            augment is None or not augment.time_stretch_enabled
        ):
            raise ValueError(
                "windows.fast_window_redraws は時間伸縮が有効なときだけ使える"
                "（伸縮率を引き直して速い窓を増やすため）"
            )
        self.seed = int(seed)
        self.mel_config = mel_config
        self.loader = loader
        self._transform: LogMelSpectrogram | None = None
        self._index = {clip.clip_id: i for i, clip in enumerate(self.clips)}

        w = settings.window_samples
        self.single_indices = [i for i, clip in enumerate(self.clips) if clip.num_samples >= w]
        if not self.single_indices:
            raise ValueError("長さが窓以上のクリップが1件も無い")
        by_speaker: dict[str, list[str]] = {}
        for clip in self.clips:
            by_speaker.setdefault(clip.client_id, []).append(clip.clip_id)
        self.clips_by_speaker = by_speaker
        low = settings.concat_group_size[0]
        self.stats: dict[str, int] = {
            "clips": len(self.clips),
            "single_windows": len(self.single_indices),
            "short_clips": len(self.clips) - len(self.single_indices),
            "speakers": len(by_speaker),
            "speakers_too_few_clips": sum(1 for v in by_speaker.values() if len(v) < low),
            "clips_unused_too_few": sum(len(v) for v in by_speaker.values() if len(v) < low),
            "short_clips_unused_too_few": sum(
                1
                for clip in self.clips
                if clip.num_samples < w and len(by_speaker[clip.client_id]) < low
            ),
        }
        self.epoch = 0
        self.groups: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = []
        self.group_stats: dict[str, int] = {}
        self.set_epoch(0)

    # -- 大きさ・長さ
    @property
    def num_single(self) -> int:
        return len(self.single_indices)

    def __len__(self) -> int:
        return 2 * self.num_single

    @property
    def frame_counts(self) -> list[int]:
        from spkrate.features.melspec import num_frames

        return [num_frames(self.settings.window_samples, self.mel_config)] * len(self)

    @property
    def durations_sec(self) -> list[float]:
        """全件が窓長（2.0秒）。無音サンプルの長さの分布に使う（009 1.5節）。"""
        return [float(self.settings.window_sec)] * len(self)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.groups, self.group_stats = build_epoch_concat_groups(
            self.clips_by_speaker,
            seed=self.seed,
            epoch=self.epoch,
            count=self.num_single,
            spec=self.settings.concat_spec,
            sample_rate=self.settings.sample_rate,
        )

    # -- 音源
    def source(self, index: int) -> WindowSource:
        """index 番目の件の音源（単一クリップか連結）。"""
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.num_single:
            clip = self.clips[self.single_indices[index]]
            return WindowSource(
                source_id=clip.clip_id,
                kind=KIND_SINGLE,
                client_id=clip.client_id,
                clip_ids=(clip.clip_id,),
                clip_samples=(clip.num_samples,),
                gap_samples=(),
            )
        client_id, members, gaps = self.groups[index - self.num_single]
        return WindowSource(
            source_id="concat:" + "+".join(members),
            kind=KIND_CONCAT,
            client_id=client_id,
            clip_ids=members,
            clip_samples=tuple(self.clips[self._index[c]].num_samples for c in members),
            gap_samples=gaps,
        )

    def source_moras(self, source: WindowSource) -> tuple[np.ndarray, np.ndarray]:
        """音源内の全モーラの (開始秒, 終了秒)。連結では各クリップの開始位置だけずらす。"""
        starts, ends = [], []
        sr = float(self.settings.sample_rate)
        for clip_id, offset in zip(source.clip_ids, source.offsets, strict=True):
            clip = self.clips[self._index[clip_id]]
            sl = slice(clip.mora_offset, clip.mora_offset + clip.mora_count)
            shift = np.float32(offset / sr)
            starts.append(self.mora_starts[sl] + shift)
            ends.append(self.mora_ends[sl] + shift)
        if not starts:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        return np.concatenate(starts).astype(np.float32), np.concatenate(ends).astype(np.float32)

    def _load_clip(self, clip_id: str) -> np.ndarray:
        clip = self.clips[self._index[clip_id]]
        if self.loader is not None:
            return self.loader(clip.audio_path)
        from spkrate.eval.audio import load_audio

        return load_audio(self.audio_root / clip.audio_path)[0]

    def _mel(self) -> LogMelSpectrogram:
        if self._transform is None:
            self._transform = LogMelSpectrogram(self.mel_config)
        return self._transform

    # -- 1件
    def window(self, index: int) -> dict[str, Any]:
        """index 番目の窓の波形・正解と途中の値（特徴量の前まで。単体テストにも使う）。

        乱数は ``default_rng((seed, epoch, index))`` の1つだけで、伸縮の抽選 → 伸縮率 →
        窓の位置 → 残りの拡張 の順に引く。呼び出し後の生成器を ``rng`` として返す
        （周波数マスクに続けて使う）。
        """
        settings = self.settings
        w_samples = settings.window_samples
        source = self.source(index)
        rng, plan = self._draw_plan(index, source)
        excerpt = assemble_excerpt(source, plan.excerpt_start, plan.excerpt_end, self._load_clip)
        applied: list[str] = []
        if plan.stretch != 1.0:
            excerpt = time_stretch(excerpt, plan.stretch, sample_rate=settings.sample_rate)
            applied.append("time_stretch")
        if self.rest_augment is not None:
            result = augment_waveform(
                excerpt,
                rng,
                config=self.rest_augment,
                noise_source=self.noise_source,
                applied_stretch=plan.stretch,
            )
            if result.samples.size != excerpt.size:
                raise RuntimeError("時刻を変えない拡張で長さが変わった")
            excerpt = result.samples
            applied.extend(result.effective)
            if "noise" in result.skipped:
                applied.append(NOISE_SKIPPED)
        k = window_start_in_excerpt(
            plan.start, plan.excerpt_start, plan.stretch, excerpt.size, w_samples
        )
        waveform = np.ascontiguousarray(excerpt[k : k + w_samples], dtype=np.float32)
        starts, ends = self.source_moras(source)
        label = stretched_window_label(
            starts,
            ends,
            excerpt_start=plan.excerpt_start,
            stretch=plan.stretch,
            window_start=k,
            window_sec=settings.window_sec,
            sample_rate=settings.sample_rate,
        )
        return {
            "source": source,
            "plan": plan,
            "excerpt_window_start": k,
            "waveform": waveform,
            "mora": label,
            "applied": applied,
            "rng": rng,
        }

    def window_label(self, index: int) -> tuple[float, WindowPlan, WindowSource]:
        """index 番目の窓の正解を、音声を読まずに求める（``window`` の ``mora`` と同じ値）。

        正解はアライメント・伸縮率・窓の位置だけで決まる（011 4.2節の学習前の確認に使う）。
        伸縮後の抜粋の長さは ``time_stretch`` と同じ ``round(抜粋長 · s)``（伸縮なしなら抜粋長）。
        乱数は ``window`` と同じ生成器から伸縮の抽選 → 伸縮率 → 窓の位置 までを引く。

        Returns:
            (窓の正解モーラ数, 窓の取り方, 音源)
        """
        source = self.source(index)
        _, plan = self._draw_plan(index, source)
        starts, ends = self.source_moras(source)
        return self._plan_label(plan, starts, ends), plan, source

    def _plan_label(self, plan: WindowPlan, starts: np.ndarray, ends: np.ndarray) -> float:
        """窓の取り方から、音声を読まずに正解モーラ数を求める（``window`` の ``mora`` と同じ値）。"""
        settings = self.settings
        w_samples = settings.window_samples
        length = plan.excerpt_end - plan.excerpt_start
        if plan.stretch != 1.0:
            length = int(round(length * plan.stretch))
        k = window_start_in_excerpt(plan.start, plan.excerpt_start, plan.stretch, length, w_samples)
        label = stretched_window_label(
            starts,
            ends,
            excerpt_start=plan.excerpt_start,
            stretch=plan.stretch,
            window_start=k,
            window_sec=settings.window_sec,
            sample_rate=settings.sample_rate,
        )
        return float(label)

    def _draw_plan(
        self, index: int, source: WindowSource
    ) -> tuple[np.random.Generator, WindowPlan]:
        """件の主の生成器と窓の取り方（速い窓の選び直しを含む）。

        主の生成器 ``default_rng((seed, epoch, index))`` から伸縮の抽選 → 伸縮率 → 窓の位置を引く。
        ``fast_window_redraws`` が正で、伸縮の抽選に当たり、正解が毎秒 ``fast_window_min_rate``
        未満なら、別の生成器 ``default_rng((seed, epoch, index, FAST_REDRAW_TAG))`` で伸縮率だけを
        最大 K 回引き直し、閾値以上になった最初の候補を使う（無ければ元の窓）。主の生成器は
        引き直しに使わないので、返した後の系列（残りの拡張・周波数マスク）は無効のときと同じ。
        """
        settings = self.settings
        rng = np.random.default_rng((self.seed, self.epoch, index))
        plan = draw_window_plan(
            rng,
            source.num_samples,
            window_samples=settings.window_samples,
            margin_samples=settings.margin_samples,
            augment=self.augment,
        )
        redraws = int(settings.fast_window_redraws)
        if redraws <= 0 or not plan.stretch_drawn or self.augment is None:
            return rng, plan
        threshold = float(settings.fast_window_min_rate) * float(settings.window_sec)
        starts, ends = self.source_moras(source)
        if self._plan_label(plan, starts, ends) >= threshold:
            return rng, plan
        sub = np.random.default_rng((self.seed, self.epoch, index, FAST_REDRAW_TAG))
        for attempt in range(1, redraws + 1):
            candidate = redraw_stretch_keep_position(
                plan,
                sub,
                source.num_samples,
                window_samples=settings.window_samples,
                margin_samples=settings.margin_samples,
                augment=self.augment,
                attempt=attempt,
            )
            if self._plan_label(candidate, starts, ends) >= threshold:
                return rng, candidate
        return rng, plan

    def __getitem__(self, index: int) -> ClipItem:
        info = self.window(index)
        rng = info["rng"]
        applied = list(info["applied"])
        feature = self._mel()(info["waveform"], self.settings.sample_rate)
        if self.normalizer is not None:
            feature = self.normalizer(feature)
        if self.augment is not None:
            feature, masked = augment_feature_with_status(feature, rng, config=self.augment)
            if masked:
                applied.append("freq_mask")
        source: WindowSource = info["source"]
        plan: WindowPlan = info["plan"]
        return ClipItem(
            features=np.asarray(feature, dtype=np.float32),
            mora=float(info["mora"]),
            duration_sec=float(self.settings.window_sec),
            clip_id=f"{source.kind}:{source.clip_ids[0]}@{index}",
            augment_applied=tuple(applied),
            kind=source.kind,
            stretch=float(plan.stretch),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_transform"] = None  # torch のモジュールをワーカーへ持ち越さない
        return state


# --------------------------------------------------------------------------------------
# エポックごとの窓の集計（metrics.jsonl と log.txt に書く。009 1.4節）


@dataclass
class WindowEpochStats:
    """学習中に見た窓の集計（無音サンプルは含めない）。"""

    kinds: Counter = field(default_factory=Counter)
    stretched: int = 0
    stretch_sum: float = 0.0
    stretch_min: float = math.inf
    stretch_max: float = -math.inf
    zero: int = 0
    bands: Counter = field(default_factory=Counter)
    # 毎秒10・12・14モーラ以上の窓の件数（高速度域。docs/directives/2026-09-28-fast-speech.md）
    high: Counter = field(default_factory=Counter)
    total: int = 0
    # 雑音を実際に重ねた窓（augment_applied に noise）と、規則で省いた窓（noise_skipped）の件数
    # （docs/decisions/012-no-noise-on-strong-stretch.md）。applied を渡した update だけで数える
    noise: int = 0
    noise_skipped: int = 0

    def update(
        self,
        kinds: Iterable[str],
        stretches: Iterable[float],
        moras: Iterable[float],
        durations: Iterable[float],
        applied: Iterable[Sequence[str]] | None = None,
    ) -> None:
        """種類が空の件（方式Aのクリップ・無音サンプル）は数えない。

        ``applied`` は件ごとの ``ClipItem.augment_applied``（``Batch.augment_applied``）。
        渡したときだけ雑音を重ねた窓・省いた窓を数える。
        """
        from spkrate.eval.metrics import band_of

        kinds = list(kinds)
        names_per_item: Iterable[Sequence[str]] = (
            [()] * len(kinds) if applied is None else applied
        )
        for kind, s, m, d, names in zip(
            kinds, stretches, moras, durations, names_per_item, strict=True
        ):
            if not kind:
                continue
            self.total += 1
            if "noise" in names:
                self.noise += 1
            if NOISE_SKIPPED in names:
                self.noise_skipped += 1
            self.kinds[kind] += 1
            if s != 1.0:
                self.stretched += 1
                self.stretch_sum += float(s)
                self.stretch_min = min(self.stretch_min, float(s))
                self.stretch_max = max(self.stretch_max, float(s))
            if float(m) == 0.0:
                self.zero += 1
            rate = float(m) / float(d)
            self.bands[band_of(rate)] += 1
            for threshold in HIGH_RATE_THRESHOLDS:
                if rate >= threshold:
                    self.high[threshold] += 1

    def as_dict(self) -> dict[str, Any]:
        from spkrate.eval.metrics import BAND_KEYS

        total = self.total
        return {
            "window_total": total,
            "window_single": int(self.kinds.get(KIND_SINGLE, 0)),
            "window_concat": int(self.kinds.get(KIND_CONCAT, 0)),
            "window_stretch_rate": self.stretched / total if total else float("nan"),
            "window_stretch_mean": self.stretch_sum / self.stretched if self.stretched else float("nan"),
            "window_stretch_min": self.stretch_min if self.stretched else float("nan"),
            "window_stretch_max": self.stretch_max if self.stretched else float("nan"),
            "window_zero": self.zero,
            "window_zero_rate": self.zero / total if total else float("nan"),
            **{f"window_band_{b}": int(self.bands.get(b, 0)) for b in BAND_KEYS},
            **{f"window_rate_ge{int(t)}": int(self.high.get(t, 0)) for t in HIGH_RATE_THRESHOLDS},
            "window_noise": self.noise,
            "window_noise_rate": self.noise / total if total else float("nan"),
            "window_noise_skipped": self.noise_skipped,
            "window_noise_skipped_rate": self.noise_skipped / total if total else float("nan"),
        }

    def describe(self, epoch: int) -> str:
        d = self.as_dict()
        return (
            f"エポック{epoch} 窓の集計: 計={d['window_total']} 単一={d['window_single']} "
            f"連結={d['window_concat']} 伸縮の適用率={d['window_stretch_rate']:.3f} "
            f"伸縮率(平均/最小/最大)={d['window_stretch_mean']:.3f}/{d['window_stretch_min']:.3f}/"
            f"{d['window_stretch_max']:.3f} 正解0の窓={d['window_zero']}({d['window_zero_rate']:.4f}) "
            f"帯別(<4/4-6/6-8/>=8)={d['window_band_under4']}/{d['window_band_4to6']}/"
            f"{d['window_band_6to8']}/{d['window_band_over8']} "
            f"毎秒10/12/14以上={d['window_rate_ge10']}/{d['window_rate_ge12']}/{d['window_rate_ge14']} "
            f"雑音を重ねた窓={d['window_noise']}({d['window_noise_rate']:.4f}) "
            f"雑音を省いた窓={d['window_noise_skipped']}({d['window_noise_skipped_rate']:.4f})"
        )
