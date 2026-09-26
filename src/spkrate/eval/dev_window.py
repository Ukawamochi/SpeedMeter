"""窓単位の評価セット dev_window の生成と波形の再生成（docs/directives/2026-09-25.md タスク5-1）。

設定は ``configs/eval/dev_window.yaml``、入口は ``scripts/build_dev_window.py``、
記録は ``results/dev_window_build.md``。

## 窓と正解

- 窓: 長さ ``window_sec``（2.0秒）の窓を ``hop_sec``（0.25秒）ずつずらす。開始は
  0, hop, 2hop, ... の標本位置で、音声の中に窓全体が収まるものだけを取る
  （詰め物をしない・末尾の端数は捨てる。``window_diag.split_into_windows`` と同じ規約）。
  ``window_sec`` 未満の音声は窓を持たない
- 正解: 窓 [t, t + W) 内のモーラ数。モーラ区間 [start, end] のうち窓内に入る時間の割合で按分する
  （``window_mora``）。end <= start の区間は点とみなし、t <= start < t + W なら1

## 単一クリップと連続発話

- 単一クリップ（``kind="single"``）: 各クリップをそのまま音源にする
- 連続発話（``kind="concat"``）: 同一話者のクリップを3〜5件、0.3〜1.5秒の無音（全標本0）を
  挟んで連結した音声を音源にする。組の作り方は ``build_concat_groups``。
  各クリップのモーラ時刻は、連結音声内でのそのクリップの開始位置だけずらして使う

## 除外と印付け

- ``ok=false`` のクリップと、先頭2モーラまたは末尾2モーラの開始時刻の差が閾値（1.0秒）を
  超えるクリップ（``isolation_flags``）は、単一からも連続発話の組からも除く
- ``known_no_speech``（人間の聴取で発話が聞き取れなかったクリップ）は除かず、印だけ付ける

## 雑音下の版

元の音源（単一クリップまたは連結音声）の全体に ``noisy.degrade_waveform``（固定残響 →
MUSAN noise 評価用の重畳。dev_noisy と同じ）を掛けてから窓を切る。雑音の乱数生成器の鍵
（``WindowSource.noise_key``）は単一クリップでは clip_id（dev_noisy と同じ雑音になる）、
連続発話では ``"concat:" + "+".join(clip_ids)``。SNR は音源全体の実効値に対する比である。

## 保存形式

窓の音声は保存しない（dev 全件で数十GBになる）。``output_dir`` に次を置く。

- ``sources.jsonl``: 音源ごとの1行（id、単一/連続の別、話者、clip_id の列、各クリップの
  標本数と連結音声内の開始標本、無音の標本数、known_no_speech）
- ``windows.npz``: 窓ごとの配列（音源の番号、開始標本、正解モーラ数、単一/連続の別、
  known_no_speech）
- ``meta.json``: 設定の写し・件数の集計

波形は ``DevWindowAudio`` で決定的に再生成する（読み込みは ``eval.audio.load_audio``）。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spkrate.eval.noisy import NoisyDevConfig, clip_key, degrade_waveform

__all__ = [
    "DEFAULT_WINDOW_CONFIG",
    "KIND_CONCAT",
    "KIND_SINGLE",
    "ConcatSpec",
    "DevWindowAudio",
    "DevWindowConfig",
    "DevWindowSet",
    "WindowSource",
    "build_concat_groups",
    "build_dev_window",
    "isolation_flags",
    "load_alignments",
    "load_dev_clips",
    "load_dev_window",
    "load_known_no_speech",
    "load_window_config",
    "save_dev_window",
    "source_moras",
    "window_mora",
    "window_starts",
]

DEFAULT_WINDOW_CONFIG = Path("configs/eval/dev_window.yaml")
KIND_SINGLE = "single"
KIND_CONCAT = "concat"
_KIND_CODES = {KIND_SINGLE: 0, KIND_CONCAT: 1}


# --------------------------------------------------------------------------------------
# 設定


@dataclass(frozen=True)
class ConcatSpec:
    group_size: tuple[int, int] = (3, 5)
    gap_sec: tuple[float, float] = (0.3, 1.5)
    gap_round_sec: float = 0.01

    def __post_init__(self) -> None:
        low, high = self.group_size
        if not 1 <= low <= high:
            raise ValueError(f"concat.group_size が不正: {self.group_size}")
        g_low, g_high = self.gap_sec
        if not 0.0 <= g_low <= g_high:
            raise ValueError(f"concat.gap_sec が不正: {self.gap_sec}")
        if self.gap_round_sec <= 0.0:
            raise ValueError(f"concat.gap_round_sec は正の値: {self.gap_round_sec}")


@dataclass(frozen=True)
class DevWindowConfig:
    """``configs/eval/dev_window.yaml`` の内容。"""

    seed: int
    noisy: NoisyDevConfig
    window_sec: float = 2.0
    hop_sec: float = 0.25
    sample_rate: int = 16000
    concat: ConcatSpec = field(default_factory=ConcatSpec)
    isolation_threshold_sec: float = 1.0
    dev_split: str = "configs/splits/dev.json"
    clips_jsonl: str = "data/processed/clips.jsonl"
    alignments: str = "data/processed/alignments/dev.jsonl"
    audio_root: str = "."
    known_no_speech_list: str | None = "results/error_cases/to_listen.tsv"
    output_dir: str = "data/processed/dev_window"

    def __post_init__(self) -> None:
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError(f"窓長・刻みは正の値: {self.window_sec}, {self.hop_sec}")
        if abs(self.window_samples / self.sample_rate - self.window_sec) > 1e-9:
            raise ValueError(f"窓長が標本の整数倍でない: {self.window_sec}")
        if abs(self.hop_samples / self.sample_rate - self.hop_sec) > 1e-9:
            raise ValueError(f"刻みが標本の整数倍でない: {self.hop_sec}")
        if "test" in Path(self.dev_split).stem:
            raise ValueError(f"dev_window に test の分割は使わない: {self.dev_split}")

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_sec * self.sample_rate))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DevWindowConfig":
        concat = mapping.get("concat") or {}
        return cls(
            seed=int(mapping["seed"]),
            noisy=NoisyDevConfig.from_mapping(mapping["noisy"]),
            window_sec=float(mapping.get("window_sec", 2.0)),
            hop_sec=float(mapping.get("hop_sec", 0.25)),
            sample_rate=int(mapping.get("sample_rate", 16000)),
            concat=ConcatSpec(
                group_size=tuple(int(v) for v in concat.get("group_size", (3, 5))),
                gap_sec=tuple(float(v) for v in concat.get("gap_sec", (0.3, 1.5))),
                gap_round_sec=float(concat.get("gap_round_sec", 0.01)),
            ),
            isolation_threshold_sec=float(mapping.get("isolation_threshold_sec", 1.0)),
            dev_split=str(mapping.get("dev_split", "configs/splits/dev.json")),
            clips_jsonl=str(mapping.get("clips_jsonl", "data/processed/clips.jsonl")),
            alignments=str(mapping.get("alignments", "data/processed/alignments/dev.jsonl")),
            audio_root=str(mapping.get("audio_root", ".")),
            known_no_speech_list=mapping.get("known_no_speech_list"),
            output_dir=str(mapping.get("output_dir", "data/processed/dev_window")),
        )


def load_window_config(path: str | Path = DEFAULT_WINDOW_CONFIG) -> DevWindowConfig:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return DevWindowConfig.from_mapping(yaml.safe_load(handle))


# --------------------------------------------------------------------------------------
# 窓と正解（純粋な計算）


def window_starts(num_samples: int, window_samples: int, hop_samples: int) -> np.ndarray:
    """窓の開始標本の列。窓全体が音声に収まるものだけ（末尾の端数は捨てる）。

    ``num_samples < window_samples`` なら空。
    """
    if num_samples < window_samples:
        return np.zeros(0, dtype=np.int64)
    count = (num_samples - window_samples) // hop_samples + 1
    return np.arange(count, dtype=np.int64) * hop_samples


def window_mora(
    starts: np.ndarray, ends: np.ndarray, window_start: np.ndarray, window_sec: float
) -> np.ndarray:
    """各窓の正解モーラ数（按分）。

    Args:
        starts, ends: モーラ区間の開始・終了（秒、音源内の時刻）
        window_start: 窓の開始（秒）の列
        window_sec: 窓長（秒）

    モーラ区間 [s, e] と窓 [t, t + W) の重なりの長さ ÷ (e − s) を足し上げる。
    ``e <= s`` の区間は点とみなし、``t <= s < t + W`` なら1、それ以外は0とする。
    """
    s = np.asarray(starts, dtype=np.float32)[None, :]
    e = np.asarray(ends, dtype=np.float32)[None, :]
    t0 = np.asarray(window_start, dtype=np.float32)[:, None]
    t1 = t0 + np.float32(window_sec)
    if s.size == 0 or t0.size == 0:
        return np.zeros(t0.shape[0], dtype=np.float32)
    length = e - s
    positive = length > 0
    overlap = np.clip(np.minimum(e, t1) - np.maximum(s, t0), 0.0, None)
    safe = np.where(positive, length, np.float32(1.0))
    fraction = np.where(positive, np.minimum(overlap / safe, np.float32(1.0)), np.float32(0.0))
    point = (~positive) & (s >= t0) & (s < t1)
    fraction = np.where(point, np.float32(1.0), fraction)
    return fraction.sum(axis=1, dtype=np.float32)


def isolation_flags(moras: Sequence[Mapping[str, float]], threshold_sec: float) -> tuple[bool, bool]:
    """(先頭が孤立, 末尾が孤立)。先頭2モーラ・末尾2モーラの開始時刻の差が閾値を超える（>）か。

    モーラが2つ未満なら判定できないので (False, False)。
    """
    if len(moras) < 2:
        return False, False
    head = float(moras[1]["start"]) - float(moras[0]["start"]) > threshold_sec
    tail = float(moras[-1]["start"]) - float(moras[-2]["start"]) > threshold_sec
    return head, tail


# --------------------------------------------------------------------------------------
# 入力の読み込み


def load_alignments(path: str | Path) -> dict[str, dict]:
    """``data/processed/alignments/dev.jsonl`` を clip_id → 行 の辞書で読む。"""
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["clip_id"] in rows:
                raise ValueError(f"アライメントに重複した clip_id がある: {row['clip_id']}")
            rows[row["clip_id"]] = row
    return rows


def load_known_no_speech(path: str | Path | None) -> set[str]:
    """印を付ける clip_id の集合（TSV の ``clip_id`` 列。``#`` で始まる行は注記として飛ばす）。"""
    if path is None:
        return set()
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return {row["clip_id"] for row in csv.DictReader(lines, delimiter="\t")}


# --------------------------------------------------------------------------------------
# 音源（単一クリップ・連結）


@dataclass(frozen=True)
class WindowSource:
    """窓を切り出す元の音声1本。"""

    source_id: str
    kind: str
    client_id: str
    clip_ids: tuple[str, ...]
    clip_samples: tuple[int, ...]  # 各クリップの標本数（16kHz）
    gap_samples: tuple[int, ...]  # クリップ間の無音の標本数（件数 − 1）
    known_no_speech: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KIND_CODES:
            raise ValueError(f"kind が不正: {self.kind}")
        if len(self.clip_ids) != len(self.clip_samples):
            raise ValueError("clip_ids と clip_samples の長さが違う")
        if len(self.gap_samples) != max(0, len(self.clip_ids) - 1):
            raise ValueError("gap_samples の個数は clip 数 − 1")

    @property
    def offsets(self) -> tuple[int, ...]:
        """各クリップの音源内の開始標本。"""
        offsets = []
        position = 0
        for index, samples in enumerate(self.clip_samples):
            offsets.append(position)
            position += samples
            if index < len(self.gap_samples):
                position += self.gap_samples[index]
        return tuple(offsets)

    @property
    def num_samples(self) -> int:
        return int(sum(self.clip_samples) + sum(self.gap_samples))

    @property
    def noise_key(self) -> str:
        """雑音の乱数生成器の鍵（``noisy.clip_rng`` に渡す）。"""
        if self.kind == KIND_SINGLE:
            return self.clip_ids[0]
        return "concat:" + "+".join(self.clip_ids)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["offsets"] = list(self.offsets)
        payload["num_samples"] = self.num_samples
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "WindowSource":
        return cls(
            source_id=str(payload["source_id"]),
            kind=str(payload["kind"]),
            client_id=str(payload["client_id"]),
            clip_ids=tuple(payload["clip_ids"]),
            clip_samples=tuple(int(v) for v in payload["clip_samples"]),
            gap_samples=tuple(int(v) for v in payload["gap_samples"]),
            known_no_speech=bool(payload.get("known_no_speech", False)),
        )


def source_moras(
    source: WindowSource, alignments: Mapping[str, Mapping[str, Any]], sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """音源内の全モーラの (開始秒, 終了秒)。連結では各クリップの開始位置だけずらす。"""
    starts: list[float] = []
    ends: list[float] = []
    for clip_id, offset in zip(source.clip_ids, source.offsets, strict=True):
        shift = offset / float(sample_rate)
        for mora in alignments[clip_id]["moras"]:
            starts.append(float(mora["start"]) + shift)
            ends.append(float(mora["end"]) + shift)
    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


def build_concat_groups(
    clips_by_speaker: Mapping[str, Sequence[str]],
    *,
    seed: int | Sequence[int],
    spec: ConcatSpec,
    sample_rate: int,
) -> tuple[list[tuple[str, tuple[str, ...], tuple[int, ...]]], dict[str, int]]:
    """同一話者のクリップから連結の組を作る。

    ``seed`` は整数か整数の列。整数なら乱数生成器の種は ``[seed, clip_key(client_id)]``、
    列なら ``[*seed, clip_key(client_id)]`` である（方式Bの学習でエポックごとに組を作り直す
    ための拡張。docs/decisions/009-method-b.md 1.3節）。整数を渡したときの結果は拡張前と同じ。

    方法（話者ごとに独立。話者の並び順や他の話者の件数に依存しない）:

    1. 話者の使えるクリップを clip_id 順に並べ、``default_rng([*seed, clip_key(client_id)])`` の
       ``permutation`` で並べ替える
    2. 残りが ``group_size`` の下限以上のあいだ、件数 k を下限〜上限の整数一様乱数で引き
       （残りより多ければ残り全部）、先頭から k 件を1組にする。続けて k − 1 個の無音長を
       ``gap_sec`` の一様乱数 [下限, 上限) から引き、``gap_round_sec`` 単位に丸めて標本数にする
    3. 各クリップは高々1回使う。下限に満たない残り（1〜2件）と、使えるクリップが下限未満の
       話者のクリップは連結に使わない

    Returns:
        (組の列 [(client_id, clip_ids, gap_samples)], 集計)
    """
    low, high = spec.group_size
    seed_prefix = [int(seed)] if isinstance(seed, (int, np.integer)) else [int(v) for v in seed]
    groups: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = []
    stats = Counter()
    for client_id in sorted(clips_by_speaker):
        clip_ids = sorted(clips_by_speaker[client_id])
        if len(clip_ids) < low:
            stats["speakers_too_few_clips"] += 1
            stats["clips_unused_too_few"] += len(clip_ids)
            continue
        rng = np.random.default_rng([*seed_prefix, clip_key(client_id)])
        order = [clip_ids[i] for i in rng.permutation(len(clip_ids))]
        position = 0
        while len(order) - position >= low:
            k = min(int(rng.integers(low, high + 1)), len(order) - position)
            members = tuple(order[position : position + k])
            position += k
            gaps_sec = rng.uniform(spec.gap_sec[0], spec.gap_sec[1], size=k - 1)
            gaps = tuple(
                int(round(round(float(g) / spec.gap_round_sec) * spec.gap_round_sec * sample_rate))
                for g in gaps_sec
            )
            groups.append((client_id, members, gaps))
        stats["speakers_grouped"] += 1
        stats["clips_unused_leftover"] += len(order) - position
    return groups, dict(stats)


# --------------------------------------------------------------------------------------
# 生成


@dataclass
class DevWindowSet:
    """dev_window の窓の定義一式。"""

    sources: list[WindowSource]
    source_index: np.ndarray  # int32
    start_sample: np.ndarray  # int64
    mora: np.ndarray  # float32
    kind: np.ndarray  # int8（0=single, 1=concat）
    known_no_speech: np.ndarray  # bool
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.source_index.size)


def load_dev_clips(
    clips_jsonl: str | Path, client_ids: Iterable[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """dev の話者のクリップについて (clip_id → client_id, clip_id → audio_path)。"""
    wanted = set(client_ids)
    to_client: dict[str, str] = {}
    to_path: dict[str, str] = {}
    with open(clips_jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["client_id"] in wanted:
                to_client[row["clip_id"]] = row["client_id"]
                to_path[row["clip_id"]] = row["audio_path"]
    return to_client, to_path


def build_dev_window(
    config: DevWindowConfig,
    *,
    alignments: Mapping[str, Mapping[str, Any]],
    clip_to_client: Mapping[str, str],
    known_no_speech: set[str] | None = None,
) -> DevWindowSet:
    """窓の定義を作る（音声は読まない。長さはアライメントの ``duration_sec`` を使う）。

    Args:
        alignments: clip_id → アライメントの行（ok, duration_sec, moras）
        clip_to_client: 対象とする clip_id → client_id（dev の話者に限る）
        known_no_speech: 印を付ける clip_id
    """
    known = set(known_no_speech or ())
    sr = config.sample_rate
    counts: Counter = Counter()
    point_moras = 0

    eligible: list[str] = []
    excluded: dict[str, list[str]] = {"failed": [], "head": [], "tail": [], "both": []}
    missing = sorted(c for c in clip_to_client if c not in alignments)
    for clip_id in sorted(clip_to_client):
        row = alignments.get(clip_id)
        if row is None:
            continue
        if not row.get("ok", False):
            excluded["failed"].append(clip_id)
            continue
        head, tail = isolation_flags(row["moras"], config.isolation_threshold_sec)
        if head and tail:
            excluded["both"].append(clip_id)
        elif head:
            excluded["head"].append(clip_id)
        elif tail:
            excluded["tail"].append(clip_id)
        else:
            eligible.append(clip_id)
            point_moras += sum(1 for m in row["moras"] if float(m["end"]) <= float(m["start"]))

    def clip_samples(clip_id: str) -> int:
        return int(round(float(alignments[clip_id]["duration_sec"]) * sr))

    sources: list[WindowSource] = []
    for clip_id in eligible:
        sources.append(
            WindowSource(
                source_id=clip_id,
                kind=KIND_SINGLE,
                client_id=clip_to_client[clip_id],
                clip_ids=(clip_id,),
                clip_samples=(clip_samples(clip_id),),
                gap_samples=(),
                known_no_speech=clip_id in known,
            )
        )

    by_speaker: dict[str, list[str]] = {}
    for clip_id in eligible:
        by_speaker.setdefault(clip_to_client[clip_id], []).append(clip_id)
    groups, group_stats = build_concat_groups(
        by_speaker, seed=config.seed, spec=config.concat, sample_rate=sr
    )
    for index, (client_id, members, gaps) in enumerate(groups):
        sources.append(
            WindowSource(
                source_id=f"concat-{index:05d}",
                kind=KIND_CONCAT,
                client_id=client_id,
                clip_ids=members,
                clip_samples=tuple(clip_samples(c) for c in members),
                gap_samples=gaps,
                known_no_speech=any(c in known for c in members),
            )
        )

    idx_parts, start_parts, mora_parts, kind_parts, known_parts = [], [], [], [], []
    for number, source in enumerate(sources):
        starts = window_starts(source.num_samples, config.window_samples, config.hop_samples)
        if starts.size == 0:
            counts[f"{source.kind}_sources_without_windows"] += 1
            continue
        mora_start, mora_end = source_moras(source, alignments, sr)
        labels = window_mora(mora_start, mora_end, starts.astype(np.float32) / sr, config.window_sec)
        idx_parts.append(np.full(starts.size, number, dtype=np.int32))
        start_parts.append(starts)
        mora_parts.append(labels)
        kind_parts.append(np.full(starts.size, _KIND_CODES[source.kind], dtype=np.int8))
        known_parts.append(np.full(starts.size, source.known_no_speech, dtype=bool))

    def cat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    window_set = DevWindowSet(
        sources=sources,
        source_index=cat(idx_parts, np.int32),
        start_sample=cat(start_parts, np.int64),
        mora=cat(mora_parts, np.float32),
        kind=cat(kind_parts, np.int8),
        known_no_speech=cat(known_parts, bool),
    )
    window_set.meta = {
        "counts": _summarize(window_set, config, excluded, missing, group_stats, known, counts),
        "point_moras_in_eligible_clips": int(point_moras),
        "excluded_clip_ids": excluded,
    }
    return window_set


def _summarize(
    ws: DevWindowSet,
    config: DevWindowConfig,
    excluded: Mapping[str, list[str]],
    missing: list[str],
    group_stats: Mapping[str, int],
    known: set[str],
    extra: Mapping[str, int],
) -> dict[str, Any]:
    from spkrate.eval.metrics import band_of

    single_sources = [s for s in ws.sources if s.kind == KIND_SINGLE]
    concat_sources = [s for s in ws.sources if s.kind == KIND_CONCAT]
    excluded_all = set().union(*excluded.values())
    summary: dict[str, Any] = {
        "clips_in_split": len(set(excluded_all) | {s.clip_ids[0] for s in single_sources}) + len(missing),
        "clips_missing_alignment": len(missing),
        "excluded_failed": len(excluded["failed"]),
        "excluded_head_only": len(excluded["head"]),
        "excluded_tail_only": len(excluded["tail"]),
        "excluded_head_and_tail": len(excluded["both"]),
        "excluded_head_total": len(excluded["head"]) + len(excluded["both"]),
        "excluded_tail_total": len(excluded["tail"]) + len(excluded["both"]),
        "excluded_total": len(excluded_all),
        "eligible_clips": len(single_sources),
        "single_sources_without_windows": int(extra.get("single_sources_without_windows", 0)),
        "concat_groups": len(concat_sources),
        "concat_group_sizes": {
            str(k): v for k, v in sorted(Counter(len(s.clip_ids) for s in concat_sources).items())
        },
        "concat_clips_used": int(sum(len(s.clip_ids) for s in concat_sources)),
        "concat_sources_without_windows": int(extra.get("concat_sources_without_windows", 0)),
        **{f"concat_{k}": int(v) for k, v in group_stats.items()},
        "known_no_speech_ids": len(known),
        "known_no_speech_excluded": len(known & excluded_all),
        "known_no_speech_single_sources": sum(1 for s in single_sources if s.known_no_speech),
        "known_no_speech_concat_groups": sum(1 for s in concat_sources if s.known_no_speech),
    }
    for name, code in _KIND_CODES.items():
        mask = ws.kind == code
        labels = ws.mora[mask]
        rates = labels / np.float32(config.window_sec)
        summary[f"{name}_windows"] = int(mask.sum())
        summary[f"{name}_windows_zero"] = int((labels == 0).sum())
        summary[f"{name}_windows_known_no_speech"] = int((mask & ws.known_no_speech).sum())
        bands = Counter(band_of(float(r)) for r in rates)
        summary[f"{name}_windows_by_band"] = {k: int(bands.get(k, 0)) for k in _band_keys()}
        summary[f"{name}_mora_per_sec_mean"] = float(rates.mean()) if rates.size else float("nan")
    summary["windows_total"] = len(ws)
    summary["windows_zero_total"] = int((ws.mora == 0).sum())
    return summary


def _band_keys() -> tuple[str, ...]:
    from spkrate.eval.metrics import BAND_KEYS

    return BAND_KEYS


# --------------------------------------------------------------------------------------
# 保存と読み込み


def save_dev_window(window_set: DevWindowSet, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sources.jsonl", "w", encoding="utf-8") as handle:
        for source in window_set.sources:
            handle.write(json.dumps(source.to_json(), ensure_ascii=False) + "\n")
    np.savez_compressed(
        out / "windows.npz",
        source_index=window_set.source_index,
        start_sample=window_set.start_sample,
        mora=window_set.mora,
        kind=window_set.kind,
        known_no_speech=window_set.known_no_speech,
    )
    with open(out / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(window_set.meta, handle, ensure_ascii=False, indent=2)
    return out


def load_dev_window(output_dir: str | Path) -> DevWindowSet:
    out = Path(output_dir)
    with open(out / "sources.jsonl", encoding="utf-8") as handle:
        sources = [WindowSource.from_json(json.loads(line)) for line in handle if line.strip()]
    with np.load(out / "windows.npz") as arrays:
        data = {key: arrays[key] for key in arrays.files}
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return DevWindowSet(sources=sources, meta=meta, **data)


# --------------------------------------------------------------------------------------
# 波形の再生成


AudioLoader = Callable[[str], np.ndarray]


class DevWindowAudio:
    """窓の波形を決定的に再生成する。

    ``clip_audio_path``（clip_id → 音声ファイルのパス）と ``loader``（パス → 16kHz 波形）で
    クリップを読み、音源（単一または連結）を組み立て、``snr_db`` があれば音源全体に
    dev_noisy の加工を掛けてから窓を切る。直前の1音源の波形を保持する
    （窓を音源の順に読むと各音源を1回だけ組み立てる）。
    """

    def __init__(
        self,
        config: DevWindowConfig,
        clip_audio_path: Mapping[str, str | Path],
        *,
        loader: AudioLoader | None = None,
        noise_source: Any = None,
    ) -> None:
        self.config = config
        self.clip_audio_path = clip_audio_path
        if loader is None:
            from spkrate.eval.audio import load_audio

            def loader(path: str) -> np.ndarray:
                return load_audio(path, target_sample_rate=config.sample_rate)[0]

        self.loader = loader
        self.noise_source = noise_source
        self._cache_key: tuple[str, float | None] | None = None
        self._cache_value: np.ndarray | None = None

    def source_waveform(self, source: WindowSource, snr_db: float | None = None) -> np.ndarray:
        key = (source.source_id, None if snr_db is None else float(snr_db))
        if self._cache_key == key and self._cache_value is not None:
            return self._cache_value
        pieces: list[np.ndarray] = []
        for index, (clip_id, expected) in enumerate(
            zip(source.clip_ids, source.clip_samples, strict=True)
        ):
            samples = np.asarray(self.loader(str(self.clip_audio_path[clip_id])), dtype=np.float32)
            if samples.size != expected:
                raise ValueError(
                    f"{clip_id} の長さがアライメント時と違う: {samples.size} != {expected} 標本"
                )
            pieces.append(samples)
            if index < len(source.gap_samples):
                pieces.append(np.zeros(source.gap_samples[index], dtype=np.float32))
        waveform = np.ascontiguousarray(np.concatenate(pieces))
        if snr_db is not None:
            if self.noise_source is None:
                raise ValueError("雑音下の版には noise_source（make_noise_source の評価用）が必要")
            waveform = degrade_waveform(
                waveform,
                source.noise_key,
                float(snr_db),
                config=self.config.noisy,
                noise_source=self.noise_source,
                sample_rate=self.config.sample_rate,
            )
        self._cache_key = key
        self._cache_value = waveform
        return waveform

    def window(
        self, source: WindowSource, start_sample: int, snr_db: float | None = None
    ) -> np.ndarray:
        """1窓（``window_samples`` 標本）の波形。"""
        waveform = self.source_waveform(source, snr_db)
        start = int(start_sample)
        end = start + self.config.window_samples
        if start < 0 or end > waveform.size:
            raise ValueError(f"窓が音源の外にはみ出す: {start}〜{end} / {waveform.size}")
        return np.ascontiguousarray(waveform[start:end])

    def window_at(
        self, window_set: DevWindowSet, index: int, snr_db: float | None = None
    ) -> np.ndarray:
        """``window_set`` の index 番目の窓の波形。"""
        source = window_set.sources[int(window_set.source_index[index])]
        return self.window(source, int(window_set.start_sample[index]), snr_db)
