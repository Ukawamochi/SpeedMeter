"""Common Voice 日本語の validated.tsv からクリップ一覧を作る。

設計上の前提（重要）

この経路は**学習データの構築**のためのものである。既定では
spkrate.labels.filter.should_exclude_from_training を適用する。同関数は
docs/spec.md「前処理」に従い学習データにのみ適用してよい除外規則であり、
検証セット・テストセットなどの評価セットには適用してはならない。
評価セットは docs/PLAN.md 第9段階で別途構築する。したがって、ここで作る
clips 一覧は学習用の母集団であって、評価セットそのものではない。

評価セットの構築などで同じ読み込み処理を再利用したい場合は
build_clip_index(..., apply_training_filter=False) を指定する。既定は
学習用（True）である。apply_training_filter=False のときは読み変換に
失敗した文（has_unconverted が True の文）も残るため、モーラ数はカタカナ以外の
文字を無視した値になる点に注意する。残った件数は FilterStats.unconverted_kept に
記録する。

除外の適用順（各段階の件数は FilterStats に記録する）

1. down_votes >= up_votes のクリップ
2. sentence が空
3. should_exclude_from_training が True を返す文
   （理由 "unconverted" と "long_digit_run" を別々に集計する）
4. clip_durations.tsv に音声長が無いクリップ
5. 毎秒モーラ数が min_mora_per_second 未満、または max_mora_per_second 超のクリップ

音声長は data/common_voice_ja/clip_durations.tsv のミリ秒値を秒に換算して使う
（実音声は読まない）。data/ 以下は読み取りのみで、変更・削除は行わない。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from spkrate.labels.filter import (
    REASON_LONG_DIGIT_RUN,
    REASON_UNCONVERTED,
    should_exclude_from_training,
)
from spkrate.labels.mora import count_mora_from_kana, has_unconverted, to_kana

__all__ = [
    "CLIPS_SUBDIR",
    "ClipRecord",
    "DEFAULT_MAX_MORA_PER_SECOND",
    "DEFAULT_MIN_MORA_PER_SECOND",
    "FilterStats",
    "SentenceInfo",
    "analyse_sentence",
    "build_clip_index",
    "iter_validated_rows",
    "read_clip_durations",
    "summarize_mora_per_second",
    "write_records",
]

# 毎秒モーラ数の外れ値判定のしきい値（docs/PLAN.md 第2段階 2-2）。
DEFAULT_MIN_MORA_PER_SECOND = 1.0
DEFAULT_MAX_MORA_PER_SECOND = 12.0

# Common Voice の音声ファイルが置かれるサブディレクトリ名。
CLIPS_SUBDIR = "clips"


@dataclass(frozen=True)
class ClipRecord:
    """1クリップ分の学習用メタデータ。"""

    clip_id: str
    audio_path: str
    client_id: str
    sentence: str
    kana: str
    mora: int
    duration_sec: float
    mora_per_second: float


@dataclass
class FilterStats:
    """除外条件ごとの件数。"""

    total_rows: int = 0
    unique_sentences: int = 0
    excluded_down_votes: int = 0
    excluded_empty_sentence: int = 0
    excluded_unconverted: int = 0
    excluded_long_digit_run: int = 0
    missing_duration: int = 0
    excluded_outlier: int = 0
    unconverted_kept: int = 0
    kept: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SentenceInfo:
    """1つの原文に対する読み変換の結果と除外判定。"""

    kana: str
    mora: int
    exclude_reason: str = ""
    unconverted: bool = False


def read_clip_durations(path: str | Path) -> dict[str, float]:
    """clip_durations.tsv を読み、クリップのファイル名から秒数への対応を返す。

    ファイルの値はミリ秒なので1000で割って秒にする。
    """
    durations: dict[str, float] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None:
            return durations
        for row in reader:
            if len(row) < 2:
                continue
            try:
                durations[row[0]] = int(row[1]) / 1000.0
            except ValueError:
                continue
    return durations


def iter_validated_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """validated.tsv を1行ずつ辞書で返す。

    Common Voice のtsvは引用符を使わないため QUOTE_NONE で解析する。
    """
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            yield row


def analyse_sentence(text: str, *, apply_training_filter: bool = True) -> SentenceInfo:
    """原文を読みに変換し、モーラ数と除外判定を返す。

    apply_training_filter が True のとき、除外判定は
    spkrate.labels.filter.should_exclude_from_training に完全に委ねる
    （除外規則の正はあちら側の1か所のみ）。
    """
    if apply_training_filter:
        excluded, reason = should_exclude_from_training(text)
        if excluded:
            return SentenceInfo(kana="", mora=0, exclude_reason=reason)
    kana = to_kana(text)
    return SentenceInfo(
        kana=kana,
        mora=count_mora_from_kana(kana),
        exclude_reason="",
        unconverted=has_unconverted(kana),
    )


def _clip_id_from_path(clip_path: str) -> str:
    return Path(clip_path).stem


def build_clip_index(
    validated_tsv: str | Path,
    clip_durations_tsv: str | Path,
    *,
    audio_dir: str = "data/common_voice_ja/clips",
    apply_training_filter: bool = True,
    min_mora_per_second: float = DEFAULT_MIN_MORA_PER_SECOND,
    max_mora_per_second: float = DEFAULT_MAX_MORA_PER_SECOND,
    progress: Callable[[int, FilterStats], None] | None = None,
    progress_every: int = 20000,
) -> tuple[list[ClipRecord], FilterStats]:
    """validated.tsv からクリップ一覧を作る。

    Args:
        validated_tsv: validated.tsv のパス。
        clip_durations_tsv: clip_durations.tsv のパス（音声長はここから取る）。
        audio_dir: 出力に書く音声ファイルの相対パスの先頭部分。
        apply_training_filter: 学習用の文除外を適用するか。既定True（学習用）。
            評価セットを作る場合のみFalseにする。
        min_mora_per_second, max_mora_per_second: 外れ値の範囲（両端を含む）。
        progress: (処理済み行数, 途中経過の統計) を受け取る進捗コールバック。

    Returns:
        (クリップ一覧, 除外件数の統計)。
    """
    durations = read_clip_durations(clip_durations_tsv)
    stats = FilterStats()
    sentence_cache: dict[str, SentenceInfo] = {}
    records: list[ClipRecord] = []

    for row in iter_validated_rows(validated_tsv):
        stats.total_rows += 1
        if progress is not None and stats.total_rows % progress_every == 0:
            progress(stats.total_rows, stats)

        if _is_down_voted(row):
            stats.excluded_down_votes += 1
            continue

        sentence = (row.get("sentence") or "").strip()
        if not sentence:
            stats.excluded_empty_sentence += 1
            continue

        info = sentence_cache.get(sentence)
        if info is None:
            info = analyse_sentence(
                sentence, apply_training_filter=apply_training_filter
            )
            sentence_cache[sentence] = info

        if info.exclude_reason == REASON_UNCONVERTED:
            stats.excluded_unconverted += 1
            continue
        if info.exclude_reason == REASON_LONG_DIGIT_RUN:
            stats.excluded_long_digit_run += 1
            continue
        if info.unconverted:
            stats.unconverted_kept += 1

        clip_path = (row.get("path") or "").strip()
        duration = durations.get(clip_path)
        if duration is None or duration <= 0.0:
            stats.missing_duration += 1
            continue

        rate = info.mora / duration
        if rate < min_mora_per_second or rate > max_mora_per_second:
            stats.excluded_outlier += 1
            continue

        records.append(
            ClipRecord(
                clip_id=_clip_id_from_path(clip_path),
                audio_path=f"{audio_dir.rstrip('/')}/{clip_path}",
                client_id=(row.get("client_id") or "").strip(),
                sentence=sentence,
                kana=info.kana,
                mora=info.mora,
                duration_sec=round(duration, 3),
                mora_per_second=rate,
            )
        )

    stats.unique_sentences = len(sentence_cache)
    stats.kept = len(records)
    return records, stats


def _is_down_voted(row: dict[str, str]) -> bool:
    """down_votes が up_votes 以上ならTrue。数値として読めない行も除外する。"""
    try:
        up = int((row.get("up_votes") or "").strip())
        down = int((row.get("down_votes") or "").strip())
    except ValueError:
        return True
    return down >= up


def write_records(records: Iterable[ClipRecord], path: str | Path) -> Path:
    """クリップ一覧をparquet（pyarrowがあれば）かjsonlに保存する。

    pyarrow が入っていない場合は拡張子を .jsonl に変えて1行1レコードで書く。
    実際に書いたパスを返す。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if path.suffix == ".parquet" and _has_pyarrow():
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return path
    if path.suffix == ".parquet":
        path = path.with_suffix(".jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _has_pyarrow() -> bool:
    import importlib.util

    return importlib.util.find_spec("pyarrow") is not None


@dataclass
class RateSummary:
    """毎秒モーラ数の分布。"""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    q1: float
    q3: float
    histogram: list[tuple[float, float, int]] = field(default_factory=list)
    total_duration_sec: float = 0.0


def summarize_mora_per_second(
    records: list[ClipRecord], *, bin_width: float = 0.5
) -> RateSummary:
    """残存クリップの毎秒モーラ数の分布と合計時間を求める。"""
    if not records:
        return RateSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], 0.0)
    rates = sorted(record.mora_per_second for record in records)
    total_duration = sum(record.duration_sec for record in records)
    lower = DEFAULT_MIN_MORA_PER_SECOND
    upper = DEFAULT_MAX_MORA_PER_SECOND
    bin_count = int(round((upper - lower) / bin_width))
    counts = [0] * bin_count
    for rate in rates:
        index = int((rate - lower) / bin_width)
        index = min(max(index, 0), bin_count - 1)
        counts[index] += 1
    histogram = [
        (lower + i * bin_width, lower + (i + 1) * bin_width, counts[i])
        for i in range(bin_count)
    ]
    return RateSummary(
        count=len(rates),
        minimum=rates[0],
        maximum=rates[-1],
        mean=sum(rates) / len(rates),
        median=_quantile(rates, 0.5),
        q1=_quantile(rates, 0.25),
        q3=_quantile(rates, 0.75),
        histogram=histogram,
        total_duration_sec=total_duration,
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    """線形補間による分位点（numpy既定と同じ定義）。"""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
