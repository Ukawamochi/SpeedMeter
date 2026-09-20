"""話者（client_id）単位で学習・検証・テストに分割する。

docs/PLAN.md 第2段階「2-3 分割の固定」に対応する。docs/spec.md「分割」の
「話者単位で学習・検証・テストに分ける。分割は configs/splits/ に固定する」を
実装したものである。

分割の性質

- 分割の単位は client_id である。同一話者のクリップが複数の分割にまたがることはない
- 比率は 8:1:1 を目安とするが、合わせるのは**話者数ではなくクリップ数**である
  （話者ごとのクリップ数の偏りが大きいため）
- 乱択は固定シード ``SPLIT_SEED`` のみを使う。入力の並び順にも依存しないよう、
  client_id で整列してからシャッフルする。したがって何度実行しても同じ分割になる

**テストセットの使用禁止（docs/PLAN.md 禁止事項）**

``configs/splits/test.json`` は docs/PLAN.md 第10段階（最終評価）まで使用禁止である。
条件の選択・ハイパーパラメータの調整・途中の評価に使ってはならない。途中の評価には
``dev.json``（検証セット）のみを使う。この禁止を読み手側のコードで分かるようにするため、
テストセットの読み出しは汎用の ``load_split`` ではなく専用の ``load_test_split`` を通す。
同関数は ``stage10_approved=True`` を明示的に渡さない限り ``RuntimeError`` を送出する。
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from spkrate.data.common_voice import ClipRecord, RateSummary, summarize_mora_per_second

__all__ = [
    "DEFAULT_RATIOS",
    "SPLIT_NAMES",
    "SPLIT_SEED",
    "SplitSummary",
    "SpeakerStats",
    "TEST_SPLIT_RESTRICTION",
    "aggregate_speakers",
    "assign_speakers",
    "build_split_payload",
    "find_overlaps",
    "load_clip_records",
    "load_split",
    "load_test_split",
    "summarize_splits",
    "write_split_files",
]

# 乱択に使う固定シード。この値を変えない限り分割は再現される。
# 値の由来は分割を固定した日付（2026-09-21）である。
SPLIT_SEED = 20260921

# 分割の名前。この順序が同点時の優先順位（先にあるものを優先）になる。
SPLIT_NAMES: tuple[str, str, str] = ("train", "dev", "test")

# クリップ数の目標比率（8:1:1）。
DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "dev": 0.1, "test": 0.1}

# テストセットの使用禁止。json にも書き込み、読み手に見えるようにする。
TEST_SPLIT_RESTRICTION = (
    "configs/splits/test.json は docs/PLAN.md 第10段階（最終評価）まで使用禁止。"
    "条件の選択・途中の評価には dev.json のみを使う。"
)


@dataclass(frozen=True)
class SpeakerStats:
    """1話者分の集計値。"""

    client_id: str
    num_clips: int
    total_duration_sec: float


@dataclass(frozen=True)
class SplitSummary:
    """1分割分の要約。毎秒モーラ数の分布は RateSummary に入る。"""

    name: str
    num_speakers: int
    num_clips: int
    total_duration_sec: float
    rate: RateSummary


def load_clip_records(path: str | Path) -> Iterator[ClipRecord]:
    """data/processed/clips.jsonl を1行ずつ ClipRecord にして返す。

    2-2 の ``write_records`` が書いた jsonl を読む。空行は読み飛ばす。
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield ClipRecord(
                clip_id=row["clip_id"],
                audio_path=row["audio_path"],
                client_id=row["client_id"],
                sentence=row["sentence"],
                kana=row["kana"],
                mora=int(row["mora"]),
                duration_sec=float(row["duration_sec"]),
                mora_per_second=float(row["mora_per_second"]),
            )


def aggregate_speakers(records: Iterable[ClipRecord]) -> list[SpeakerStats]:
    """クリップ一覧を話者ごとに集計する。戻り値は client_id の昇順。"""
    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    for record in records:
        counts[record.client_id] = counts.get(record.client_id, 0) + 1
        durations[record.client_id] = (
            durations.get(record.client_id, 0.0) + record.duration_sec
        )
    return [
        SpeakerStats(
            client_id=client_id,
            num_clips=counts[client_id],
            total_duration_sec=durations[client_id],
        )
        for client_id in sorted(counts)
    ]


def assign_speakers(
    speakers: Sequence[SpeakerStats],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """話者を分割に割り当てる（貪欲法）。

    手順。

    1. client_id の昇順に整列する（入力の並び順に依存させないため）
    2. ``random.Random(seed)`` でシャッフルする（同じクリップ数の話者の並びを
       固定シードで決めるため。シードが同じなら結果は常に同じ）
    3. クリップ数の多い順に安定ソートする（2の順序が同数時の並びとして残る）
    4. 先頭から、目標クリップ数に対する不足が最も大きい分割へ割り当てる

    クリップ数の多い話者から置くため、最後に残る誤差は小さい話者の分だけになり、
    クリップ数の比率が目標に近づく。

    Returns:
        分割名から client_id の一覧（クリップ数の多い順）への辞書。
    """
    names = [name for name in SPLIT_NAMES if name in ratios]
    for name in ratios:
        if name not in SPLIT_NAMES:
            raise ValueError(f"未知の分割名: {name}")
    ratio_sum = sum(ratios[name] for name in names)
    if ratio_sum <= 0.0:
        raise ValueError("比率の合計が0以下である")

    total_clips = sum(speaker.num_clips for speaker in speakers)
    targets = {name: total_clips * ratios[name] / ratio_sum for name in names}
    priority = {name: index for index, name in enumerate(names)}

    ordered = sorted(speakers, key=lambda speaker: speaker.client_id)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda speaker: speaker.num_clips, reverse=True)

    assigned: dict[str, list[str]] = {name: [] for name in names}
    current: dict[str, int] = {name: 0 for name in names}
    for speaker in ordered:
        # 不足が最大の分割へ。同点なら SPLIT_NAMES の順で先にあるものへ。
        target = max(names, key=lambda n: (targets[n] - current[n], -priority[n]))
        assigned[target].append(speaker.client_id)
        current[target] += speaker.num_clips
    return assigned


def find_overlaps(assignment: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    """複数の分割に現れる client_id を探す。

    Returns:
        "train&dev" のような組の名前から、重複した client_id の集合への辞書。
        重複が無ければ空の辞書。
    """
    overlaps: dict[str, set[str]] = {}
    names = list(assignment)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(assignment[left]) & set(assignment[right])
            if shared:
                overlaps[f"{left}&{right}"] = shared
    return overlaps


def summarize_splits(
    records: Iterable[ClipRecord], assignment: Mapping[str, Sequence[str]]
) -> dict[str, SplitSummary]:
    """分割ごとの話者数・クリップ数・合計時間・毎秒モーラ数の分布を求める。

    どの分割にも属さない client_id のクリップは無視する。
    """
    owner: dict[str, str] = {}
    for name, client_ids in assignment.items():
        for client_id in client_ids:
            owner[client_id] = name

    grouped: dict[str, list[ClipRecord]] = {name: [] for name in assignment}
    for record in records:
        name = owner.get(record.client_id)
        if name is not None:
            grouped[name].append(record)

    summaries: dict[str, SplitSummary] = {}
    for name, group in grouped.items():
        rate = summarize_mora_per_second(group)
        summaries[name] = SplitSummary(
            name=name,
            num_speakers=len(assignment[name]),
            num_clips=len(group),
            total_duration_sec=rate.total_duration_sec,
            rate=rate,
        )
    return summaries


def build_split_payload(
    name: str,
    client_ids: Sequence[str],
    *,
    seed: int = SPLIT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    source: str = "data/processed/clips.jsonl",
    summary: SplitSummary | None = None,
) -> dict:
    """1分割分の json の中身を作る。

    ``client_ids`` は保存時に昇順へ整列する（ファイルの差分を安定させるため）。
    """
    payload: dict = {
        "split": name,
        "unit": "client_id",
        "seed": seed,
        "method": "greedy-largest-deficit (クリップ数が目標比率に近づくよう貪欲に割り当て)",
        "target_ratio": ratios.get(name),
        "source": source,
        "generated_by": "scripts/build_splits.py",
        "implementation": "src/spkrate/data/splits.py",
        "num_speakers": len(client_ids),
    }
    if summary is not None:
        payload["num_clips"] = summary.num_clips
        payload["total_duration_sec"] = round(summary.total_duration_sec, 3)
    if name == "test":
        payload["usage_restriction"] = TEST_SPLIT_RESTRICTION
    payload["client_ids"] = sorted(client_ids)
    return payload


def write_split_files(
    out_dir: str | Path,
    assignment: Mapping[str, Sequence[str]],
    *,
    seed: int = SPLIT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    source: str = "data/processed/clips.jsonl",
    summaries: Mapping[str, SplitSummary] | None = None,
) -> dict[str, Path]:
    """configs/splits/<name>.json を書き、分割名からパスへの辞書を返す。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, client_ids in assignment.items():
        payload = build_split_payload(
            name,
            client_ids,
            seed=seed,
            ratios=ratios,
            source=source,
            summary=None if summaries is None else summaries.get(name),
        )
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written[name] = path
    return written


def load_split(path: str | Path) -> list[str]:
    """分割ファイルを読み、client_id の一覧を返す。

    **テストセットにはこの関数を使わないこと。** ``configs/splits/test.json`` は
    docs/PLAN.md 第10段階まで使用禁止であり、読み出しは ``load_test_split`` を通す。
    この関数は test.json を渡された場合 ``ValueError`` を送出する。
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") == "test":
        raise ValueError(
            "テストセットは load_split では読めない。" + TEST_SPLIT_RESTRICTION
        )
    return list(payload["client_ids"])


def load_test_split(path: str | Path, *, stage10_approved: bool = False) -> list[str]:
    """テストセット（configs/splits/test.json）の client_id を読む。

    **使用禁止の注意（docs/PLAN.md 禁止事項）**

    ``configs/splits/test.json`` は docs/PLAN.md 第10段階（最終評価）まで使用禁止である。
    条件の選択、ハイパーパラメータの調整、学習途中の評価、モデルの採否の判断に
    使ってはならない。それらには ``dev.json``（検証セット）のみを使う。

    この禁止を破りにくくするため、本関数は ``stage10_approved=True`` を明示的に
    渡されない限り ``RuntimeError`` を送出する。第10段階に到達するまで、この引数を
    True にするコードを書かないこと。
    """
    if not stage10_approved:
        raise RuntimeError(
            "テストセットの読み出しが第10段階より前に呼ばれた。" + TEST_SPLIT_RESTRICTION
        )
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["client_ids"])
