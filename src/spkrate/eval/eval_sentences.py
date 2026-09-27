"""実環境評価（docs/plan.md 第9段階）で読み上げる文を JSUT から選ぶ。

モーラ数の数え方は第1段階の測定（scripts/eval_mora_labels.py）に合わせる。

- 正解のモーラ数: data/jsut-label の人手注釈の仮名 kana_level0 を、NFKC正規化・
  ひらがな→カタカナ変換・カタカナと長音記号以外の除去をしてから
  count_mora_from_kana で数える
- 読み変換の確認: 原文を to_kana（pyopenjtalk）で読みに変え、has_unconverted が
  True の文は選ばない（docs/plan.md 9-1）
- 追加の条件: 人手注釈と pyopenjtalk のモーラ数が一致する文だけを選ぶ
  （どちらで数えても同じ正解になる文に限り、ラベルの誤りの余地を減らす）
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from spkrate.labels.mora import (
    _KANA_PATTERN,
    count_mora_from_kana,
    has_unconverted,
    to_kana,
)

_HIRAGANA_START = ord("ぁ")
_HIRAGANA_END = ord("ゖ")
_TO_KATAKANA_OFFSET = ord("ァ") - ord("ぁ")

LABEL_LEVEL = "kana_level0"


@dataclass(frozen=True)
class Candidate:
    utt_id: str
    text: str
    kana: str  # 人手注釈の仮名（カタカナのみに整えたもの）
    mora: int  # 人手注釈から数えたモーラ数（正解）
    g2p_kana: str  # pyopenjtalk の読み
    g2p_mora: int


def normalize_annotation(kana: str) -> str:
    """人手注釈の仮名をカタカナと長音記号だけに整える（eval_mora_labels.py と同じ手順）。"""
    text = unicodedata.normalize("NFKC", kana)
    text = "".join(
        chr(ord(ch) + _TO_KATAKANA_OFFSET)
        if _HIRAGANA_START <= ord(ch) <= _HIRAGANA_END
        else ch
        for ch in text
    )
    return "".join(ch for ch in text if _KANA_PATTERN.fullmatch(ch) is not None)


def load_transcript(path: Path) -> dict[str, str]:
    """JSUT の transcript_utf8.txt（ID:文）を読む。"""
    records: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, sep, text = line.partition(":")
            if not sep:
                raise ValueError(f"区切り「:」がない行: {line!r}")
            records[utt_id] = text
    return records


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=loader)


def build_candidates(
    transcript: dict[str, str], labels: dict[str, dict[str, str]]
) -> list[Candidate]:
    """両方にある文について、人手注釈と pyopenjtalk の読み・モーラ数を求める。"""
    out: list[Candidate] = []
    for utt_id in sorted(set(transcript) & set(labels)):
        raw = labels[utt_id].get(LABEL_LEVEL)
        if raw is None:
            continue
        kana = normalize_annotation(str(raw))
        g2p_kana = to_kana(transcript[utt_id])
        out.append(
            Candidate(
                utt_id=utt_id,
                text=transcript[utt_id],
                kana=kana,
                mora=count_mora_from_kana(kana),
                g2p_kana=g2p_kana,
                g2p_mora=count_mora_from_kana(g2p_kana),
            )
        )
    return out


def is_eligible(c: Candidate, min_mora: int, max_mora: int) -> bool:
    return (
        min_mora <= c.mora <= max_mora
        and not has_unconverted(c.g2p_kana)
        and c.g2p_mora == c.mora
    )


def select(
    candidates: list[Candidate],
    n: int,
    seed: int,
    min_mora: int,
    max_mora: int,
) -> list[Candidate]:
    """条件を満たす文から、乱数の種 seed で n 文を選ぶ。結果は文ID順。"""
    eligible = [c for c in candidates if is_eligible(c, min_mora, max_mora)]
    if len(eligible) < n:
        raise ValueError(f"条件を満たす文が {len(eligible)} 件しかない（必要 {n} 件）")
    rng = random.Random(seed)
    chosen = rng.sample(eligible, n)
    return sorted(chosen, key=lambda c: c.utt_id)
