"""不一致行の分類を補助する。gold仮名と推定読みの差分区間を並べて表示する。

docs/plan.md 第1段階「1-2 原因の分類」の作業補助。読み取りのみで、
results/mora_mismatch.tsv は変更しない。
"""

from __future__ import annotations

import csv
import difflib
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.labels.mora import _KANA_PATTERN, count_mora_from_kana  # noqa: E402

TSV = ROOT / "results" / "mora_mismatch.tsv"
_HIRA_START, _HIRA_END = ord("ぁ"), ord("ゖ")
_OFFSET = ord("ァ") - ord("ぁ")


def norm_gold(kana: str) -> str:
    text = unicodedata.normalize("NFKC", kana)
    text = "".join(
        chr(ord(c) + _OFFSET) if _HIRA_START <= ord(c) <= _HIRA_END else c
        for c in text
    )
    return "".join(c for c in text if _KANA_PATTERN.fullmatch(c) is not None)


def has_unconv(kana: str) -> bool:
    return any(_KANA_PATTERN.fullmatch(c) is None for c in kana)


def main() -> None:
    rows = list(csv.DictReader(TSV.open(encoding="utf-8"), delimiter="\t"))
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    mode = sys.argv[2] if len(sys.argv) > 2 else "show"

    if mode == "stats":
        rest = rows[top:]
        print("残り件数:", len(rest))
        print("差の値ごとの件数（残り）:")
        for d, n in sorted(Counter(int(r["差"]) for r in rest).items()):
            print(f"  {d}\t{n}")
        print("読み変換失敗（非カタカナ残存）: 上位%d件中 %d 件 / 全%d件中 %d 件" % (
            top,
            sum(1 for r in rows[:top] if has_unconv(r["pyopenjtalkの読み"])),
            len(rows),
            sum(1 for r in rows if has_unconv(r["pyopenjtalkの読み"])),
        ))
        return

    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    hi = int(sys.argv[4]) if len(sys.argv) > 4 else top
    for i, r in enumerate(rows[:top][lo:hi], start=lo + 1):
        g = norm_gold(r["人手注釈の仮名"])
        e = r["pyopenjtalkの読み"]
        sm = difflib.SequenceMatcher(None, g, e, autojunk=False)
        parts = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            gs, es = g[i1:i2], e[j1:j2]
            dm = count_mora_from_kana(es) - count_mora_from_kana(gs)
            ctx = g[max(0, i1 - 4):i1]
            parts.append(f"[…{ctx}] 人手<{gs}> | pyo<{es}> ({dm:+d})")
        flag = "!NONKATA" if has_unconv(e) else ""
        print(f"#{i}\t{r['文ID']}\t差{r['差']}\t{flag}")
        print(f"  原文: {r['原文']}")
        for p in parts:
            print(f"  {p}")


if __name__ == "__main__":
    main()
