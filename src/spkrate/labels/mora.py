import re

import pyopenjtalk

_KANA_PATTERN = re.compile(r"[ァ-ヺー]")
_SMALL_COMBINING = frozenset("ァィゥェォャュョ")


def read_kana(text: str) -> str:
    raw = pyopenjtalk.g2p(text, kana=True)
    return "".join(_KANA_PATTERN.findall(raw))


def count_mora(text: str) -> int:
    return sum(1 for ch in text if ch not in _SMALL_COMBINING)
