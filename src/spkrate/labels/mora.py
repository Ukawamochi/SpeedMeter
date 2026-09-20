"""テキストからモーラ数を数える。

docs/spec.md の「モーラの数え方」と「前処理」に対応する。

- 読み変換の前に unicodedata.normalize("NFKC", text) を適用する
- 小書き文字「ャュョァィゥェォヮ」は直前のかなと合わせて1モーラ
- 小書き文字が先頭に出現した場合は1モーラとして数え、警告を出す
- 「ヵ」「ヶ」は単独で1モーラ
- 撥音「ン」・促音「ッ」・長音「ー」はそれぞれ1モーラ
- 読み変換後にカタカナ以外の文字が残る文は学習データから除外する
  （ここでは判定ヘルパー has_unconverted のみ提供する。
  除外の判定は src/spkrate/labels/filter.py が行う）
"""

import logging
import re
import unicodedata

import pyopenjtalk

logger = logging.getLogger(__name__)

# カタカナ（ァ-ヺ）と長音記号。モーラとして数える文字の集合。
_KANA_PATTERN = re.compile(r"[ァ-ヺー]")

# 直前のかなと結合して1モーラになる小書き文字。
_SMALL_COMBINING = frozenset("ャュョァィゥェォヮ")

# 読みとして意味を持たない文字のUnicodeカテゴリ先頭文字。
# P（約物）、Z（空白）、C（制御）は読み変換の失敗ではないため取り除く。
_DISCARDED_CATEGORIES = ("P", "Z", "C")

# pyopenjtalk（open_jtalk）は入力が長すぎるとプロセスごと異常終了する
# （SIGABRT。実測ではUTF-8で約8192バイトを超えると落ちる）。
# 実データには1万字を超える文が混ざっているため、余裕を持った長さで
# 分割してから読み変換し、結果を連結する。
_MAX_G2P_BYTES = 2048

# 分割位置の候補。この文字の直後で切ると読みへの影響が小さい。
_SPLIT_AFTER = "。．！？!?、，,・…\u3000 \t\n"


def _drop_non_reading(raw: str) -> str:
    """読みに関係しない約物・空白・制御文字を取り除く。

    読み変換に失敗して残った文字（ラテン文字・ギリシャ文字・絵文字など）は
    has_unconverted で検出できるよう、あえて残す。
    """
    return "".join(
        ch for ch in raw if unicodedata.category(ch)[0] not in _DISCARDED_CATEGORIES
    )


def _split_for_g2p(text: str, max_bytes: int = _MAX_G2P_BYTES) -> list[str]:
    """読み変換に渡せる長さの塊に分ける。

    各塊のUTF-8バイト長が max_bytes 以下になるようにする。できるだけ
    句読点などの直後で切り、切れる場所が無ければ文字境界で強制的に切る。
    """
    chunks: list[str] = []
    current = ""
    current_bytes = 0
    boundary = -1  # current の中で最後に現れた分割候補の直後の位置
    for ch in text:
        size = len(ch.encode("utf-8"))
        if current_bytes + size > max_bytes and current:
            if boundary > 0:
                chunks.append(current[:boundary])
                current = current[boundary:]
            else:
                chunks.append(current)
                current = ""
            current_bytes = len(current.encode("utf-8"))
            boundary = -1
        current += ch
        current_bytes += size
        if ch in _SPLIT_AFTER:
            boundary = len(current)
    if current:
        chunks.append(current)
    return chunks


def to_kana(text: str) -> str:
    """生テキストをNFKC正規化し、pyopenjtalkでカタカナ読みに変換する。

    長すぎる入力は pyopenjtalk がプロセスごと異常終了させるため、
    _MAX_G2P_BYTES 以下の塊に分けて変換し、結果を連結する。分割した場合は
    境界をまたぐ読みが変わりうるので警告を出す。
    """
    normalized = unicodedata.normalize("NFKC", text)
    chunks = _split_for_g2p(normalized)
    if len(chunks) > 1:
        logger.warning(
            "入力が長いため%d個に分割して読み変換する（%d文字）。"
            "分割境界の読みは正確でない可能性がある。",
            len(chunks),
            len(normalized),
        )
    raw = "".join(pyopenjtalk.g2p(chunk, kana=True) for chunk in chunks)
    return _drop_non_reading(raw)


def has_unconverted(kana: str) -> bool:
    """カタカナ読み文字列にカタカナ・長音記号以外の文字が残っていればTrue。"""
    return any(_KANA_PATTERN.fullmatch(ch) is None for ch in kana)


def count_mora_from_kana(kana: str) -> int:
    """カタカナ読み文字列のモーラ数を数える。

    カタカナ・長音記号以外の文字は数えない。そのような文字を含む文は
    has_unconverted で検出して学習データから除外する。
    """
    mora = 0
    for index, ch in enumerate(kana):
        if _KANA_PATTERN.fullmatch(ch) is None:
            continue
        if ch in _SMALL_COMBINING:
            if _combines_with_previous(kana, index):
                continue
            logger.warning(
                "小書き文字「%s」が結合先のかなを持たない（位置%d、読み「%s」）。1モーラとして数える。",
                ch,
                index,
                kana,
            )
        mora += 1
    return mora


def count_mora(text: str) -> int:
    """生テキストのモーラ数を数える。"""
    return count_mora_from_kana(to_kana(text))


def _combines_with_previous(kana: str, index: int) -> bool:
    """位置indexの小書き文字が直前のかなと結合できるか判定する。"""
    if index == 0:
        return False
    previous = kana[index - 1]
    if _KANA_PATTERN.fullmatch(previous) is None:
        return False
    return previous not in _SMALL_COMBINING
