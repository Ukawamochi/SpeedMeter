"""学習データから除外する文を判定する。

このモジュールは**学習データ構築の経路からのみ**呼ばれる前提である。
docs/spec.md の「前処理」節に定めるとおり、ここでの除外は学習データにのみ適用し、
検証セット・テストセットなどの評価セットには適用しない。評価セットは実際の入力分布を
そのまま反映させる必要があるため、評価セットの構築処理からこのモジュールの関数を
呼んではならない。

除外条件（docs/spec.md「前処理」）:

1. 読み変換後にカタカナ・長音記号以外の文字が残る文（理由 "unconverted"）
2. NFKC正規化後のテキストに3桁以上連続するASCII数字を含む文（理由 "long_digit_run"）
"""

import re
import unicodedata

from spkrate.labels.mora import has_unconverted, to_kana

__all__ = [
    "PURPOSE",
    "REASON_LONG_DIGIT_RUN",
    "REASON_UNCONVERTED",
    "should_exclude",
    "should_exclude_from_training",
]

# このモジュールの適用範囲。学習データ専用であり評価セットには適用しない。
PURPOSE = "training_data_only"

# 除外理由の識別子。集計しやすいよう短い固定文字列とする。
REASON_UNCONVERTED = "unconverted"
REASON_LONG_DIGIT_RUN = "long_digit_run"

# NFKC正規化後のテキストに対して照合する、3桁以上連続するASCII数字。
# 区切り文字（カンマ・小数点など）は連続を断ち切る。挙動は
# should_exclude_from_training のdocstringを参照。
_LONG_DIGIT_RUN_PATTERN = re.compile(r"[0-9]{3,}")


def should_exclude_from_training(text: str) -> tuple[bool, str]:
    """学習データから除外すべき文かを判定する。評価セットには使わない。

    この関数は学習データ構築の経路からのみ呼ぶこと。評価セット（検証・テスト）の
    構築では呼ばない。docs/spec.md の「前処理」節がこの適用範囲を定めている。

    Args:
        text: 書き起こしの生テキスト。NFKC正規化はこの関数の内部で行う。

    Returns:
        (除外するか, 理由の識別子)。除外しない場合は (False, "")。

    判定順（複数条件に該当する場合は先に一致したものを理由として返す）:

    1. "unconverted": to_kana(text) の結果に has_unconverted が True を返す。
       読み変換に失敗した痕跡が残る文であり、モーラ数が数えられない。
    2. "long_digit_run": NFKC正規化後のテキストに正規表現 [0-9]{3,} が一致する。
       電話番号・部屋番号などで位取り読みと桁読みが食い違い、モーラ数が大きくずれる。

    数字判定の詳細:

    - 判定はNFKC正規化後に行う。全角数字「１２３」や丸囲み数字「①②③」は
      正規化でASCII数字になるため除外対象となる。
    - 判定対象はASCII数字0-9が3文字以上連続する箇所のみである。カンマ・小数点などの
      区切り文字は数字の連続を断ち切る。したがって次のように扱う。
        - "1,234" は "234" が3桁連続するため除外する
        - "3.14" は "3" と "14" に分かれるため除外しない
        - "3.14159" は "14159" が5桁連続するため除外する
        - "99" のような2桁以下の数字のみの文は除外しない
    """
    if has_unconverted(to_kana(text)):
        return True, REASON_UNCONVERTED
    normalized = unicodedata.normalize("NFKC", text)
    if _LONG_DIGIT_RUN_PATTERN.search(normalized) is not None:
        return True, REASON_LONG_DIGIT_RUN
    return False, ""


# docs/PLAN.md が指定する名前。実体は should_exclude_from_training と同一であり、
# 学習データ専用であることは同関数のdocstringに記載している。
should_exclude = should_exclude_from_training
