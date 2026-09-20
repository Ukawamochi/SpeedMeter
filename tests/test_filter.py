from spkrate.labels.filter import (
    PURPOSE,
    REASON_LONG_DIGIT_RUN,
    REASON_UNCONVERTED,
    should_exclude,
    should_exclude_from_training,
)


# 学習データ専用であることの表明
def test_purpose_is_training_data_only():
    assert PURPOSE == "training_data_only"


def test_should_exclude_is_the_training_only_function():
    assert should_exclude is should_exclude_from_training


# 3桁以上の数字列（除外する）
def test_exactly_three_digits_is_excluded():
    assert should_exclude("123人が集まりました") == (True, REASON_LONG_DIGIT_RUN)


def test_four_digits_is_excluded():
    assert should_exclude("西暦2026年です") == (True, REASON_LONG_DIGIT_RUN)


def test_long_digit_run_is_excluded():
    assert should_exclude("電話番号は0312345678です") == (True, REASON_LONG_DIGIT_RUN)


def test_digits_only_text_is_excluded():
    assert should_exclude("123456") == (True, REASON_LONG_DIGIT_RUN)


# 2桁以下の数字（除外しない）
def test_two_digits_is_not_excluded():
    assert should_exclude("99人が集まりました") == (False, "")


def test_one_and_two_digits_are_not_excluded():
    assert should_exclude("12個買いました") == (False, "")


# 全角数字・丸囲み数字はNFKC正規化でASCII数字になるため判定対象になる
def test_fullwidth_digits_are_excluded():
    assert should_exclude("１２３人が集まりました") == (True, REASON_LONG_DIGIT_RUN)


def test_fullwidth_two_digits_are_not_excluded():
    assert should_exclude("９９人が集まりました") == (False, "")


def test_circled_digits_are_excluded():
    assert should_exclude("①②③") == (True, REASON_LONG_DIGIT_RUN)


# 区切り文字は数字の連続を断ち切る
def test_comma_separated_number_is_excluded_by_its_three_digit_group():
    # "1,234" のうち "234" が3桁連続するため除外する
    assert should_exclude("1,234人が集まりました") == (True, REASON_LONG_DIGIT_RUN)


def test_decimal_point_splits_the_digit_run():
    # "3.14" は "3" と "14" に分かれるため除外しない
    assert should_exclude("円周率は3.14です") == (False, "")


def test_long_decimal_fraction_is_excluded():
    # "3.14159" は小数部の "14159" が5桁連続するため除外する
    assert should_exclude("円周率は3.14159です") == (True, REASON_LONG_DIGIT_RUN)


# 数字を含まない通常文（除外しない）
def test_plain_sentence_is_not_excluded():
    assert should_exclude("今日は快晴です") == (False, "")


def test_plain_sentence_with_punctuation_is_not_excluded():
    assert should_exclude("それは、本当に良かった。") == (False, "")


def test_hiragana_sentence_is_not_excluded():
    assert should_exclude("さくらが咲きました") == (False, "")


def test_alphabet_sentence_is_not_excluded():
    # ラテン文字はpyopenjtalkがカタカナ読みに変換するため未変換にはならない
    assert should_exclude("CNNモデルを学習します") == (False, "")


# has_unconverted に該当する文（除外する）
def test_unconvertible_text_is_excluded():
    assert should_exclude("αβγを読みます") == (True, REASON_UNCONVERTED)


# 両条件に該当する文は unconverted を優先する
def test_both_conditions_return_unconverted_first():
    assert should_exclude("αβγと123") == (True, REASON_UNCONVERTED)


# 空文字列
def test_empty_string_is_not_excluded():
    assert should_exclude("") == (False, "")


def test_whitespace_only_string_is_not_excluded():
    assert should_exclude("   ") == (False, "")
