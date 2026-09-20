import logging

from spkrate.labels.mora import (
    count_mora,
    count_mora_from_kana,
    has_unconverted,
    to_kana,
)


# 清音のみ
def test_seion_sakura():
    assert count_mora_from_kana("サクラ") == 3


def test_seion_aiueo():
    assert count_mora_from_kana("アイウエオ") == 5


def test_seion_katakana():
    assert count_mora_from_kana("カタカナ") == 4


def test_seion_hana():
    assert count_mora_from_kana("ハナ") == 2


def test_seion_mizu():
    assert count_mora_from_kana("ミズ") == 2


# 拗音
def test_youon_kyou():
    assert count_mora_from_kana("キョウ") == 2


def test_youon_shuppatsu():
    assert count_mora_from_kana("シュッパツ") == 4


def test_youon_kyaku():
    assert count_mora_from_kana("キャク") == 2


def test_youon_ryokou():
    assert count_mora_from_kana("リョコウ") == 3


def test_youon_jouzu():
    assert count_mora_from_kana("ジョウズ") == 3


def test_youon_chocolate():
    assert count_mora_from_kana("チョコレート") == 5


# 小書きヮの結合（ヮは常に直前の文字と合わせて1モーラ）
def test_small_wa_kuwa():
    assert count_mora_from_kana("クヮ") == 1


def test_small_wa_shiwa():
    assert count_mora_from_kana("シヮ") == 1


def test_small_wa_in_word():
    assert count_mora_from_kana("クヮガタ") == 3


# ヵ・ヶは単独で1モーラ
def test_small_ka_is_one_mora():
    assert count_mora_from_kana("ヵ") == 1


def test_small_ke_is_one_mora():
    assert count_mora_from_kana("ヶ") == 1


def test_small_ka_ke_are_counted_independently():
    assert count_mora_from_kana("ヵヶ") == 2


def test_small_ke_in_word():
    assert count_mora_from_kana("ヶツ") == 2


# 小書き文字が先頭に出現した場合は1モーラとし、警告を出す
def test_leading_small_letter_counts_as_one(caplog):
    with caplog.at_level(logging.WARNING, logger="spkrate.labels.mora"):
        assert count_mora_from_kana("ャ") == 1
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_leading_small_letter_in_string(caplog):
    with caplog.at_level(logging.WARNING, logger="spkrate.labels.mora"):
        assert count_mora_from_kana("ャクシャ") == 3
    assert len(caplog.records) == 1


def test_leading_small_wa_counts_as_one(caplog):
    with caplog.at_level(logging.WARNING, logger="spkrate.labels.mora"):
        assert count_mora_from_kana("ヮ") == 1
    assert len(caplog.records) == 1


def test_normal_kana_emits_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="spkrate.labels.mora"):
        assert count_mora_from_kana("キョウ") == 2
    assert caplog.records == []


# 促音
def test_sokuon_gakkou():
    assert count_mora_from_kana("ガッコウ") == 4


def test_sokuon_kippu():
    assert count_mora_from_kana("キップ") == 3


def test_sokuon_rappa():
    assert count_mora_from_kana("ラッパ") == 3


def test_sokuon_zettai():
    assert count_mora_from_kana("ゼッタイ") == 4


# 撥音
def test_hatsuon_nihon():
    assert count_mora_from_kana("ニホン") == 3


def test_hatsuon_sanpo():
    assert count_mora_from_kana("サンポ") == 3


def test_hatsuon_honya():
    assert count_mora_from_kana("ホンヤ") == 3


# 長音
def test_chouon_coffee():
    assert count_mora_from_kana("コーヒー") == 4


def test_chouon_super():
    assert count_mora_from_kana("スーパー") == 4


def test_chouon_ramen():
    assert count_mora_from_kana("ラーメン") == 4


# 漢字かな混じり文からの変換（to_kana + count_mora_from_kanaの組み合わせ）
def test_kanji_kana_sakura_ga_sakimashita():
    kana = to_kana("桜が咲きました")
    assert kana == "サクラガサキマシタ"
    assert count_mora_from_kana(kana) == 9
    assert count_mora("桜が咲きました") == 9


def test_kanji_kana_kyou_wa_kaisei():
    kana = to_kana("今日は快晴です")
    assert kana == "キョーワカイセーデス"
    assert count_mora_from_kana(kana) == 9


def test_kanji_kana_eki_made():
    kana = to_kana("駅まで歩いて行きます")
    assert kana == "エキマデアルイテイキマス"
    assert count_mora_from_kana(kana) == 12


def test_kanji_kana_kanojo():
    kana = to_kana("彼女は毎朝走ります")
    assert kana == "カノジョワマイアサハシリマス"
    assert count_mora_from_kana(kana) == 13


# 数字を含む文
def test_number_date():
    kana = to_kana("2026年9月20日に会議があります")
    assert kana == "ニセンニジューロクネンクガツハツカニカイギガアリマス"
    assert count_mora_from_kana(kana) == 25


def test_number_people_minutes():
    kana = to_kana("3人で30分話しました")
    assert kana == "サンニンデサンジュップンハナシマシタ"
    assert count_mora_from_kana(kana) == 17


def test_number_meters():
    kana = to_kana("100メートル走りました")
    assert kana == "ヒャクメートルハシリマシタ"
    assert count_mora_from_kana(kana) == 12


# 英字を含む文
def test_alphabet_only():
    kana = to_kana("ABC")
    assert kana == "エイビーシー"
    assert count_mora_from_kana(kana) == 6


def test_alphabet_cnn_model():
    kana = to_kana("CNNモデルを学習します")
    assert kana == "シーエヌエヌモデルヲガクシューシマス"
    assert count_mora_from_kana(kana) == 17


def test_alphabet_gpu():
    kana = to_kana("GPUを使って計算します")
    assert kana == "ジーピーユーヲツカッテケーサンシマス"
    assert count_mora_from_kana(kana) == 18


# NFKC正規化（読み変換の前に適用される）
def test_nfkc_fullwidth_latin():
    assert to_kana("ＣＮＮモデルを学習します") == to_kana("CNNモデルを学習します")


def test_nfkc_fullwidth_latin_reading():
    assert to_kana("ＧＰＵ") == "ジーピーユー"


def test_nfkc_fullwidth_digits():
    kana = to_kana("１２３人")
    assert kana == "ヒャクニジューサンニン"
    assert count_mora_from_kana(kana) == 9


def test_nfkc_circled_digit():
    kana = to_kana("①番目です")
    assert kana == "イチバンメデス"
    assert count_mora_from_kana(kana) == 7


def test_nfkc_halfwidth_katakana():
    kana = to_kana("ｱｲｳｴｵ")
    assert kana == "アイウエオ"
    assert count_mora_from_kana(kana) == 5


def test_nfkc_halfwidth_katakana_in_sentence():
    assert to_kana("ﾃﾞｰﾀを読みます") == to_kana("データを読みます")


# has_unconverted
def test_has_unconverted_false_for_katakana():
    assert has_unconverted("サクラ") is False


def test_has_unconverted_false_for_chouon_and_small_letters():
    assert has_unconverted("キョーヒャクヮヶ") is False


def test_has_unconverted_false_for_empty():
    assert has_unconverted("") is False


def test_has_unconverted_false_for_converted_sentence():
    assert has_unconverted(to_kana("今日は快晴です")) is False


def test_has_unconverted_true_for_latin():
    assert has_unconverted("エヌＣエヌ") is True


def test_has_unconverted_true_for_hiragana():
    assert has_unconverted("さくら") is True


def test_has_unconverted_true_for_unreadable_text():
    kana = to_kana("αβγ")
    assert kana == "αβγ"
    assert has_unconverted(kana) is True


# 記号・句読点を含む文
def test_symbol_question_mark():
    kana = to_kana("こんにちは、元気ですか?")
    assert kana == "コンニチワゲンキデスカ"
    assert count_mora_from_kana(kana) == 11


def test_symbol_exclamation():
    kana = to_kana("走った!すごい!")
    assert kana == "ハシッタスゴイ"
    assert count_mora_from_kana(kana) == 7


def test_symbol_period_touten():
    kana = to_kana("それは、本当に良かった。")
    assert kana == "ソレワホントーニヨカッタ"
    assert count_mora_from_kana(kana) == 12


def test_symbol_only_string_is_empty():
    assert to_kana("!!!") == ""
    assert count_mora("!!!") == 0


def test_whitespace_only_string_is_empty():
    assert to_kana("   ") == ""


def test_empty_string_is_empty():
    assert to_kana("") == ""
    assert count_mora_from_kana("") == 0
    assert count_mora("") == 0


# フィラーを含む文
def test_filler_eeto():
    kana = to_kana("えーと、それではお願いします")
    assert kana == "エートソレデワオネガイシマス"
    assert count_mora_from_kana(kana) == 14


def test_filler_anoo():
    kana = to_kana("あのー、少し待ってください")
    assert kana == "アノースコシマッテクダサイ"
    assert count_mora_from_kana(kana) == 13


def test_filler_maa_sono():
    kana = to_kana("まあ、その、なんというか")
    assert kana == "マーソノナントイウカ"
    assert count_mora_from_kana(kana) == 10


# 長文の分割（pyopenjtalkは長すぎる入力でプロセスごと異常終了する）
def test_split_for_g2p_keeps_chunks_within_byte_limit():
    from spkrate.labels.mora import _MAX_G2P_BYTES, _split_for_g2p

    text = "あいうえお、" * 2000
    chunks = _split_for_g2p(text)
    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= _MAX_G2P_BYTES for chunk in chunks)
    assert "".join(chunks) == text


def test_split_for_g2p_splits_without_punctuation():
    """区切り文字が無い場合も強制的に分割し、元の文字列を保つ。"""
    from spkrate.labels.mora import _MAX_G2P_BYTES, _split_for_g2p

    text = "あ" * 3000
    chunks = _split_for_g2p(text)
    assert all(len(chunk.encode("utf-8")) <= _MAX_G2P_BYTES for chunk in chunks)
    assert "".join(chunks) == text


def test_split_for_g2p_returns_single_chunk_for_short_text():
    from spkrate.labels.mora import _split_for_g2p

    assert _split_for_g2p("今日はいい天気です") == ["今日はいい天気です"]


def test_to_kana_handles_very_long_text():
    """1万字規模の文でも異常終了せずカタカナ読みを返す。"""
    text = "今日はいい天気です。" * 1000
    kana = to_kana(text)
    assert kana
    assert not has_unconverted(kana)
    assert count_mora_from_kana(kana) > 1000
