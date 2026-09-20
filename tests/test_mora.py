from spkrate.labels.mora import count_mora, read_kana


# 清音のみ
def test_seion_sakura():
    assert count_mora("サクラ") == 3


def test_seion_aiueo():
    assert count_mora("アイウエオ") == 5


def test_seion_katakana():
    assert count_mora("カタカナ") == 4


def test_seion_hana():
    assert count_mora("ハナ") == 2


def test_seion_mizu():
    assert count_mora("ミズ") == 2


# 拗音
def test_youon_kyou():
    assert count_mora("キョウ") == 2


def test_youon_shuppatsu():
    assert count_mora("シュッパツ") == 4


def test_youon_kyaku():
    assert count_mora("キャク") == 2


def test_youon_ryokou():
    assert count_mora("リョコウ") == 3


def test_youon_jouzu():
    assert count_mora("ジョウズ") == 3


def test_youon_chocolate():
    assert count_mora("チョコレート") == 5


# 促音
def test_sokuon_gakkou():
    assert count_mora("ガッコウ") == 4


def test_sokuon_kippu():
    assert count_mora("キップ") == 3


def test_sokuon_rappa():
    assert count_mora("ラッパ") == 3


def test_sokuon_zettai():
    assert count_mora("ゼッタイ") == 4


# 撥音
def test_hatsuon_nihon():
    assert count_mora("ニホン") == 3


def test_hatsuon_sanpo():
    assert count_mora("サンポ") == 3


def test_hatsuon_honya():
    assert count_mora("ホンヤ") == 3


# 長音
def test_chouon_coffee():
    assert count_mora("コーヒー") == 4


def test_chouon_super():
    assert count_mora("スーパー") == 4


def test_chouon_ramen():
    assert count_mora("ラーメン") == 4


# 漢字かな混じり文からの変換（read_kana + count_moraの組み合わせ）
def test_kanji_kana_sakura_ga_sakimashita():
    kana = read_kana("桜が咲きました")
    assert kana == "サクラガサキマシタ"
    assert count_mora(kana) == 9


def test_kanji_kana_kyou_wa_kaisei():
    kana = read_kana("今日は快晴です")
    assert kana == "キョーワカイセーデス"
    assert count_mora(kana) == 9


def test_kanji_kana_eki_made():
    kana = read_kana("駅まで歩いて行きます")
    assert kana == "エキマデアルイテイキマス"
    assert count_mora(kana) == 12


def test_kanji_kana_kanojo():
    kana = read_kana("彼女は毎朝走ります")
    assert kana == "カノジョワマイアサハシリマス"
    assert count_mora(kana) == 13


# 数字を含む文
def test_number_date():
    kana = read_kana("2026年9月20日に会議があります")
    assert kana == "ニセンニジューロクネンクガツハツカニカイギガアリマス"
    assert count_mora(kana) == 25


def test_number_people_minutes():
    kana = read_kana("3人で30分話しました")
    assert kana == "サンニンデサンジュップンハナシマシタ"
    assert count_mora(kana) == 17


def test_number_meters():
    kana = read_kana("100メートル走りました")
    assert kana == "ヒャクメートルハシリマシタ"
    assert count_mora(kana) == 12


# 英字を含む文
def test_alphabet_only():
    kana = read_kana("ABC")
    assert kana == "エイビーシー"
    assert count_mora(kana) == 6


def test_alphabet_cnn_model():
    kana = read_kana("CNNモデルを学習します")
    assert kana == "シーエヌエヌモデルヲガクシューシマス"
    assert count_mora(kana) == 17


def test_alphabet_gpu():
    kana = read_kana("GPUを使って計算します")
    assert kana == "ジーピーユーヲツカッテケーサンシマス"
    assert count_mora(kana) == 18


# 記号・句読点を含む文
def test_symbol_question_mark():
    kana = read_kana("こんにちは、元気ですか?")
    assert kana == "コンニチワゲンキデスカ"
    assert count_mora(kana) == 11


def test_symbol_exclamation():
    kana = read_kana("走った!すごい!")
    assert kana == "ハシッタスゴイ"
    assert count_mora(kana) == 7


def test_symbol_period_touten():
    kana = read_kana("それは、本当に良かった。")
    assert kana == "ソレワホントーニヨカッタ"
    assert count_mora(kana) == 12


def test_symbol_only_string_is_empty():
    assert read_kana("!!!") == ""
    assert count_mora(read_kana("!!!")) == 0


def test_whitespace_only_string_is_empty():
    assert read_kana("   ") == ""


def test_empty_string_is_empty():
    assert read_kana("") == ""
    assert count_mora("") == 0


# フィラーを含む文
def test_filler_eeto():
    kana = read_kana("えーと、それではお願いします")
    assert kana == "エートソレデワオネガイシマス"
    assert count_mora(kana) == 14


def test_filler_anoo():
    kana = read_kana("あのー、少し待ってください")
    assert kana == "アノースコシマッテクダサイ"
    assert count_mora(kana) == 13


def test_filler_maa_sono():
    kana = read_kana("まあ、その、なんというか")
    assert kana == "マーソノナントイウカ"
    assert count_mora(kana) == 10
