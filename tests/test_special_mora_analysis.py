"""scripts/special_mora_analysis.py のかなから特殊なモーラを数える関数のテスト。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("special_mora_analysis", REPO / "scripts" / "special_mora_analysis.py")
sma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sma)


def test_classify_single_special():
    assert sma.classify_mora("ー") == {"chouon": True, "hatsuon": False, "sokuon": False,
                                       "special": True, "yoon": False, "yoon_ya": False}
    assert sma.classify_mora("ン")["hatsuon"] and sma.classify_mora("ン")["special"]
    assert sma.classify_mora("ッ")["sokuon"] and sma.classify_mora("ッ")["special"]
    assert not any(sma.classify_mora("カ").values())


def test_classify_yoon():
    assert sma.classify_mora("キャ")["yoon"] and sma.classify_mora("キャ")["yoon_ya"]
    # 外来音の結合（ティ・ファ）と合拗音（クヮ）は拗音に含め、狭い意味の拗音（ャュョ）には含めない
    for mora in ("ティ", "ファ", "クヮ"):
        kind = sma.classify_mora(mora)
        assert kind["yoon"] and not kind["yoon_ya"] and not kind["special"]
    # 結合先の無い小書き文字（単独の1モーラ）は拗音ではない
    assert not sma.classify_mora("ャ")["yoon"]


def test_count_special_morae():
    # コンピューター: コ ン ピュ ー タ ー
    c = sma.count_special_morae("コンピューター")
    assert c["mora"] == 6
    assert (c["hatsuon"], c["chouon"], c["sokuon"], c["special"]) == (1, 2, 0, 3)
    assert (c["yoon"], c["yoon_ya"]) == (1, 1)
    # チャット: チャ ッ ト
    c = sma.count_special_morae("チャット")
    assert (c["mora"], c["sokuon"], c["special"], c["yoon"]) == (3, 1, 1, 1)
    # 先頭の小書き文字は単独で1モーラ（拗音に数えない）
    c = sma.count_special_morae("ャア")
    assert (c["mora"], c["yoon"]) == (2, 0)
    # カタカナ以外は数えない
    assert sma.count_special_morae("ア、ン")["mora"] == 2


def test_word_mora_lengths_skips_symbols():
    nodes = [
        {"string": "先生", "pron": "センセー", "pos": "名詞"},
        {"string": "、", "pron": "、", "pos": "記号"},
        {"string": "人", "pron": "ヒ’ト", "pos": "名詞"},
        {"string": "チャット", "pron": "チャット", "pos": "名詞"},
    ]
    lengths, prons = sma.word_mora_lengths(nodes)
    assert lengths == [4, 2, 3]
    assert prons == ["センセー", "ヒト", "チャット"]


def test_mora_fractions_sum_matches_window_mora():
    from spkrate.eval.dev_window import window_mora

    starts = np.array([0.1, 1.9, 2.5, 3.0], dtype=np.float32)
    ends = np.array([0.2, 2.1, 2.5, 3.4], dtype=np.float32)
    window_start = np.array([0.0, 0.25, 1.5], dtype=np.float32)
    frac = sma.mora_fractions(starts, ends, window_start, 2.0)
    np.testing.assert_allclose(frac.sum(axis=1), window_mora(starts, ends, window_start, 2.0), atol=1e-6)
    # 窓 [0, 2) で 1.9〜2.1 のモーラは半分
    np.testing.assert_allclose(frac[0, 1], 0.5, atol=1e-5)
