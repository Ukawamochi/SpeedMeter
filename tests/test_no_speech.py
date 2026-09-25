"""spkrate.eval.no_speech の単体テスト（かなの正規化、文字誤り率、判定規則の適用）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spkrate.eval.no_speech import (
    Condition,
    Rule,
    apply_rule,
    cer,
    ctc_greedy_decode,
    ctc_metrics,
    edit_distance,
    energy_metrics,
    hyp_to_text,
    normalize_reference,
    windows_from_clips,
)

# モデルの語彙の一部を模したもの（「ゎゐゑ」は無く、「ー」と区切り「|」がある）
VOCAB = {c: i for i, c in enumerate("あいうえおかきくけこしゃょわーん|")}
VOCAB["[UNK]"] = len(VOCAB)
VOCAB["[PAD]"] = len(VOCAB)
BLANK = VOCAB["[PAD]"]


# ---------------------------------------------------------------- かなの正規化
def test_normalize_katakana_to_hiragana_keeps_long_vowel_and_small():
    assert normalize_reference("キャーコン", VOCAB) == "きゃーこん"


def test_normalize_fallback_for_missing_kana():
    # ヮ→ゎ→わ、ヰ→ゐ→い、ヱ→ゑ→え（アライメントと同じ置き換え）
    assert normalize_reference("クヮヰヱ", VOCAB) == "くわいえ"


def test_normalize_drops_unknown_and_delimiter():
    # 語彙に無い文字（ト）・記号・区切りは捨てる
    assert normalize_reference("ア、ト|イ!", VOCAB) == "あい"


# ---------------------------------------------------------------- 貪欲デコード
def test_greedy_decode_collapses_repeats_and_blank():
    a, i = VOCAB["あ"], VOCAB["い"]
    frames = [BLANK, a, a, BLANK, a, i, i, BLANK]
    assert ctc_greedy_decode(frames, BLANK) == [a, a, i]


def test_hyp_to_text_drops_non_chars():
    ids = [VOCAB["あ"], VOCAB["|"], VOCAB["[UNK]"], VOCAB["い"]]
    inv = {v: k for k, v in VOCAB.items()}
    assert hyp_to_text(ids, inv) == "あい"


# ---------------------------------------------------------------- 文字誤り率
@pytest.mark.parametrize(
    ("ref", "hyp", "dist"),
    [("", "", 0), ("あい", "", 2), ("", "あい", 2), ("あいう", "あいう", 0),
     ("あいう", "あえう", 1), ("あいう", "あう", 1), ("あいう", "あいうえ", 1), ("かきくけこ", "きくけこか", 2)],
)
def test_edit_distance(ref, hyp, dist):
    assert edit_distance(ref, hyp) == dist
    assert edit_distance(hyp, ref) == dist


def test_cer_values():
    assert cer("あいう", "あいう") == 0.0
    assert cer("あいう", "") == 1.0
    assert cer("あいう", "あえうおか") == pytest.approx(3 / 3)
    assert cer("あいうえ", "あうえ") == pytest.approx(1 / 4)
    assert cer("", "") == 0.0
    assert cer("", "あ") == 1.0


# ---------------------------------------------------------------- CTC の指標
def _emission(frame_ids):
    lp = torch.full((len(frame_ids), len(VOCAB)), -20.0)
    for f, t in enumerate(frame_ids):
        lp[f, t] = 0.0
    return torch.log_softmax(lp, dim=-1)[None]


def test_ctc_metrics_perfect_and_silent():
    a, i = VOCAB["あ"], VOCAB["い"]
    good = ctc_metrics(_emission([BLANK, a, BLANK, i, BLANK]), "あい", VOCAB, BLANK, mora=2)
    assert good["hyp"] == "あい" and good["cer"] == 0.0 and good["hyp_mora_ratio"] == 1.0
    assert good["ctc_nonblank_frac"] == pytest.approx(2 / 5)
    silent = ctc_metrics(_emission([BLANK] * 5), "あい", VOCAB, BLANK, mora=2)
    assert silent["hyp"] == "" and silent["cer"] == 1.0 and silent["ctc_nonblank_frac"] == 0.0
    assert silent["ctc_nll_per_token"] > good["ctc_nll_per_token"]


def test_ctc_metrics_too_few_frames_gives_nan():
    out = ctc_metrics(_emission([BLANK]), "あい", VOCAB, BLANK, mora=2)
    assert math.isnan(out["ctc_nll_per_token"])


# ---------------------------------------------------------------- 音量の指標
def test_energy_metrics_silence_vs_burst():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 1e-3).astype(np.float32)
    flat = energy_metrics(noise)
    burst = noise.copy()
    burst[8000:12000] += (0.3 * np.sin(np.arange(4000) * 0.1)).astype(np.float32)
    loud = energy_metrics(burst)
    assert flat["energy_speech_frac"] == 0.0
    assert loud["energy_speech_frac"] == pytest.approx(0.25, abs=0.03)
    assert loud["frame_db_range"] > flat["frame_db_range"] + 20
    assert flat["rms_dbfs"] == pytest.approx(-60.0, abs=0.5)
    assert loud["peak_dbfs"] == pytest.approx(20 * math.log10(0.3), abs=0.2)


# ---------------------------------------------------------------- 判定規則
ROWS = [
    {"clip_id": "a", "cer": 0.95, "ctc_nonblank_frac": 0.02},
    {"clip_id": "b", "cer": 0.95, "ctc_nonblank_frac": 0.30},
    {"clip_id": "c", "cer": 0.10, "ctc_nonblank_frac": 0.01},
    {"clip_id": "d", "cer": 0.90, "ctc_nonblank_frac": float("nan")},
]


def test_single_condition_threshold_is_inclusive_by_op():
    assert apply_rule(ROWS, Rule((Condition("cer", ">=", 0.95),))) == ["a", "b"]
    assert apply_rule(ROWS, Rule((Condition("cer", ">", 0.95),))) == []


def test_conjunction_and_nan_never_matches():
    rule = Rule((Condition("cer", ">=", 0.9), Condition("ctc_nonblank_frac", "<", 0.05)))
    assert apply_rule(ROWS, rule) == ["a"]
    assert str(rule) == "cer >= 0.9 かつ ctc_nonblank_frac < 0.05"


def test_rule_from_mapping_and_validation():
    rule = Rule.from_mapping({"conditions": [{"metric": "cer", "op": ">=", "threshold": 0.9}]})
    assert apply_rule(ROWS, rule) == ["a", "b", "d"]
    with pytest.raises(ValueError):
        Condition("cer", "==", 1.0)
    with pytest.raises(ValueError):
        Rule(())


def test_windows_from_clips_marks_whole_group():
    class S:
        def __init__(self, ids):
            self.clip_ids = ids

    class WS:
        sources = [S(("x",)), S(("y",)), S(("y", "z", "w"))]
        source_index = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)

    assert windows_from_clips(WS, {"y"}).tolist() == [False, False, True, True, True, True]
    assert windows_from_clips(WS, {"w"}).tolist() == [False, False, False, True, True, True]
    assert windows_from_clips(WS, set()).tolist() == [False] * 6
