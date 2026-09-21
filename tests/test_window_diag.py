"""D1・D2・D3 の計算の単体テスト（src/spkrate/eval/window_diag.py）。

合成データと偽の推定器だけを使い、実データ（data/ 以下）には依存させない。
"""

from __future__ import annotations

import numpy as np
import pytest

from spkrate.eval.window_diag import (
    concat_additivity,
    concatenate_with_silence,
    split_consistency,
    split_into_windows,
    summarize,
)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000  # 2.0秒


def ramp(n: int) -> np.ndarray:
    """0,1,2,... という判別しやすい合成波形（値そのものに意味はない）。"""
    return np.arange(n, dtype=np.float32)


# --------------------------------------------------------------------------------------
# 切り出し


def test_split_into_windows_counts_and_slices():
    # 5.3秒 → 2窓、対象区間は4.0秒ちょうど
    waveform = ramp(int(5.3 * SAMPLE_RATE))
    windows, covered = split_into_windows(waveform)
    assert len(windows) == 2
    assert all(window.size == WINDOW_SAMPLES for window in windows)
    assert covered.size == 2 * WINDOW_SAMPLES
    np.testing.assert_array_equal(covered, waveform[: 2 * WINDOW_SAMPLES])
    np.testing.assert_array_equal(np.concatenate(windows), covered)
    # 先頭から順に並んでいる（重なりなし）
    np.testing.assert_array_equal(windows[0], waveform[:WINDOW_SAMPLES])
    np.testing.assert_array_equal(windows[1], waveform[WINDOW_SAMPLES : 2 * WINDOW_SAMPLES])


def test_split_into_windows_exact_multiple_keeps_everything():
    waveform = ramp(3 * WINDOW_SAMPLES)
    windows, covered = split_into_windows(waveform)
    assert len(windows) == 3
    assert covered.size == waveform.size


def test_split_into_windows_rejects_too_short():
    with pytest.raises(ValueError):
        split_into_windows(ramp(WINDOW_SAMPLES - 1))


def test_split_into_windows_rejects_2d():
    with pytest.raises(ValueError):
        split_into_windows(np.zeros((2, WINDOW_SAMPLES), dtype=np.float32))


def test_concatenate_with_silence_shape_and_gap():
    first = ramp(1000) + 1.0
    second = ramp(2000) + 1.0
    joined, gap = concatenate_with_silence(first, second, gap_sec=0.2)
    assert gap.size == 3200
    assert joined.size == 1000 + 3200 + 2000
    np.testing.assert_array_equal(joined[:1000], first)
    np.testing.assert_array_equal(joined[1000:4200], np.zeros(3200, dtype=np.float32))
    np.testing.assert_array_equal(joined[4200:], second)
    assert joined.dtype == np.float32


def test_concatenate_with_silence_zero_gap():
    joined, gap = concatenate_with_silence(ramp(10), ramp(20), gap_sec=0.0)
    assert gap.size == 0
    assert joined.size == 30


# --------------------------------------------------------------------------------------
# 要約


def test_summarize_basic():
    stats = summarize([0.0, 1.0, 2.0, 3.0, 100.0])
    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(21.2)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(100.0)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


# --------------------------------------------------------------------------------------
# D1


def additive_predictor(scale: float = 1e-4):
    """長さに比例した値を返す完全に加法的な推定器（Σ = P になるはず）。"""

    def predict(waveforms):
        return np.asarray([w.size * scale for w in waveforms], dtype=np.float32)

    return predict


def biased_predictor(scale: float = 1e-4, *, per_call_offset: float = 0.0):
    """呼び出しごとに一定量を差し引く推定器（窓の端の取りこぼしの模擬）。"""

    def predict(waveforms):
        return np.asarray(
            [w.size * scale - per_call_offset for w in waveforms], dtype=np.float32
        )

    return predict


def test_split_consistency_is_zero_for_additive_predictor():
    waveforms = [("a", ramp(int(5.3 * SAMPLE_RATE))), ("b", ramp(int(9.0 * SAMPLE_RATE)))]
    results = split_consistency(additive_predictor(), waveforms)
    assert [r.key for r in results] == ["a", "b"]
    assert [r.num_windows for r in results] == [2, 4]
    assert [r.covered_sec for r in results] == [4.0, 8.0]
    for result in results:
        assert result.diff == pytest.approx(0.0, abs=1e-5)
        assert result.error_mora_per_sec == pytest.approx(0.0, abs=1e-6)


def test_split_consistency_matches_hand_computed_bias():
    # 窓ごとに 0.3 モーラ取りこぼす推定器。m窓なら Σ は m×0.3 少なく、P は 0.3 少ない。
    # Σ − P = −(m−1)×0.3 なので、診断値は (m−1)×0.3 / (2.0m)。
    waveform = ramp(int(6.5 * SAMPLE_RATE))  # m = 3
    results = split_consistency(
        biased_predictor(per_call_offset=0.3), [("a", waveform)]
    )
    result = results[0]
    assert result.num_windows == 3
    assert result.covered_sec == pytest.approx(6.0)
    assert result.diff == pytest.approx(-(3 - 1) * 0.3, abs=1e-4)
    assert result.error_mora_per_sec == pytest.approx(2 * 0.3 / 6.0, abs=1e-5)


def test_split_consistency_rejects_wrong_prediction_count():
    def broken(waveforms):
        return np.zeros(len(waveforms) - 1, dtype=np.float32)

    with pytest.raises(ValueError):
        split_consistency(broken, [("a", ramp(WINDOW_SAMPLES))])


# --------------------------------------------------------------------------------------
# D3


def test_concat_additivity_is_zero_for_additive_predictor():
    pairs = [("a+b", ramp(int(3.0 * SAMPLE_RATE)), ramp(int(4.0 * SAMPLE_RATE)))]
    results = concat_additivity(additive_predictor(), pairs)
    result = results[0]
    assert result.total_sec == pytest.approx(3.0 + 0.2 + 4.0)
    assert result.diff == pytest.approx(0.0, abs=1e-5)
    assert result.error_mora_per_sec == pytest.approx(0.0, abs=1e-6)


def test_concat_additivity_matches_hand_computed_bias():
    # 1回の呼び出しにつき 0.3 引かれる。P(AB) は −0.3、右辺は 3回分で −0.9。
    # 差 = P(AB) − 右辺 = −0.3 − (−0.9) = +0.6。
    pairs = [("a+b", ramp(int(3.0 * SAMPLE_RATE)), ramp(int(4.0 * SAMPLE_RATE)))]
    results = concat_additivity(biased_predictor(per_call_offset=0.3), pairs)
    result = results[0]
    assert result.diff == pytest.approx(0.6, abs=1e-4)
    assert result.error_mora_per_sec == pytest.approx(0.6 / 7.2, abs=1e-5)


def test_concat_additivity_predicts_gap_through_the_same_path():
    seen: list[int] = []

    def predict(waveforms):
        seen.extend(int(w.size) for w in waveforms)
        return np.asarray([w.size * 1e-4 for w in waveforms], dtype=np.float32)

    pairs = [("a+b", ramp(1000), ramp(2000))]
    concat_additivity(predict, pairs, gap_sec=0.2)
    # 連結・A・B・無音 の4本が渡る
    assert seen == [1000 + 3200 + 2000, 1000, 2000, 3200]


def test_concat_additivity_rejects_wrong_prediction_count():
    def broken(waveforms):
        return np.zeros(len(waveforms) + 1, dtype=np.float32)

    with pytest.raises(ValueError):
        concat_additivity(broken, [("a+b", ramp(100), ramp(100))])
