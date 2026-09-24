"""低出力事例の診断の計算（src/spkrate/eval/low_output_diag.py）の単体テスト。

合成データのみを使い、data/ 以下には依存させない。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spkrate.eval.low_output_diag import (
    feature_stats,
    frame_output_stats,
    group_summary,
    holm_adjust,
    normalized_feature_stats,
    select_comparison,
    waveform_stats,
)


def test_waveform_stats_values():
    wave = np.array([0.0, 0.0005, -0.5, 0.5], dtype=np.float32)
    stats = waveform_stats(wave, threshold=1e-3)
    assert stats["peak"] == pytest.approx(0.5)
    assert stats["silence_frac"] == pytest.approx(0.5)
    assert stats["rms"] == pytest.approx(math.sqrt((0.0005**2 + 0.25 + 0.25) / 4), rel=1e-5)


def test_waveform_stats_rejects_empty():
    with pytest.raises(ValueError):
        waveform_stats(np.zeros(0, dtype=np.float32))


def test_feature_stats_and_normalized():
    feature = np.array([[1.0, -3.0], [2.0, 0.0]], dtype=np.float32)
    stats = feature_stats(feature)
    assert stats["logmel_mean"] == pytest.approx(0.0)
    assert stats["logmel_max"] == pytest.approx(2.0)
    assert stats["logmel_min"] == pytest.approx(-3.0)
    norm = normalized_feature_stats(feature)
    assert norm["norm_absmax"] == pytest.approx(3.0)
    assert norm["norm_std"] == pytest.approx(float(np.std(feature)))


def test_frame_output_stats_distinguishes_local_peak():
    flat = np.full(100, 0.002, dtype=np.float32)
    peaky = flat.copy()
    peaky[:5] = 0.2
    s_flat = frame_output_stats(flat)
    s_peaky = frame_output_stats(peaky)
    assert s_flat["frame_frac_gt_0p05"] == 0.0
    assert s_peaky["frame_frac_gt_0p05"] == pytest.approx(0.05)
    assert s_peaky["frame_max"] == pytest.approx(0.2)
    assert s_flat["frame_sum"] == pytest.approx(0.2, rel=1e-5)
    assert s_flat["frame_top10pct_share"] == pytest.approx(0.1, rel=1e-5)
    assert s_peaky["frame_top10pct_share"] > 0.8
    assert s_flat["n_frames"] == 100


def test_select_comparison_is_reproducible_and_below_median():
    errors = {f"c{i:03d}": float(i) for i in range(101)}  # 中央値は50
    chosen, median, pool = select_comparison(errors, 10, seed=7, exclude=["c000"])
    assert median == pytest.approx(50.0)
    assert pool == 50  # c001..c050
    assert all(errors[c] <= 50.0 for c in chosen)
    assert "c000" not in chosen
    assert len(set(chosen)) == 10
    again, _, _ = select_comparison(dict(reversed(list(errors.items()))), 10, seed=7,
                                    exclude=["c000"])
    assert chosen == again


def test_select_comparison_rejects_small_pool():
    with pytest.raises(ValueError):
        select_comparison({"a": 1.0, "b": 2.0}, 5, seed=0)


def test_holm_adjust_matches_hand_calculation():
    # p = [0.01, 0.04, 0.03, nan] → m=3, 並べると 0.01, 0.03, 0.04
    # 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 → 単調化で 0.06
    adjusted = holm_adjust([0.01, 0.04, 0.03, float("nan")])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert math.isnan(adjusted[3])


def test_group_summary_quartiles():
    summary = group_summary([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")])
    assert summary["n"] == 5
    assert summary["median"] == pytest.approx(3.0)
    assert summary["q1"] == pytest.approx(2.0)
    assert summary["q3"] == pytest.approx(4.0)
