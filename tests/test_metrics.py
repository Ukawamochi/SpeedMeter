"""評価指標の検証（src/spkrate/eval/metrics.py）。

手計算できる小さな入力で、平均絶対誤差・話速帯別の平均絶対誤差・相関係数を確かめる。
併せて、帯に該当が0件の場合と分散0の場合の取り決め（どちらも NaN）を確かめる。
"""

import math

import pytest

from spkrate.eval.metrics import (
    BAND_KEYS,
    METRIC_COLUMNS,
    SpeedRateMetrics,
    band_of,
    compute_metrics,
    mora_per_second,
    pearson_correlation,
)


def test_mora_per_second_divides_by_duration():
    assert mora_per_second([6, 10], [2.0, 2.5]) == [3.0, 4.0]


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_mora_per_second_rejects_non_positive_duration(duration):
    with pytest.raises(ValueError):
        mora_per_second([3], [duration])


def test_mora_per_second_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mora_per_second([1, 2], [1.0])


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0, "under4"),
        (3.999, "under4"),
        (4.0, "4to6"),
        (5.9, "4to6"),
        (6.0, "6to8"),
        (7.999, "6to8"),
        (8.0, "over8"),
        (20.0, "over8"),
    ],
)
def test_band_of_boundaries(rate, expected):
    """境界は「以上・未満」。4.0 は 4to6、6.0 は 6to8、8.0 は over8 に入る。"""
    assert band_of(rate) == expected


def test_band_of_rejects_nan():
    with pytest.raises(ValueError):
        band_of(math.nan)


def test_compute_metrics_hand_calculated():
    """手計算の例。

    区間長は全て2.0秒。
    正解モーラ数 [6, 10, 16, 20] -> 正解の毎秒モーラ数 [3, 5, 8, 10]
    推定モーラ数 [8, 9, 14, 24] -> 推定の毎秒モーラ数 [4, 4.5, 7, 12]
    絶対誤差 [1, 0.5, 1, 2] -> 全体の平均絶対誤差 = 4.5 / 4 = 1.125
    話速帯（正解で区分）: under4 -> [1]、4to6 -> [0.5]、6to8 -> 該当なし、
    over8 -> [1, 2] の平均 1.5
    """
    metrics = compute_metrics([6, 10, 16, 20], [8, 9, 14, 24], [2.0, 2.0, 2.0, 2.0])

    assert metrics.num_segments == 4
    assert metrics.mae_moras_per_sec == pytest.approx(1.125)
    assert metrics.mae_band_under4 == pytest.approx(1.0)
    assert metrics.mae_band_4to6 == pytest.approx(0.5)
    assert math.isnan(metrics.mae_band_6to8)
    assert metrics.mae_band_over8 == pytest.approx(1.5)
    assert (
        metrics.n_band_under4,
        metrics.n_band_4to6,
        metrics.n_band_6to8,
        metrics.n_band_over8,
    ) == (1, 1, 0, 2)


def test_bands_use_true_rate_not_prediction():
    """話速帯は正解の毎秒モーラ数で区分する。推定が別の帯でも区分は変わらない。"""
    # 正解 3 モーラ/秒（under4）、推定 9 モーラ/秒（over8 相当）。
    metrics = compute_metrics([3], [9], [1.0])
    assert metrics.n_band_under4 == 1
    assert metrics.n_band_over8 == 0
    assert metrics.mae_band_under4 == pytest.approx(6.0)
    assert math.isnan(metrics.mae_band_over8)


def test_empty_band_is_nan_with_zero_count():
    """該当区間が0件の帯は NaN、件数は0。0.0（誤差なし）とは区別する。"""
    metrics = compute_metrics([10, 12], [10, 12], [2.0, 2.0])  # いずれも 5、6 モーラ/秒
    assert metrics.n_band_under4 == 0
    assert math.isnan(metrics.mae_band_under4)
    assert metrics.mae_band_4to6 == pytest.approx(0.0)


def test_empty_input_gives_nan_metrics():
    metrics = compute_metrics([], [], [])
    assert metrics.num_segments == 0
    assert math.isnan(metrics.mae_moras_per_sec)
    assert math.isnan(metrics.correlation)
    assert all(getattr(metrics, f"n_band_{key}") == 0 for key in BAND_KEYS)


def test_compute_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics([1, 2], [1, 2], [1.0])


@pytest.mark.parametrize(
    ("preds", "expected"),
    [
        ([2.0, 4.0, 6.0], 1.0),  # 完全な正の比例
        ([6.0, 4.0, 2.0], -1.0),  # 完全な負の比例
    ],
)
def test_correlation_perfect(preds, expected):
    metrics = compute_metrics([1.0, 2.0, 3.0], preds, [1.0, 1.0, 1.0])
    assert metrics.correlation == pytest.approx(expected)


def test_correlation_hand_calculated():
    """xs=[1,2,3,4], ys=[2,4,5,4] のとき r = 3.5 / sqrt(5 * 4.75) = 0.718421..."""
    expected = 3.5 / math.sqrt(5.0 * 4.75)
    assert pearson_correlation([1, 2, 3, 4], [2, 4, 5, 4]) == pytest.approx(expected)
    # 区間長1秒ならモーラ数がそのまま毎秒モーラ数になる。
    metrics = compute_metrics([1, 2, 3, 4], [2, 4, 5, 4], [1.0, 1.0, 1.0, 1.0])
    assert metrics.correlation == pytest.approx(expected)


def test_correlation_is_nan_when_variance_is_zero():
    """推定が定数（分散0）なら相関は定義できないので NaN。0.0 とは区別する。"""
    metrics = compute_metrics([2, 4, 6], [5, 5, 5], [1.0, 1.0, 1.0])
    assert math.isnan(metrics.correlation)
    # 正解側の分散が0の場合も同じ。
    assert math.isnan(pearson_correlation([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]))


def test_correlation_is_nan_for_single_segment():
    metrics = compute_metrics([5], [4], [1.0])
    assert math.isnan(metrics.correlation)
    assert metrics.mae_moras_per_sec == pytest.approx(1.0)


def test_metric_columns_match_dataclass_keys():
    """metrics.csv の列名と dataclass のキーが一致していること。"""
    metrics = compute_metrics([5], [4], [1.0])
    assert tuple(metrics.as_dict()) == METRIC_COLUMNS
    assert METRIC_COLUMNS[0] == "num_segments"
    assert set(SpeedRateMetrics.__dataclass_fields__) == set(METRIC_COLUMNS)
