"""信号処理ベースラインの単体テスト（docs/PLAN.md 第3段階 3-2）。

合成信号のみを使い、実データには依存させない。中心となる確認は
「既知の周期で振幅変調した信号から、その周期どおりの個数が数えられること」である。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spkrate.baselines.envelope import (
    EnvelopeParams,
    EnvelopeSpeedEstimator,
    bandpass_envelope,
    count_syllables_from_log_envelope,
    frame_rms,
    load_params,
    log_envelope,
    smooth,
    syllable_rate,
)

SAMPLE_RATE = 16000


def pulse_train(
    pulses_per_sec: float, duration_sec: float, *, amplitude: float = 1.0, seed: int = 0
) -> np.ndarray:
    """既知の周期で振幅変調した雑音。1秒あたり ``pulses_per_sec`` 個の山ができる。

    変調は ``(0.5 - 0.5 cos)^2`` とし、山と山の間が十分に落ち込むようにしている
    （実際の発話でも音節の境界では音量が落ちる）。
    """
    rng = np.random.default_rng(seed)
    times = np.arange(int(duration_sec * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    carrier = rng.normal(0.0, 1.0, times.size).astype(np.float32)
    envelope = (0.5 - 0.5 * np.cos(2.0 * math.pi * pulses_per_sec * times)) ** 2
    return (amplitude * carrier * envelope).astype(np.float32)


class TestFrameRms:
    def test_frame_count_and_rate(self) -> None:
        signal = np.ones(SAMPLE_RATE, dtype=np.float32)
        envelope = frame_rms(signal, SAMPLE_RATE, frame_length_sec=0.025, hop_sec=0.010)
        # 1秒 / 10ミリ秒 = 100フレーム（包絡の標本化周波数は100Hz）。
        assert envelope.size == 100
        assert envelope.dtype == np.float32

    def test_constant_signal_gives_constant_rms(self) -> None:
        signal = np.full(SAMPLE_RATE, 0.5, dtype=np.float32)
        envelope = frame_rms(signal, SAMPLE_RATE)
        # 端のフレームは0埋めの影響を受けるので中央だけ見る。
        assert np.allclose(envelope[10:-10], 0.5, atol=1e-5)

    def test_empty_signal(self) -> None:
        assert frame_rms(np.zeros(0, dtype=np.float32), SAMPLE_RATE).size == 0

    def test_invalid_sample_rate(self) -> None:
        with pytest.raises(ValueError):
            frame_rms(np.ones(10, dtype=np.float32), 0)


class TestLogEnvelope:
    def test_decibel_conversion(self) -> None:
        values = np.array([1.0, 0.1], dtype=np.float32)
        decibels = log_envelope(values, floor_db=-60.0)
        assert decibels[0] == pytest.approx(0.0, abs=1e-4)
        assert decibels[1] == pytest.approx(-20.0, abs=1e-4)

    def test_floor_is_relative_to_maximum(self) -> None:
        values = np.array([1.0, 1e-8], dtype=np.float32)
        decibels = log_envelope(values, floor_db=-40.0)
        assert decibels[1] == pytest.approx(-40.0, abs=1e-4)

    def test_all_zero_is_finite(self) -> None:
        decibels = log_envelope(np.zeros(20, dtype=np.float32))
        assert np.all(np.isfinite(decibels))


class TestSmooth:
    def test_zero_time_constant_is_identity(self) -> None:
        values = np.array([1.0, 5.0, 2.0], dtype=np.float32)
        assert np.array_equal(smooth(values, 100.0, 0.0), values)

    def test_constant_is_preserved_including_edges(self) -> None:
        values = np.full(50, -3.0, dtype=np.float32)
        smoothed = smooth(values, 100.0, 0.05)
        assert np.allclose(smoothed, -3.0, atol=1e-5)

    def test_reduces_single_sample_spike(self) -> None:
        values = np.zeros(50, dtype=np.float32)
        values[25] = 10.0
        smoothed = smooth(values, 100.0, 0.05)
        assert smoothed[25] < 10.0
        assert smoothed[24] > 0.0


class TestBandpass:
    def test_removes_constant_offset(self) -> None:
        values = np.full(200, 7.0, dtype=np.float32)
        filtered = bandpass_envelope(values, 100.0)
        assert np.max(np.abs(filtered)) < 1e-3

    def test_keeps_in_band_oscillation(self) -> None:
        times = np.arange(300, dtype=np.float32) / 100.0
        in_band = np.sin(2.0 * math.pi * 5.0 * times).astype(np.float32)
        out_of_band = np.sin(2.0 * math.pi * 0.2 * times).astype(np.float32)
        assert np.std(bandpass_envelope(in_band, 100.0)) > 0.5
        assert np.std(bandpass_envelope(out_of_band, 100.0)) < 0.1

    def test_too_short_input_returns_zeros(self) -> None:
        filtered = bandpass_envelope(np.ones(5, dtype=np.float32), 100.0)
        assert filtered.size == 5
        assert np.all(filtered == 0.0)


class TestCountSyllables:
    @pytest.mark.parametrize("pulses_per_sec", [3.0, 4.0, 5.0, 6.0, 8.0])
    @pytest.mark.parametrize("duration_sec", [2.0, 3.0])
    def test_peak_count_matches_known_pulse_count(
        self, pulses_per_sec: float, duration_sec: float
    ) -> None:
        signal = pulse_train(pulses_per_sec, duration_sec)
        estimator = EnvelopeSpeedEstimator(EnvelopeParams(count_method="peak"))
        assert estimator.count_syllables(signal, SAMPLE_RATE) == round(
            pulses_per_sec * duration_sec
        )

    @pytest.mark.parametrize("pulses_per_sec", [3.0, 5.0, 8.0])
    def test_zero_cross_count_is_within_one(self, pulses_per_sec: float) -> None:
        signal = pulse_train(pulses_per_sec, 3.0)
        estimator = EnvelopeSpeedEstimator(EnvelopeParams(count_method="zero_cross"))
        count = estimator.count_syllables(signal, SAMPLE_RATE)
        # ゼロ交差は区間の端の半周期を落とすことがあるため1個のずれを許す。
        assert abs(count - round(pulses_per_sec * 3.0)) <= 1

    def test_count_is_invariant_to_gain(self) -> None:
        """デシベル領域で処理するため、録音レベルを変えても個数は変わらない。"""
        signal = pulse_train(5.0, 3.0)
        estimator = EnvelopeSpeedEstimator()
        loud = estimator.count_syllables(signal, SAMPLE_RATE)
        quiet = estimator.count_syllables((signal * 0.01).astype(np.float32), SAMPLE_RATE)
        assert loud == quiet

    def test_silence_counts_zero(self) -> None:
        estimator = EnvelopeSpeedEstimator()
        assert estimator.count_syllables(np.zeros(2 * SAMPLE_RATE, np.float32), SAMPLE_RATE) == 0

    def test_trailing_silence_is_not_counted(self) -> None:
        """無音ゲートが効いていれば、後ろに無音を足しても個数は増えない。"""
        speech = pulse_train(5.0, 2.0)
        padded = np.concatenate([speech, np.zeros(SAMPLE_RATE, dtype=np.float32)])
        estimator = EnvelopeSpeedEstimator()
        assert estimator.count_syllables(padded, SAMPLE_RATE) == estimator.count_syllables(
            speech, SAMPLE_RATE
        )

    def test_empty_log_envelope(self) -> None:
        assert count_syllables_from_log_envelope(np.zeros(0, np.float32), 100.0, EnvelopeParams()) == 0

    def test_cached_envelope_path_matches_waveform_path(self) -> None:
        """探索で使う「包絡から数える」経路が、波形から数えた結果と一致すること。"""
        signal = pulse_train(6.0, 3.0)
        params = EnvelopeParams(smoothing_sec=0.03, peak_prominence=2.0)
        estimator = EnvelopeSpeedEstimator(params)
        cached = count_syllables_from_log_envelope(
            estimator.log_envelope_of(signal, SAMPLE_RATE), params.envelope_rate, params
        )
        assert cached == estimator.count_syllables(signal, SAMPLE_RATE)


class TestSyllableRate:
    def test_rate(self) -> None:
        assert syllable_rate(10, 2.0) == pytest.approx(5.0)

    def test_non_positive_duration(self) -> None:
        assert syllable_rate(10, 0.0) == 0.0


class TestEstimator:
    def test_returns_mora_per_second_with_conversion_factor(self) -> None:
        signal = pulse_train(5.0, 2.0)
        params = EnvelopeParams(mora_per_syllable=1.4)
        estimator = EnvelopeSpeedEstimator(params)
        count = estimator.count_syllables(signal, SAMPLE_RATE)
        assert estimator(signal, SAMPLE_RATE) == pytest.approx(1.4 * count / 2.0)

    def test_conversion_factor_scales_output(self) -> None:
        signal = pulse_train(5.0, 2.0)
        single = EnvelopeSpeedEstimator(EnvelopeParams(mora_per_syllable=1.0))
        double = EnvelopeSpeedEstimator(EnvelopeParams(mora_per_syllable=2.0))
        assert double(signal, SAMPLE_RATE) == pytest.approx(2.0 * single(signal, SAMPLE_RATE))

    def test_faster_speech_gives_higher_rate(self) -> None:
        estimator = EnvelopeSpeedEstimator()
        slow = estimator(pulse_train(3.0, 3.0), SAMPLE_RATE)
        fast = estimator(pulse_train(7.0, 3.0), SAMPLE_RATE)
        assert fast > slow

    def test_empty_input_returns_zero(self) -> None:
        assert EnvelopeSpeedEstimator()(np.zeros(0, dtype=np.float32), SAMPLE_RATE) == 0.0

    def test_matches_runner_estimator_interface(self) -> None:
        """runner.evaluate が呼ぶ形（samples, sample_rate）で有限の値を返すこと。"""
        from spkrate.eval.runner import evaluate

        value = EnvelopeSpeedEstimator()(pulse_train(5.0, 2.0), SAMPLE_RATE)
        assert np.isfinite(value) and value >= 0.0
        assert callable(evaluate)


class TestParams:
    def test_envelope_rate(self) -> None:
        assert EnvelopeParams(hop_sec=0.01).envelope_rate == pytest.approx(100.0)

    def test_replace_keeps_other_values(self) -> None:
        params = EnvelopeParams(band_low_hz=2.5).replace(peak_prominence=3.0)
        assert params.band_low_hz == 2.5
        assert params.peak_prominence == 3.0

    def test_rejects_unknown_count_method(self) -> None:
        with pytest.raises(ValueError):
            EnvelopeParams(count_method="onset")

    def test_rejects_inverted_band(self) -> None:
        with pytest.raises(ValueError):
            EnvelopeParams(band_low_hz=10.0, band_high_hz=2.0)

    def test_rejects_band_above_nyquist_of_envelope(self) -> None:
        with pytest.raises(ValueError):
            EnvelopeParams(hop_sec=0.1, band_high_hz=10.0)

    def test_load_params_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "envelope.yaml"
        path.write_text(
            "params:\n  band_low_hz: 2.5\n  count_method: zero_cross\n  mora_per_syllable: 1.8\n",
            encoding="utf-8",
        )
        params = load_params(path)
        assert params.band_low_hz == 2.5
        assert params.count_method == "zero_cross"
        assert params.mora_per_syllable == 1.8

    def test_load_params_rejects_unknown_key(self, tmp_path) -> None:
        path = tmp_path / "envelope.yaml"
        path.write_text("params:\n  unknown_key: 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_params(path)
