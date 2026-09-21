"""対数メルスペクトログラムの単体テスト（docs/PLAN.md 第4段階 4-1）。

実データには依存しない。入力はすべてこの場で作る合成波形である。
docs/spec.md の特徴量パラメータが実際に使われていることを、
numpy で独立に書いた参照実装との一致で確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

from spkrate.features.melspec import (
    MEL_DEFAULTS,
    LogMelSpectrogram,
    MelSpecConfig,
    log_mel_spectrogram,
    num_frames,
)

SAMPLE_RATE = 16000


def sine(frequency: float, num_samples: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


# --------------------------------------------------------------------------------------
# 参照実装（numpy）。torchaudio を使わずに docs/spec.md のパラメータを書き下す。


def htk_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def htk_mel_inverse(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def reference_filterbank(n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """頂点が1の三角フィルタ（torchaudio の norm=None, mel_scale="htk" と同じ形）。"""
    bin_freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(htk_mel(np.array(f_min)), htk_mel(np.array(f_max)), n_mels + 2)
    freq_points = htk_mel_inverse(mel_points)
    fb = np.zeros((len(bin_freqs), n_mels), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = freq_points[index : index + 3]
        rising = (bin_freqs - left) / (center - left)
        falling = (right - bin_freqs) / (right - center)
        fb[:, index] = np.maximum(0.0, np.minimum(rising, falling))
    return fb


def reference_log_mel(samples: np.ndarray) -> np.ndarray:
    """docs/spec.md のパラメータをそのまま書き下した参照実装。"""
    n_fft, hop_length, n_mels = 400, 160, 80
    # center=True, pad_mode="reflect"
    padded = np.pad(samples.astype(np.float64), n_fft // 2, mode="reflect")
    # periodic hann
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    frames = 1 + len(samples) // hop_length
    spectrum = np.empty((frames, n_fft // 2 + 1), dtype=np.float64)
    for index in range(frames):
        chunk = padded[index * hop_length : index * hop_length + n_fft]
        spectrum[index] = np.abs(np.fft.rfft(chunk * window)) ** 2.0  # power=2.0
    mel = spectrum @ reference_filterbank(n_fft, n_mels, 0.0, 8000.0)
    return np.log(mel + 1e-6)


# --------------------------------------------------------------------------------------
# 形状


@pytest.mark.parametrize(
    ("num_samples", "expected_frames"),
    [(32000, 201), (16000, 101), (1600, 11), (640, 5), (480, 4)],
)
def test_output_shape_follows_spec(num_samples: int, expected_frames: int) -> None:
    """フレーム数は 1 + 標本数 // 160、メル次元は 80。"""
    feature = log_mel_spectrogram(sine(440.0, num_samples), SAMPLE_RATE)
    assert feature.shape == (expected_frames, 80)
    assert num_frames(num_samples) == expected_frames


def test_output_dtype_is_float32() -> None:
    """CLAUDE.md「float64は使わない」。"""
    feature = log_mel_spectrogram(sine(440.0, 32000), SAMPLE_RATE)
    assert feature.dtype == np.float32


def test_two_second_window_is_201_frames() -> None:
    """docs/spec.md「モデルの入力単位: 固定長窓2.0秒」に対応する形。"""
    feature = log_mel_spectrogram(sine(300.0, 2 * SAMPLE_RATE), SAMPLE_RATE)
    assert feature.shape == (201, 80)


# --------------------------------------------------------------------------------------
# 仕様のパラメータが実際に使われているか


def test_config_defaults_match_spec() -> None:
    assert MEL_DEFAULTS.sample_rate == 16000
    assert MEL_DEFAULTS.n_fft == 400
    assert MEL_DEFAULTS.win_length == 400
    assert MEL_DEFAULTS.hop_length == 160
    assert MEL_DEFAULTS.n_mels == 80
    assert MEL_DEFAULTS.f_min == 0.0
    assert MEL_DEFAULTS.f_max == 8000.0
    assert MEL_DEFAULTS.power == 2.0
    assert MEL_DEFAULTS.log_offset == 1e-6
    # 仕様に無く torchaudio の既定に合わせた値（docstring と一致すること）。
    assert MEL_DEFAULTS.window == "hann"
    assert MEL_DEFAULTS.window_periodic is True
    assert MEL_DEFAULTS.center is True
    assert MEL_DEFAULTS.pad_mode == "reflect"
    assert MEL_DEFAULTS.mel_scale == "htk"
    assert MEL_DEFAULTS.mel_norm is None
    assert MEL_DEFAULTS.normalized is False


def test_matches_numpy_reference_implementation() -> None:
    """窓・詰め方・メル尺度・対数まで含め、独立な参照実装と一致する。"""
    rng = np.random.default_rng(20260921)
    samples = rng.standard_normal(8000).astype(np.float32) * 0.1
    feature = log_mel_spectrogram(samples, SAMPLE_RATE)
    reference = reference_log_mel(samples)
    assert feature.shape == reference.shape
    assert np.allclose(feature, reference, atol=2e-3, rtol=0.0)


def test_filterbank_is_htk_with_unit_peaks() -> None:
    """メルフィルタはHTK尺度・norm=None（三角の頂点が1、面積正規化なし）である。"""
    transform = LogMelSpectrogram()
    fb = transform.mel_filterbank
    assert fb.shape == (201, 80)
    # norm=None なので重みは1を超えない（slaney 正規化なら帯域幅で割るため1を超える列が出る）。
    assert fb.max() <= 1.0 + 1e-6
    # どの列にも正の重みがある（f_min=0, f_max=8000 の80分割）。
    assert (fb.max(axis=0) > 0.0).all()
    assert np.allclose(fb, reference_filterbank(400, 80, 0.0, 8000.0), atol=1e-5)


def test_silence_equals_log_offset() -> None:
    """無音のメルは0なので log(0 + 1e-6) になる（対数の下駄が 1e-6）。"""
    feature = log_mel_spectrogram(np.zeros(1600, dtype=np.float32), SAMPLE_RATE)
    assert np.allclose(feature, np.log(1e-6), atol=1e-5)


def test_peak_mel_bin_tracks_tone_frequency() -> None:
    """純音の峰の位置が周波数とともに上がる（f_min=0, f_max=8000 のメル配置）。"""
    peaks = []
    for frequency in (250.0, 1000.0, 4000.0):
        feature = log_mel_spectrogram(sine(frequency, 16000), SAMPLE_RATE)
        peaks.append(int(np.argmax(feature.mean(axis=0))))
    assert peaks[0] < peaks[1] < peaks[2]
    # 4000Hz は f_max=8000 のメル尺度上で80次元中のおおむね60番目に来る。
    assert 50 <= peaks[2] <= 70


def test_hop_length_is_160_samples() -> None:
    """160標本増えるごとにフレームが1つ増える（hop_length=160）。"""
    base = sine(440.0, 16000)
    short = log_mel_spectrogram(base, SAMPLE_RATE)
    longer = log_mel_spectrogram(sine(440.0, 16000 + 160), SAMPLE_RATE)
    assert longer.shape[0] == short.shape[0] + 1


# --------------------------------------------------------------------------------------
# 入力の検査


def test_rejects_wrong_sample_rate() -> None:
    with pytest.raises(ValueError):
        log_mel_spectrogram(sine(440.0, 16000), 44100)


def test_rejects_multichannel_input() -> None:
    with pytest.raises(ValueError):
        log_mel_spectrogram(np.zeros((16000, 2), dtype=np.float32), SAMPLE_RATE)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        log_mel_spectrogram(np.zeros(0, dtype=np.float32), SAMPLE_RATE)


def test_custom_config_changes_shape() -> None:
    """設定を差し替えれば実際に反映される（既定値が固定で埋め込まれていない）。"""
    config = MelSpecConfig(n_mels=40, hop_length=320)
    feature = log_mel_spectrogram(sine(440.0, 16000), SAMPLE_RATE, config)
    assert feature.shape == (51, 40)
