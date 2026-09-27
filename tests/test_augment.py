"""データ拡張の単体テスト（docs/PLAN.md 第4段階 4-2）。

実データ（MUSAN本体）には依存しない。雑音は合成信号で作る。
MUSAN 本体を読むテストは、ファイルが無い環境では skip する。

docs/PLAN.md の完了条件は「各拡張について、適用前後でモーラ数のラベルが変わらないことを
確認する単体テスト」である。モーラ数は聴かないと数えられないので、ここでは次の2つを
機械的に確かめる。

1. ラベルの読み替え関数がモーラ数を変えないこと（``AugmentResult.mora_count`` が恒等）
2. **モーラの数え上げに相当する量が保たれること**。音節を模した ``N`` 個のパルス列
   （母音の塊を想定した正弦波の塊と無音の繰り返し）を作り、拡張の前後で包絡から
   数えた塊の個数が ``N`` のまま変わらないことを確かめる。これが変わる拡張は
   モーラ数のラベルと矛盾する

時間伸縮については、上に加えて「モーラ数が不変」かつ「毎秒モーラ数が伸縮率どおり
変わる（``mora_per_second / stretch``）」の両方を検証する。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from spkrate.data import augment as aug
from spkrate.data.augment import (
    AugmentConfig,
    AugmentResult,
    ArrayNoiseSource,
    MusanNoiseSource,
    add_noise,
    apply_reverb,
    augment_feature,
    augment_waveform,
    band_limit,
    change_volume,
    fit_noise,
    frequency_mask,
    generate_rir,
    mora_count_after_time_stretch,
    mora_per_second_after_time_stretch,
    time_stretch,
)

SAMPLE_RATE = 16000
MUSAN_ROOT = Path(__file__).resolve().parents[1] / "data" / "musan"


# --------------------------------------------------------------------------------------
# 音節を模した合成信号と、その塊を数える関数（テスト側の独立実装）


def syllable_train(
    num_syllables: int,
    *,
    syllable_sec: float = 0.12,
    gap_sec: float = 0.13,
    frequency: float = 220.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 0,
) -> np.ndarray:
    """``num_syllables`` 個の音の塊と無音を交互に並べた擬似発話を作る。

    1塊は基本周波数 ``frequency`` の倍音を持つ正弦波に、立ち上がり・立ち下がりの
    なだらかな窓をかけたもの。実音声ではないが「塊がいくつあるか」を数えるには足りる。
    """
    rng = np.random.default_rng(seed)
    syllable_len = int(round(syllable_sec * sample_rate))
    gap_len = int(round(gap_sec * sample_rate))
    envelope = np.hanning(syllable_len).astype(np.float32)
    time = np.arange(syllable_len, dtype=np.float32) / sample_rate
    pieces: list[np.ndarray] = [np.zeros(gap_len, dtype=np.float32)]
    for index in range(num_syllables):
        tone = np.zeros(syllable_len, dtype=np.float32)
        base = frequency * float(rng.uniform(0.9, 1.1))
        for harmonic in (1, 2, 3, 4):
            tone += (0.5 / harmonic) * np.sin(
                2.0 * np.pi * base * harmonic * time + index
            ).astype(np.float32)
        pieces.append((tone * envelope).astype(np.float32))
        pieces.append(np.zeros(gap_len, dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def count_bursts(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    smooth_sec: float = 0.03,
    high_ratio: float = 0.30,
    low_ratio: float = 0.15,
) -> int:
    """短時間実効値の包絡から音の塊の個数を数える（モーラ数の代理）。

    立ち上がりと立ち下がりで別の閾値を使う（履歴を持たせる）。1つの塊の中で包絡が
    わずかに凹んでも2つに割れないようにするため。
    """
    power = np.asarray(samples, dtype=np.float64) ** 2
    width = max(1, int(round(smooth_sec * sample_rate)))
    kernel = np.ones(width, dtype=np.float64) / width
    envelope = np.sqrt(np.convolve(power, kernel, mode="same"))
    peak = float(envelope.max())
    if peak <= 0.0:
        return 0
    normalized = envelope / peak
    count = 0
    inside = False
    for value in normalized:
        if not inside and value > high_ratio:
            inside = True
            count += 1
        elif inside and value < low_ratio:
            inside = False
    return count


def dominant_frequency(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """振幅スペクトルが最大になる周波数（音程保持の確認用）。"""
    array = np.asarray(samples, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(array * np.hanning(array.size)))
    freqs = np.fft.rfftfreq(array.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def test_syllable_train_is_countable() -> None:
    """テスト用の合成信号そのものが、狙った個数の塊として数えられること。"""
    for num in (5, 8, 12):
        assert count_bursts(syllable_train(num)) == num


# --------------------------------------------------------------------------------------
# 時間伸縮


@pytest.mark.parametrize("stretch", [0.7, 0.85, 1.0, 1.2, 1.5])
def test_time_stretch_changes_duration_by_the_ratio(stretch: float) -> None:
    samples = syllable_train(8)
    stretched = time_stretch(samples, stretch)
    assert stretched.dtype == np.float32
    # 位相ボコーダはフレーム単位なので、長さは厳密には一致しない。2%の幅を許す。
    assert stretched.size == pytest.approx(samples.size * stretch, rel=0.02)


@pytest.mark.parametrize("stretch", [0.7, 1.5])
def test_time_stretch_preserves_pitch(stretch: float) -> None:
    """音程が保たれること。再標本化で長さを変えた場合と対比する。"""
    samples = (
        0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)
    ).astype(np.float32)
    stretched = time_stretch(samples, stretch)
    assert dominant_frequency(stretched) == pytest.approx(440.0, abs=5.0)

    # 素朴な再標本化（音程が 1/stretch 倍にずれる方式）は採らない、ということの確認。
    import librosa

    resampled = librosa.resample(
        samples, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE / stretch)
    )
    shifted = dominant_frequency(np.asarray(resampled, dtype=np.float32))
    assert abs(shifted - 440.0) > 20.0


@pytest.mark.parametrize("stretch", [0.7, 0.9, 1.0, 1.3, 1.5])
def test_time_stretch_keeps_mora_count_and_scales_rate(stretch: float) -> None:
    """モーラ数は不変、毎秒モーラ数は伸縮率で割った値になる（docs/spec.md）。"""
    mora_count = 24.0
    duration = 4.0
    mora_per_second = mora_count / duration

    assert mora_count_after_time_stretch(mora_count, stretch) == mora_count
    assert mora_per_second_after_time_stretch(mora_per_second, stretch) == pytest.approx(
        mora_per_second / stretch
    )
    # 定義どおり「不変のモーラ数 ÷ 伸縮後の音声長」と一致すること。
    assert mora_per_second_after_time_stretch(mora_per_second, stretch) == pytest.approx(
        mora_count / (duration * stretch)
    )
    # 縮める（速口にする）と毎秒モーラ数は増える。
    if stretch < 1.0:
        assert mora_per_second_after_time_stretch(mora_per_second, stretch) > mora_per_second
    if stretch > 1.0:
        assert mora_per_second_after_time_stretch(mora_per_second, stretch) < mora_per_second


@pytest.mark.parametrize("stretch", [0.7, 1.0, 1.5])
def test_time_stretch_preserves_number_of_syllables(stretch: float) -> None:
    """伸縮しても音の塊の個数（＝モーラ数の代理）は変わらない。"""
    num_syllables = 8
    samples = syllable_train(num_syllables)
    stretched = time_stretch(samples, stretch)
    assert count_bursts(stretched) == num_syllables

    # 実測した音声長から計算した毎秒モーラ数が、読み替え関数の値と一致すること。
    before = num_syllables / (samples.size / SAMPLE_RATE)
    after = num_syllables / (stretched.size / SAMPLE_RATE)
    assert after == pytest.approx(
        mora_per_second_after_time_stretch(before, stretch), rel=0.02
    )


def test_time_stretch_rejects_invalid_ratio() -> None:
    samples = syllable_train(3)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            time_stretch(samples, bad)


# --------------------------------------------------------------------------------------
# 雑音重畳


def pink_like_noise(num_samples: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(num_samples).astype(np.float32)
    kernel = np.ones(8, dtype=np.float32) / 8.0
    return np.convolve(white, kernel, mode="same").astype(np.float32) * 0.3


@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 20.0])
def test_add_noise_achieves_requested_snr(snr_db: float) -> None:
    samples = syllable_train(8)
    noise = pink_like_noise(samples.size)
    mixed = add_noise(samples, noise, snr_db)
    residual = mixed - samples
    signal_power = float(np.mean(samples.astype(np.float64) ** 2))
    noise_power = float(np.mean(residual.astype(np.float64) ** 2))
    measured = 10.0 * math.log10(signal_power / noise_power)
    assert measured == pytest.approx(snr_db, abs=0.1)


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_add_noise_keeps_length_and_labels(snr_db: float) -> None:
    """雑音は重なるだけで音声長を変えないので、モーラ数も毎秒モーラ数も不変。"""
    samples = syllable_train(8)
    mixed = add_noise(samples, pink_like_noise(samples.size), snr_db)
    assert mixed.size == samples.size
    assert mixed.dtype == np.float32
    result = AugmentResult(samples=mixed, stretch=1.0, applied=("noise",))
    assert result.mora_count(8.0) == 8.0
    assert result.mora_per_second(4.5) == pytest.approx(4.5)
    assert result.duration_sec() == pytest.approx(samples.size / SAMPLE_RATE)


def test_add_noise_preserves_number_of_syllables() -> None:
    """SN比が高い側では、塊の個数が保たれることも確かめる。

    SN比0dB では雑音が無音区間を埋めるため包絡からは数えられなくなるが、
    **実際に発話されたモーラの数は変わらない**ので正解ラベルは不変のままでよい。
    低SN比で数えられなくなるのは拡張の狙い（難しい入力を作ること）そのものである。
    """
    num_syllables = 8
    samples = syllable_train(num_syllables)
    noise = pink_like_noise(samples.size)
    for snr_db in (15.0, 20.0):
        assert count_bursts(add_noise(samples, noise, snr_db)) == num_syllables


def test_add_noise_on_silence_is_safe() -> None:
    silence = np.zeros(1600, dtype=np.float32)
    mixed = add_noise(silence, pink_like_noise(1600), 10.0)
    assert np.array_equal(mixed, silence)


def test_fit_noise_repeats_and_crops() -> None:
    short = np.arange(10, dtype=np.float32)
    assert fit_noise(short, 25).size == 25
    assert np.array_equal(fit_noise(short, 25)[:10], short)
    long = np.arange(100, dtype=np.float32)
    rng = np.random.default_rng(0)
    cropped = fit_noise(long, 30, rng)
    assert cropped.size == 30
    assert np.all(np.diff(cropped) == 1.0)  # 連続した区間を切り出している


def test_array_noise_source() -> None:
    source = ArrayNoiseSource([pink_like_noise(4000, seed=s) for s in range(3)])
    rng = np.random.default_rng(0)
    assert len(source) == 3
    assert source.sample(1234, rng).shape == (1234,)


def test_musan_source_rejects_speech_subset() -> None:
    """speech を混ぜると別話者の声が入りモーラ数のラベルと矛盾するため拒否する。"""
    with pytest.raises(ValueError, match="speech"):
        MusanNoiseSource(MUSAN_ROOT, subsets=("noise", "speech"))
    with pytest.raises(ValueError):
        MusanNoiseSource(MUSAN_ROOT, subsets=("unknown",))


@pytest.mark.skipif(
    not (MUSAN_ROOT / "noise").is_dir(), reason="MUSAN が配置されていない"
)
def test_musan_noise_source_reads_real_files() -> None:
    source = MusanNoiseSource(MUSAN_ROOT)
    assert len(source.paths) > 0
    rng = np.random.default_rng(0)
    noise = source.sample(SAMPLE_RATE, rng)
    assert noise.shape == (SAMPLE_RATE,)
    assert noise.dtype == np.float32
    samples = syllable_train(8)
    mixed = add_noise(samples, source.sample(samples.size, rng), 10.0)
    assert mixed.size == samples.size


# --------------------------------------------------------------------------------------
# 残響


def test_generate_rir_shape_and_normalization() -> None:
    rng = np.random.default_rng(0)
    rir = generate_rir(rng)
    assert rir.ndim == 1
    assert rir.dtype == np.float32
    assert rir.size > 1
    assert float(np.max(np.abs(rir))) == pytest.approx(1.0)


def test_apply_reverb_keeps_length_and_alignment() -> None:
    """長さが変わらず、直接音の時刻もずれないこと（＝毎秒モーラ数が変わらない）。"""
    samples = syllable_train(8)
    rng = np.random.default_rng(3)
    reverbed = apply_reverb(samples, generate_rir(rng))
    assert reverbed.size == samples.size
    assert reverbed.dtype == np.float32
    # 直接音の位置合わせにより、先頭の立ち上がりの時刻が保たれる。
    def onset(signal: np.ndarray) -> int:
        envelope = np.abs(signal)
        return int(np.argmax(envelope > 0.1 * float(envelope.max())))

    assert abs(onset(reverbed) - onset(samples)) < int(0.01 * SAMPLE_RATE)


def test_apply_reverb_preserves_number_of_syllables_and_labels() -> None:
    num_syllables = 8
    samples = syllable_train(num_syllables)
    rng = np.random.default_rng(7)
    for _ in range(5):
        rir = generate_rir(rng, rt60_range=(0.1, 0.3))
        reverbed = apply_reverb(samples, rir)
        assert count_bursts(reverbed) == num_syllables
        result = AugmentResult(samples=reverbed, stretch=1.0, applied=("reverb",))
        assert result.mora_count(num_syllables) == num_syllables
        assert result.mora_per_second(4.0) == pytest.approx(4.0)


def test_apply_reverb_with_unit_impulse_is_identity() -> None:
    samples = syllable_train(4)
    identity = np.array([1.0], dtype=np.float32)
    assert np.allclose(apply_reverb(samples, identity), samples, atol=1e-5)


# --------------------------------------------------------------------------------------
# 音量変化


@pytest.mark.parametrize("gain_db", [-12.0, -6.0, 0.0, 6.0])
def test_change_volume_applies_gain(gain_db: float) -> None:
    samples = (syllable_train(8) * 0.2).astype(np.float32)
    louder = change_volume(samples, gain_db)
    expected = 10.0 ** (gain_db / 20.0)
    ratio = float(np.max(np.abs(louder))) / float(np.max(np.abs(samples)))
    assert ratio == pytest.approx(expected, rel=1e-4)
    assert louder.size == samples.size


def test_change_volume_prevents_clipping() -> None:
    samples = (syllable_train(4) / np.max(np.abs(syllable_train(4)))).astype(np.float32)
    louder = change_volume(samples, 20.0)
    assert float(np.max(np.abs(louder))) <= 1.0


def test_change_volume_keeps_labels() -> None:
    num_syllables = 8
    samples = syllable_train(num_syllables)
    for gain_db in (-12.0, 0.0, 6.0):
        changed = change_volume(samples, gain_db)
        assert changed.size == samples.size
        assert count_bursts(changed) == num_syllables  # 相対的な包絡の形は不変
        result = AugmentResult(samples=changed, stretch=1.0, applied=("volume",))
        assert result.mora_count(num_syllables) == num_syllables
        assert result.mora_per_second(5.5) == pytest.approx(5.5)


# --------------------------------------------------------------------------------------
# 帯域制限


def test_band_limit_removes_out_of_band_energy() -> None:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    low_tone = (0.5 * np.sin(2.0 * np.pi * 1000.0 * time)).astype(np.float32)
    high_tone = (0.5 * np.sin(2.0 * np.pi * 7000.0 * time)).astype(np.float32)
    mixed = (low_tone + high_tone).astype(np.float32)
    filtered = band_limit(mixed, low_hz=300.0, high_hz=3400.0)

    def band_energy(signal: np.ndarray, low: float, high: float) -> float:
        spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / SAMPLE_RATE)
        selected = (freqs >= low) & (freqs <= high)
        return float(np.sum(spectrum[selected] ** 2))

    assert band_energy(filtered, 6500.0, 7500.0) < 1e-3 * band_energy(mixed, 6500.0, 7500.0)
    assert band_energy(filtered, 900.0, 1100.0) == pytest.approx(
        band_energy(mixed, 900.0, 1100.0), rel=0.1
    )


def test_band_limit_keeps_length_and_labels() -> None:
    num_syllables = 8
    samples = syllable_train(num_syllables)
    for low_hz, high_hz in ((0.0, 3400.0), (300.0, 3400.0), (100.0, 7800.0)):
        filtered = band_limit(samples, low_hz=low_hz, high_hz=high_hz)
        assert filtered.size == samples.size
        assert filtered.dtype == np.float32
        assert count_bursts(filtered) == num_syllables
        result = AugmentResult(samples=filtered, stretch=1.0, applied=("band_limit",))
        assert result.mora_count(num_syllables) == num_syllables
        assert result.mora_per_second(4.0) == pytest.approx(4.0)


def test_band_limit_without_band_is_identity() -> None:
    samples = syllable_train(4)
    assert np.array_equal(band_limit(samples, low_hz=0.0, high_hz=None), samples)


def test_band_limit_rejects_empty_band() -> None:
    with pytest.raises(ValueError):
        band_limit(syllable_train(4), low_hz=4000.0, high_hz=1000.0)


# --------------------------------------------------------------------------------------
# 周波数方向のマスク（時間方向のマスクは実装しない）


def test_frequency_mask_masks_only_frequency_axis() -> None:
    feature = np.ones((201, 80), dtype=np.float32)
    rng = np.random.default_rng(0)
    masked = frequency_mask(feature, rng, num_masks=2, max_width=12)
    assert masked.shape == feature.shape
    # どのフレームも丸ごとは消えない（時間方向の情報が残る）。
    assert np.all(masked.max(axis=1) == 1.0)
    # マスクされた列は全フレームで同じ扱いになる（列単位のマスク）。
    column_is_masked = np.all(masked == 0.0, axis=0)
    column_is_kept = np.all(masked == 1.0, axis=0)
    assert np.all(column_is_masked | column_is_kept)
    assert column_is_masked.sum() > 0


def test_frequency_mask_does_not_modify_input() -> None:
    feature = np.ones((50, 80), dtype=np.float32)
    frequency_mask(feature, np.random.default_rng(0), num_masks=2, max_width=20)
    assert np.all(feature == 1.0)


def test_frequency_mask_keeps_labels() -> None:
    """フレーム数が変わらないのでモーラ数も毎秒モーラ数も不変。"""
    feature = np.random.default_rng(0).standard_normal((201, 80)).astype(np.float32)
    masked = frequency_mask(feature, np.random.default_rng(1))
    assert masked.shape[0] == feature.shape[0]
    result = AugmentResult(samples=np.zeros(32000, dtype=np.float32), stretch=1.0)
    assert result.mora_count(10.0) == 10.0
    assert result.mora_per_second(5.0) == pytest.approx(5.0)


def test_no_time_direction_mask_is_provided() -> None:
    """docs/spec.md「時間方向のマスクは正解と矛盾するため禁止」。API を持たないこと。"""
    names = set(dir(aug))
    for forbidden in ("time_mask", "time_masking", "spec_augment", "TimeMask"):
        assert forbidden not in names
    assert "time_mask" not in aug.__all__


# --------------------------------------------------------------------------------------
# 設定とパイプライン


def test_augment_config_from_mapping() -> None:
    config = AugmentConfig.from_mapping({"noise_prob": 1.0, "snr_db_range": [5.0, 15.0]})
    assert config.noise_prob == 1.0
    assert config.snr_db_range == (5.0, 15.0)
    assert AugmentConfig.from_mapping(None) == AugmentConfig()
    with pytest.raises(ValueError):
        AugmentConfig.from_mapping({"time_mask_prob": 1.0})


def test_augment_config_defaults_match_spec() -> None:
    """docs/spec.md が数値で定める範囲。"""
    config = AugmentConfig()
    assert config.time_stretch_range == (0.7, 1.5)
    assert config.snr_db_range == (0.0, 20.0)
    assert config.sample_rate == 16000


def test_disabled_config_is_identity() -> None:
    samples = syllable_train(8)
    config = AugmentConfig().disabled()
    result = augment_waveform(samples, np.random.default_rng(0), config=config)
    assert np.array_equal(result.samples, samples)
    assert result.applied == ()
    assert result.stretch == 1.0
    assert result.mora_count(8.0) == 8.0


def test_augment_waveform_is_reproducible_and_labels_follow_stretch() -> None:
    samples = syllable_train(10)
    noise_source = ArrayNoiseSource([pink_like_noise(SAMPLE_RATE * 2, seed=s) for s in range(3)])
    config = AugmentConfig()
    outputs = []
    for _ in range(2):
        result = augment_waveform(
            samples, np.random.default_rng(12345), config=config, noise_source=noise_source
        )
        outputs.append(result)
    assert np.array_equal(outputs[0].samples, outputs[1].samples)
    assert outputs[0].applied == outputs[1].applied
    assert outputs[0].stretch == outputs[1].stretch

    result = outputs[0]
    assert result.samples.dtype == np.float32
    # モーラ数は常に不変。毎秒モーラ数は伸縮率ぶんだけ変わる。
    assert result.mora_count(30.0) == 30.0
    assert result.mora_per_second(6.0) == pytest.approx(6.0 / result.stretch)
    # 音声長は伸縮率ぶんだけ変わり、それ以外の拡張では変わらない。
    assert result.samples.size == pytest.approx(samples.size * result.stretch, rel=0.02)


def test_augment_waveform_covers_all_augmentations() -> None:
    """既定の確率でひととおりの拡張が現れ、いずれの場合もモーラ数が不変であること。"""
    samples = syllable_train(10)
    noise_source = ArrayNoiseSource([pink_like_noise(SAMPLE_RATE * 2, seed=0)])
    rng = np.random.default_rng(0)
    seen: set[str] = set()
    for _ in range(40):
        result = augment_waveform(samples, rng, noise_source=noise_source)
        seen.update(result.applied)
        assert result.mora_count(30.0) == 30.0
        assert np.all(np.isfinite(result.samples))
        assert result.samples.size > 0
        expected_rate = 30.0 / (samples.size / SAMPLE_RATE) / result.stretch
        assert result.mora_per_second(30.0 / (samples.size / SAMPLE_RATE)) == pytest.approx(
            expected_rate
        )
    assert seen == {"time_stretch", "reverb", "noise", "band_limit", "volume"}


def test_augment_waveform_without_noise_source_skips_noise() -> None:
    samples = syllable_train(6)
    rng = np.random.default_rng(1)
    for _ in range(10):
        result = augment_waveform(samples, rng)
        assert "noise" not in result.applied


def test_augment_feature_applies_only_frequency_mask() -> None:
    feature = np.ones((201, 80), dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(10):
        masked = augment_feature(feature, rng)
        assert masked.shape == feature.shape
        # 時間方向は1フレームも消えない。
        assert np.all(masked.max(axis=1) == 1.0)


def test_augmented_waveform_feeds_the_mel_front_end() -> None:
    """拡張した波形がそのまま docs/spec.md の特徴量計算に渡せること。"""
    from spkrate.features.melspec import log_mel_spectrogram, num_frames

    samples = syllable_train(8)
    noise_source = ArrayNoiseSource([pink_like_noise(SAMPLE_RATE * 2, seed=0)])
    result = augment_waveform(samples, np.random.default_rng(4), noise_source=noise_source)
    feature = log_mel_spectrogram(result.samples, SAMPLE_RATE)
    assert feature.shape == (num_frames(result.samples.size), 80)
    assert feature.dtype == np.float32
    assert np.all(np.isfinite(feature))


# --------------------------------------------------------------------------------------
# 拡張ごとの有効フラグ（docs/questions.md 2026-09-24 回答2）


def test_individually_disabled_augmentations_are_not_applied() -> None:
    from spkrate.data.augment import AUGMENTATIONS

    rng_signal = np.random.default_rng(0)
    samples = (0.1 * rng_signal.standard_normal(16000)).astype(np.float32)
    noise = ArrayNoiseSource([(0.1 * rng_signal.standard_normal(16000)).astype(np.float32)])
    waveform_names = [name for name, _ in AUGMENTATIONS if name != "freq_mask"]
    for disabled in waveform_names:
        config = AugmentConfig.from_mapping(
            {
                f"{name}_prob": 1.0 for name, _ in AUGMENTATIONS
            }
            | {f"{disabled}_enabled": False}
        )
        result = augment_waveform(samples, np.random.default_rng(1), config=config, noise_source=noise)
        assert disabled not in result.applied
        assert set(result.applied) == set(waveform_names) - {disabled}


def test_freq_mask_disabled_returns_feature_unchanged() -> None:
    feature = np.random.default_rng(0).standard_normal((50, 80)).astype(np.float32)
    config = AugmentConfig.from_mapping({"freq_mask_prob": 1.0, "freq_mask_enabled": False})
    out = augment_feature(feature, np.random.default_rng(0), config=config)
    np.testing.assert_array_equal(out, feature)


# ======================================================================================
# 伸縮率の分布（docs/experiments/011-search-plan.md 4.2節、exp010）


def test_time_stretch_distribution_default_is_uniform_and_unknown_is_rejected() -> None:
    assert aug.AugmentConfig().time_stretch_distribution == "uniform"
    assert aug.AugmentConfig.from_mapping({}).time_stretch_distribution == "uniform"
    config = aug.AugmentConfig.from_mapping({"time_stretch_distribution": "log_uniform"})
    assert config.time_stretch_distribution == "log_uniform"
    with pytest.raises(ValueError):
        aug.AugmentConfig.from_mapping({"time_stretch_distribution": "normal"})
    with pytest.raises(ValueError):
        aug.draw_time_stretch(np.random.default_rng(0), 0.7, 1.5, "normal")


def test_uniform_stretch_draws_are_unchanged_from_before() -> None:
    """既定（uniform）の方式Aの抽選の列が変更前と同じ（変更前のコードで記録した値）。"""
    config = aug.AugmentConfig()
    rng = np.random.default_rng(7)
    values = []
    for _ in range(8):
        if rng.random() < config.time_stretch_prob:
            values.append(
                aug.draw_time_stretch(rng, *config.time_stretch_range, config.time_stretch_distribution)
            )
    assert values == [0.9401330279289803, 1.3569827347062131, 0.9424259414554508]


def test_log_uniform_consumes_same_random_numbers_as_uniform() -> None:
    """どちらの分布も乱数の消費は1回で、後に続く乱数の系列が同じになる。"""
    for seed in range(20):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        aug.draw_time_stretch(a, 0.7, 1.5, "uniform")
        aug.draw_time_stretch(b, 0.7, 1.5, "log_uniform")
        assert a.random() == b.random()


def test_log_uniform_stretch_is_log_uniform_on_range() -> None:
    """対数一様: [0.7, 1.5] に収まり、log s が一様（中央値 √(0.7·1.5)、s<1 の割合 log(1/0.7)/log(1.5/0.7)）。"""
    low, high = 0.7, 1.5
    rng = np.random.default_rng(20260928)
    values = np.array([aug.draw_time_stretch(rng, low, high, "log_uniform") for _ in range(40000)])
    assert values.min() >= low and values.max() <= high
    assert np.median(values) == pytest.approx(math.sqrt(low * high), abs=0.01)
    assert np.mean(values < 1.0) == pytest.approx(math.log(1 / low) / math.log(high / low), abs=0.01)
    # log s の十分位が等間隔（一様）
    logs = np.log(values)
    edges = np.linspace(math.log(low), math.log(high), 11)
    counts, _ = np.histogram(logs, bins=edges)
    assert np.all(np.abs(counts / values.size - 0.1) < 0.01)
    # 一様とは違う（中央値が (low + high) / 2 = 1.1 より小さい）
    assert np.median(values) < 1.05


def test_log_uniform_is_used_by_augment_waveform_and_rate_estimate() -> None:
    """方式Aの経路（augment_waveform）でも分布の指定が効き、他の拡張の乱数はずれない。"""
    samples = (np.random.default_rng(1).standard_normal(8000) * 0.1).astype(np.float32)
    base = aug.AugmentConfig(
        time_stretch_prob=1.0,
        reverb_enabled=False,
        noise_enabled=False,
        band_limit_enabled=False,
        volume_prob=1.0,
    )
    log_cfg = aug.AugmentConfig(**{**base.__dict__, "time_stretch_distribution": "log_uniform"})
    ru = aug.augment_waveform(samples, np.random.default_rng(3), config=base)
    rl = aug.augment_waveform(samples, np.random.default_rng(3), config=log_cfg)
    u = float(np.random.default_rng(3).uniform(0.7, 1.5, size=2)[1])  # random() の次の1回
    assert ru.stretch == pytest.approx(u)
    assert rl.stretch == pytest.approx(math.exp(math.log(0.7) + (u - 0.7) / 0.8 * math.log(1.5 / 0.7)))
    assert ru.params["gain_db"] == rl.params["gain_db"]
    aug.validate_augment_config(log_cfg)
