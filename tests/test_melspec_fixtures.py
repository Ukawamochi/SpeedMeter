"""対数メルスペクトログラムの参照値（tests/fixtures/melspec/）との照合（docs/plan.md 第8段階 8-2）。

参照値は ``scripts/make_melspec_fixtures.py`` が書いた合成波形（サイン波）の値である。
ブラウザ側の実装も同じ参照値・同じ許容誤差で照合する想定である。

## 許容誤差（対数メルの値、自然対数の単位、絶対誤差）

- 全要素: ``ATOL_ALL = 2e-3``
- 値が ``-5`` より大きい要素（メルのパワーが1e-6のオフセットより十分大きい帯）: ``ATOL_ENERGY = 1e-4``

根拠: 同じ参照値を numpy（float64）で独立に書いた実装（tests/test_melspec.py の
``reference_log_mel``）と比べると、差の最大は、エネルギーのほとんど無い帯（値が −13 前後、
log(mel + 1e-6) のオフセットの近く）で約1.0e-3、値が −5 より大きい帯で約1.8e-5 だった
（results/onnx_export.md）。オフセットの近くでは FFT の丸め誤差が相対的に大きく効くため、
計算の精度（float32 と float64）や FFT の実装の違いで 1e-3 程度ずれうる。値の大きい帯は
話速の推定に効く帯であり、そこはより厳しく照合する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram, log_mel_spectrogram

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "melspec"
sys.path.insert(0, str(ROOT / "scripts"))

from make_melspec_fixtures import SIGNALS, int16_to_float32  # noqa: E402

ATOL_ALL = 2e-3
ATOL_ENERGY = 1e-4
ENERGY_THRESHOLD = -5.0
SIGNAL_NAMES = sorted(SIGNALS)


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fixture_matches_implementation(name: str) -> None:
    """実装の出力が保存した参照値と許容誤差内で一致する。"""
    payload = _load(name)
    reference = np.asarray(payload["log_mel"], dtype=np.float32)
    feature = log_mel_spectrogram(int16_to_float32(payload["waveform_int16"]), 16000)
    assert feature.shape == reference.shape == (payload["num_frames"], 80)
    diff = np.abs(feature - reference)
    assert float(diff.max()) <= ATOL_ALL
    energy = reference > ENERGY_THRESHOLD
    assert energy.any()
    assert float(diff[energy].max()) <= ATOL_ENERGY


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fixture_waveform_is_reproducible(name: str) -> None:
    """保存した波形が式（scripts/make_melspec_fixtures.py）から同じ int16 で作り直せる。"""
    payload = _load(name)
    samples, _ = SIGNALS[name]()
    assert np.array_equal(samples, np.asarray(payload["waveform_int16"], dtype=np.int16))
    assert payload["num_samples"] == samples.size
    assert payload["num_frames"] == 1 + samples.size // 160


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fixture_records_current_config(name: str) -> None:
    """参照値を作った特徴量の設定が現在の既定（docs/spec.md「特徴量」）と同じ。"""
    payload = _load(name)
    for key, value in payload["config"].items():
        assert getattr(MEL_DEFAULTS, key) == value, key


def test_silent_frames_equal_log_offset() -> None:
    """無音（厳密に0）だけのフレームは log(0 + 1e-6) になる（対数のオフセットの確認）。

    center=True なのでフレーム i は標本 [160 i - 200, 160 i + 200) を見る。先頭1600標本が
    0なので、i <= 8 のフレーム（反射の詰め物を含めて0だけを見る）が該当する。
    """
    payload = _load("multitone_with_silence")
    reference = np.asarray(payload["log_mel"], dtype=np.float32)
    assert np.all(reference[:9] == np.float32(np.log(np.float32(1e-6))))
    assert np.all(reference[10:].max(axis=1) > ENERGY_THRESHOLD)


def test_filterbank_fixture_matches_implementation() -> None:
    """メルフィルタバンクの参照値（0でない要素）が実装と一致する。"""
    payload = json.loads((FIXTURE_DIR / "mel_filterbank.json").read_text(encoding="utf-8"))
    fb = LogMelSpectrogram(MEL_DEFAULTS).mel_filterbank
    assert (payload["num_bins"], payload["n_mels"]) == fb.shape == (201, 80)
    rebuilt = np.zeros_like(fb)
    for mel, band in enumerate(payload["bands"]):
        weights = np.asarray(band["weights"], dtype=np.float32)
        rebuilt[band["start_bin"]:band["start_bin"] + weights.size, mel] = weights
    np.testing.assert_allclose(rebuilt, fb, rtol=0.0, atol=1e-6)
