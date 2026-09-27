"""ブラウザの確認用ページ（web/）の照合テスト。

- Node のテスト（``web/tests/melspec.test.mjs``。JS の対数メルと tests/fixtures/melspec/ の参照値の照合、
  窓の切り出し、再標本化、出力のパース）を呼ぶ。node が無ければ skip
- ONNX の参照値（``web/reference/onnx_exp005_fp32.json``）: 波形が式から作り直せること、Python の経路の
  出力と一致すること、JS で作った入力（``web/tests/dump_reference_logmel.mjs``）を ONNX Runtime
  （Python、CPU）に通した出力が参照値と一致すること。モデル（``runs/onnx_exp005/model_fp32.onnx``）
  が無い・別物なら skip

## 許容誤差

- 対数メル（JS と Python）: tests/test_melspec_fixtures.py と同じ（全要素 2e-3、値が −5 より大きい要素 1e-4）
- ONNX の出力 mora（JS の入力 → ONNX Runtime と、参照値）: 1e-3 モーラ。作成時の実測は最大 1.9e-6
  モーラ（対数メルの差 最大 1.7e-4 が出力に伝わった分）
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
REFERENCE = WEB / "reference" / "onnx_exp005_fp32.json"
MODEL = ROOT / "runs" / "onnx_exp005" / "model_fp32.onnx"
sys.path.insert(0, str(ROOT / "scripts"))

from make_web_reference import int16_to_float32, python_inputs, reference_waveform_int16, run_model, window_starts  # noqa: E402

ATOL_ALL = 2e-3
ATOL_ENERGY = 1e-4
ENERGY_THRESHOLD = -5.0
MORA_ATOL = 1e-3
NODE = shutil.which("node")


def _reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _model_or_skip(reference: dict) -> Path:
    if not MODEL.is_file():
        pytest.skip(f"{MODEL.relative_to(ROOT)} が無い")
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != reference["model_sha256"]:
        pytest.skip("モデルの SHA-256 が参照値の作成時と異なる")
    return MODEL


@pytest.mark.skipif(NODE is None, reason="node が無い")
def test_node_unit_tests() -> None:
    """web/tests/melspec.test.mjs（node:test）がすべて通る。"""
    result = subprocess.run([NODE, "--test", str(WEB / "tests" / "melspec.test.mjs")],
                            capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


def test_reference_waveform_is_reproducible() -> None:
    reference = _reference()
    samples = reference_waveform_int16()
    assert np.array_equal(samples, np.asarray(reference["waveform_int16"], dtype=np.int16))
    assert reference["window_starts"] == window_starts(samples.size) == [0, 4000, 8000, 12000, 16000]
    assert reference["window_input_shape"] == [5, 201, 80]
    assert reference["whole_input_shape"] == [1, 301, 80]


def test_reference_outputs_match_python_pipeline() -> None:
    reference = _reference()
    model = _model_or_skip(reference)
    windows, whole = python_inputs(int16_to_float32(reference["waveform_int16"]))
    np.testing.assert_allclose(run_model(model, windows), reference["window_mora"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(run_model(model, whole), reference["whole_mora"], rtol=0, atol=1e-5)


@pytest.fixture(scope="module")
def js_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, np.ndarray, dict]:
    if NODE is None:
        pytest.skip("node が無い")
    prefix = tmp_path_factory.mktemp("web_ref") / "ref"
    subprocess.run([NODE, str(WEB / "tests" / "dump_reference_logmel.mjs"), str(REFERENCE), str(prefix)],
                   check=True, cwd=ROOT)
    meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    windows = np.fromfile(f"{prefix}.window.f32", dtype="<f4").reshape(meta["window_dims"])
    whole = np.fromfile(f"{prefix}.whole.f32", dtype="<f4").reshape(meta["whole_dims"])
    return windows, whole, meta


def test_js_inputs_match_python_log_mel(js_inputs) -> None:  # noqa: ANN001
    """ページが作る ONNX の入力（窓ごと・全体）が Python の対数メルと許容誤差内で一致する。"""
    windows, whole, meta = js_inputs
    reference = _reference()
    assert meta["window_starts"] == reference["window_starts"]
    expected_windows, expected_whole = python_inputs(int16_to_float32(reference["waveform_int16"]))
    for got, expected in ((windows, expected_windows), (whole, expected_whole)):
        assert got.shape == expected.shape
        diff = np.abs(got - expected)
        assert float(diff.max()) <= ATOL_ALL
        assert float(diff[expected > ENERGY_THRESHOLD].max()) <= ATOL_ENERGY


def test_js_inputs_through_onnx_match_reference(js_inputs) -> None:  # noqa: ANN001
    """JS の入力を ONNX Runtime（Python）に通した mora が参照値と一致する。"""
    windows, whole, _ = js_inputs
    reference = _reference()
    model = _model_or_skip(reference)
    np.testing.assert_allclose(run_model(model, windows), reference["window_mora"], rtol=0, atol=MORA_ATOL)
    np.testing.assert_allclose(run_model(model, whole), reference["whole_mora"], rtol=0, atol=MORA_ATOL)
