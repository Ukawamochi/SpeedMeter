"""ONNX への書き出しと int8 の動的量子化の単体テスト（docs/plan.md 第8段階 8-1）。

同じ入力（正規化前の対数メル）に対する PyTorch（``NormalizedModel``、cpu）と ONNX Runtime
（CPU の実行プロバイダ）の出力（モーラ数）を比べる。

## 許容誤差（出力はモーラ数。2.0秒窓なら毎秒モーラ数の2倍の単位）

- 量子化前（fp32）: ``|onnx - torch| <= FP32_ATOL + FP32_RTOL * |torch|``、
  ``FP32_ATOL = 1e-4`` モーラ、``FP32_RTOL = 1e-5``。
  根拠: 同じ float32 の計算で、違いは演算の順序（畳み込みの実装）による丸め誤差だけである。
  exp005 で実測した差の最大は、合成の入力で約7.6e-6 モーラ、dev_window の clean 全842,900窓
  （PyTorch は mps）で約7.6e-6 モーラ、無作為の初期値のモデル（出力約139）で約3.1e-5 だった。
- 量子化後（int8、無作為の初期値のモデル）: ``INT8_RANDOM_RTOL = 1e-3``（出力に対する比）。
  無作為の初期値のモデルは出力がほぼ一定で、量子化の誤差が小さく出る（実測の比 約1.9e-5）。
  量子化の配線（``ConvInteger`` への置き換え）が壊れていないことだけを確かめる。
- 量子化後（int8、exp005）: 窓ごとの差 ``|int8 - torch|`` の平均 ``<= INT8_EXP005_MEAN_ATOL = 0.15``
  モーラ、最大 ``<= INT8_EXP005_MAX_ATOL = 1.5`` モーラ。量子化は近似なので要素ごとの厳密な一致は
  求めない。精度への影響は dev_window の MAE で確かめる（results/onnx_export.md）。この値は
  合成の入力（下の ``_speechlike_windows``）での実測（平均約0.07、最大約0.95 モーラ）と、
  dev_window の clean 全842,900窓での窓ごとの差（平均0.092、99.9%点0.319、最大0.528 モーラ）に
  余裕を持たせて決めた。量子化を誤ると（例: 重みの刻みの取り違え）差は数モーラ以上になる。
  ``runs/exp005/checkpoint_best.pt`` が無い環境では飛ばす。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from spkrate.export.to_onnx import (
    INPUT_NAME,
    OUTPUT_NAME,
    NormalizedModel,
    export_onnx,
    load_normalized_model,
    make_onnx_inference,
    make_onnx_predictor,
    make_session,
    quantize_int8,
)
from spkrate.features.melspec import log_mel_spectrogram
from spkrate.models.cnn import CnnConfig, SpeechRateCNN
from spkrate.train.data import load_normalization

ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION = ROOT / "configs" / "normalization.yaml"
EXP005 = ROOT / "runs" / "exp005" / "checkpoint_best.pt"

FP32_ATOL = 1e-4
FP32_RTOL = 1e-5
INT8_RANDOM_RTOL = 1e-3
INT8_EXP005_MEAN_ATOL = 0.15
INT8_EXP005_MAX_ATOL = 1.5


def _speechlike_windows(count: int, num_samples: int = 32000, seed: int = 0) -> list[np.ndarray]:
    """音量が断続的に変わる正弦波と雑音の和（音声の代わりの合成入力）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples, dtype=np.float32) / np.float32(16000)
    windows = []
    for _ in range(count):
        amplitude = rng.uniform(0.01, 0.5)
        frequency = rng.uniform(100.0, 3000.0)
        gate = (np.sin(2.0 * np.pi * rng.uniform(2.0, 8.0) * t) > 0).astype(np.float32)
        noise = rng.standard_normal(num_samples).astype(np.float32) * np.float32(0.01 * rng.uniform(0.0, 3.0))
        windows.append((amplitude * np.sin(2.0 * np.pi * frequency * t) * gate + noise).astype(np.float32))
    return windows


def _features(windows: list[np.ndarray]) -> np.ndarray:
    return np.stack([log_mel_spectrogram(w, 16000) for w in windows]).astype(np.float32)


def _torch_output(module: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return module(torch.from_numpy(features)).numpy()


def _run(session, features: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return session.run([OUTPUT_NAME], {INPUT_NAME: features})[0]


@pytest.fixture(scope="module")
def random_model_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[NormalizedModel, Path, Path]:
    torch.manual_seed(0)
    normalizer = load_normalization(NORMALIZATION)
    module = NormalizedModel(SpeechRateCNN(CnnConfig()).eval(), normalizer.mean, normalizer.std).eval()
    out = tmp_path_factory.mktemp("onnx")
    fp32 = export_onnx(module, out / "model_fp32.onnx")
    int8 = quantize_int8(fp32, out / "model_int8.onnx")
    return module, fp32, int8


def test_graph_inputs_and_outputs(random_model_files) -> None:  # noqa: ANN001
    """入力は log_mel (batch, frames, 80)、出力は mora (batch,)。バッチとフレームは可変。"""
    _, fp32, int8 = random_model_files
    for path in (fp32, int8):
        session = make_session(path)
        (inp,), (out,) = session.get_inputs(), session.get_outputs()
        assert inp.name == INPUT_NAME and out.name == OUTPUT_NAME
        assert inp.shape == ["batch", "frames", 80]
        assert out.shape == ["batch"]
        assert session.get_providers() == ["CPUExecutionProvider"]


def test_quantized_graph_uses_integer_convolution(random_model_files) -> None:  # noqa: ANN001
    import onnx

    _, fp32, int8 = random_model_files
    ops_fp32 = {node.op_type for node in onnx.load(str(fp32)).graph.node}
    ops_int8 = {node.op_type for node in onnx.load(str(int8)).graph.node}
    assert "Conv" in ops_fp32 and "ConvInteger" not in ops_fp32
    assert "ConvInteger" in ops_int8 and "Conv" not in ops_int8
    assert int8.stat().st_size < fp32.stat().st_size / 3


@pytest.mark.parametrize("num_frames", [201, 57])
def test_fp32_matches_pytorch(random_model_files, num_frames: int) -> None:  # noqa: ANN001
    """量子化前: PyTorch と ONNX Runtime の出力が FP32_ATOL・FP32_RTOL で一致する（可変長も）。"""
    module, fp32, _ = random_model_files
    features = _features(_speechlike_windows(3))[:, :num_frames]
    np.testing.assert_allclose(_run(make_session(fp32), features), _torch_output(module, features),
                               rtol=FP32_RTOL, atol=FP32_ATOL)


def test_int8_close_to_pytorch_random_model(random_model_files) -> None:  # noqa: ANN001
    """量子化後（無作為の初期値のモデル）: 出力に対する比で INT8_RANDOM_RTOL 以内。"""
    module, _, int8 = random_model_files
    features = _features(_speechlike_windows(4))
    np.testing.assert_allclose(_run(make_session(int8), features), _torch_output(module, features),
                               rtol=INT8_RANDOM_RTOL, atol=0.0)


def test_predictor_and_inference_match_session(random_model_files) -> None:  # noqa: ANN001
    """波形からの推定器（長さの違う入力の混在）と1回推論の関数が、セッションの直接の実行と同じ値。"""
    module, fp32, _ = random_model_files
    session = make_session(fp32)
    windows = _speechlike_windows(3) + _speechlike_windows(2, num_samples=12345, seed=1)
    predicted = make_onnx_predictor(session, batch_size=2)(windows)
    expected = np.array([_torch_output(module, _features([w]))[0] for w in windows], dtype=np.float32)
    np.testing.assert_allclose(predicted, expected, rtol=FP32_RTOL, atol=FP32_ATOL)
    single = make_onnx_inference(session)(windows[0])
    np.testing.assert_allclose(single, expected[:1], rtol=FP32_RTOL, atol=FP32_ATOL)


@pytest.fixture(scope="module")
def exp005_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[NormalizedModel, Path, Path]:
    if not EXP005.is_file():
        pytest.skip("runs/exp005/checkpoint_best.pt が無い")
    module = load_normalized_model(EXP005, NORMALIZATION)
    out = tmp_path_factory.mktemp("onnx_exp005")
    fp32 = export_onnx(module, out / "model_fp32.onnx")
    int8 = quantize_int8(fp32, out / "model_int8.onnx")
    return module, fp32, int8


def test_exp005_fp32_matches_pytorch(exp005_files) -> None:  # noqa: ANN001
    module, fp32, _ = exp005_files
    features = _features(_speechlike_windows(16))
    np.testing.assert_allclose(_run(make_session(fp32), features), _torch_output(module, features),
                               rtol=FP32_RTOL, atol=FP32_ATOL)


def test_exp005_int8_close_to_fp32(exp005_files) -> None:  # noqa: ANN001
    """量子化後（exp005）: 窓ごとの差の平均と最大が許容誤差以内。"""
    module, _, int8 = exp005_files
    features = _features(_speechlike_windows(16))
    diff = np.abs(_run(make_session(int8), features) - _torch_output(module, features))
    assert float(diff.mean()) <= INT8_EXP005_MEAN_ATOL
    assert float(diff.max()) <= INT8_EXP005_MAX_ATOL
