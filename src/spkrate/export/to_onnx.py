"""ONNX への書き出しと int8 の動的量子化（docs/plan.md 第8段階 8-1、docs/spec.md「書き出し」）。

## 書き出すグラフ

    入力 ``log_mel`` (バッチ, フレーム, 80) float32  … 正規化**前**の対数メル（docs/spec.md「特徴量」）
      → 固定値の正規化 ``(log_mel - mean) / std``（configs/normalization.yaml。グラフに定数として埋め込む）
      → SpeechRateCNN（全フレーム有効。``lengths`` は渡さない）
    出力 ``mora`` (バッチ,) float32 … 入力区間のモーラ数（毎秒モーラ数ではない）

正規化をグラフに入れたのは、ブラウザ側で再現する処理を「対数メルの計算」だけにし、
正規化の80組の値を別に持ち運ばずに済ませるためである。対数メルの計算（``LogMelSpectrogram``）は
グラフに入れない（STFT をブラウザ側で実装する前提。8-2 の参照値で照合する）。

バッチとフレーム数は可変（動的な軸）である。詰め物のマスクは持たないので、1回の実行に入れる
入力はすべて同じフレーム数にする（2.0秒窓なら201フレーム）。

## 量子化

``onnxruntime.quantization.quantize_dynamic``（重み int8 = ``QuantType.QInt8``、チャネルごとでない、
``reduce_range=False``）。畳み込みは ``ConvInteger`` になり、活性は実行時に
``DynamicQuantizeLinear`` で uint8 に量子化される。

## 書き出し器

``torch.onnx.export(..., dynamo=False)``（TorchScript による従来の書き出し器）を使う。
torch 2.14 の既定（dynamo）は onnxscript を要し、依存を増やさないためである。opset は17。

## 使い方（リポジトリ直下から）

    uv run python -m spkrate.export.to_onnx --checkpoint runs/exp005/checkpoint_best.pt \\
        --out-dir runs/onnx_exp005

``<out-dir>/model_fp32.onnx``・``model_int8.onnx``・``export_meta.json`` を書く。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "DEFAULT_OPSET",
    "INPUT_NAME",
    "OUTPUT_NAME",
    "NormalizedModel",
    "export_onnx",
    "make_onnx_inference",
    "make_onnx_predictor",
    "make_session",
    "quantize_int8",
]

INPUT_NAME = "log_mel"
OUTPUT_NAME = "mora"
DEFAULT_OPSET = 17
SAMPLE_RATE = 16000
N_MELS = 80
EXPORT_FRAMES = 201  # 書き出しの試しの入力。2.0秒窓（32000標本）のフレーム数


class NormalizedModel(nn.Module):
    """固定値の正規化とモデルをまとめた書き出し用のモジュール（モジュール docstring）。"""

    def __init__(self, model: nn.Module, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.model = model
        mean32 = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        std32 = torch.as_tensor(np.asarray(std, dtype=np.float32))
        self.register_buffer("mean", mean32.reshape(1, 1, -1) if mean32.ndim else mean32)
        self.register_buffer("std", std32.reshape(1, 1, -1) if std32.ndim else std32)

    def forward(self, log_mel: Tensor) -> Tensor:
        return self.model((log_mel - self.mean) / self.std)


def load_normalized_model(checkpoint: str | Path, normalization: str | Path) -> NormalizedModel:
    """チェックポイントと正規化の設定から ``NormalizedModel`` を作る（cpu、eval 状態）。"""
    from spkrate.train.data import load_normalization
    from spkrate.train.train import load_checkpoint

    model, _ = load_checkpoint(str(checkpoint), map_location="cpu")
    normalizer = load_normalization(normalization)
    return NormalizedModel(model.eval(), normalizer.mean, normalizer.std).eval()


def export_onnx(module: nn.Module, out_path: str | Path, *, opset: int = DEFAULT_OPSET) -> Path:
    """``module``（``NormalizedModel``）を ONNX に書き出し、``onnx.checker`` で検査する。"""
    import warnings

    import onnx

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval().to("cpu")
    example = torch.zeros((1, EXPORT_FRAMES, N_MELS), dtype=torch.float32)
    with torch.no_grad(), warnings.catch_warnings():
        # 従来の書き出し器の非推奨の警告と、形の検査（Python の分岐）に対する追跡の警告は抑える。
        # 形の検査は入力の検査だけで、計算の経路は変わらない。
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            module,
            (example,),
            str(out),
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch", 1: "frames"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(str(out)))
    return out


def quantize_int8(in_path: str | Path, out_path: str | Path) -> Path:
    """int8 の動的量子化（重み ``QInt8``、テンソル単位、``reduce_range=False``）。"""
    import logging

    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 「前処理を勧める」旨の警告は root ロガーに出る。前処理はしない（記録: results/onnx_export.md）。
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        quantize_dynamic(str(in_path), str(out), weight_type=QuantType.QInt8,
                         per_channel=False, reduce_range=False)
    finally:
        root.setLevel(level)
    return out


def make_session(path: str | Path, *, intra_op_threads: int | None = None):  # noqa: ANN201
    """ONNX Runtime の推論セッション（CPU の実行プロバイダだけを使う）。

    ``intra_op_threads`` を省くと ONNX Runtime の既定（物理コア数）。
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    if intra_op_threads is not None:
        options.intra_op_num_threads = int(intra_op_threads)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def make_onnx_predictor(session, *, batch_size: int = 256, mel=None) -> Callable[[Sequence[np.ndarray]], np.ndarray]:  # noqa: ANN001
    """``波形の列 → モーラ数の列`` の推定器（``spkrate.eval.window_diag.make_predictor`` の ONNX 版）。

    経路は「波形 → 対数メル → （グラフ内で正規化 → モデル）」。グラフは詰め物のマスクを
    持たないので、フレーム数の同じ入力ごとにまとめてバッチにする（詰め物をしない）。
    """
    from spkrate.features.melspec import LogMelSpectrogram

    transform = mel if mel is not None else LogMelSpectrogram()

    def predict(waveforms: Sequence[np.ndarray]) -> np.ndarray:
        features = [np.asarray(transform(np.asarray(w, dtype=np.float32), SAMPLE_RATE), dtype=np.float32)
                    for w in waveforms]
        output = np.zeros(len(features), dtype=np.float32)
        groups: dict[int, list[int]] = {}
        for index, feature in enumerate(features):
            groups.setdefault(feature.shape[0], []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                batch = np.stack([features[i] for i in chunk]).astype(np.float32, copy=False)
                values = session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]
                output[chunk] = np.asarray(values, dtype=np.float32)
        return output

    return predict


def make_onnx_inference(session, mel=None) -> Callable[[np.ndarray], object]:  # noqa: ANN001
    """2.0秒窓の波形から1回推論する関数（推論時間の測定用。``make_cnn_inference`` の ONNX 版）。

    範囲は docs/decisions/007-latency-measurement.md 1節と同じ（対数メル計算 → 正規化 →
    前向き計算）。正規化はグラフの中で行う。ONNX Runtime の CPU 実行は同期的に終わる。
    """
    from spkrate.features.melspec import LogMelSpectrogram

    transform = mel if mel is not None else LogMelSpectrogram()

    def infer(samples: np.ndarray):
        feature = transform(samples, SAMPLE_RATE)
        return session.run([OUTPUT_NAME], {INPUT_NAME: np.ascontiguousarray(feature[None], dtype=np.float32)})[0]

    return infer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    import onnx
    import onnxruntime

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization", default="configs/normalization.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    module = load_normalized_model(args.checkpoint, args.normalization)
    fp32 = export_onnx(module, out_dir / "model_fp32.onnx", opset=args.opset)
    int8 = quantize_int8(fp32, out_dir / "model_int8.onnx")

    root = Path(__file__).resolve().parents[3]
    meta = {
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "commit": _commit(root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "normalization": str(args.normalization),
        "opset": args.opset,
        "input": {"name": INPUT_NAME, "shape": ["batch", "frames", N_MELS], "dtype": "float32",
                  "meaning": "正規化前の対数メル（docs/spec.md「特徴量」）"},
        "output": {"name": OUTPUT_NAME, "shape": ["batch"], "dtype": "float32", "meaning": "モーラ数"},
        "quantization": {"method": "onnxruntime.quantization.quantize_dynamic", "weight_type": "QInt8",
                         "per_channel": False, "reduce_range": False},
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in (fp32, int8)},
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": onnxruntime.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "command": " ".join([sys.executable.split("/")[-1], "-m", "spkrate.export.to_onnx", *(argv or sys.argv[1:])]),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["files"], ensure_ascii=False))
    print("EXPORT_DONE", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
