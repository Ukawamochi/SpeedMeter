"""ブラウザの確認用ページ（web/）で ONNX の出力を照合するための参照値を作る。

固定の合成波形（3.0秒、音声の代わりの断続的な調波音と雑音。int16 で保存）を、Python 側の経路
（``spkrate.features.melspec.LogMelSpectrogram`` → ONNX Runtime の ``model_fp32.onnx``）に通した
出力 ``mora`` を ``web/reference/onnx_exp005_fp32.json`` に書く。ページの「参照入力で実行」は
同じ波形を JS の対数メル → onnxruntime-web に通し、この値との差を表示する。

入力は2通り（web/app.js と同じ。どちらも確認用）:

- 窓ごと: docs/spec.md の窓（2.0秒、0.25秒ずらし、窓全体が収まるものだけ）。3.0秒なら5窓、
  各窓 201 フレームをまとめて (5, 201, 80) で1回実行する
- 全体: クリップ全体（301 フレーム）を (1, 301, 80) で1回実行する

モデルのファイルはリポジトリに入れないので、参照値にはモデルの SHA-256 を記録する。

実行（リポジトリ直下から）:
    uv run python scripts/make_web_reference.py --model runs/onnx_exp005/model_fp32.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spkrate.features.melspec import LogMelSpectrogram  # noqa: E402

SAMPLE_RATE = 16000
NUM_SAMPLES = 48000  # 3.0秒
WINDOW_SAMPLES = 32000  # docs/spec.md: 2.0秒窓
HOP_SAMPLES = 4000  # docs/spec.md: 推論間隔 0.25秒
SEED = 20260927
DEFAULT_OUT = ROOT / "web" / "reference" / "onnx_exp005_fp32.json"
DEFAULT_MODEL = ROOT / "runs" / "onnx_exp005" / "model_fp32.onnx"
INT16_SCALE_OUT = 32767.0
INT16_SCALE_IN = 32768.0


def reference_waveform_int16() -> np.ndarray:
    """断続的な調波音（基本周波数が揺れる）と弱い雑音の和。音節のような 0.12〜0.3秒の区切り。"""
    rng = np.random.default_rng(SEED)
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    signal = np.zeros(NUM_SAMPLES)
    position = int(0.2 * SAMPLE_RATE)
    while position < NUM_SAMPLES - int(0.2 * SAMPLE_RATE):
        length = int(rng.uniform(0.12, 0.3) * SAMPLE_RATE)
        gap = int(rng.uniform(0.02, 0.1) * SAMPLE_RATE)
        f0 = rng.uniform(110.0, 240.0)
        segment = slice(position, min(position + length, NUM_SAMPLES))
        local_t = t[segment] - t[position]
        envelope = np.sin(np.pi * local_t / (length / SAMPLE_RATE)) ** 2
        tone = sum((0.6 / k) * np.sin(2 * np.pi * k * f0 * local_t * (1 + 0.05 * local_t)) for k in range(1, 12))
        signal[segment] += 0.3 * envelope * tone
        position += length + gap
    signal += 0.005 * rng.standard_normal(NUM_SAMPLES)
    return np.clip(np.round(signal * INT16_SCALE_OUT), -32768, 32767).astype(np.int16)


def int16_to_float32(values) -> np.ndarray:  # noqa: ANN001
    return (np.asarray(values, dtype=np.int16).astype(np.float32) / np.float32(INT16_SCALE_IN)).astype(np.float32)


def window_starts(num_samples: int) -> list[int]:
    return list(range(0, num_samples - WINDOW_SAMPLES + 1, HOP_SAMPLES))


def python_inputs(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(窓ごとの入力 (窓数, 201, 80), 全体の入力 (1, フレーム数, 80))。"""
    mel = LogMelSpectrogram()
    windows = np.stack([mel(samples[s:s + WINDOW_SAMPLES], SAMPLE_RATE) for s in window_starts(samples.size)])
    whole = mel(samples, SAMPLE_RATE)[None]
    return windows.astype(np.float32), whole.astype(np.float32)


def run_model(model: Path, features: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return np.asarray(session.run(["mora"], {"log_mel": np.ascontiguousarray(features, dtype=np.float32)})[0],
                      dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    import onnxruntime

    samples_int16 = reference_waveform_int16()
    samples = int16_to_float32(samples_int16)
    windows, whole = python_inputs(samples)
    payload = {
        "description": "確認用ページ（web/）の ONNX 出力の参照値。Python の対数メル → ONNX Runtime（CPU）",
        "generator": "scripts/make_web_reference.py",
        "model_file": "model_fp32.onnx",
        "model_sha256": sha256(args.model),
        "onnxruntime": onnxruntime.__version__,
        "sample_rate": SAMPLE_RATE,
        "input_conversion": "float32(waveform_int16) / 32768",
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "window_starts": window_starts(samples.size),
        "window_input_shape": list(windows.shape),
        "window_mora": [float(v) for v in run_model(args.model, windows)],
        "whole_input_shape": list(whole.shape),
        "whole_mora": [float(v) for v in run_model(args.model, whole)],
        "waveform_int16": [int(v) for v in samples_int16],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: window_mora={payload['window_mora']} whole_mora={payload['whole_mora']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
