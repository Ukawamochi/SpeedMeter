"""対数メルスペクトログラムの参照値を作る（docs/plan.md 第8段階 8-2）。

ブラウザ側の実装と特徴量計算を照合するための参照データを ``tests/fixtures/melspec/`` に書く。
合成したサイン波だけを扱う（実際のクリップの値は ``--clip`` でリポジトリ外に書く。下記）。

- ``sine_1000hz.json``: 1000Hz、振幅 0.5 の正弦波 0.5秒（8000標本）
- ``multitone_with_silence.json``: 先頭0.1秒が無音（厳密に0）、続く0.4秒が 250Hz・1500Hz・5000Hz
  （振幅 0.3・0.2・0.1）の和。無音のフレームで対数のオフセット（log(0 + 1e-6)）を確かめる
- ``mel_filterbank.json``: メルフィルタバンク（201ビン × 80）の0でない要素（帯ごとの開始ビンと重み）

波形は int16 に丸めた値（``round(x * 32767)``）で保存し、特徴量の入力は ``int16 / 32768``（float32）
とする。ブラウザ側でも同じ標本値を厳密に再現できるようにするためである（wav の int16 を
float32 にする変換と同じ）。対数メルの値は float32 の値を10進で書く（``float(np.float32)``、
往復で同じ float32 に戻る）。形は ``[フレーム][80]``。

実際のクリップ（``--clip <clip_id>``、dev のクリップ）: Common Voice の再配布が禁止されているため、
音声は保存しない。対数メルの値をリポジトリに入れてよいかは docs/questions.md で人間の判断を仰いでいる
ので、``--clip-out`` で指定したリポジトリ外（runs/ の下）に書くだけにする。

実行（リポジトリ直下から）:
    uv run python scripts/make_melspec_fixtures.py
    uv run python scripts/make_melspec_fixtures.py --clip <clip_id> --clip-out runs/melspec_fixture_clip
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "melspec"
SAMPLE_RATE = 16000
INT16_SCALE_OUT = 32767.0  # 丸めるときの倍率
INT16_SCALE_IN = 32768.0  # 特徴量の入力に戻すときの除数


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * INT16_SCALE_OUT), -32768, 32767).astype(np.int16)


def int16_to_float32(values) -> np.ndarray:  # noqa: ANN001
    """保存した int16 の標本を特徴量の入力（float32）に戻す。"""
    return (np.asarray(values, dtype=np.int16).astype(np.float32) / np.float32(INT16_SCALE_IN)).astype(np.float32)


def sine_1000hz() -> tuple[np.ndarray, str]:
    n = np.arange(8000)
    x = 0.5 * np.sin(2.0 * np.pi * 1000.0 * n / SAMPLE_RATE)
    return _to_int16(x), "int16(round(32767 * 0.5 * sin(2*pi*1000*n/16000)))、n=0..7999"


def multitone_with_silence() -> tuple[np.ndarray, str]:
    n = np.arange(8000)
    x = (0.3 * np.sin(2.0 * np.pi * 250.0 * n / SAMPLE_RATE)
         + 0.2 * np.sin(2.0 * np.pi * 1500.0 * n / SAMPLE_RATE)
         + 0.1 * np.sin(2.0 * np.pi * 5000.0 * n / SAMPLE_RATE))
    x[:1600] = 0.0
    return _to_int16(x), ("n<1600 は0、それ以外は int16(round(32767 * (0.3*sin(2*pi*250*n/16000) "
                          "+ 0.2*sin(2*pi*1500*n/16000) + 0.1*sin(2*pi*5000*n/16000))))、n=0..7999")


SIGNALS = {"sine_1000hz": sine_1000hz, "multitone_with_silence": multitone_with_silence}


def _config_dict() -> dict:
    return {k: v for k, v in asdict(MEL_DEFAULTS).items()}


def _log_mel_list(feature: np.ndarray) -> list[list[float]]:
    return [[float(np.float32(v)) for v in row] for row in feature.astype(np.float32)]


def write_signal_fixture(name: str, out_dir: Path, transform: LogMelSpectrogram) -> Path:
    samples_int16, formula = SIGNALS[name]()
    feature = transform(int16_to_float32(samples_int16), SAMPLE_RATE)
    payload = {
        "description": "対数メルスペクトログラムの参照値（docs/plan.md 8-2、docs/spec.md「特徴量」）",
        "generator": "scripts/make_melspec_fixtures.py",
        "sample_rate": SAMPLE_RATE,
        "signal": formula,
        "input_conversion": "float32(waveform_int16) / 32768",
        "config": _config_dict(),
        "num_samples": int(samples_int16.size),
        "num_frames": int(feature.shape[0]),
        "n_mels": int(feature.shape[1]),
        "log_mel_layout": "[frame][mel]、float32 の値",
        "waveform_int16": [int(v) for v in samples_int16],
        "log_mel": _log_mel_list(feature),
    }
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return out


def write_filterbank_fixture(out_dir: Path, transform: LogMelSpectrogram) -> Path:
    fb = transform.mel_filterbank  # (201, 80)
    bands = []
    for mel in range(fb.shape[1]):
        nonzero = np.flatnonzero(fb[:, mel])
        start = int(nonzero[0]) if nonzero.size else 0
        stop = int(nonzero[-1]) + 1 if nonzero.size else 0
        bands.append({"start_bin": start, "weights": [float(np.float32(v)) for v in fb[start:stop, mel]]})
    payload = {
        "description": "メルフィルタバンク（torchaudio の mel_scale='htk'、norm=None）の0でない要素",
        "generator": "scripts/make_melspec_fixtures.py",
        "num_bins": int(fb.shape[0]),
        "n_mels": int(fb.shape[1]),
        "layout": "bands[mel] = {start_bin, weights}。mel[m] = Σ_k power[start_bin + k] * weights[k]",
        "config": _config_dict(),
        "bands": bands,
    }
    out = out_dir / "mel_filterbank.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return out


def write_clip_values(clip_id: str, out_dir: Path, transform: LogMelSpectrogram) -> Path:
    """dev のクリップ1件の対数メルをリポジトリ外に書く（音声は書かない）。"""
    from spkrate.eval.audio import load_audio
    from spkrate.eval.dev_window import load_dev_clips, load_window_config
    from spkrate.data.splits import load_split

    config = load_window_config(ROOT / "configs/eval/dev_window.yaml")
    _, clip_paths = load_dev_clips(ROOT / config.clips_jsonl, load_split(ROOT / config.dev_split))
    if clip_id not in clip_paths:
        raise SystemExit(f"dev のクリップではない: {clip_id}")
    samples, _ = load_audio(ROOT / config.audio_root / clip_paths[clip_id])
    feature = transform(samples, SAMPLE_RATE)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Common Voice dev のクリップ1件の対数メル（音声は含めない。リポジトリに入れない）",
        "clip_id": clip_id,
        "source_path": str(clip_paths[clip_id]),
        "num_samples": int(np.asarray(samples).size),
        "num_frames": int(feature.shape[0]),
        "config": _config_dict(),
        "log_mel": _log_mel_list(feature),
    }
    out = out_dir / f"clip_{clip_id}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(FIXTURE_DIR))
    parser.add_argument("--clip", default=None, help="dev のクリップの clip_id（値はリポジトリ外に書く）")
    parser.add_argument("--clip-out", default=None, help="--clip の出力先（runs/ の下などリポジトリ外）")
    args = parser.parse_args(argv)

    transform = LogMelSpectrogram(MEL_DEFAULTS)
    if args.clip:
        if not args.clip_out:
            raise SystemExit("--clip には --clip-out（リポジトリ外）が要る")
        print(write_clip_values(args.clip, Path(args.clip_out), transform))
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SIGNALS:
        print(write_signal_fixture(name, out_dir, transform))
    print(write_filterbank_fixture(out_dir, transform))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
