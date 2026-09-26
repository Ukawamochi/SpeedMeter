"""書き起こしベースラインの計測と評価（docs/PLAN.md 第3段階 3-3）。

実装の本体は src/spkrate/baselines/asr.py にある。このスクリプトは3つの動作を持つ。

- ``compare``: 少数のクリップで複数のモデル・精度・ビーム幅を比べる（選定用）
- ``timeit``: 少数のクリップで1クリップあたりの所要時間を測り、全件評価の見積もりを出す
- ``eval``: 選んだ設定で検証セット（dev）を評価し、``results/metrics.csv`` に1行追記する

**テストセット（configs/splits/test.json）は使わない。** 分割の指定は dev のみであり、
``spkrate.eval.runner`` 側にもテスト分割を拒否する仕組みがある。

## 評価の規模

音声認識は信号処理ベースラインより3桁遅い。dev 全件（29,518件）を逐次で書き起こすと
6時間近くかかる見込みだったため、**固定シードで抽出した dev の部分集合**で評価する。
件数とシードは experiment_id と docs/decisions/003-asr-baseline.md に明記する。
同じ部分集合で信号処理ベースラインも再評価して比較する
（``scripts/run_envelope_baseline.py eval --size ... --seed ...``）。

## 処理時間の測り方

``runner.evaluate`` が測る1推論あたりの処理時間を歪めないため、**音声の復号だけ**を
複数プロセスで先読みし、音声認識そのものは主プロセスで逐次実行する
（信号処理ベースラインの評価と同じ作り）。したがって
``latency_ms_per_inference`` は「書き起こし＋モーラ計数」だけの時間であり、
mp3 の復号と再標本化は含まない。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.baselines.asr import (  # noqa: E402
    AsrParams,
    AsrSpeedEstimator,
    FasterWhisperTranscriber,
    load_params,
)
from spkrate.eval.audio import load_audio  # noqa: E402
from spkrate.eval.runner import EvalSegment, build_eval_segments, run_and_record  # noqa: E402

CLIPS_JSONL = ROOT / "data" / "processed" / "clips.jsonl"
DEV_SPLIT = ROOT / "configs" / "splits" / "dev.json"
CONFIG_PATH = ROOT / "configs" / "baselines" / "asr.yaml"
METRICS_CSV = ROOT / "results" / "metrics.csv"

# 評価に使う部分集合を選ぶ固定シードと件数。この値を変えない限り同じ部分集合になる。
# 信号処理ベースラインの調整に使った seed（20260921）とは別の値にしてある。
EVAL_SEED = 20260922
EVAL_SIZE = 5000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_dev_segments() -> list[EvalSegment]:
    return build_eval_segments(CLIPS_JSONL, DEV_SPLIT, ROOT)


def sample_segments(segments: list[EvalSegment], size: int, seed: int) -> list[EvalSegment]:
    """固定シードで部分集合を選ぶ。件数が足りなければ全件を返す。"""
    if size >= len(segments):
        return list(segments)
    indices = sorted(random.Random(seed).sample(range(len(segments)), size))
    return [segments[index] for index in indices]


def _load_audio_worker(path: str) -> tuple[np.ndarray, int]:
    return load_audio(path)


class BatchParallelLoader:
    """``runner.evaluate`` に渡す音声読み込み。復号だけを複数プロセスで行う。

    ``evaluate`` は区間の順に ``loader(path)`` を呼ぶので、同じ順に一定数ずつ
    まとめて並列復号し、その結果を1件ずつ返す。推定器の実行は主プロセスのままなので、
    測定される1推論あたりの処理時間には影響しない。
    """

    def __init__(self, segments: list[EvalSegment], workers: int, batch_size: int = 128) -> None:
        self._paths = [str(segment.audio_path) for segment in segments]
        self._batch_size = batch_size
        self._pool = mp.Pool(workers) if workers > 1 else None
        self._buffer: list[tuple[np.ndarray, int]] = []
        self._next = 0
        self._served = 0
        self._started = time.perf_counter()

    def __call__(self, path) -> tuple[np.ndarray, int]:
        if not self._buffer:
            chunk = self._paths[self._next : self._next + self._batch_size]
            if not chunk:
                raise RuntimeError("読み込む音声が残っていない")
            if self._pool is None:
                self._buffer = [_load_audio_worker(item) for item in chunk]
            else:
                self._buffer = list(self._pool.map(_load_audio_worker, chunk, chunksize=8))
            self._next += len(chunk)
        self._served += 1
        if self._served % 100 == 0:
            elapsed = time.perf_counter() - self._started
            total = len(self._paths)
            rate = self._served / elapsed if elapsed > 0 else 0.0
            remaining = (total - self._served) / rate if rate > 0 else float("nan")
            log(
                f"  進捗 {self._served}/{total} 件, "
                f"{elapsed / self._served * 1000:.0f}ミリ秒/件, "
                f"経過 {elapsed / 60:.1f}分, 残り約 {remaining / 60:.1f}分"
            )
        return self._buffer.pop(0)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


# --------------------------------------------------------------------------------------
# 選定用の比較


# 比較する候補。(モデル名, 計算精度, ビーム幅)。
COMPARE_CANDIDATES: tuple[tuple[str, str, int], ...] = (
    ("Systran/faster-whisper-tiny", "int8", 1),
    ("Systran/faster-whisper-tiny", "float32", 1),
    ("Systran/faster-whisper-base", "int8", 1),
    ("Systran/faster-whisper-base", "float32", 1),
    ("Systran/faster-whisper-base", "int8", 5),
    ("Systran/faster-whisper-small", "int8", 1),
    ("Systran/faster-whisper-small", "float32", 1),
    ("Systran/faster-whisper-small", "int8", 5),
    ("Systran/faster-whisper-small", "float32", 5),
)


def compare(args: argparse.Namespace) -> None:
    segments = load_dev_segments()
    subset = sample_segments(segments, args.size, args.seed)
    log(f"比較に使うクリップ: {len(subset)}件 (seed={args.seed})")
    loaded = [load_audio(segment.audio_path) for segment in subset]
    true_rates = np.array(
        [segment.true_mora / segment.duration_sec for segment in subset], dtype=np.float64
    )

    rows: list[dict[str, object]] = []
    for model_name, compute_type, beam_size in COMPARE_CANDIDATES:
        params = AsrParams(
            model_name=model_name,
            compute_type=compute_type,
            beam_size=beam_size,
            cpu_threads=args.cpu_threads,
        )
        estimator = AsrSpeedEstimator(FasterWhisperTranscriber(params))
        estimator(loaded[0][0], loaded[0][1])  # 読み込みと暖機
        predictions = []
        started = time.perf_counter()
        for samples, sample_rate in loaded:
            predictions.append(estimator(samples, sample_rate))
        elapsed = time.perf_counter() - started
        predicted = np.array(predictions, dtype=np.float64)
        errors = np.abs(predicted - true_rates)
        row = {
            "model": model_name,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "ms_per_clip": round(elapsed / len(loaded) * 1000.0, 1),
            "mae": round(float(errors.mean()), 4),
            "correlation": round(float(np.corrcoef(predicted, true_rates)[0, 1]), 4),
            "max_abs_error": round(float(errors.max()), 3),
        }
        rows.append(row)
        log(f"  {model_name} / {compute_type} / beam={beam_size} -> {row}")
    Path(args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"比較結果を書いた: {args.out}")


# --------------------------------------------------------------------------------------
# 所要時間の計測


def timeit(args: argparse.Namespace) -> None:
    """少数のクリップで所要時間を測り、dev 全件の見積もりを出す。"""
    segments = load_dev_segments()
    subset = sample_segments(segments, args.size, args.seed)
    params = load_params(CONFIG_PATH) if CONFIG_PATH.exists() else AsrParams()
    log(f"パラメータ: {params.as_dict()}")

    decode_started = time.perf_counter()
    loaded = [load_audio(segment.audio_path) for segment in subset]
    decode_elapsed = time.perf_counter() - decode_started

    estimator = AsrSpeedEstimator(FasterWhisperTranscriber(params))
    estimator(loaded[0][0], loaded[0][1])  # 読み込みと暖機

    infer_started = time.perf_counter()
    for samples, sample_rate in loaded:
        estimator(samples, sample_rate)
    infer_elapsed = time.perf_counter() - infer_started

    count = len(subset)
    audio_sec = sum(len(samples) / rate for samples, rate in loaded)
    log(f"計測: {count}件, 音声合計 {audio_sec:.1f}秒")
    log(f"  復号+再標本化(逐次): {decode_elapsed / count * 1000:.1f}ミリ秒/件")
    log(f"  書き起こし+モーラ計数(逐次): {infer_elapsed / count * 1000:.1f}ミリ秒/件")
    log(f"  実時間比: {audio_sec / infer_elapsed:.2f}倍速")
    log(f"  dev 全件({len(segments)}件)の見積もり: {infer_elapsed / count * len(segments) / 3600:.2f}時間")
    log(f"  {EVAL_SIZE}件の見積もり: {infer_elapsed / count * EVAL_SIZE / 60:.1f}分")


# --------------------------------------------------------------------------------------
# 評価


def evaluate_split(args: argparse.Namespace) -> None:
    params = load_params(CONFIG_PATH)
    log(f"パラメータ: {params.as_dict()}")
    segments = load_dev_segments()
    total_dev = len(segments)
    if args.size and args.size < total_dev:
        segments = sample_segments(segments, args.size, args.seed)
        experiment_id = f"{args.experiment_id}-subset{len(segments)}-seed{args.seed}"
        note = f"dev から seed={args.seed} で抽出した {len(segments)}/{total_dev} 件"
    else:
        experiment_id = args.experiment_id
        note = f"dev 全件 {total_dev} 件"
    log(f"評価対象: {note}")
    log(f"experiment_id = {experiment_id}")

    estimator = AsrSpeedEstimator(FasterWhisperTranscriber(params))
    transcripts: list[dict[str, object]] = []

    def recording_estimator(samples: np.ndarray, sample_rate: int) -> float:
        rate = estimator(samples, sample_rate)
        transcripts.append({"transcript": estimator.last_transcript, "rate": rate})
        return rate

    loader = BatchParallelLoader(segments, args.workers, batch_size=args.batch_size)
    started = time.perf_counter()
    try:
        # モデルの読み込みを計測の外で済ませる。
        estimator(np.zeros(16000, dtype=np.float32), 16000)
        result = run_and_record(
            recording_estimator,
            segments,
            experiment_id=experiment_id,
            method=f"asr-baseline (faster-whisper {params.model_name.split('-')[-1]} + count_mora)",
            config_path=str(CONFIG_PATH.relative_to(ROOT)),
            split="dev",
            csv_path=METRICS_CSV,
            repo_dir=ROOT,
            audio_loader=loader,
            device=params.device,
        )
    finally:
        loader.close()
    elapsed = time.perf_counter() - started

    metrics = result.metrics.as_dict()
    log(f"評価が終わった: {elapsed / 60:.1f}分")
    for key, value in metrics.items():
        log(f"  {key} = {value}")
    log(f"  latency_ms_per_inference = {result.latency_ms_per_inference:.3f}")
    log(f"metrics.csv に追記した: {METRICS_CSV}")

    summary = {
        "experiment_id": experiment_id,
        "note": note,
        "params": params.as_dict(),
        "elapsed_min": round(elapsed / 60, 2),
        "latency_ms_per_inference": result.latency_ms_per_inference,
        "empty_transcript_ratio": round(
            sum(1 for item in transcripts if not str(item["transcript"]).strip())
            / max(1, len(transcripts)),
            5,
        ),
        **metrics,
    }
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with Path(args.transcripts).open("w", encoding="utf-8") as handle:
        for segment, item in zip(segments, transcripts):
            handle.write(
                json.dumps(
                    {
                        "clip_id": segment.segment_id,
                        "true_mora": segment.true_mora,
                        "duration_sec": segment.duration_sec,
                        **item,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log(f"要約を書いた: {args.summary} / 書き起こしを書いた: {args.transcripts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    compare_parser = sub.add_parser("compare", help="少数のクリップでモデル候補を比べる")
    compare_parser.add_argument("--size", type=int, default=100)
    compare_parser.add_argument("--seed", type=int, default=12345)
    compare_parser.add_argument("--cpu-threads", type=int, default=0)
    compare_parser.add_argument("--out", default="/tmp/asr_model_comparison.json")
    compare_parser.set_defaults(func=compare)

    time_parser = sub.add_parser("timeit", help="1クリップあたりの所要時間を測る")
    time_parser.add_argument("--size", type=int, default=30)
    time_parser.add_argument("--seed", type=int, default=7)
    time_parser.set_defaults(func=timeit)

    eval_parser = sub.add_parser("eval", help="dev を評価して metrics.csv に追記する")
    eval_parser.add_argument("--size", type=int, default=EVAL_SIZE, help="0 なら dev 全件")
    eval_parser.add_argument("--seed", type=int, default=EVAL_SEED)
    eval_parser.add_argument("--workers", type=int, default=4)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--experiment-id", default="004-asr-baseline")
    eval_parser.add_argument("--summary", default="/tmp/asr_eval_summary.json")
    eval_parser.add_argument("--transcripts", default="/tmp/asr_baseline_transcripts.jsonl")
    eval_parser.set_defaults(func=evaluate_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
