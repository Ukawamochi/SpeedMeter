"""複数のチェックポイントの推論時間を同一プロセス内で連続して測り、metrics.csv に追記する。

測定規則は docs/decisions/007-latency-measurement.md（実装は ``spkrate.eval.latency``）。
2.0秒窓1回の推論（対数メル計算 → 正規化 → 転送 → mps での前向き計算、計時の前後で同期）を
各モデル100回測り、中央値と最大値を記録する。ウォームアップの後、モデルを交互に測る。
全モデルの行に同じ ``latency_session_id`` を付ける。精度評価（``scripts/eval_dev_full.py``）とは
別の入口であり、比較に使う推論時間はこの入口で測った値である。

実行例（リポジトリ直下から）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/measure_latency.py \\
        --checkpoint runs/exp001/checkpoint_best.pt --experiment-id 005-first-model \\
        --checkpoint runs/exp002/checkpoint_best.pt --experiment-id 007-augmentation

動作確認（results/metrics.csv に書かない）:
    ... --metrics-csv /tmp/check/metrics.csv --output-json /tmp/check/latency.json

ONNX（第8段階 8-3）: ``--onnx <model.onnx>`` を繰り返して、ONNX Runtime（CPU の実行プロバイダ）での
推論も同じセッションで測る（``spkrate.export.to_onnx.make_onnx_inference``。範囲は同じく
対数メル計算 → 正規化（グラフ内）→ 前向き計算）。``--onnx-experiment-id``・``--onnx-threads``
（intra_op_num_threads。0 は ONNX Runtime の既定）・``--onnx-config``・``--onnx-method`` は
``--onnx`` と同じ数だけ与える。metrics.csv の device 列は cpu、model_size_bytes は ONNX ファイルの大きさ。
``--checkpoint`` を省いて ONNX だけを測ることもできる（そのときは同期をしない）。例:
    uv run python scripts/measure_latency.py \\
        --checkpoint runs/exp005/checkpoint_best.pt --experiment-id 010-exp005-method-b \\
        --onnx runs/onnx_exp005/model_fp32.onnx --onnx-experiment-id onnx-exp005-fp32 --onnx-threads 0 \\
        --onnx runs/onnx_exp005/model_int8.onnx --onnx-experiment-id onnx-exp005-int8 --onnx-threads 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from spkrate.eval.latency import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    device_synchronizer,
    latency_csv_row,
    make_cnn_inference,
    make_window,
    measure_session,
)
from spkrate.eval.runner import (
    DEFAULT_METRICS_CSV,
    append_metrics_row,
    git_commit_info,
    model_size_bytes,
)
from spkrate.train.data import load_normalization
from spkrate.train.train import MpsFallbackWatcher, load_checkpoint

NORMALIZATION = "configs/normalization.yaml"
DEFAULT_METHOD = "cnn (話速推定CNN)"


def _default_experiment_id(checkpoint: Path) -> str:
    return checkpoint.parent.name


def _default_config(checkpoint: Path) -> str:
    snapshot = checkpoint.parent / "config_snapshot.yaml"
    return str(snapshot) if snapshot.is_file() else ""


def _per_model(values: list[str] | None, count: int, name: str) -> list[str | None]:
    if not values:
        return [None] * count
    if len(values) != count:
        raise SystemExit(f"--{name} は --checkpoint と同じ数だけ与える（{len(values)} != {count}）")
    return list(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="測るチェックポイント（2つ以上を同じセッションで測る。繰り返して指定）")
    parser.add_argument("--experiment-id", action="append",
                        help="metrics.csv の実験ID（--checkpoint と同じ順・同じ数。省略時は runs/ 以下のディレクトリ名）")
    parser.add_argument("--config", action="append",
                        help="metrics.csv の設定ファイルのパス（省略時は同じディレクトリの config_snapshot.yaml）")
    parser.add_argument("--method", action="append",
                        help=f"metrics.csv の手法名（省略時は {DEFAULT_METHOD!r}）")
    parser.add_argument("--onnx", action="append", default=[], help="ONNX のモデル（繰り返して指定）")
    parser.add_argument("--onnx-experiment-id", action="append", help="--onnx ごとの実験ID（必須）")
    parser.add_argument("--onnx-threads", action="append", type=int,
                        help="--onnx ごとの intra_op_num_threads（0 は ONNX Runtime の既定。省略時はすべて0）")
    parser.add_argument("--onnx-config", action="append", help="--onnx ごとの設定ファイルのパス（metrics.csv）")
    parser.add_argument("--onnx-method", action="append", help="--onnx ごとの手法名（metrics.csv）")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--no-interleave", action="store_true",
                        help="交互測定をせずモデルごとに連続して測る（比較用。既定は交互測定）")
    parser.add_argument("--device", default=None, help="mps・cuda・cpu（既定 mps）。使えない場合は開始前に止める")
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV),
                        help="追記先の csv（動作確認ではテスト用の出力先を渡す）")
    parser.add_argument("--no-metrics-csv", action="store_true", help="csv に追記しない")
    parser.add_argument("--output-json", default=None,
                        help="各回の測定値を含む詳細の出力先（省略時は書かない）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("measure_latency")

    checkpoints = [Path(p) for p in args.checkpoint]
    onnx_paths = [Path(p) for p in args.onnx]
    if not checkpoints and not onnx_paths:
        raise SystemExit("--checkpoint か --onnx を1つ以上与える")
    onnx_ids = _per_model(args.onnx_experiment_id, len(onnx_paths), "onnx-experiment-id")
    if any(e is None for e in onnx_ids):
        raise SystemExit("--onnx には --onnx-experiment-id を同じ数だけ与える")
    onnx_threads = [t or 0 for t in _per_model(args.onnx_threads, len(onnx_paths), "onnx-threads")]
    onnx_configs = [c or "" for c in _per_model(args.onnx_config, len(onnx_paths), "onnx-config")]
    onnx_methods = [m or f"cnn (話速推定CNN) ONNX Runtime CPUExecutionProvider {p.name} threads={t or 'default'}"
                    for m, p, t in zip(_per_model(args.onnx_method, len(onnx_paths), "onnx-method"),
                                       onnx_paths, onnx_threads)]
    ids = [e or _default_experiment_id(c) for e, c in
           zip(_per_model(args.experiment_id, len(checkpoints), "experiment-id"), checkpoints)]
    if len(set(ids + onnx_ids)) != len(ids) + len(onnx_ids):
        raise SystemExit(f"実験IDが重複している: {ids + onnx_ids}")
    configs = [c or _default_config(p) for c, p in
               zip(_per_model(args.config, len(checkpoints), "config"), checkpoints)]
    methods = [m or DEFAULT_METHOD for m in _per_model(args.method, len(checkpoints), "method")]

    from spkrate.device import setup_device

    # 使えないデバイスを指定したら止める（cpu に落とさない）。比較に使う推論時間は Mac の mps で測る
    # （docs/directives/2026-09-26-rtx3060.md 0節4）。
    device, _ = setup_device(args.device or "mps", logger)
    if device.type != "mps":
        logger.warning("%s で測る（比較に使う推論時間は Mac の mps で測る）", device)

    normalizer = load_normalization(NORMALIZATION)
    inferences = {}
    for experiment_id, checkpoint in zip(ids, checkpoints):
        model, _ = load_checkpoint(checkpoint, map_location="cpu")
        inferences[experiment_id] = make_cnn_inference(model.to(device).eval(), normalizer, device)

    onnx_record = {}
    if onnx_paths:
        import onnxruntime

        from spkrate.export.to_onnx import make_onnx_inference, make_session

        for experiment_id, path, threads in zip(onnx_ids, onnx_paths, onnx_threads):
            ort_session = make_session(path, intra_op_threads=threads or None)
            inferences[experiment_id] = make_onnx_inference(ort_session)
            onnx_record[experiment_id] = {"path": str(path), "intra_op_threads": threads,
                                          "providers": ort_session.get_providers()}
        onnx_record["onnxruntime"] = onnxruntime.__version__
        logger.info("onnx: %s", json.dumps(onnx_record, ensure_ascii=False))

    # ONNX だけを測るときは同期しない（ONNX Runtime の CPU 実行は呼び出しの中で終わる）。
    synchronize = device_synchronizer(device) if checkpoints else (lambda: None)
    commit = git_commit_info()
    with MpsFallbackWatcher(logger) as watcher:
        session = measure_session(inferences, make_window(), repeats=args.repeats, warmup=args.warmup,
                                  interleave=not args.no_interleave, synchronize=synchronize)
    logger.info("latency_session_id: %s", session.session_id)
    for experiment_id in ids + onnx_ids:
        logger.info("latency %s", json.dumps(session.results[experiment_id].summary(), ensure_ascii=False))
    logger.info("mps_cpu_fallback_events: %d %s", len(watcher.events),
                json.dumps(watcher.events, ensure_ascii=False) if watcher.events else "")

    rows = [latency_csv_row(session.results[e], session.session_id, experiment_id=e, method=m,
                            config_path=c, model_size_bytes=model_size_bytes(p),
                            commit_hash=commit.commit_hash, commit_dirty=commit.dirty_flag,
                            device=device.type)
            for e, m, c, p in zip(ids, methods, configs, checkpoints)]
    rows += [latency_csv_row(session.results[e], session.session_id, experiment_id=e, method=m,
                             config_path=c, model_size_bytes=model_size_bytes(p),
                             commit_hash=commit.commit_hash, commit_dirty=commit.dirty_flag, device="cpu")
             for e, m, c, p in zip(onnx_ids, onnx_methods, onnx_configs, onnx_paths)]
    all_ids = ids + onnx_ids
    if not args.no_metrics_csv:
        for row in rows:
            append_metrics_row(row, csv_path=args.metrics_csv)
        logger.info("追記した: %s（%d行）", args.metrics_csv, len(rows))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        detail = {"latency_session_id": session.session_id, "device": str(device),
                  "interleave": session.interleave, "repeats": args.repeats, "warmup": args.warmup,
                  "commit_hash": commit.commit_hash, "commit_dirty": commit.dirty_flag,
                  "checkpoints": {e: str(p) for e, p in zip(ids, checkpoints)},
                  "onnx": onnx_record, "synchronize": "device" if checkpoints else "none",
                  "summary": {e: session.results[e].summary() for e in all_ids},
                  "times_ms": {e: list(session.results[e].times_ms) for e in all_ids},
                  "rounds_first": [list(r) for r in session.rounds[: len(all_ids)]],
                  "mps_cpu_fallback_events": watcher.events}
        output.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LATENCY_DONE", session.session_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
