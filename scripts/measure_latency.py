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
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="測るチェックポイント（2つ以上を同じセッションで測る。繰り返して指定）")
    parser.add_argument("--experiment-id", action="append",
                        help="metrics.csv の実験ID（--checkpoint と同じ順・同じ数。省略時は runs/ 以下のディレクトリ名）")
    parser.add_argument("--config", action="append",
                        help="metrics.csv の設定ファイルのパス（省略時は同じディレクトリの config_snapshot.yaml）")
    parser.add_argument("--method", action="append",
                        help=f"metrics.csv の手法名（省略時は {DEFAULT_METHOD!r}）")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--no-interleave", action="store_true",
                        help="交互測定をせずモデルごとに連続して測る（比較用。既定は交互測定）")
    parser.add_argument("--device", default=None, help="既定は mps（使えなければ cpu で警告）")
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
    ids = [e or _default_experiment_id(c) for e, c in
           zip(_per_model(args.experiment_id, len(checkpoints), "experiment-id"), checkpoints)]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"実験IDが重複している: {ids}")
    configs = [c or _default_config(p) for c, p in
               zip(_per_model(args.config, len(checkpoints), "config"), checkpoints)]
    methods = [m or DEFAULT_METHOD for m in _per_model(args.method, len(checkpoints), "method")]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        logger.warning("%s で測る（CLAUDE.md の学習デバイスは mps）", device)

    normalizer = load_normalization(NORMALIZATION)
    inferences = {}
    for experiment_id, checkpoint in zip(ids, checkpoints):
        model, _ = load_checkpoint(checkpoint, map_location="cpu")
        inferences[experiment_id] = make_cnn_inference(model.to(device).eval(), normalizer, device)

    commit = git_commit_info()
    with MpsFallbackWatcher(logger) as watcher:
        session = measure_session(inferences, make_window(), repeats=args.repeats, warmup=args.warmup,
                                  interleave=not args.no_interleave,
                                  synchronize=device_synchronizer(device))
    logger.info("latency_session_id: %s", session.session_id)
    for experiment_id in ids:
        logger.info("latency %s", json.dumps(session.results[experiment_id].summary(), ensure_ascii=False))
    logger.info("mps_cpu_fallback_events: %d %s", len(watcher.events),
                json.dumps(watcher.events, ensure_ascii=False) if watcher.events else "")

    rows = [latency_csv_row(session.results[e], session.session_id, experiment_id=e, method=m,
                            config_path=c, model_size_bytes=model_size_bytes(p),
                            commit_hash=commit.commit_hash, commit_dirty=commit.dirty_flag)
            for e, m, c, p in zip(ids, methods, configs, checkpoints)]
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
                  "summary": {e: session.results[e].summary() for e in ids},
                  "times_ms": {e: list(session.results[e].times_ms) for e in ids},
                  "rounds_first": [list(r) for r in session.rounds[: len(ids)]],
                  "mps_cpu_fallback_events": watcher.events}
        output.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LATENCY_DONE", session.session_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
