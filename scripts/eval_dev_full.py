"""学習済みチェックポイントを dev 全件で評価し results/metrics.csv に追記する。

``runs/exp001/eval_dev_full.py``（第5段階5-3、exp001 の評価）と同じ手順を、
実験ごとに引数で切り替えられるようにしたもの。手順は変えていない。

- 評価データ: data/processed/features/dev（configs/splits/dev.json の話者）。test.json は使わない
- 指標: ``spkrate.train.train.evaluate_dev``（学習中の dev 評価と同じ）。バッチ64、長さ順バケッティング
- 1推論あたりの処理時間: docs/spec.md の推論単位である 2.0秒窓（32000標本）1回あたり。
  対数メル計算 → 正規化 → mps でのモデル前向き計算 までを1回として測る
  （音声の読み込み・再標本化は含めない。第3段階のベースラインの測り方に合わせる）
- モデルサイズ: ``spkrate.eval.runner.model_size_bytes``（checkpoint_best.pt のファイルサイズ）

実行例（リポジトリ直下から）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_full.py \\
        --run-dir runs/exp002 --experiment-id 007-augmentation --config configs/exp002.yaml \\
        --method "cnn (話速推定CNN、方式A: クリップ全体入力、拡張あり)"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spkrate.data.splits import load_split
from spkrate.eval.runner import MetricsRow, append_metrics_row, model_size_bytes
from spkrate.features.melspec import log_mel_spectrogram
from spkrate.train.data import (
    FeatureClipDataset,
    LengthBucketBatchSampler,
    collate_clips,
    load_normalization,
)
from spkrate.train.train import MpsFallbackWatcher, build_loss, evaluate_dev, load_checkpoint

BATCH = 64
WINDOW_SEC = 2.0
SAMPLE_RATE = 16000
LATENCY_SEED = 20260921
FEATURES_DIR = "data/processed/features/dev"
DEV_SPLIT = "configs/splits/dev.json"
NORMALIZATION = "configs/normalization.yaml"


def measure_latency(model, normalizer, device, *, warmup=20, repeats=200):
    rng = np.random.default_rng(LATENCY_SEED)
    samples = (rng.standard_normal(int(WINDOW_SEC * SAMPLE_RATE)) * 0.05).astype(np.float32)
    feature = normalizer(log_mel_spectrogram(samples, SAMPLE_RATE))
    tensor = torch.from_numpy(np.ascontiguousarray(feature[None], dtype=np.float32)).to(device)
    lengths = torch.tensor([feature.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(tensor, lengths)
        if device.type == "mps":
            torch.mps.synchronize()
        # 前向き計算のみ
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(tensor, lengths)
            if device.type == "mps":
                torch.mps.synchronize()
        forward_ms = (time.perf_counter() - t0) / repeats * 1000.0
        # 対数メル計算 → 正規化 → 前向き計算（ベースラインと同じ範囲）
        t0 = time.perf_counter()
        for _ in range(repeats):
            f = normalizer(log_mel_spectrogram(samples, SAMPLE_RATE))
            x = torch.from_numpy(np.ascontiguousarray(f[None], dtype=np.float32)).to(device)
            n = torch.tensor([f.shape[0]], dtype=torch.long, device=device)
            model(x, n)
            if device.type == "mps":
                torch.mps.synchronize()
        total_ms = (time.perf_counter() - t0) / repeats * 1000.0
    return {"frames": int(feature.shape[0]), "forward_only_ms": forward_ms, "total_ms": total_ms,
            "warmup": warmup, "repeats": repeats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="checkpoint_best.pt のある runs/ 以下")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", required=True, help="metrics.csv に記録する設定ファイルのパス")
    parser.add_argument("--method", default="cnn (話速推定CNN、方式A: クリップ全体入力)")
    parser.add_argument("--output-dir", default=None,
                        help="eval_dev_full.json / predictions_dev_full.json の出力先（既定は --run-dir）")
    parser.add_argument("--no-metrics-csv", action="store_true",
                        help="results/metrics.csv に追記しない（再現確認用）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("eval_dev_full")

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        logger.warning("mps が使えないため %s で実行する（CLAUDE.md の規定は mps）", device)
    checkpoint = str(run_dir / "checkpoint_best.pt")
    model, payload = load_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device).eval()
    print("best epoch:", payload["epoch"], flush=True)

    normalizer = load_normalization(NORMALIZATION)
    dataset = FeatureClipDataset(FEATURES_DIR, client_ids=load_split(DEV_SPLIT),
                                 normalizer=normalizer)
    print("dev clips:", len(dataset), flush=True)
    sampler = LengthBucketBatchSampler(dataset.frame_counts, BATCH, shuffle=False,
                                       pool_batches=20, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_clips, num_workers=4)
    with MpsFallbackWatcher(logger) as watcher:
        metrics, extras, predictions = evaluate_dev(model, loader, build_loss("mse"), device)
        print("metrics:", json.dumps(metrics.as_dict(), ensure_ascii=False), flush=True)
        latency = measure_latency(model, normalizer, device)
    print("latency:", json.dumps(latency), flush=True)
    print("mps_cpu_fallback_events:", len(watcher.events), flush=True)
    size = model_size_bytes(checkpoint)

    if not args.no_metrics_csv:
        append_metrics_row(MetricsRow(
            experiment_id=args.experiment_id,
            method=args.method,
            config_path=args.config,
            split="dev",
            metrics=metrics,
            latency_ms_per_inference=latency["total_ms"],
            model_size_bytes=size,
        ))
    summary = {"experiment_id": args.experiment_id, "best_epoch": int(payload["epoch"]),
               "metrics": metrics.as_dict(), "extras": extras, "latency": latency,
               "model_size_bytes": size, "checkpoint": checkpoint,
               "mps_cpu_fallback_events": watcher.events}
    with open(output_dir / "eval_dev_full.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(output_dir / "predictions_dev_full.json", "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, ensure_ascii=False)
    print("EVAL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
