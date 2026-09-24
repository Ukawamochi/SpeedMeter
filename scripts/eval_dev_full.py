"""学習済みチェックポイントを dev 全件で評価し results/metrics.csv に追記する。

``runs/exp001/eval_dev_full.py``（第5段階5-3、exp001 の評価）と同じ手順を、
実験ごとに引数で切り替えられるようにしたもの。手順は変えていない。

- 評価データ: data/processed/features/dev（configs/splits/dev.json の話者）。test.json は使わない
- 指標: ``spkrate.train.train.evaluate_dev``（学習中の dev 評価と同じ）。バッチ64、長さ順バケッティング
- 1推論あたりの処理時間（旧方式）: docs/spec.md の推論単位である 2.0秒窓（32000標本）1回あたり。
  対数メル計算 → 正規化 → mps でのモデル前向き計算 までを1回として測る
  （音声の読み込み・再標本化は含めない。第3段階のベースラインの測り方に合わせる）。
  ウォームアップ20回の後200回の**平均**を ``latency_ms_per_inference`` 列に書く。
  この値はモデルごとに別のプロセス・時刻で測るため、**モデル間の比較には使わない**
  （参考値として残す。``latency_session_id`` 等の新方式の列は空欄）。
  比較に使う推論時間は、精度評価とは別に ``scripts/measure_latency.py`` で比較対象の全モデルを
  同一プロセス内で測った新方式の値（``latency_session_id`` 付きの中央値・最大値）である
  （docs/decisions/007-latency-measurement.md）
- モデルサイズ: ``spkrate.eval.runner.model_size_bytes``（checkpoint_best.pt のファイルサイズ）
- 雑音下評価 dev_noisy（``configs/eval/dev_noisy.yaml``、``spkrate.eval.noisy``）: clean と同じ
  クリップに固定の残響と MUSAN noise（SNR 5/10/15 dB）を掛け、波形から都度対数メルを計算して
  ``evaluate_dev`` で評価する。metrics.csv には clean の ``dev`` 行に続けて
  ``dev_noisy_snr5`` / ``dev_noisy_snr10`` / ``dev_noisy_snr15`` / ``dev_noisy_all``
  （3条件をまとめた指標）の行を追記する（記録形式は ``spkrate.eval.runner`` の docstring）。
  処理時間とモデルサイズは clean と同じ値を書く。``--no-noisy`` で従来どおり clean のみ

実行例（リポジトリ直下から）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_full.py \\
        --run-dir runs/exp002 --experiment-id 007-augmentation --config configs/exp002.yaml \\
        --method "cnn (話速推定CNN、方式A: クリップ全体入力、拡張あり)"

動作確認（先頭200件、metrics.csv には書かず別の csv へ）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_full.py \
        --run-dir runs/exp001 --experiment-id check --config configs/exp001.yaml \
        --limit 200 --output-dir /tmp/check --metrics-csv /tmp/check/metrics.csv
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

from spkrate.data.splits import load_clip_records, load_split
from spkrate.eval.noisy import (
    DEFAULT_NOISY_CONFIG,
    NoisyClipDataset,
    load_noisy_config,
    make_noise_source,
)
from spkrate.eval.runner import (
    DEFAULT_METRICS_CSV,
    NOISY_POOLED_SPLIT,
    MetricsRow,
    append_metrics_row,
    model_size_bytes,
    noisy_config_path,
    noisy_split_name,
    pool_metrics,
)
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
CLIPS_JSONL = "data/processed/clips.jsonl"
AUDIO_ROOT = "."  # clips.jsonl の audio_path はリポジトリ直下からの相対パス
NORMALIZATION = "configs/normalization.yaml"


def measure_latency(model, normalizer, device, *, warmup=20, repeats=200):
    """旧方式の推論時間（平均）。比較には使わない（モジュール docstring、spkrate.eval.latency）。"""
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
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV),
                        help="追記先の csv（動作確認ではテスト用の出力先を渡す）")
    parser.add_argument("--noisy-config", default=str(DEFAULT_NOISY_CONFIG),
                        help="dev_noisy の設定（configs/eval/dev_noisy.yaml）")
    parser.add_argument("--no-noisy", action="store_true",
                        help="dev_noisy を評価しない（clean のみ）")
    parser.add_argument("--limit", type=int, default=None,
                        help="dev の先頭この件数だけを評価する（動作確認用）")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--noisy-num-workers", type=int, default=0,
                        help="dev_noisy の DataLoader のワーカー数。実測（dev 2,000件、加工と対数メルのみ）で"
                             "0が約316件/秒、2が約137件/秒、4が約80件/秒と、0が最も速かった")
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
                                 normalizer=normalizer, limit=args.limit)
    print("dev clips:", len(dataset), flush=True)
    sampler = LengthBucketBatchSampler(dataset.frame_counts, BATCH, shuffle=False,
                                       pool_batches=20, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_clips,
                        num_workers=args.num_workers)

    noisy_config = None
    noisy_records = []
    if not args.no_noisy:
        noisy_config = load_noisy_config(args.noisy_config)
        # clean と同じクリップを同じ順に評価する（clips.jsonl から clip_id で引く）
        wanted = [entry["clip_id"] for entry in dataset.entries]
        wanted_set = set(wanted)
        by_id = {r.clip_id: r for r in load_clip_records(CLIPS_JSONL) if r.clip_id in wanted_set}
        missing = [clip_id for clip_id in wanted if clip_id not in by_id]
        if missing:
            raise ValueError(f"clips.jsonl に無いクリップがある: {len(missing)}件（例: {missing[:3]}）")
        noisy_records = [by_id[clip_id] for clip_id in wanted]
        noise_source = make_noise_source(noisy_config)

    noisy_metrics = {}
    noisy_extras = {}
    noisy_predictions = {}
    with MpsFallbackWatcher(logger) as watcher:
        metrics, extras, predictions = evaluate_dev(model, loader, build_loss("mse"), device)
        print("metrics:", json.dumps(metrics.as_dict(), ensure_ascii=False), flush=True)
        if noisy_config is not None:
            pooled_parts = []
            for snr_db in noisy_config.snr_db:
                split = noisy_split_name(snr_db)
                noisy_dataset = NoisyClipDataset(noisy_records, AUDIO_ROOT, snr_db,
                                                 config=noisy_config, noise_source=noise_source,
                                                 normalizer=normalizer)
                noisy_sampler = LengthBucketBatchSampler(noisy_dataset.frame_counts, BATCH,
                                                         shuffle=False, pool_batches=20, seed=0)
                noisy_loader = DataLoader(noisy_dataset, batch_sampler=noisy_sampler,
                                          collate_fn=collate_clips,
                                          num_workers=args.noisy_num_workers)
                started = time.perf_counter()
                m, e, p = evaluate_dev(model, noisy_loader, build_loss("mse"), device)
                e["wall_sec"] = time.perf_counter() - started
                print(f"metrics {split}:", json.dumps(m.as_dict(), ensure_ascii=False),
                      f"({e['wall_sec']:.1f}s)", flush=True)
                noisy_metrics[split], noisy_extras[split], noisy_predictions[split] = m, e, p
                pooled_parts.append((
                    [float(r.mora) for r in noisy_records],
                    [p[r.clip_id] * float(r.duration_sec) for r in noisy_records],
                    [float(r.duration_sec) for r in noisy_records],
                ))
            noisy_metrics[NOISY_POOLED_SPLIT] = pool_metrics(pooled_parts)
            print(f"metrics {NOISY_POOLED_SPLIT}:",
                  json.dumps(noisy_metrics[NOISY_POOLED_SPLIT].as_dict(), ensure_ascii=False),
                  flush=True)
        latency = measure_latency(model, normalizer, device)
    print("latency:", json.dumps(latency), flush=True)
    print("mps_cpu_fallback_events:", len(watcher.events), flush=True)
    size = model_size_bytes(checkpoint)

    if not args.no_metrics_csv:
        rows = [MetricsRow(
            experiment_id=args.experiment_id,
            method=args.method,
            config_path=args.config,
            split="dev",
            metrics=metrics,
            latency_ms_per_inference=latency["total_ms"],
            model_size_bytes=size,
        )]
        for split, noisy_metric in noisy_metrics.items():
            rows.append(MetricsRow(
                experiment_id=args.experiment_id,
                method=args.method,
                config_path=noisy_config_path(args.config, args.noisy_config),
                split=split,
                metrics=noisy_metric,
                latency_ms_per_inference=latency["total_ms"],
                model_size_bytes=size,
            ))
        for row in rows:
            append_metrics_row(row, csv_path=args.metrics_csv)
    summary = {"experiment_id": args.experiment_id, "best_epoch": int(payload["epoch"]),
               "metrics": metrics.as_dict(), "extras": extras, "latency": latency,
               "model_size_bytes": size, "checkpoint": checkpoint,
               "limit": args.limit,
               "noisy_config": None if args.no_noisy else args.noisy_config,
               "noisy_metrics": {k: v.as_dict() for k, v in noisy_metrics.items()},
               "noisy_extras": noisy_extras,
               "mps_cpu_fallback_events": watcher.events}
    with open(output_dir / "eval_dev_full.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(output_dir / "predictions_dev_full.json", "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, ensure_ascii=False)
    if noisy_predictions:
        with open(output_dir / "predictions_dev_noisy.json", "w", encoding="utf-8") as fh:
            json.dump(noisy_predictions, fh, ensure_ascii=False)
    print("EVAL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
