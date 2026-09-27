"""方式Bのエポックの学習窓の正解の分布（docs/experiments/011-search-plan.md 2.2節・4.2節）。

exp010（時間伸縮の伸縮率を対数一様にする）の学習の前に、エポック1の学習窓の正解のうち
16 モーラ（毎秒8）以上の割合と4帯（metrics.SPEED_BANDS）の割合を、一様と対数一様で比べる。
正解はアライメント・伸縮率・窓の位置だけで決まるので、音声は読まない
（``WindowTrainDataset.window_label``）。学習時と同じ種 (seed, epoch, index) の生成器を使うので、
抽出した窓の正解は学習で使われる窓の正解そのものである。

- 対象は方式Bの窓（単一と連結）だけ。無音・雑音のみのサンプル（silence_samples。正解0）は含めない
- ``--sample`` を与えると全 2 × N_single 件から種 ``--sample-seed`` で非復元抽出する（負荷を抑える見積もり）
- ``--distribution`` を複数与えると、同じ窓の番号でそれぞれの分布の値を出す
  （設定ファイルの ``augment.time_stretch_distribution`` を上書きする）
- configs/splits/test.json は使わない（学習用の一覧と train.json だけを読む）

実行例:
    uv run python scripts/epoch_window_label_distribution.py --config configs/exp005.yaml \\
        --epoch 1 --sample 20000 --distribution uniform log_uniform
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np

from spkrate.eval.dev_window import KIND_CONCAT, KIND_SINGLE
from spkrate.eval.metrics import SPEED_BANDS, band_of
from spkrate.train.method_b import WindowTrainDataset, load_train_clips, read_clip_list
from spkrate.train.train import load_train_config

FAST_MORA = 16.0  # 2.0秒窓で毎秒8モーラ


def summarize(labels: np.ndarray, kinds: list[str], stretches: np.ndarray, window_sec: float) -> dict[str, Any]:
    """窓の正解の列から、16 モーラ以上の割合・4帯の割合・伸縮の集計を求める。"""
    n = int(labels.size)
    rates = labels / float(window_sec)
    bands = Counter(band_of(float(r)) for r in rates)
    out: dict[str, Any] = {
        "windows": n,
        "fast_ratio": float(np.mean(labels >= FAST_MORA)) if n else float("nan"),
        "fast_count": int(np.sum(labels >= FAST_MORA)),
        "bands": {key: bands.get(key, 0) / n for key, _, _ in SPEED_BANDS},
        "mean_mora": float(np.mean(labels)) if n else float("nan"),
        "stretched_ratio": float(np.mean(stretches != 1.0)) if n else float("nan"),
        "stretch_below1_among_stretched": (
            float(np.mean(stretches[stretches != 1.0] < 1.0)) if np.any(stretches != 1.0) else float("nan")
        ),
    }
    for kind in (KIND_SINGLE, KIND_CONCAT):
        mask = np.asarray([k == kind for k in kinds])
        out[f"fast_ratio_{kind}"] = float(np.mean(labels[mask] >= FAST_MORA)) if mask.any() else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="方式Bの学習の設定（data.source: windows）")
    parser.add_argument("--epoch", type=int, default=1, help="エポック（学習と同じ番号。既定1）")
    parser.add_argument("--sample", type=int, default=None, help="抽出する窓の数（既定は全件）")
    parser.add_argument("--sample-seed", type=int, default=20260928)
    parser.add_argument(
        "--distribution",
        nargs="+",
        default=None,
        help="time_stretch_distribution を上書きして比べる（既定は設定ファイルの値のみ）",
    )
    parser.add_argument("--json", default=None, help="結果を JSON で書く先")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    if config.data.source != "windows":
        raise SystemExit(f"data.source=windows の設定だけを扱う: {config.data.source}")
    augment = config.augment.build()
    started = time.perf_counter()
    clip_ids = read_clip_list(config.data.train_clip_list)
    clips, starts, ends = load_train_clips(
        clip_ids,
        clips_jsonl=config.data.clips_jsonl,
        train_split=config.data.train_split,
        alignments_path=config.data.train_alignments,
        sample_rate=config.windows.sample_rate,
        limit=config.data.max_train_clips,
    )
    dataset = WindowTrainDataset(
        clips, starts, ends, settings=config.windows, augment=augment, seed=config.seed
    )
    dataset.set_epoch(args.epoch)
    total = len(dataset)
    if args.sample is not None and args.sample < total:
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(rng.choice(total, size=int(args.sample), replace=False))
    else:
        indices = np.arange(total)
    print(
        f"設定={args.config} エポック={args.epoch} 窓の総数={total}（単一 {dataset.num_single}・連結 "
        f"{dataset.num_single}） 対象={indices.size}件（抽出の種 {args.sample_seed}） "
        f"読み込み {time.perf_counter() - started:.1f}秒",
        flush=True,
    )

    distributions = args.distribution or [
        augment.time_stretch_distribution if augment is not None else "uniform"
    ]
    results: dict[str, Any] = {
        "config": args.config,
        "epoch": args.epoch,
        "total_windows": total,
        "sampled": int(indices.size),
        "sample_seed": args.sample_seed,
        "by_distribution": {},
    }
    for name in distributions:
        dataset.augment = None if augment is None else replace(augment, time_stretch_distribution=name)
        labels = np.empty(indices.size, dtype=np.float32)
        stretches = np.empty(indices.size, dtype=np.float32)
        kinds: list[str] = []
        for j, index in enumerate(indices):
            label, plan, source = dataset.window_label(int(index))
            labels[j] = label
            stretches[j] = plan.stretch
            kinds.append(source.kind)
        summary = summarize(labels, kinds, stretches, config.windows.window_sec)
        results["by_distribution"][name] = summary
        bands = " ".join(f"{k}={v:.4f}" for k, v in summary["bands"].items())
        print(
            f"分布={name} 16モーラ以上={summary['fast_ratio']:.4f}（{summary['fast_count']}件。単一 "
            f"{summary['fast_ratio_single']:.4f}・連結 {summary['fast_ratio_concat']:.4f}） 帯: {bands} "
            f"平均={summary['mean_mora']:.3f}モーラ 伸縮の割合={summary['stretched_ratio']:.3f} "
            f"伸縮のうち s<1={summary['stretch_below1_among_stretched']:.3f}",
            flush=True,
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
