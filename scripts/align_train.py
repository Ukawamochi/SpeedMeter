"""train 全件にアライメントと無発話判定の指標を付ける（指示書 2026-09-26 タスク2の1・2）。

dev と同じ手段・同じ定義（scripts/align_dev.py・scripts/detect_no_speech.py）で、
ひらがなCTCの推論を1クリップにつき1回だけ行い、2つの JSONL に書く
（実装は src/spkrate/labels/train_selection.py）。

- アライメント: --align-out（既定 data/processed/alignments/train.jsonl。dev.jsonl と同じ形式）
- 無発話の指標: --metrics-out（既定 data/processed/no_speech/train_metrics.jsonl。dev_metrics.jsonl と同じ形式）

途中で止まっても同じコマンドで再開できる（出力ごとに既存の clip_id を飛ばす）。
処理中の警告（MPS の CPU フォールバックを含む）はログに clip_id つきで記録する。
configs/splits/test.json は使わない。

使い方:
    nohup uv run python scripts/align_train.py > runs/alignment_train/log.txt 2>&1 &
    # dev で既存出力との一致を確かめる試行
    uv run python scripts/align_train.py --split configs/splits/dev.json --num 30 \\
        --align-out <scratch>/a.jsonl --metrics-out <scratch>/m.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from spkrate.labels.alignment import HIRAGANA_REVISION, HiraganaAligner, select_dev_clips
from spkrate.labels.train_selection import measure_clip, measure_clips_to_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align-out", default="data/processed/alignments/train.jsonl")
    ap.add_argument("--metrics-out", default="data/processed/no_speech/train_metrics.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--split", default="configs/splits/train.json", help="対象の話者の分割（test.json は不可）")
    ap.add_argument("--num", type=int, default=None, help="処理する件数（省略時は全件。試行用）")
    ap.add_argument("--seed", type=int, default=20260926, help="--num 指定時の抽出の種")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()
    if Path(args.split).name == "test.json":
        ap.error("configs/splits/test.json は使わない")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("align_train")

    records = select_dev_clips(args.clips, args.split, num=args.num, seed=args.seed)
    log.info(
        "対象 %d 件（%s）→ %s, %s（モデルの rev %s）",
        len(records), args.split, args.align_out, args.metrics_out, HIRAGANA_REVISION,
    )

    t0 = time.perf_counter()
    aligner = HiraganaAligner(args.device)
    aligner.emission(np.zeros(16000 * 2, dtype=np.float32))  # MPS の初回カーネル生成
    log.info("モデル読み込み %.1f 秒（device=%s）", time.perf_counter() - t0, args.device)

    fn = partial(measure_clip, emission_fn=aligner.emission, vocab=aligner.vocab, blank=aligner.blank)
    stats = measure_clips_to_jsonl(
        records, args.align_out, args.metrics_out,
        fn,
        log_every=args.log_every,
    )
    per = stats.seconds / stats.processed if stats.processed else float("nan")
    log.info(
        "完了: 処理 %d 件（既存で飛ばした %d 件）アライメント成功 %d 失敗 %s 警告 %d、%.1f 秒（%.3f 秒/件、読み込み込み）",
        stats.processed, stats.skipped, stats.ok, stats.failed, stats.warnings, stats.seconds, per,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
