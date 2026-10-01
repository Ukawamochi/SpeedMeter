"""dev のクリップにモーラ単位の強制アライメントを付けて JSONL に書く（タスク4-1の入口）。

手段と規則は docs/decisions/008-forced-alignment.md、実装は src/spkrate/labels/alignment.py。
出力は1行1クリップ。成功: clip_id・ok=true・moras（kana/start/end 秒）・speech_start・
speech_end。失敗: clip_id・ok=false・reason（008 の理由コード）・detail。
既存の出力にある clip_id は飛ばす（--no-resume で上書き）。既定は dev。
``--split test`` で configs/splits/test.json の話者を対象にし、既定の出力先は
data/processed/alignments/test.jsonl になる（第10段階、2026-10-02 の人間の指示）。

使い方:
    uv run python scripts/align_dev.py --num 20 --seed 20260925 --out data/processed/alignments/dev.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from spkrate.device import describe_environment
from spkrate.labels.alignment import HiraganaAligner, align_clips_to_jsonl, select_dev_clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), default="dev", help="対象の分割。既定は dev")
    ap.add_argument("--out", default=None, help="省略時は data/processed/alignments/<split>.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--dev", default=None, help="分割ファイル。省略時は configs/splits/<split>.json")
    ap.add_argument("--num", type=int, default=None, help="処理する件数（省略時は dev 全件）")
    ap.add_argument("--seed", type=int, default=20260925, help="--num 指定時の抽出の種")
    ap.add_argument("--device", default="mps", help="mps・cuda・cpu。使えない場合は開始前に止める")
    ap.add_argument("--no-resume", action="store_true", help="既存の出力を上書きする")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("align_dev")

    if args.out is None:
        args.out = f"data/processed/alignments/{args.split}.jsonl"
    if args.dev is None:
        args.dev = f"configs/splits/{args.split}.json"
    records = select_dev_clips(
        args.clips, args.dev, num=args.num, seed=args.seed, allow_test=args.split == "test"
    )
    log.info("対象 %d 件 → %s", len(records), args.out)

    t0 = time.perf_counter()
    aligner = HiraganaAligner(args.device)
    aligner.emission(np.zeros(16000 * 2, dtype=np.float32))  # MPS の初回カーネル生成
    log.info("モデル読み込み %.1f 秒（device=%s）", time.perf_counter() - t0, args.device)
    log.info("%s", describe_environment(aligner.environment))

    stats = align_clips_to_jsonl(
        records, Path(args.out), aligner.align, resume=not args.no_resume, log_every=args.log_every
    )
    per = stats.seconds / stats.processed if stats.processed else float("nan")
    log.info(
        "完了: 処理 %d 件（既存で飛ばした %d 件）成功 %d 失敗 %s、%.1f 秒（%.3f 秒/件、読み込み込み）",
        stats.processed, stats.skipped, stats.ok, stats.failed, stats.seconds, per,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
