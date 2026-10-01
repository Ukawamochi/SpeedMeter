"""dev 全件の無発話判定用の指標を求め、JSONL に1行1クリップで書く（追加指示 2026-09-26 の1）。

指標の定義は src/spkrate/eval/no_speech.py。認識はアライメントと同じひらがなCTCモデル
（spkrate.labels.alignment のコミット固定、mps）で、文を与えずに貪欲デコードする。
既存の出力にある clip_id は飛ばす（途中で止まっても再開できる。--no-resume で上書き）。
処理中の警告（MPS の CPU フォールバックを含む）はログに clip_id つきで記録する。
既定は dev。``--split test`` で configs/splits/test.json の話者を対象にする（第10段階、2026-10-02 の人間の指示。
出力は data/processed/no_speech/test_metrics.jsonl）。

使い方:
    uv run python scripts/detect_no_speech.py --out data/processed/no_speech/dev_metrics.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from spkrate.device import describe_environment
from spkrate.eval.metrics import band_of
from spkrate.eval.no_speech import ctc_metrics, energy_metrics, normalize_reference
from spkrate.labels.alignment import (
    HIRAGANA_REVISION,
    HiraganaAligner,
    align_clips_to_jsonl,
    select_dev_clips,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), default="dev", help="対象の分割。既定は dev")
    ap.add_argument("--out", default=None, help="省略時は data/processed/no_speech/<split>_metrics.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--dev", default=None, help="分割ファイル。省略時は configs/splits/<split>.json")
    ap.add_argument("--num", type=int, default=None, help="処理する件数（省略時は dev 全件。試行用）")
    ap.add_argument("--seed", type=int, default=20260926, help="--num 指定時の抽出の種")
    ap.add_argument("--device", default="mps", help="mps・cuda・cpu。使えない場合は開始前に止める")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("detect_no_speech")

    if args.out is None:
        args.out = f"data/processed/no_speech/{args.split}_metrics.jsonl"
    if args.dev is None:
        args.dev = f"configs/splits/{args.split}.json"
    records = select_dev_clips(
        args.clips, args.dev, num=args.num, seed=args.seed, allow_test=args.split == "test"
    )
    log.info("対象 %d 件 → %s（モデルの rev %s）", len(records), args.out, HIRAGANA_REVISION)

    t0 = time.perf_counter()
    aligner = HiraganaAligner(args.device)
    aligner.emission(np.zeros(16000 * 2, dtype=np.float32))  # MPS の初回カーネル生成
    log.info("モデル読み込み %.1f 秒（device=%s）", time.perf_counter() - t0, args.device)
    log.info("%s", describe_environment(aligner.environment))

    def measure(record: dict, wav: np.ndarray) -> dict:
        ref = normalize_reference(record["kana"], aligner.vocab)
        mora = int(record["mora"])
        out = {
            "clip_id": record["clip_id"],
            "ok": True,
            "duration_sec": len(wav) / 16000,
            "mora": mora,
            "mora_per_second": float(record["mora_per_second"]),
            "band": band_of(float(record["mora_per_second"])),
            "ref": ref,
        }
        out.update(energy_metrics(wav))
        out.update(ctc_metrics(aligner.emission(wav), ref, aligner.vocab, aligner.blank, mora))
        return out

    stats = align_clips_to_jsonl(
        records, Path(args.out), measure, resume=not args.no_resume, log_every=args.log_every
    )
    per = stats.seconds / stats.processed if stats.processed else float("nan")
    log.info(
        "完了: 処理 %d 件（既存で飛ばした %d 件）、%.1f 秒（%.3f 秒/件、読み込み込み）",
        stats.processed, stats.skipped, stats.seconds, per,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
