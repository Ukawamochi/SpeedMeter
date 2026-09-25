"""dev の強制アライメント結果を音声を聴かずに検査する（docs/directives/2026-09-25.md タスク4-2）。

入力は scripts/align_dev.py の出力（data/processed/alignments/dev.jsonl）と
data/processed/clips.jsonl。音声は読まない。結果は JSON で標準出力（または --out）へ書く。

検査項目:
- 失敗の件数と理由コード別の内訳
- アライメント後のモーラ数（moras の要素数）とラベル（clips.jsonl の mora）の不一致件数
- モーラ区間長（end - start）の分布。フレーム数（区間長 ÷ フレーム長）別の割合
- モーラ間隔（同じクリップ内で隣接するモーラの開始時刻の差）の分布
- 極端なモーラの閾値は、モーラ間隔の対数の分布から頑健に定める:
  中央値 ± 3 × (1.4826 × MAD)（対数領域）。区間長は約9割が1フレームで MAD が0になり
  下側の閾値が定義できないため、区間長の「極端に長い」には同じ上側閾値（間隔の上限）を使う
- 先頭・末尾のモーラの孤立: 最初の2モーラ・最後の2モーラの開始時刻の差の分布
- 先頭の無音（speech_start）と末尾の無音（duration_sec - speech_end）の分布

configs/splits/test.json は使わない。

使い方:
    uv run python scripts/inspect_alignment_dev.py --out runs/alignment_dev/inspect.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

FRAME_SEC = 0.02  # wav2vec2 の出力フレーム（008 4節）。フレーム数の丸めにのみ使う
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def pct(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    q = np.percentile(values, PCTS)
    out = {f"p{p}": float(v) for p, v in zip(PCTS, q)}
    out.update(mean=float(values.mean()), min=float(values.min()), max=float(values.max()), n=int(values.size))
    return out


def robust_log_fences(values: np.ndarray, k: float = 3.0) -> tuple[float, float, float, float]:
    """対数領域の中央値 ± k × 1.4826 × MAD を秒に戻して返す（下限, 上限, 中央値, σ）。"""
    lv = np.log(values[values > 0])
    med = float(np.median(lv))
    sigma = 1.4826 * float(np.median(np.abs(lv - med)))
    return math.exp(med - k * sigma), math.exp(med + k * sigma), math.exp(med), sigma


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignments", default="data/processed/alignments/dev.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = [json.loads(line) for line in Path(args.alignments).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {r["clip_id"] for r in recs}
    label: dict[str, dict] = {}
    with Path(args.clips).open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["clip_id"] in ids:
                label[c["clip_id"]] = c

    ok = [r for r in recs if r["ok"]]
    failed = [r for r in recs if not r["ok"]]
    reasons = Counter(r["reason"] for r in failed)
    fail_examples: dict[str, list] = {}
    for r in failed:
        fail_examples.setdefault(r["reason"], [])
        if len(fail_examples[r["reason"]]) < 5:
            c = label[r["clip_id"]]
            fail_examples[r["reason"]].append(
                {"clip_id": r["clip_id"], "detail": r["detail"], "duration_sec": c["duration_sec"], "mora": c["mora"]}
            )
    # 失敗クリップの長さ・モーラ数の要約（align_failed が短いクリップに偏るかを見る）
    fail_profile = {
        reason: {
            "duration_sec": pct(np.array([label[r["clip_id"]]["duration_sec"] for r in failed if r["reason"] == reason])),
            "mora_per_second": pct(np.array([label[r["clip_id"]]["mora_per_second"] for r in failed if r["reason"] == reason])),
        }
        for reason in reasons
    }

    mismatch = [r["clip_id"] for r in ok if len(r["moras"]) != int(label[r["clip_id"]]["mora"])]

    durs: list[float] = []
    intervals: list[float] = []
    lead: list[float] = []
    trail: list[float] = []
    per_clip_max_interval: list[float] = []
    per_clip_max_dur: list[float] = []
    first_iv: list[float] = []  # 1番目→2番目のモーラの開始時刻の差（先頭モーラの孤立を見る）
    last_iv: list[float] = []  # 最後から2番目→最後のモーラの開始時刻の差
    frame_mismatch_dur = 0
    for r in ok:
        m = r["moras"]
        starts = np.array([x["start"] for x in m])
        d = np.array([x["end"] - x["start"] for x in m])
        durs.extend(d.tolist())
        iv = np.diff(starts)
        intervals.extend(iv.tolist())
        per_clip_max_interval.append(float(iv.max()) if iv.size else float("nan"))
        if iv.size:
            first_iv.append(float(iv[0]))
            last_iv.append(float(iv[-1]))
        per_clip_max_dur.append(float(d.max()))
        lead.append(r["speech_start"])
        trail.append(r["duration_sec"] - r["speech_end"])
        if abs(r["duration_sec"] - label[r["clip_id"]]["duration_sec"]) > 0.05:
            frame_mismatch_dur += 1
    durs_a = np.array(durs)
    iv_a = np.array(intervals)
    lead_a = np.array(lead)
    trail_a = np.array(trail)

    frames = np.rint(durs_a / FRAME_SEC).astype(int)
    frame_hist = {
        "1": float(np.mean(frames <= 1)),
        "2": float(np.mean(frames == 2)),
        "3": float(np.mean(frames == 3)),
        "4-5": float(np.mean((frames >= 4) & (frames <= 5))),
        "6-10": float(np.mean((frames >= 6) & (frames <= 10))),
        "11-25": float(np.mean((frames >= 11) & (frames <= 25))),
        ">25": float(np.mean(frames > 25)),
    }
    iv_frames = np.rint(iv_a / FRAME_SEC).astype(int)
    iv_zero = int(np.sum(iv_a <= 1e-9))

    lo, hi, med, sigma = robust_log_fences(iv_a)
    pcm_max_iv = np.array(per_clip_max_interval)
    pcm_max_iv = pcm_max_iv[~np.isnan(pcm_max_iv)]

    def frac_lt(a: np.ndarray, t: float) -> float:
        return float(np.mean(a < t))

    def frac_gt(a: np.ndarray, t: float) -> float:
        return float(np.mean(a > t))

    out = {
        "n_records": len(recs),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "fail_reasons": dict(reasons),
        "fail_examples": fail_examples,
        "fail_profile": fail_profile,
        "n_mora_mismatch": len(mismatch),
        "mora_mismatch_examples": mismatch[:10],
        "n_duration_mismatch_vs_clips_gt_50ms": frame_mismatch_dur,
        "mora_duration_sec": pct(durs_a),
        "mora_duration_frame_hist": frame_hist,
        "mora_interval_sec": pct(iv_a),
        "mora_interval_zero": iv_zero,
        "mora_interval_frame_1": float(np.mean(iv_frames == 1)),
        "mora_interval_frame_le2": float(np.mean(iv_frames <= 2)),
        "interval_fence": {"low": lo, "high": hi, "median": med, "sigma_log": sigma, "k": 3.0},
        "interval_frac_below_low": frac_lt(iv_a, lo),
        "interval_frac_above_high": frac_gt(iv_a, hi),
        "duration_frac_above_high": frac_gt(durs_a, hi),
        "clip_frac_any_interval_above_high": frac_gt(pcm_max_iv, hi),
        "clip_frac_any_duration_above_high": frac_gt(np.array(per_clip_max_dur), hi),
        "clip_frac_any_interval_above_1s": frac_gt(pcm_max_iv, 1.0),
        "clip_max_interval_sec": pct(pcm_max_iv),
        "first_interval_sec": pct(np.array(first_iv)),
        "last_interval_sec": pct(np.array(last_iv)),
        "first_interval_frac_above_high": frac_gt(np.array(first_iv), hi),
        "last_interval_frac_above_high": frac_gt(np.array(last_iv), hi),
        "first_interval_frac_above_1s": frac_gt(np.array(first_iv), 1.0),
        "last_interval_frac_above_1s": frac_gt(np.array(last_iv), 1.0),
        "lead_silence_sec": pct(lead_a),
        "trail_silence_sec": pct(trail_a),
        "lead_frac_lt_0.05": frac_lt(lead_a, 0.05),
        "trail_frac_lt_0.05": frac_lt(trail_a, 0.05),
        "lead_frac_gt_3": frac_gt(lead_a, 3.0),
        "trail_frac_gt_3": frac_gt(trail_a, 3.0),
        "speech_ratio": pct(np.array([(r["speech_end"] - r["speech_start"]) / r["duration_sec"] for r in ok])),
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
