"""強制アライメントの連結試験（docs/directives/2026-09-25.md タスク4-2）。

dev のアライメント成功クリップから2件ずつ無作為に組を作り（種固定、同じクリップ同士は組まない）、
「クリップ1 ＋ 既知の長さの無音（0値）＋ クリップ2」を連結して1本としてアライメントする。
推定された発話の境界
- 1つ目の発話の終了 = 連結結果の、クリップ1の最後のモーラの終了
- 2つ目の発話の開始 = 連結結果の、クリップ2の最初のモーラの開始
を、次の2つの既知の境界と比べ、ずれ（推定 − 既知、ミリ秒）を出す。
- 基準A（単独アライメント）: クリップ1 単独の speech_end、クリップ2 単独の speech_start ＋ オフセット
  （オフセット = クリップ1の長さ ＋ 無音長）。単独の結果は data/processed/alignments/dev.jsonl
- 基準B（クリップ端）: クリップ1の末尾（＝無音の開始）、クリップ2の先頭（＝無音の終了）

無音長は 0.3〜1.5 秒の一様乱数を 10ms に丸めたもの（種固定）。連結時のモーラ数検査は
クリップ1・2 の mora の和で行う（連結した文を読み変換し直すと読みが変わりうるため）。
あわせて、連結の前後で各モーラの開始時刻がどれだけ動いたか（全モーラ）も出す。
configs/splits/test.json は使わない。

使い方:
    uv run python scripts/alignment_concat_test.py --pairs 100 --seed 20260926 \
        --out runs/alignment_dev/concat_test.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from spkrate.eval.audio import load_audio
from spkrate.labels.alignment import SAMPLE_RATE, HiraganaAligner, align_clip


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float32)
    ab = np.abs(a)
    return {
        "n": int(a.size),
        "signed_median": float(np.median(a)),
        "signed_mean": float(a.mean()),
        "abs_median": float(np.median(ab)),
        "abs_p90": float(np.percentile(ab, 90)),
        "abs_p95": float(np.percentile(ab, 95)),
        "abs_max": float(ab.max()),
        "signed_p5": float(np.percentile(a, 5)),
        "signed_p95": float(np.percentile(a, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignments", default="data/processed/alignments/dev.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--sil-min", type=float, default=0.3)
    ap.add_argument("--sil-max", type=float, default=1.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="runs/alignment_dev/concat_test.json")
    args = ap.parse_args()

    single = {}
    for line in Path(args.alignments).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["ok"]:
            single[r["clip_id"]] = r
    clips = {}
    with Path(args.clips).open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["clip_id"] in single:
                clips[c["clip_id"]] = c

    rng = random.Random(args.seed)
    ids = sorted(single)
    aligner = HiraganaAligner(args.device)

    rows = []
    failures = []
    for k in range(args.pairs):
        a, b = rng.sample(ids, 2)
        sil = round(rng.uniform(args.sil_min, args.sil_max), 2)
        ca, cb = clips[a], clips[b]
        wa = load_audio(ca["audio_path"])[0]
        wb = load_audio(cb["audio_path"])[0]
        n_sil = int(round(sil * SAMPLE_RATE))
        wav = np.concatenate([wa, np.zeros(n_sil, dtype=np.float32), wb]).astype(np.float32)
        na = int(ca["mora"])
        total = na + int(cb["mora"])
        rec = {
            "clip_id": f"{a}+{b}",
            "kana": ca["kana"] + cb["kana"],
            "mora": total,
            "sentence": ca["sentence"] + cb["sentence"],
        }
        out = aligner.align(rec, wav, count_mora_fn=lambda _s, t=total: t)
        if not out["ok"]:
            failures.append({"pair": k, "a": a, "b": b, "reason": out["reason"], "detail": out["detail"]})
            continue
        m = out["moras"]
        len_a = len(wa) / SAMPLE_RATE
        offset = len_a + n_sil / SAMPLE_RATE
        est_end1 = m[na - 1]["end"]
        est_start2 = m[na]["start"]
        sa, sb = single[a], single[b]
        # 連結前後の全モーラの開始時刻の変化
        shift = [m[i]["start"] - sa["moras"][i]["start"] for i in range(na)]
        shift += [m[na + i]["start"] - (sb["moras"][i]["start"] + offset) for i in range(len(sb["moras"]))]
        rows.append(
            {
                "pair": k, "a": a, "b": b, "silence_sec": sil,
                "end1_vs_single_ms": (est_end1 - sa["speech_end"]) * 1000,
                "start2_vs_single_ms": (est_start2 - (sb["speech_start"] + offset)) * 1000,
                "end1_vs_edge_ms": (est_end1 - len_a) * 1000,
                "start2_vs_edge_ms": (est_start2 - offset) * 1000,
                "single_trail_ms": (sa["duration_sec"] - sa["speech_end"]) * 1000,
                "single_lead_ms": sb["speech_start"] * 1000,
                "mora_start_shift_abs_ms": [abs(x) * 1000 for x in shift],
            }
        )

    all_shift = [x for r in rows for x in r["mora_start_shift_abs_ms"]]
    res = {
        "seed": args.seed,
        "pairs": args.pairs,
        "silence_range_sec": [args.sil_min, args.sil_max],
        "n_ok": len(rows),
        "failures": failures,
        "end1_vs_single": summarize([r["end1_vs_single_ms"] for r in rows]),
        "start2_vs_single": summarize([r["start2_vs_single_ms"] for r in rows]),
        "end1_vs_edge": summarize([r["end1_vs_edge_ms"] for r in rows]),
        "start2_vs_edge": summarize([r["start2_vs_edge_ms"] for r in rows]),
        "both_vs_single": summarize([r["end1_vs_single_ms"] for r in rows] + [r["start2_vs_single_ms"] for r in rows]),
        "mora_start_shift_abs_ms": {
            "n": len(all_shift),
            "median": float(np.median(all_shift)),
            "p90": float(np.percentile(all_shift, 90)),
            "p99": float(np.percentile(all_shift, 99)),
            "frac_gt_100ms": float(np.mean(np.asarray(all_shift) > 100)),
        },
        "n_pairs_boundary_vs_single_gt_100ms": sum(
            1 for r in rows if abs(r["end1_vs_single_ms"]) > 100 or abs(r["start2_vs_single_ms"]) > 100
        ),
        "rows": [{k: v for k, v in r.items() if k != "mora_start_shift_abs_ms"} for r in rows],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
