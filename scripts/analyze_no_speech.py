"""無発話判定の指標の分布と判定規則の候補を比べ、規則を適用する（追加指示 2026-09-26 の1）。

入力: scripts/detect_no_speech.py の出力（data/processed/no_speech/dev_metrics.jsonl）、
正例の37件（results/error_cases/to_listen.tsv。聴取で発話が聞き取れないと確定）、
data/processed/clips.jsonl（文）、data/processed/dev_window/（窓の定義。読むだけで作り直さない）。

- ``candidates``: 指標ごとの分布（37件と残りの dev）と、37件をすべて含む閾値での規則の候補
  （1指標・2指標の論理積・3指標の論理積）の該当件数を Markdown で書く
- ``apply``: configs/eval/no_speech.yaml の規則を適用し、results/no_speech_suspect_dev.tsv と
  件数の要約（話速帯別・dev_window の窓数）を JSON で書く

configs/splits/test.json は使わない。

使い方:
    uv run python scripts/analyze_no_speech.py candidates --out <scratch>/candidates.md
    uv run python scripts/analyze_no_speech.py apply --summary <scratch>/apply.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spkrate.eval.dev_window import load_dev_window, load_known_no_speech
from spkrate.eval.metrics import SPEED_BANDS
from spkrate.eval.no_speech import Condition, Rule, apply_rule, load_rule, windows_from_clips

# 判定に使う指標と、無発話で値が「高い」「低い」どちらに寄ると期待するか（候補の向き）。
# 向きは 37件の中央値と残りの中央値の大小から決める（期待とは独立に、データで決める）。
METRICS = (
    "rms_dbfs",
    "peak_dbfs",
    "frame_db_p10",
    "frame_db_p50",
    "frame_db_p90",
    "frame_db_p99",
    "frame_db_range",
    "energy_speech_frac",
    "abs_speech_frac",
    "cer",
    "hyp_len",
    "hyp_mora_ratio",
    "ctc_nonblank_frac",
    "ctc_nll_per_token",
)
# 閾値を外側（37件を含む側）へ丸める刻み
_STEP = {"cer": 0.01, "hyp_mora_ratio": 0.01, "ctc_nonblank_frac": 0.005, "energy_speech_frac": 0.01,
         "abs_speech_frac": 0.01, "hyp_len": 1.0, "ctc_nll_per_token": 0.01}
_DB_STEP = 0.1
TSV_COLUMNS = ("clip_id", "band", "mora_per_second", "known_no_speech") + METRICS + ("hyp", "ref")


def load_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _step(metric: str) -> float:
    return _STEP.get(metric, _DB_STEP)


def outward(value: float, metric: str, op: str) -> float:
    """37件の端の値を、37件を含む向きに刻みで丸める。"""
    s = _step(metric)
    return math.ceil(value / s - 1e-9) * s if op == "<=" else math.floor(value / s + 1e-9) * s


def _values(rows: list[dict], metric: str) -> np.ndarray:
    v = np.array([r[metric] for r in rows], dtype=np.float64)
    return v


def direction(pos: list[dict], neg: list[dict], metric: str) -> str:
    """37件の中央値が残りより低ければ ``<=``（低いほど無発話）、高ければ ``>=``。"""
    return "<=" if np.nanmedian(_values(pos, metric)) < np.nanmedian(_values(neg, metric)) else ">="


def cover_condition(pos: list[dict], neg: list[dict], metric: str) -> Condition:
    op = direction(pos, neg, metric)
    v = _values(pos, metric)
    edge = float(np.nanmax(v) if op == "<=" else np.nanmin(v))
    return Condition(metric, op, round(outward(edge, metric, op), 6))


def _fmt(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"


def cmd_candidates(rows: list[dict], known: set[str], out: Path) -> None:
    pos = [r for r in rows if r["clip_id"] in known]
    neg = [r for r in rows if r["clip_id"] not in known]
    lines = [f"- dev の指標のあるクリップ: {len(rows)}（37件のうち該当 {len(pos)}、残り {len(neg)}）", ""]

    lines += ["## 分布", "",
              "| 指標 | 向き | 37件 最小 | 37件 中央値 | 37件 最大 | 残り 1% | 残り 5% | 残り 中央値 | 残り 95% | 残り 99% | 残りの NaN |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in METRICS:
        p, n = _values(pos, m), _values(neg, m)
        q = np.nanpercentile(n, [1, 5, 50, 95, 99])
        lines.append(
            f"| {m} | {direction(pos, neg, m)} | {_fmt(np.nanmin(p))} | {_fmt(np.nanmedian(p))} | "
            f"{_fmt(np.nanmax(p))} | " + " | ".join(_fmt(x) for x in q) + f" | {int(np.isnan(n).sum())} |"
        )

    conds = {m: cover_condition(pos, neg, m) for m in METRICS}
    neg_hits = {m: np.array([conds[m](r) for r in neg]) for m in METRICS}
    for c in conds.values():
        assert all(c(r) for r in pos), c

    def table(k: int, top: int) -> list[str]:
        res = []
        for combo in itertools.combinations(METRICS, k):
            hit = np.logical_and.reduce([neg_hits[m] for m in combo])
            res.append((int(hit.sum()), combo))
        res.sort()
        out_lines = [f"## {k}指標の論理積（該当件数の少ない順に上位{top}）", "",
                     "| 規則（37件をすべて含む閾値） | 37件以外の該当件数 |", "| --- | ---: |"]
        for cnt, combo in res[:top]:
            out_lines.append(f"| {' かつ '.join(str(conds[m]) for m in combo)} | {cnt} |")
        return out_lines + [""]

    lines += [""] + table(1, len(METRICS)) + table(2, 15) + table(3, 10)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"書き出し: {out}")


def _band_table(ids: set[str], by_id: dict[str, dict]) -> dict[str, int]:
    c = Counter(by_id[i]["band"] for i in ids)
    return {key: int(c.get(key, 0)) for key, _, _ in SPEED_BANDS}


def cmd_apply(rows: list[dict], known: set[str], args: argparse.Namespace) -> None:
    rule = load_rule(args.rule)
    by_id = {r["clip_id"]: r for r in rows}
    suspect = apply_rule(rows, rule)
    suspect_set = set(suspect)
    missed = sorted(known - suspect_set)

    tsv = Path(args.tsv)
    with tsv.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# no_speech_suspect（規則: {rule}。configs/eval/no_speech.yaml）。"
                "known_no_speech は results/error_cases/to_listen.tsv の37件\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(TSV_COLUMNS)
        for cid in suspect:
            r = by_id[cid]
            row = []
            for col in TSV_COLUMNS:
                if col == "known_no_speech":
                    row.append("1" if cid in known else "0")
                elif col in ("hyp", "ref", "clip_id", "band"):
                    row.append(r[col])
                else:
                    v = float(r[col])
                    row.append("NaN" if math.isnan(v) else f"{v:.4f}")
            w.writerow(row)

    all_bands = _band_table(set(by_id), by_id)
    ws = load_dev_window(args.dev_window)
    in_window_clips = {c for s in ws.sources for c in s.clip_ids}
    single_clips = {s.clip_ids[0] for s in ws.sources if s.kind == "single"}
    kind = np.asarray(ws.kind)

    def window_counts(ids: set[str]) -> dict[str, int]:
        mask = windows_from_clips(ws, ids)
        return {
            "single": int((mask & (kind == 0)).sum()),
            "concat": int((mask & (kind == 1)).sum()),
            "total": int(mask.sum()),
            "zero_label": int((mask & (np.asarray(ws.mora) == 0)).sum()),
            "concat_groups": sum(1 for s in ws.sources if s.kind == "concat" and any(c in ids for c in s.clip_ids)),
            "single_sources_with_windows": len({
                ws.sources[int(i)].clip_ids[0] for i in np.asarray(ws.source_index)[mask & (kind == 0)]
            }),
        }

    new = suspect_set - known
    summary = {
        "rule": str(rule),
        "dev_clips": len(rows),
        "suspect": len(suspect_set),
        "suspect_known": len(suspect_set & known),
        "suspect_new": len(new),
        "known_missed": missed,
        "bands_dev": all_bands,
        "bands_suspect": _band_table(suspect_set, by_id),
        "bands_suspect_new": _band_table(new, by_id),
        "dev_window": {
            "suspect_clips_in_dev_window": len(suspect_set & in_window_clips),
            "suspect_clips_single_source": len(suspect_set & single_clips),
            "suspect_new_clips_in_dev_window": len(new & in_window_clips),
            "windows_suspect": window_counts(suspect_set),
            "windows_suspect_new_only": window_counts(new),
            "windows_known": window_counts(known),
            "windows_known_or_suspect": window_counts(suspect_set | known),
            "windows_all": {"single": int((kind == 0).sum()), "concat": int((kind == 1).sum()),
                            "total": len(ws)},
        },
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("candidates", "apply"))
    ap.add_argument("--metrics", default="data/processed/no_speech/dev_metrics.jsonl")
    ap.add_argument("--known", default="results/error_cases/to_listen.tsv")
    ap.add_argument("--rule", default="configs/eval/no_speech.yaml")
    ap.add_argument("--dev-window", default="data/processed/dev_window")
    ap.add_argument("--tsv", default="results/no_speech_suspect_dev.tsv")
    ap.add_argument("--out", default="candidates.md", help="candidates の出力")
    ap.add_argument("--summary", default="data/processed/no_speech/apply_summary.json")
    args = ap.parse_args()

    rows = load_metrics(Path(args.metrics))
    known = load_known_no_speech(args.known)
    missing = known - {r["clip_id"] for r in rows}
    if missing:
        raise SystemExit(f"37件のうち指標の無いクリップがある: {sorted(missing)}")
    if args.command == "candidates":
        cmd_candidates(rows, known, Path(args.out))
    else:
        cmd_apply(rows, known, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
