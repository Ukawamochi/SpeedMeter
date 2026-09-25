"""train のアライメント検査・無発話の印・学習に使うクリップの一覧を作る（指示書 2026-09-26 タスク2の2〜4）。

入力（scripts/align_train.py の出力。読むだけ）:
- data/processed/alignments/train.jsonl（アライメント）
- data/processed/no_speech/train_metrics.jsonl（無発話判定の指標）
- data/processed/clips.jsonl と configs/splits/train.json（対象と正解のモーラ数・話速）
- configs/eval/no_speech.yaml（無発話疑いの規則）、configs/eval/dev_window.yaml の
  isolation_threshold_sec（端の孤立の閾値。dev と同じ 1.0 秒、>）

除外の理由（src/spkrate/labels/train_selection.py。排他の件数はこの優先順で数える）:
align_failed（ok=false）→ mora_mismatch → head_isolated → tail_isolated → no_speech_suspect

出力:
- --suspect-out（既定 data/processed/no_speech/train_suspect.tsv）: 無発話疑いのクリップと判定に使った指標
- --out-dir（既定 data/processed/train_selection/）: usable_clip_ids.txt（学習に使う clip_id、clips.jsonl の順）、
  exclusions.tsv（除外したクリップと理由）、summary.json（件数）
- --md（既定 results/train_alignment.md）: 件数の記録

未処理（アライメントか指標のどちらかが無い）のクリップは一覧に入れず、件数を別に書く
（本処理の途中で実行した場合は「途中結果」と明記する）。configs/splits/test.json は使わない。

使い方:
    uv run python scripts/summarize_train_alignment.py --log runs/alignment_train/log.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spkrate.eval.dev_window import load_window_config
from spkrate.eval.metrics import BAND_KEYS, band_of
from spkrate.eval.no_speech import load_rule
from spkrate.labels.alignment import select_dev_clips
from spkrate.labels.train_selection import (
    EXCLUSION_PRIORITY,
    exclusion_reasons,
    primary_reason,
)

REASON_LABELS = {
    "align_failed": "アライメントの失敗（ok=false）",
    "mora_mismatch": "モーラ数の不一致（moras の要素数 ≠ clips.jsonl の mora）",
    "head_isolated": "端の孤立・先頭（1→2番目のモーラの開始時刻の差 > 閾値）",
    "tail_isolated": "端の孤立・末尾（最後から2番目→最後の差 > 閾値）",
    "no_speech_suspect": "無発話疑い（configs/eval/no_speech.yaml）",
}
BAND_LABELS = {"under4": "4未満", "4to6": "4以上6未満", "6to8": "6以上8未満", "over8": "8以上"}
SUSPECT_COLUMNS = ("clip_id", "band", "mora_per_second", "frame_db_p90", "frame_db_range", "cer", "hyp", "ref")


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # 書き込み中の最終行
                continue
            if r["clip_id"] in rows:
                raise ValueError(f"{path} に重複した clip_id: {r['clip_id']}")
            rows[r["clip_id"]] = r
    return rows


def count_log(path: Path | None) -> dict[str, int] | None:
    if path is None or not path.exists():
        return None
    c = Counter()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "MPS_CPU_FALLBACK" in line:
                c["mps_cpu_fallback"] += 1
            elif " WARNING " in line and "clip_id=" in line:
                c["other_warning"] += 1
            elif "Traceback" in line:
                c["traceback"] += 1
    return {k: int(c.get(k, 0)) for k in ("mps_cpu_fallback", "other_warning", "traceback")}


def pct(v: list[float], ps=(50, 99)) -> list[float]:
    return [float(x) for x in np.percentile(np.asarray(v, dtype=np.float64), ps)] if v else [float("nan")] * len(ps)


def band_row(counter: Counter, total: int | None = None) -> list[int]:
    vals = [int(counter.get(b, 0)) for b in BAND_KEYS]
    return vals + [sum(vals) if total is None else total]


def fmt_row(label: str, vals: list[int], base: list[int] | None = None) -> str:
    cells = []
    for i, v in enumerate(vals):
        if base is not None and base[i]:
            cells.append(f"{v:,}（{100 * v / base[i]:.2f}%）")
        else:
            cells.append(f"{v:,}")
    return f"| {label} | " + " | ".join(cells) + " |"


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignments", default="data/processed/alignments/train.jsonl")
    ap.add_argument("--metrics", default="data/processed/no_speech/train_metrics.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--split", default="configs/splits/train.json")
    ap.add_argument("--rule", default="configs/eval/no_speech.yaml")
    ap.add_argument("--window-config", default="configs/eval/dev_window.yaml")
    ap.add_argument("--suspect-out", default="data/processed/no_speech/train_suspect.tsv")
    ap.add_argument("--out-dir", default="data/processed/train_selection")
    ap.add_argument("--md", default="results/train_alignment.md")
    ap.add_argument("--log", default=None, help="scripts/align_train.py のログ（警告の件数だけ数える）")
    args = ap.parse_args()
    if Path(args.split).name == "test.json":
        ap.error("configs/splits/test.json は使わない")

    clips = select_dev_clips(args.clips, args.split)
    align = read_jsonl(Path(args.alignments))
    metrics = read_jsonl(Path(args.metrics))
    rule = load_rule(args.rule)
    threshold = load_window_config(args.window_config).isolation_threshold_sec

    n_total = len(clips)
    total_band: Counter = Counter()
    done_band: Counter = Counter()
    any_count: Counter = Counter()  # 理由 → 話速帯の Counter（重複を含む）
    any_by_band: dict[str, Counter] = {k: Counter() for k in EXCLUSION_PRIORITY}
    prim_by_band: dict[str, Counter] = {k: Counter() for k in EXCLUSION_PRIORITY}
    excluded_band: Counter = Counter()
    usable_band: Counter = Counter()
    fail_codes: Counter = Counter()
    combos: Counter = Counter()
    head_iv: list[float] = []
    tail_iv: list[float] = []
    pending = 0
    usable: list[str] = []
    excl_rows: list[list] = []
    suspect_rows: list[list] = []

    for c in clips:
        cid = c["clip_id"]
        band = band_of(float(c["mora_per_second"]))
        total_band[band] += 1
        a, m = align.get(cid), metrics.get(cid)
        if a is None or m is None:
            pending += 1
            continue
        done_band[band] += 1
        suspect = rule(m)
        if suspect:
            suspect_rows.append([cid, band, f"{c['mora_per_second']:.4f}"] + [m[k] for k in SUSPECT_COLUMNS[3:]])
        if a["ok"]:
            ms = a["moras"]
            if len(ms) >= 2:
                head_iv.append(ms[1]["start"] - ms[0]["start"])
                tail_iv.append(ms[-1]["start"] - ms[-2]["start"])
        else:
            fail_codes[a["reason"]] += 1
        reasons = exclusion_reasons(a, int(c["mora"]), suspect, threshold)
        if not reasons:
            usable.append(cid)
            usable_band[band] += 1
            continue
        excluded_band[band] += 1
        for r in reasons:
            any_count[r] += 1
            any_by_band[r][band] += 1
        p = primary_reason(reasons)
        prim_by_band[p][band] += 1
        combos["+".join(reasons)] += 1
        excl_rows.append([cid, band, f"{c['mora_per_second']:.4f}", p, ",".join(reasons)])

    n_done = n_total - pending
    complete = pending == 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "usable_clip_ids.txt").write_text("".join(f"{x}\n" for x in usable), encoding="utf-8")
    with (out_dir / "exclusions.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["clip_id", "band", "mora_per_second", "primary_reason", "reasons"])
        w.writerows(excl_rows)
    suspect_path = Path(args.suspect_out)
    suspect_path.parent.mkdir(parents=True, exist_ok=True)
    with suspect_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(SUSPECT_COLUMNS)
        w.writerows(suspect_rows)

    log_counts = count_log(Path(args.log)) if args.log else None
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": git_head(),
        "complete": complete,
        "train_clips": n_total,
        "processed": n_done,
        "pending": pending,
        "alignment_rows": len(align),
        "metrics_rows": len(metrics),
        "isolation_threshold_sec": threshold,
        "rule": str(rule),
        "alignment_fail_codes": dict(fail_codes),
        "excluded": sum(excluded_band.values()),
        "usable": len(usable),
        "reasons_with_overlap": {k: int(any_count.get(k, 0)) for k in EXCLUSION_PRIORITY},
        "reasons_exclusive": {k: int(sum(prim_by_band[k].values())) for k in EXCLUSION_PRIORITY},
        "reason_combinations": dict(combos.most_common()),
        "by_band": {
            "train": dict(total_band), "processed": dict(done_band),
            "excluded": dict(excluded_band), "usable": dict(usable_band),
            "with_overlap": {k: dict(v) for k, v in any_by_band.items()},
            "exclusive": {k: dict(v) for k, v in prim_by_band.items()},
        },
        "log": log_counts,
        "outputs": {
            "usable": str(out_dir / "usable_clip_ids.txt"),
            "exclusions": str(out_dir / "exclusions.tsv"),
            "suspect": str(suspect_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- Markdown
    heads = "| 項目 | " + " | ".join(BAND_LABELS[b] for b in BAND_KEYS) + " | 計 |"
    sep = "| --- | " + " | ".join("---:" for _ in BAND_KEYS) + " | ---: |"
    done_vals = band_row(done_band)
    L: list[str] = []
    L.append("# train のアライメントと学習に使うクリップの選別（指示書 2026-09-26 タスク2）")
    L.append("")
    if not complete:
        L.append(f"> **途中結果**: train {n_total:,}件のうち {n_done:,}件の処理済み分のみを集計した（未処理 {pending:,}件）。本処理の完了後に作り直す。")
        L.append("")
    L += [
        f"- 生成: {summary['generated_at']}（`scripts/summarize_train_alignment.py`、コミット {summary['commit']}）",
        "- 手段: dev と同じ（`docs/decisions/008-forced-alignment.md` のひらがなCTC＋`forced_align`、無発話の指標は `scripts/detect_no_speech.py` と同じ定義）。"
        "推論は1クリップ1回で、アライメントと cer の貪欲デコードに共通に使った（`scripts/align_train.py`、`src/spkrate/labels/train_selection.py`）",
        f"- 対象: `configs/splits/train.json` の話者に属する `data/processed/clips.jsonl` の {n_total:,}件。`configs/splits/test.json` は使っていない。音声を聴く判断はしていない",
        f"- 入力: `{args.alignments}`（{len(align):,}行）、`{args.metrics}`（{len(metrics):,}行）（Git管理外）",
        f"- 出力（Git管理外）: 学習に使う clip_id `{out_dir / 'usable_clip_ids.txt'}`、除外の一覧 `{out_dir / 'exclusions.tsv'}`、"
        f"件数 `{out_dir / 'summary.json'}`、無発話疑い `{suspect_path}`",
        f"- 規則: 無発話疑い＝`{args.rule}` の「{rule}」。端の孤立＝先頭2モーラまたは末尾2モーラの開始時刻の差 > {threshold:g} 秒（`{args.window_config}` の `isolation_threshold_sec`）",
        "",
        "## 1. 件数",
        "",
        "| 項目 | 件数 |",
        "| --- | ---: |",
        f"| train | {n_total:,} |",
        f"| 処理済み（アライメントと指標の両方あり） | {n_done:,} |",
        f"| 未処理 | {pending:,} |",
        f"| 除外 | {summary['excluded']:,} |",
        f"| 学習に使う | {len(usable):,} |",
        "",
    ]
    if log_counts is not None:
        L += [
            f"ログ（`{args.log}`）: MPS の CPU フォールバック（`MPS_CPU_FALLBACK`）{log_counts['mps_cpu_fallback']}件、"
            f"その他の clip_id つき警告 {log_counts['other_warning']}件、Traceback {log_counts['traceback']}件。",
            "",
        ]
    L += ["## 2. アライメントの自動検査", "", "| 検査 | 件数 |", "| --- | ---: |"]
    for code in ("mora_mismatch_label", "unknown_token", "align_failed", "mora_mismatch_alignment"):
        L.append(f"| 失敗 `{code}` | {fail_codes.get(code, 0):,} |")
    L.append(f"| 成功レコードのモーラ数 ≠ ラベル | {any_count.get('mora_mismatch', 0):,} |")
    L.append(f"| 端の孤立・先頭（> {threshold:g} 秒） | {any_count.get('head_isolated', 0):,} |")
    L.append(f"| 端の孤立・末尾（> {threshold:g} 秒） | {any_count.get('tail_isolated', 0):,} |")
    hp, tp = pct(head_iv, (50, 99, 100)), pct(tail_iv, (50, 99, 100))
    L += [
        "",
        f"端の2モーラの開始時刻の差（秒）: 先頭 中央値 {hp[0]:.3f}・99%点 {hp[1]:.3f}・最大 {hp[2]:.2f}、"
        f"末尾 中央値 {tp[0]:.3f}・99%点 {tp[1]:.3f}・最大 {tp[2]:.2f}（dev は results/alignment_dev.md 3.4節）。",
        "",
        "## 3. 除外の理由別・話速帯別の件数",
        "",
        "話速帯はラベルの毎秒モーラ数（`clips.jsonl` の `mora_per_second`）による。括弧内は処理済みのその帯の件数に対する割合。",
        "",
        "### 3.1 重複を含む件数（1クリップが複数の理由に該当すれば、それぞれに数える）",
        "",
        heads, sep,
        fmt_row("処理済み", done_vals),
    ]
    for k in EXCLUSION_PRIORITY:
        L.append(fmt_row(REASON_LABELS[k], band_row(any_by_band[k]), done_vals))
    L += [
        "",
        "### 3.2 排他の件数（優先順: 失敗 → 不一致 → 先頭の孤立 → 末尾の孤立 → 無発話疑い。最初に該当した理由にだけ数える）",
        "",
        heads, sep,
    ]
    for k in EXCLUSION_PRIORITY:
        L.append(fmt_row(REASON_LABELS[k], band_row(prim_by_band[k]), done_vals))
    L.append(fmt_row("除外の計", band_row(excluded_band), done_vals))
    L.append(fmt_row("学習に使う", band_row(usable_band), done_vals))
    L += ["", "### 3.3 理由の組み合わせ", "", "| 理由の組み合わせ | 件数 |", "| --- | ---: |"]
    for key, v in combos.most_common():
        L.append(f"| {key} | {v:,} |")
    if not combos:
        L.append("| （なし） | 0 |")
    L.append("")
    md = Path(args.md)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("complete", "train_clips", "processed", "pending", "excluded", "usable",
                                             "reasons_with_overlap", "reasons_exclusive")}, ensure_ascii=False))
    print(f"書き出し: {md}, {out_dir}, {suspect_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
