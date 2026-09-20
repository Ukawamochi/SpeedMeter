"""話者単位で学習・検証・テストに分割し、configs/splits/ に固定する。

docs/PLAN.md 第2段階「2-3 分割の固定」に対応する。
処理の本体は src/spkrate/data/splits.py にある。

入力: data/processed/clips.jsonl（2-2 の出力）
出力:
- configs/splits/train.json, dev.json, test.json
- results/splits_overview.md

固定シードを使うため、同じ入力に対して何度実行しても同じ分割になる。

注意: configs/splits/test.json は docs/PLAN.md 第10段階まで使用禁止である。
条件の選択や途中の評価には dev.json のみを使う。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.data.splits import (  # noqa: E402
    DEFAULT_RATIOS,
    SPLIT_NAMES,
    SPLIT_SEED,
    SplitSummary,
    TEST_SPLIT_RESTRICTION,
    aggregate_speakers,
    assign_speakers,
    find_overlaps,
    load_clip_records,
    summarize_splits,
    write_split_files,
)

CLIPS_JSONL = ROOT / "data" / "processed" / "clips.jsonl"
OUT_DIR = ROOT / "configs" / "splits"
OUT_MD = ROOT / "results" / "splits_overview.md"


def write_report(
    out_md: Path,
    summaries: dict[str, SplitSummary],
    *,
    source: Path,
    seed: int,
    total_clips: int,
    total_speakers: int,
) -> None:
    lines: list[str] = []
    lines.append("# 話者単位の分割（train / dev / test）")
    lines.append("")
    lines.append(f"- 入力: `{source.relative_to(ROOT)}`（2-2 のクリップ一覧）")
    lines.append("- 出力: `configs/splits/train.json`, `dev.json`, `test.json`")
    lines.append("- 生成スクリプト: `scripts/build_splits.py`")
    lines.append("- 実装: `src/spkrate/data/splits.py`（`assign_speakers`）")
    lines.append(f"- 乱数シード: **{seed}**（固定。同じ入力なら常に同じ分割になる）")
    lines.append(f"- 対象: 話者 {total_speakers:,} 人、クリップ {total_clips:,} 件")
    lines.append("")
    lines.append(f"**禁止事項: {TEST_SPLIT_RESTRICTION}**")
    lines.append("")
    lines.append("## 1. 分割方法")
    lines.append("")
    lines.append("- 分割の単位は `client_id`。同一話者が複数の分割に現れることはない")
    lines.append(
        "- 比率 8:1:1 は**クリップ数**で合わせる（話者ごとのクリップ数の偏りが"
        "大きいため、話者数の比率では合わせない）"
    )
    lines.append("- 手順（貪欲法）")
    lines.append("    1. 話者を `client_id` の昇順に整列する（入力の並び順に依存させない）")
    lines.append(f"    2. `random.Random({seed})` でシャッフルする")
    lines.append("    3. クリップ数の多い順に安定ソートする")
    lines.append(
        "    4. 先頭から、目標クリップ数に対する不足が最大の分割へ割り当てる"
        "（同点なら train, dev, test の順）"
    )
    lines.append("")
    lines.append("## 2. json の形式")
    lines.append("")
    lines.append("各ファイルは次のキーを持つ1つのオブジェクトである。")
    lines.append("")
    lines.append("| キー | 内容 |")
    lines.append("| --- | --- |")
    lines.append("| `split` | 分割名（train / dev / test） |")
    lines.append("| `unit` | 分割の単位。常に `client_id` |")
    lines.append("| `seed` | 生成に使った乱数シード |")
    lines.append("| `method` | 割り当て方法 |")
    lines.append("| `target_ratio` | 目標のクリップ数比率 |")
    lines.append("| `source` | 元にしたクリップ一覧のパス |")
    lines.append("| `generated_by` / `implementation` | 生成スクリプトと実装のパス |")
    lines.append("| `num_speakers` / `num_clips` / `total_duration_sec` | 実際の規模 |")
    lines.append("| `usage_restriction` | test.json のみ。使用禁止の注記 |")
    lines.append("| `client_ids` | **分割に含まれる client_id の一覧（昇順）** |")
    lines.append("")
    lines.append(
        "読み出しは `spkrate.data.splits.load_split`（train / dev）を使う。"
        "test.json は `load_test_split` 専用で、`stage10_approved=True` を"
        "明示しない限り例外になる。"
    )
    lines.append("")
    lines.append("## 3. 各分割の規模")
    lines.append("")
    lines.append("| 分割 | 話者数 | クリップ数 | クリップ数の割合(%) | 合計時間(時間) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in SPLIT_NAMES:
        summary = summaries[name]
        share = summary.num_clips / total_clips * 100 if total_clips else 0.0
        lines.append(
            f"| {name} | {summary.num_speakers:,} | {summary.num_clips:,} | "
            f"{share:.2f} | {summary.total_duration_sec / 3600.0:,.2f} |"
        )
    total_duration = sum(s.total_duration_sec for s in summaries.values())
    lines.append(
        f"| 合計 | {total_speakers:,} | {total_clips:,} | 100.00 | "
        f"{total_duration / 3600.0:,.2f} |"
    )
    lines.append("")
    lines.append("目標比率は train 80.00 / dev 10.00 / test 10.00 パーセントである。")
    lines.append("")
    lines.append("## 4. 毎秒モーラ数の分布")
    lines.append("")
    lines.append("| 分割 | 最小 | 第1四分位 | 中央値 | 平均 | 第3四分位 | 最大 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in SPLIT_NAMES:
        rate = summaries[name].rate
        lines.append(
            f"| {name} | {rate.minimum:.3f} | {rate.q1:.3f} | {rate.median:.3f} | "
            f"{rate.mean:.3f} | {rate.q3:.3f} | {rate.maximum:.3f} |"
        )
    lines.append("")
    lines.append(
        "外れ値（1未満・12超）は 2-2 の段階で除外済みのため、最小・最大は"
        "その範囲に収まる。"
    )
    lines.append("")
    lines.append("## 5. 重複の検証")
    lines.append("")
    lines.append(
        "同一話者が複数の分割に現れないことは `tests/test_splits.py` で検証する。"
        "生成済みの json を読む検証と、合成データで分割関数の性質"
        "（話者単位であること・決定的であること）を確かめる検証の両方を持つ。"
    )
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=CLIPS_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if not args.clips.exists():
        print(f"[エラー] クリップ一覧が見つからない: {args.clips}", flush=True)
        return 1

    print(f"[開始] clips={args.clips} seed={args.seed}", flush=True)
    records = list(load_clip_records(args.clips))
    print(f"[読み込み] クリップ {len(records):,} 件", flush=True)

    speakers = aggregate_speakers(records)
    print(f"[集計] 話者 {len(speakers):,} 人", flush=True)

    assignment = assign_speakers(speakers, ratios=DEFAULT_RATIOS, seed=args.seed)
    overlaps = find_overlaps(assignment)
    if overlaps:
        print(f"[エラー] 話者が複数の分割に現れた: {list(overlaps)}", flush=True)
        return 1

    summaries = summarize_splits(records, assignment)
    for name in SPLIT_NAMES:
        summary = summaries[name]
        print(
            f"[分割] {name}: 話者={summary.num_speakers:,} "
            f"クリップ={summary.num_clips:,} "
            f"({summary.num_clips / len(records) * 100:.2f}%) "
            f"時間={summary.total_duration_sec / 3600.0:.2f}h",
            flush=True,
        )

    written = write_split_files(
        args.out_dir,
        assignment,
        seed=args.seed,
        ratios=DEFAULT_RATIOS,
        source=str(args.clips.relative_to(ROOT)),
        summaries=summaries,
    )
    for name in SPLIT_NAMES:
        print(f"[出力] {written[name]}", flush=True)

    write_report(
        args.out_md,
        summaries,
        source=args.clips,
        seed=args.seed,
        total_clips=len(records),
        total_speakers=len(speakers),
    )
    print(f"[出力] {args.out_md}", flush=True)
    print(f"[注意] {TEST_SPLIT_RESTRICTION}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
