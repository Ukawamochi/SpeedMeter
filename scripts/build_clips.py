"""Common Voice のvalidated.tsvから学習用クリップ一覧を作り、結果を書き出す。

docs/PLAN.md 第2段階「2-2 クリップ一覧の作成」に対応する。
処理の本体は src/spkrate/data/common_voice.py にある。

この一覧は**学習データ構築のための母集団**であり、既定で学習専用の文除外
（spkrate.labels.filter.should_exclude_from_training）を適用する。
評価セットは第9段階で別に作る（--no-training-filter で除外を外せる）。

data/common_voice_ja/ は読み取りのみ行い、一切変更しない。

出力:
- data/processed/clips.parquet（pyarrowが無い場合は clips.jsonl）
- results/clip_filtering.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.data.common_voice import (  # noqa: E402
    DEFAULT_MAX_MORA_PER_SECOND,
    DEFAULT_MIN_MORA_PER_SECOND,
    FilterStats,
    RateSummary,
    build_clip_index,
    summarize_mora_per_second,
    write_records,
)

CV_DIR = ROOT / "data" / "common_voice_ja"
VALIDATED_TSV = CV_DIR / "validated.tsv"
DURATIONS_TSV = CV_DIR / "clip_durations.tsv"
AUDIO_DIR_REL = "data/common_voice_ja/clips"
OUT_DATA = ROOT / "data" / "processed" / "clips.parquet"
OUT_MD = ROOT / "results" / "clip_filtering.md"


def make_progress(started: float):
    def report(processed: int, stats: FilterStats) -> None:
        elapsed = time.time() - started
        print(
            f"[進捗] {processed:,}行 処理 経過 {elapsed:.1f}秒 "
            f"除外(votes={stats.excluded_down_votes:,}, "
            f"empty={stats.excluded_empty_sentence:,}, "
            f"unconverted={stats.excluded_unconverted:,}, "
            f"digits={stats.excluded_long_digit_run:,}, "
            f"no_duration={stats.missing_duration:,}, "
            f"outlier={stats.excluded_outlier:,})",
            flush=True,
        )

    return report


def write_report(
    out_md: Path,
    stats: FilterStats,
    summary: RateSummary,
    data_path: Path,
    apply_training_filter: bool,
) -> None:
    hours = summary.total_duration_sec / 3600.0
    lines: list[str] = []
    lines.append("# Common Voice クリップ一覧の作成と除外")
    lines.append("")
    lines.append("- 対象: `data/common_voice_ja/validated.tsv`")
    lines.append(f"- 出力: `{data_path.relative_to(ROOT)}`")
    lines.append("- 生成スクリプト: `scripts/build_clips.py`")
    lines.append(
        "- 実装: `src/spkrate/data/common_voice.py`（`build_clip_index`）"
    )
    lines.append(
        "- 学習専用の文除外（`should_exclude_from_training`）の適用: "
        + ("あり（学習データ構築のため）" if apply_training_filter else "なし")
    )
    lines.append(
        "- この一覧は学習データ構築の経路である。評価セット（検証・テスト）は"
        "docs/PLAN.md 第9段階で別に作り、学習専用の除外は適用しない"
    )
    lines.append("- data/ 以下は読み取りのみ行い、変更・削除はしていない")
    lines.append("")
    lines.append("## 1. 入力と音声長の取得方法")
    lines.append("")
    lines.append(
        f"- 使用した `validated.tsv` のデータ行数（ヘッダを除く）: {stats.total_rows:,}"
    )
    lines.append(
        "- 音声長: **`data/common_voice_ja/clip_durations.tsv` の値を使用**した。"
        "同ファイルの `duration[ms]` をミリ秒として読み、1000で割って秒に換算した。"
        "音声ファイル本体（mp3）は読んでいない"
    )
    lines.append(
        f"- `clip_durations.tsv` に音声長が無かったクリップ: {stats.missing_duration:,} 件"
        "（除外した）"
    )
    lines.append(
        f"- 読み変換したユニークな原文の数: {stats.unique_sentences:,}"
        "（同一原文の結果は再利用した）"
    )
    lines.append(
        "- 読み変換の注意: pyopenjtalk はUTF-8で約8192バイトを超える入力で"
        "プロセスごと異常終了する。validated.tsv には1万字規模の文が含まれるため、"
        "`spkrate.labels.mora.to_kana` は長い文を2048バイト以下に分割して変換し、"
        "結果を連結している（該当する文は毎秒モーラ数の外れ値として除外される）"
    )
    lines.append("")
    lines.append("## 2. 除外件数")
    lines.append("")
    lines.append("除外は上から順に適用し、先に該当した条件で計上している。")
    lines.append("")
    lines.append("| 順 | 除外条件 | 件数 |")
    lines.append("| ---: | --- | ---: |")
    lines.append(f"| - | 元のクリップ数（validated.tsv のデータ行数） | {stats.total_rows:,} |")
    lines.append(f"| 1 | down_votes が up_votes 以上 | {stats.excluded_down_votes:,} |")
    lines.append(f"| 2 | sentence が空 | {stats.excluded_empty_sentence:,} |")
    lines.append(
        f"| 3a | 読み変換後にカタカナ以外が残る（`unconverted`） | {stats.excluded_unconverted:,} |"
    )
    lines.append(
        f"| 3b | 3桁以上連続する数字列を含む（`long_digit_run`） | {stats.excluded_long_digit_run:,} |"
    )
    lines.append(
        f"| 4 | `clip_durations.tsv` に音声長が無い | {stats.missing_duration:,} |"
    )
    lines.append(
        f"| 5 | 毎秒モーラ数が {DEFAULT_MIN_MORA_PER_SECOND:g} 未満または "
        f"{DEFAULT_MAX_MORA_PER_SECOND:g} 超の外れ値 | {stats.excluded_outlier:,} |"
    )
    excluded_total = stats.total_rows - stats.kept
    lines.append(f"| - | 除外合計 | {excluded_total:,} |")
    lines.append(f"| - | **残存件数** | **{stats.kept:,}** |")
    lines.append("")
    lines.append(
        f"残存率: {stats.kept / stats.total_rows * 100:.2f} パーセント"
        if stats.total_rows
        else "残存率: -"
    )
    lines.append("")
    lines.append("## 3. 残存クリップの毎秒モーラ数の分布")
    lines.append("")
    lines.append("| 統計量 | 値 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 件数 | {summary.count:,} |")
    lines.append(f"| 最小 | {summary.minimum:.3f} |")
    lines.append(f"| 第1四分位 | {summary.q1:.3f} |")
    lines.append(f"| 中央値 | {summary.median:.3f} |")
    lines.append(f"| 平均 | {summary.mean:.3f} |")
    lines.append(f"| 第3四分位 | {summary.q3:.3f} |")
    lines.append(f"| 最大 | {summary.maximum:.3f} |")
    lines.append("")
    lines.append("### ヒストグラム（幅0.5、左端を含み右端を含まない。最後の階級のみ両端を含む）")
    lines.append("")
    lines.append("| 階級（毎秒モーラ数） | 件数 | 割合(%) |")
    lines.append("| --- | ---: | ---: |")
    last_index = len(summary.histogram) - 1
    for index, (low, high, count) in enumerate(summary.histogram):
        share = count / summary.count * 100 if summary.count else 0.0
        edge = "以下" if index == last_index else "未満"
        lines.append(f"| {low:.1f} 以上 {high:.1f} {edge} | {count:,} | {share:.2f} |")
    lines.append("")
    lines.append("## 4. 残存クリップの合計時間")
    lines.append("")
    lines.append(f"- 合計: {summary.total_duration_sec:,.1f} 秒")
    lines.append(f"- 合計: {hours:,.2f} 時間")
    if summary.count:
        lines.append(
            f"- 1クリップあたり平均: {summary.total_duration_sec / summary.count:.3f} 秒"
        )
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated", type=Path, default=VALIDATED_TSV)
    parser.add_argument("--durations", type=Path, default=DURATIONS_TSV)
    parser.add_argument("--out-data", type=Path, default=OUT_DATA)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument(
        "--no-training-filter",
        action="store_true",
        help="学習専用の文除外を適用しない（評価セット用。既定は適用する）",
    )
    args = parser.parse_args()

    for path in (args.validated, args.durations):
        if not path.exists():
            print(f"[エラー] 入力が見つからない: {path}", flush=True)
            return 1

    apply_training_filter = not args.no_training_filter
    started = time.time()
    print(f"[開始] validated={args.validated}", flush=True)
    print(f"[開始] 学習専用フィルタ適用={apply_training_filter}", flush=True)

    records, stats = build_clip_index(
        args.validated,
        args.durations,
        audio_dir=AUDIO_DIR_REL,
        apply_training_filter=apply_training_filter,
        progress=make_progress(started),
    )
    print(f"[完了] 読み込みと除外 経過 {time.time() - started:.1f}秒", flush=True)
    for key, value in stats.as_dict().items():
        print(f"[統計] {key}={value}", flush=True)

    data_path = write_records(records, args.out_data)
    print(f"[出力] {data_path}", flush=True)

    summary: RateSummary = summarize_mora_per_second(records)
    write_report(args.out_md, stats, summary, data_path, apply_training_filter)
    print(f"[出力] {args.out_md}", flush=True)
    print(
        f"[要約] 残存={stats.kept:,} 平均={summary.mean:.3f} 中央値={summary.median:.3f} "
        f"合計時間={summary.total_duration_sec / 3600.0:.2f}時間",
        flush=True,
    )
    print(f"[終了] 総経過 {time.time() - started:.1f}秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
