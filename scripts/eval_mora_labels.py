"""人手注釈の仮名から数えたモーラ数と、pyopenjtalkでの推定値を比較して測定する。

docs/plan.md 第1段階「1-1 測定」に対応する。

- 原文: data/jsut/basic5000/transcript_utf8.txt（text_level1 に対応）
- 人手注釈: data/jsut-label/text_kana/basic5000.yaml
- 主の正解は kana_level0、参考値として kana_level2 でも同じ測定を行う

出力:
- results/mora_label_eval.md
- results/mora_mismatch.tsv（主の正解で不一致だった全文、差の絶対値の降順）
"""

from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.labels.mora import (  # noqa: E402
    _KANA_PATTERN,
    count_mora,
    count_mora_from_kana,
    has_unconverted,
    to_kana,
)

TRANSCRIPT = ROOT / "data" / "jsut" / "basic5000" / "transcript_utf8.txt"
LABEL_YAML = ROOT / "data" / "jsut-label" / "text_kana" / "basic5000.yaml"
OUT_MD = ROOT / "results" / "mora_label_eval.md"
OUT_TSV = ROOT / "results" / "mora_mismatch.tsv"

# ひらがな（ぁ-ゖ）をカタカナへ移すコードポイント差。
_HIRAGANA_START = ord("ぁ")
_HIRAGANA_END = ord("ゖ")
_TO_KATAKANA_OFFSET = ord("ァ") - ord("ぁ")

LEVELS = ("kana_level0", "kana_level2")


def hiragana_to_katakana(text: str) -> str:
    """ひらがなをカタカナへ変換する。それ以外の文字はそのまま残す。"""
    return "".join(
        chr(ord(ch) + _TO_KATAKANA_OFFSET)
        if _HIRAGANA_START <= ord(ch) <= _HIRAGANA_END
        else ch
        for ch in text
    )


def normalize_annotation(kana: str) -> str:
    """人手注釈の仮名を、モーラ数を数えられる形（カタカナのみ）に整える。

    1. NFKC正規化（半角カタカナ・全角記号を正規化する）
    2. ひらがな→カタカナ変換
    3. カタカナ・長音記号以外の文字（読点「、」、全角カンマ「，」、
       空白、その他の記号）をすべて除去する
    """
    text = unicodedata.normalize("NFKC", kana)
    text = hiragana_to_katakana(text)
    return "".join(ch for ch in text if _KANA_PATTERN.fullmatch(ch) is not None)


def load_transcript() -> dict[str, str]:
    records: dict[str, str] = {}
    with TRANSCRIPT.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, _, text = line.partition(":")
            records[utt_id] = text
    return records


def load_labels() -> dict[str, dict[str, str]]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with LABEL_YAML.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=loader)


def one_line(text: str) -> str:
    """TSVのフィールドに入れるため、タブ・改行を空白へ置換する。"""
    return " ".join(text.split())


def bucket(diff: int) -> str:
    return str(diff) if diff <= 3 else "4以上"


def main() -> None:
    transcript = load_transcript()
    labels = load_labels()

    missing_in_transcript = sorted(set(labels) - set(transcript))
    missing_in_labels = sorted(set(transcript) - set(labels))
    common_ids = sorted(set(transcript) & set(labels))

    # 原文側の推定は1回だけ計算して両レベルで共有する。
    estimates: dict[str, tuple[str, int]] = {}
    unconverted_ids: list[str] = []
    for utt_id in common_ids:
        kana = to_kana(transcript[utt_id])
        estimates[utt_id] = (kana, count_mora_from_kana(kana))
        if has_unconverted(kana):
            unconverted_ids.append(utt_id)

    # count_mora と count_mora_from_kana(to_kana(...)) の一致を念のため確認する。
    sample_id = common_ids[0]
    assert count_mora(transcript[sample_id]) == estimates[sample_id][1]

    stats: dict[str, dict] = {}
    rows_main: list[tuple[int, str, str, str, str, int, int]] = []

    for level in LEVELS:
        diffs: list[int] = []
        counter: Counter[str] = Counter()
        total_true = 0
        total_est = 0
        total_abs = 0
        exact = 0
        missing_field: list[str] = []
        used = 0
        for utt_id in common_ids:
            raw = labels[utt_id].get(level)
            if raw is None:
                missing_field.append(utt_id)
                continue
            gold_kana = normalize_annotation(str(raw))
            gold = count_mora_from_kana(gold_kana)
            est_kana, est = estimates[utt_id]
            diff = est - gold
            abs_diff = abs(diff)
            used += 1
            diffs.append(abs_diff)
            counter[bucket(abs_diff)] += 1
            total_true += gold
            total_est += est
            total_abs += abs_diff
            if abs_diff == 0:
                exact += 1
            elif level == "kana_level0":
                rows_main.append(
                    (
                        abs_diff,
                        utt_id,
                        one_line(transcript[utt_id]),
                        one_line(str(raw)),
                        one_line(est_kana),
                        gold,
                        est,
                    )
                )
        stats[level] = {
            "used": used,
            "exact": exact,
            "mean_abs": total_abs / used if used else 0.0,
            "counter": counter,
            "mean_gold": total_true / used if used else 0.0,
            "mean_est": total_est / used if used else 0.0,
            "total_true": total_true,
            "total_abs": total_abs,
            "rel": total_abs / total_true if total_true else 0.0,
            "missing_field": missing_field,
        }

    # 差の絶対値の降順。同じ差の中では文IDの昇順で安定させる。
    rows_main.sort(key=lambda row: (-row[0], row[1]))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8") as handle:
        handle.write(
            "文ID\t原文\t人手注釈の仮名\tpyopenjtalkの読み\t正解モーラ数\t推定モーラ数\t差\n"
        )
        for abs_diff, utt_id, text, gold_kana, est_kana, gold, est in rows_main:
            handle.write(
                f"{utt_id}\t{text}\t{gold_kana}\t{est_kana}\t{gold}\t{est}\t{est - gold}\n"
            )

    def table(level: str) -> str:
        s = stats[level]
        used = s["used"]
        lines = [
            f"- 対象文数: {used}",
            f"- 完全一致率: {s['exact'] / used:.4f}（{s['exact']} / {used}）",
            f"- 差の絶対値の平均: {s['mean_abs']:.4f} モーラ",
            "- 差の分布（差の絶対値）:",
            "",
            "| 差の絶対値 | 件数 | 割合 |",
            "| --- | --- | --- |",
        ]
        for key in ("0", "1", "2", "3", "4以上"):
            n = s["counter"].get(key, 0)
            lines.append(f"| {key} | {n} | {n / used:.4f} |")
        lines += [
            "",
            f"- 文あたり平均モーラ数（正解側 {level}）: {s['mean_gold']:.3f}",
            f"- 文あたり平均モーラ数（推定側 pyopenjtalk）: {s['mean_est']:.3f}",
            f"- 相対誤差（差の絶対値の合計 {s['total_abs']} ÷ 正解モーラ数の合計"
            f" {s['total_true']}）: {s['rel']:.4f}",
        ]
        if s["missing_field"]:
            lines.append(
                f"- {level} が欠けていた文: {len(s['missing_field'])} 件"
                f"（{', '.join(s['missing_field'][:10])}）"
            )
        else:
            lines.append(f"- {level} が欠けていた文: 0 件")
        return "\n".join(lines)

    n_common = len(common_ids)
    md = f"""# モーラ数ラベルの測定（JSUT BASIC5000）

pyopenjtalkでテキストから数えたモーラ数と、人手注釈の仮名から数えた値の一致を測定した。

- 原文（推定の入力）: `data/jsut/basic5000/transcript_utf8.txt`（{len(transcript)} 行、`text_level1` に対応）
- 人手注釈: `data/jsut-label/text_kana/basic5000.yaml`（{len(labels)} ID）
- 推定値: 原文に `count_mora`（`to_kana` → `count_mora_from_kana`）を適用した値
- 生成スクリプト: `scripts/eval_mora_labels.py`

## 正解の定義

- **主の正解は `kana_level0`**（原文に対応する人手注釈の仮名）。`transcript_utf8.txt` の原文は
  `text_level1` に対応するため、読み変換の品質測定にはこちらを主とする。
- **参考値は `kana_level2`**（実発話文 `text_level2` に対応する仮名）。実発話に合わせた読みの差が
  どれだけ含まれるかを見るために併記する。

## 主（正解 = kana_level0）

{table("kana_level0")}

## 参考（正解 = kana_level2）

{table("kana_level2")}

## 前処理

推定側（原文 → 読み）:

1. `unicodedata.normalize("NFKC", text)` を適用する（`src/spkrate/labels/mora.py` の `to_kana` の先頭。
   全角ラテン文字・全角数字・丸囲み数字・半角カタカナを正規化してから読み変換に渡す）。
2. `pyopenjtalk.g2p(..., kana=True)` でカタカナ読みへ変換する。
3. Unicodeカテゴリが P（約物）・Z（空白）・C（制御）の文字を除去する（`_drop_non_reading`）。
   読み変換に失敗して残った文字（ラテン文字など）は `has_unconverted` で検出できるよう残す。
4. `count_mora_from_kana` はカタカナ（ァ-ヺ）と長音記号のみを数える。小書き文字
   「ャュョァィゥェォヮ」は直前のかなと合わせて1モーラ、結合先がなければ単独1モーラで警告。

正解側（人手注釈の仮名 → モーラ数）:

1. `unicodedata.normalize("NFKC", kana)` を適用する（半角カタカナ・全角記号の正規化）。
2. ひらがな（ぁ-ゖ, U+3041–U+3096）をカタカナへ変換する（コードポイント +0x60）。
   長音記号「ー」(U+30FC) はそのまま残る。
3. カタカナ（ァ-ヺ）と長音記号以外の文字をすべて除去する。読点「、」、全角カンマ「，」（ポーズ）、
   半角・全角空白、その他の記号がこれで落ちる。
4. 整えた文字列に `count_mora_from_kana` を適用する（推定側と同じ数え方）。

## IDの対応

- 両方に存在したID: {n_common}
- 注釈YAMLにあり原文（transcript_utf8.txt）に無かったID: {len(missing_in_transcript)}
- 原文にあり注釈YAMLに無かったID: {len(missing_in_labels)}

## 参考: 読み変換の失敗

- `has_unconverted` が True になった文（読み変換後にカタカナ・長音記号以外の文字が残った文）:
  {len(unconverted_ids)} 件 / {n_common} 件（{len(unconverted_ids) / n_common:.4f}）

## 不一致の一覧

主の正解（`kana_level0`）で不一致だった全文を、差の絶対値の降順で
`results/mora_mismatch.tsv` に出力した（{len(rows_main)} 行、ヘッダ行を除く）。
列は 文ID / 原文 / 人手注釈の仮名 / pyopenjtalkの読み / 正解モーラ数 / 推定モーラ数 / 差。
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_TSV} ({len(rows_main)} rows)")
    for level in LEVELS:
        s = stats[level]
        print(
            f"{level}: n={s['used']} exact={s['exact'] / s['used']:.4f} "
            f"mae={s['mean_abs']:.4f} rel={s['rel']:.4f}"
        )
    print(f"unconverted={len(unconverted_ids)}")


if __name__ == "__main__":
    main()
