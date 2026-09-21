"""検証セット（dev）で誤差の大きい上位100件を抽出する（docs/PLAN.md 第6段階 6-1）。

入力は `runs/exp001/predictions_dev_full.json`（第5段階5-3の初回学習の最良チェックポイントを
dev 全29,518件で評価した予測値。値は**毎秒モーラ数**）と `data/processed/clips.jsonl`
（正解のメタデータ）である。前向き計算はここでは行わない。

## 誤差の定義

**毎秒モーラ数の絶対誤差** `|推定毎秒モーラ数 − 正解毎秒モーラ数|` を順位付けに使う。
docs/spec.md の評価指標の第1項が「毎秒モーラ数の平均絶対誤差」であり、表示する量もこれなので、
誤差の大きい事例＝この指標を押し上げている事例、という対応をとるためである。
モーラ数そのものの絶対誤差 `|推定モーラ数 − 正解モーラ数|` も列として同時に出す
（こちらはクリップが長いほど大きくなりやすく、順位付けには使わない）。

## 出力

- `results/error_cases/<clip_id>.mp3` … 音声のコピー（元ファイルは読むだけ）
- `results/error_cases/index.tsv` … 一覧
- `results/error_cases/README.md` … 誤差の定義と、音声をコミットしない理由

**音声は Common Voice の再配布にあたるためコミットしない**（`.gitignore` に追加済み）。

実行:
    uv run python scripts/extract_errors.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from spkrate.data.splits import load_clip_records, load_split

PREDICTIONS = "runs/exp001/predictions_dev_full.json"
CLIPS_JSONL = "data/processed/clips.jsonl"
DEV_SPLIT = "configs/splits/dev.json"
OUTPUT_DIR = "results/error_cases"
TOP_N = 100

#: index.tsv の列。誤差の定義は abs_error_mora_per_second（順位付けに使う方）である。
COLUMNS = (
    "rank",
    "clip_id",
    "client_id",
    "audio_file",
    "duration_sec",
    "mora_true",
    "mora_pred",
    "mora_per_second_true",
    "mora_per_second_pred",
    "abs_error_mora_per_second",
    "abs_error_mora",
    "signed_error_mora_per_second",
    "band_true",
    "sentence",
    "kana",
)


@dataclass(frozen=True)
class ErrorCase:
    """1件の誤り事例。"""

    clip_id: str
    client_id: str
    audio_path: str
    sentence: str
    kana: str
    mora_true: float
    mora_pred: float
    duration_sec: float
    mps_true: float
    mps_pred: float

    @property
    def abs_error_mps(self) -> float:
        """毎秒モーラ数の絶対誤差（順位付けに使う誤差）。"""
        return abs(self.mps_pred - self.mps_true)

    @property
    def abs_error_mora(self) -> float:
        """モーラ数の絶対誤差（参考列）。"""
        return abs(self.mora_pred - self.mora_true)

    @property
    def signed_error_mps(self) -> float:
        """符号つきの毎秒モーラ数の誤差（推定 − 正解）。"""
        return self.mps_pred - self.mps_true


def band_of(mora_per_second: float) -> str:
    """docs/spec.md の話速帯（4未満 / 4以上6未満 / 6以上8未満 / 8以上）。"""
    if mora_per_second < 4.0:
        return "under4"
    if mora_per_second < 6.0:
        return "4to6"
    if mora_per_second < 8.0:
        return "6to8"
    return "over8"


def build_cases(
    predictions: dict[str, float], clips_jsonl: str | Path, dev_split: str | Path
) -> list[ErrorCase]:
    """予測値と clips.jsonl を突き合わせて誤り事例を作る（dev の話者のみ）。"""
    allowed = set(load_split(dev_split))
    cases: list[ErrorCase] = []
    for record in load_clip_records(clips_jsonl):
        if record.client_id not in allowed:
            continue
        predicted_mps = predictions.get(record.clip_id)
        if predicted_mps is None:
            continue
        cases.append(
            ErrorCase(
                clip_id=record.clip_id,
                client_id=record.client_id,
                audio_path=record.audio_path,
                sentence=record.sentence,
                kana=record.kana,
                mora_true=float(record.mora),
                mora_pred=float(predicted_mps) * float(record.duration_sec),
                duration_sec=float(record.duration_sec),
                mps_true=float(record.mora_per_second),
                mps_pred=float(predicted_mps),
            )
        )
    if len(cases) != len(predictions):
        raise ValueError(
            f"予測値 {len(predictions)} 件に対し突き合わせできたのは {len(cases)} 件。"
            "clips.jsonl と予測の対応を確認すること"
        )
    return cases


def select_top(cases: list[ErrorCase], top_n: int = TOP_N) -> list[ErrorCase]:
    """毎秒モーラ数の絶対誤差の大きい順に上位 ``top_n`` 件（同値は clip_id 順で安定化）。"""
    return sorted(cases, key=lambda case: (-case.abs_error_mps, case.clip_id))[:top_n]


def format_row(rank: int, case: ErrorCase) -> str:
    """index.tsv の1行にする。タブと改行は原文から除く。"""

    def clean(text: str) -> str:
        return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

    values = [
        str(rank),
        case.clip_id,
        case.client_id,
        f"{case.clip_id}.mp3",
        f"{case.duration_sec:.3f}",
        f"{case.mora_true:.0f}",
        f"{case.mora_pred:.3f}",
        f"{case.mps_true:.4f}",
        f"{case.mps_pred:.4f}",
        f"{case.abs_error_mps:.4f}",
        f"{case.abs_error_mora:.3f}",
        f"{case.signed_error_mps:+.4f}",
        band_of(case.mps_true),
        clean(case.sentence),
        clean(case.kana),
    ]
    return "\t".join(values)


README_TEXT = """\
# 誤り事例（docs/PLAN.md 第6段階 6-1）

生成: `uv run python scripts/extract_errors.py`

## 対象

- 検証セット dev の全29,518件。予測は `runs/exp001/predictions_dev_full.json`
  （第5段階5-3の初回学習 exp001 の最良チェックポイント、方式A、拡張なし）
- `configs/splits/test.json` は使っていない

## 誤差の定義

順位付けに使った誤差は **毎秒モーラ数の絶対誤差** である。

    誤差 = |推定毎秒モーラ数 − 正解毎秒モーラ数|

`docs/spec.md` の評価指標の第1項が「毎秒モーラ数の平均絶対誤差」であり、実際に表示する量も
毎秒モーラ数なので、この指標を押し上げている事例をそのまま取り出せる定義を選んだ。
参考として **モーラ数の絶対誤差** `|推定モーラ数 − 正解モーラ数|` も `abs_error_mora` 列に
出してある（こちらはクリップが長いほど大きくなりやすいため、順位付けには使っていない）。

推定モーラ数は `推定毎秒モーラ数 × クリップ長` で戻した値である（モデルの出力はモーラ数だが、
保存されている予測値が毎秒モーラ数のため）。

## index.tsv の列

| 列 | 内容 |
| --- | --- |
| `rank` | 誤差の大きい順の順位（1が最大） |
| `clip_id` | クリップの識別子 |
| `client_id` | 話者の識別子 |
| `audio_file` | 同ディレクトリにコピーした音声のファイル名 |
| `duration_sec` | クリップ長（秒） |
| `mora_true` | 正解モーラ数 |
| `mora_pred` | 推定モーラ数 |
| `mora_per_second_true` | 正解の毎秒モーラ数 |
| `mora_per_second_pred` | 推定の毎秒モーラ数 |
| `abs_error_mora_per_second` | **順位付けに使った誤差**（毎秒モーラ数の絶対誤差） |
| `abs_error_mora` | モーラ数の絶対誤差（参考） |
| `signed_error_mora_per_second` | 符号つきの誤差（推定 − 正解） |
| `band_true` | 正解の話速帯（under4 / 4to6 / 6to8 / over8） |
| `sentence` | 原文 |
| `kana` | カタカナ読み |

## 音声ファイルをコミットしない理由

`docs/PLAN.md`「データセットと役割」のとおり **Common Voice は再ホスト・再共有が禁止**されている。
本リポジトリは Public で運用しているため、`results/error_cases/` に置いた音声をコミットすると
Common Voice の音声を再配布することになる。

対処として `.gitignore` に次を追加し、**音声だけを追跡対象から外した**。

    /results/error_cases/*.mp3

`index.tsv` と本 README.md はテキストなのでコミットする。音声は上のコマンドを実行すれば
`data/common_voice_ja/clips/` から手元に再生成できる（元ファイルは読むだけで変更しない）。
第6段階6-3の聴取を行う人は、このディレクトリのコピーを聴くこと。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=PREDICTIONS)
    parser.add_argument("--clips", default=CLIPS_JSONL)
    parser.add_argument("--dev-split", default=DEV_SPLIT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument(
        "--no-audio", action="store_true", help="音声のコピーを行わない（一覧だけ作る）"
    )
    args = parser.parse_args(argv)

    predictions: dict[str, float] = json.loads(
        Path(args.predictions).read_text(encoding="utf-8")
    )
    cases = build_cases(predictions, args.clips, args.dev_split)
    print(f"dev 突き合わせ件数: {len(cases)}", flush=True)
    top = select_top(cases, args.top_n)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []
    if not args.no_audio:
        for case in top:
            source = Path(case.audio_path)
            if not source.is_file():
                missing.append(case.clip_id)
                continue
            shutil.copy2(source, output_dir / f"{case.clip_id}.mp3")
            copied += 1

    lines = [
        "# 誤差の定義: abs_error_mora_per_second = |推定毎秒モーラ数 - 正解毎秒モーラ数|"
        "（順位付けに使用）。詳細は README.md",
        "\t".join(COLUMNS),
    ]
    lines.extend(format_row(rank, case) for rank, case in enumerate(top, start=1))
    (output_dir / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")

    errors = [case.abs_error_mps for case in top]
    print(f"上位 {len(top)} 件: 誤差 最大 {max(errors):.4f} / 最小 {min(errors):.4f} mora/s")
    print(f"音声コピー: {copied} 件（欠損 {len(missing)} 件）", flush=True)
    if missing:
        print("欠損した clip_id:", ", ".join(missing[:10]), flush=True)
    print("EXTRACT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
