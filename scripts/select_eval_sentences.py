"""実環境評価で読み上げる文を JSUT basic5000 から選ぶ（docs/plan.md 9-1）。

条件（docs/directives/2026-09-27.md 系統C）:
- 出典 data/jsut/basic5000/transcript_utf8.txt、正解の読みは
  data/jsut-label/text_kana/basic5000.yaml の kana_level0
- モーラ数 20〜40、has_unconverted が True の文は除く、
  人手注釈と pyopenjtalk のモーラ数が一致する文に限る
- 乱数の種 SEED で 20 文を選ぶ

出力（data/eval_scripts/）:
- sentences.txt: 「文ID:文」を1行1文（scripts/record_eval.py が読む）
- sentences_labeled.tsv: 文ID・文・カタカナ読み・モーラ数
- selection.json: 選び方（条件・乱数の種・入力ファイル）と選んだ文ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.eval.eval_sentences import (  # noqa: E402
    LABEL_LEVEL,
    build_candidates,
    is_eligible,
    load_labels,
    load_transcript,
    select,
)

TRANSCRIPT = ROOT / "data" / "jsut" / "basic5000" / "transcript_utf8.txt"
LABEL_YAML = ROOT / "data" / "jsut-label" / "text_kana" / "basic5000.yaml"
OUT_DIR = ROOT / "data" / "eval_scripts"

SEED = 20260927
N_SENTENCES = 20
MIN_MORA = 20
MAX_MORA = 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    for path in (TRANSCRIPT, LABEL_YAML):
        if not path.is_file():
            sys.exit(f"入力がない: {path}（data/DATASETS.md を参照）")

    transcript = load_transcript(TRANSCRIPT)
    labels = load_labels(LABEL_YAML)
    candidates = build_candidates(transcript, labels)
    eligible = [c for c in candidates if is_eligible(c, MIN_MORA, MAX_MORA)]
    chosen = select(candidates, N_SENTENCES, SEED, MIN_MORA, MAX_MORA)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sentences.txt").open("w", encoding="utf-8") as handle:
        for c in chosen:
            handle.write(f"{c.utt_id}:{c.text}\n")
    with (out_dir / "sentences_labeled.tsv").open("w", encoding="utf-8") as handle:
        handle.write("sentence_id\ttext\tkana\tmora\n")
        for c in chosen:
            handle.write(f"{c.utt_id}\t{c.text}\t{c.kana}\t{c.mora}\n")
    selection = {
        "source_transcript": str(TRANSCRIPT.relative_to(ROOT)),
        "source_transcript_sha256": sha256(TRANSCRIPT),
        "source_labels": str(LABEL_YAML.relative_to(ROOT)),
        "source_labels_sha256": sha256(LABEL_YAML),
        "label_level": LABEL_LEVEL,
        "seed": SEED,
        "n_sentences": N_SENTENCES,
        "min_mora": MIN_MORA,
        "max_mora": MAX_MORA,
        "conditions": [
            f"{MIN_MORA} <= kana_level0 のモーラ数 <= {MAX_MORA}",
            "pyopenjtalk の読みに has_unconverted が True となる文字がない",
            "pyopenjtalk の読みのモーラ数が kana_level0 のモーラ数と一致する",
        ],
        "method": "条件を満たす文を文ID順に並べ、random.Random(seed).sample で選び、文ID順に並べ直す",
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
        "selected_ids": [c.utt_id for c in chosen],
        "selected_mora": [c.mora for c in chosen],
        "script": "scripts/select_eval_sentences.py",
    }
    (out_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    moras = [c.mora for c in chosen]
    print(
        f"候補 {len(candidates)} 文、条件を満たす {len(eligible)} 文から {len(chosen)} 文を選んだ。"
        f"モーラ数 {min(moras)}〜{max(moras)}、平均 {sum(moras) / len(moras):.1f}。出力: {out_dir}"
    )


if __name__ == "__main__":
    main()
