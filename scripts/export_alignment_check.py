"""人間の聴取確認用に、アライメント結果を wav と Audacity ラベルで書き出す（タスク4-2）。

dev のアライメント成功クリップから、clips.jsonl の mora_per_second による仕様の4つの話速帯
（4未満、4以上6未満、6以上8未満、8以上）ごとに無作為に抽出する（種固定）。
30件は4で割り切れないため、配分は 4未満 7件、4以上6未満 8件、6以上8未満 8件、8以上 7件とする
（端の2帯を1件ずつ減らす）。

出力（results/alignment_check/）:
- <clip_id>.wav: 16kHz モノラル PCM16（.gitignore で除外。Common Voice は再共有禁止）
- <clip_id>.txt: Audacity のラベル（1行に 開始秒<TAB>終了秒<TAB>かな）。モーラごと
- index.tsv: clip_id・話速帯・毎秒モーラ数・モーラ数・長さ・文（テキストのみ）

configs/splits/test.json は使わない。

使い方:
    uv run python scripts/export_alignment_check.py --seed 20260926
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf

from spkrate.eval.audio import load_audio
from spkrate.eval.metrics import band_of

ALLOCATION = {"under4": 7, "4to6": 8, "6to8": 8, "over8": 7}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignments", default="data/processed/alignments/dev.jsonl")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--out-dir", default="results/alignment_check")
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    ok = {}
    for line in Path(args.alignments).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["ok"]:
            ok[r["clip_id"]] = r
    by_band: dict[str, list[dict]] = {k: [] for k in ALLOCATION}
    with Path(args.clips).open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["clip_id"] in ok:
                by_band[band_of(c["mora_per_second"])].append(c)

    rng = random.Random(args.seed)
    chosen: list[tuple[str, dict]] = []
    for band, n in ALLOCATION.items():
        pool = sorted(by_band[band], key=lambda c: c["clip_id"])
        chosen.extend((band, c) for c in rng.sample(pool, n))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = ["clip_id\tband\tmora_per_second\tmora\tduration_sec\tsentence"]
    for band, c in chosen:
        wav, sr = load_audio(c["audio_path"])
        sf.write(out / f"{c['clip_id']}.wav", wav, sr, subtype="PCM_16")
        lines = [f"{m['start']:.3f}\t{m['end']:.3f}\t{m['kana']}" for m in ok[c["clip_id"]]["moras"]]
        (out / f"{c['clip_id']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        index.append(
            f"{c['clip_id']}\t{band}\t{c['mora_per_second']:.3f}\t{c['mora']}\t{c['duration_sec']:.3f}\t{c['sentence']}"
        )
    (out / "index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    print("\n".join(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
