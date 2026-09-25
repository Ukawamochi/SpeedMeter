"""no_speech_suspect のクリップ（37件を除く）から無作為に選び、聴取用の wav を書く（追加指示 2026-09-26 の1）。

作り方は listening/1_誤り事例_37件/ と同じ: 16kHz モノラル PCM16、ピークが −6dBFS になるよう
利得を掛け、前に0.5秒・後に1秒の無音を足す。ファイル名は「番号_文の先頭.wav」（文から記号・空白を
除いた先頭18文字）。listening/*/ は .gitignore で除外される（Common Voice は再共有禁止）。
記録用の表 results/no_speech_check.md の行（#・clip_id・文・増幅量(dB)）も書き出す。

使い方:
    uv run python scripts/export_no_speech_listening.py --num 20 --seed 20260926
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from spkrate.eval.audio import load_audio

SAMPLE_RATE = 16000
PEAK_DBFS = -6.0
PAD_BEFORE_SEC = 0.5
PAD_AFTER_SEC = 1.0
NAME_CHARS = 18


_DROP = set("、。，．,.！？!?/\\")


def name_head(sentence: str, n: int = NAME_CHARS) -> str:
    """文から句読点・疑問符・感嘆符・空白・パス区切りを除いた先頭 n 文字（既存の listening/ と同じ）。

    「・」「『」などはそのまま残す（listening/1・2 のファイル名に合わせた）。
    """
    kept = [c for c in sentence if c not in _DROP and not c.isspace()]
    return "".join(kept)[:n]


def normalize_for_listening(wav: np.ndarray) -> tuple[np.ndarray, float]:
    """ピーク −6dBFS に揃え、前後に無音を足す。戻り値は（波形、掛けた利得 dB）。"""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    gain_db = PEAK_DBFS - 20.0 * np.log10(peak) if peak > 0 else 0.0
    y = wav.astype(np.float64) * (10.0 ** (gain_db / 20.0))
    y = np.concatenate([
        np.zeros(int(PAD_BEFORE_SEC * SAMPLE_RATE)), y, np.zeros(int(PAD_AFTER_SEC * SAMPLE_RATE))
    ])
    return y.astype(np.float32), float(gain_db)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspect", default="results/no_speech_suspect_dev.tsv")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--out-dir", default="listening/3_無発話疑い_20件")
    ap.add_argument("--rows-out", default="data/processed/no_speech/listening_rows.tsv")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    with open(args.suspect, encoding="utf-8") as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    pool = sorted(r["clip_id"] for r in rows if r["known_no_speech"] == "0")
    chosen = random.Random(args.seed).sample(pool, args.num)
    clips = {}
    with open(args.clips, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["clip_id"] in chosen:
                clips[r["clip_id"]] = r

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for number, cid in enumerate(chosen, 1):
        rec = clips[cid]
        wav, _ = load_audio(rec["audio_path"])
        y, gain = normalize_for_listening(wav)
        name = f"{number:02d}_{name_head(rec['sentence'])}.wav"
        sf.write(out_dir / name, y, SAMPLE_RATE, subtype="PCM_16")
        table.append((number, cid, rec["sentence"], f"{gain:.1f}", name))
    Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.rows_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(("#", "clip_id", "sentence", "gain_db", "file"))
        w.writerows(table)
    print(f"母集団 {len(pool)} 件から種 {args.seed} で {len(chosen)} 件 → {out_dir}")
    for row in table:
        print("\t".join(map(str, row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
