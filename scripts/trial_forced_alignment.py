"""強制アライメント手段の比較試行（docs/directives/2026-09-25.md タスク3）。

dev から乱数の種を固定して少数のクリップを選び、次の2候補でモーラ単位の
アライメントを行い、動作・1件あたり処理時間・候補間の境界の差を記録する。

- hiragana: vumichien/wav2vec2-large-xlsr-japanese-hiragana（ひらがな1文字単位のCTC）
            の出力に torchaudio.functional.forced_align を掛ける
- mms:      torchaudio.pipelines.MMS_FA（ローマ字単位のCTC）。かなは本スクリプト内の
            表でモーラごとにローマ字へ変換する

これは選定のための試行であり、本実装（src/spkrate/labels/）ではない。
出力は Git 管理外（既定: data/processed/alignment_trial/）に書く。
configs/splits/test.json は使わない。

使い方:
    uv run python scripts/trial_forced_alignment.py --num-clips 20 --device mps
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

from spkrate.data.splits import load_split
from spkrate.eval.audio import load_audio
from spkrate.labels.mora import count_mora

SEED = 20260925
HIRAGANA_MODEL = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
HIRAGANA_REVISION = "017225bb128a6b1c6de9d58391f891908d69bc7b"
SAMPLE_RATE = 16000
_SMALL_COMBINING = frozenset("ャュョァィゥェォヮ")


# ---------------------------------------------------------------- モーラ分割
def split_mora(kana: str) -> list[str]:
    """カタカナ列をモーラに分ける（count_mora_from_kana と同じ結合規則）。"""
    moras: list[str] = []
    for ch in kana:
        if ch in _SMALL_COMBINING and moras and moras[-1][-1] not in _SMALL_COMBINING:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


# ---------------------------------------------------------------- 候補1: ひらがなCTC
# 語彙に無い文字の置き換え（読みの近い文字へ）。
_HIRA_FALLBACK = {"ゎ": "わ", "ゕ": "か", "ゖ": "け", "ゐ": "い", "ゑ": "え"}


def kata_to_hira(ch: str) -> str:
    code = ord(ch)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    if ch == "ヴ":
        return "ゔ"
    return ch  # 「ー」など


class HiraganaAligner:
    name = "hiragana"

    def __init__(self, device: str) -> None:
        from transformers import Wav2Vec2ForCTC

        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(HIRAGANA_MODEL, revision=HIRAGANA_REVISION).to(device).eval()
        vocab_path = Path(
            __import__("huggingface_hub").hf_hub_download(HIRAGANA_MODEL, "vocab.json", revision=HIRAGANA_REVISION)
        )
        self.vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.blank = self.vocab["[PAD]"]  # CTC の blank は pad_token

    def tokens_for_mora(self, mora: str) -> list[int] | None:
        ids = []
        for ch in mora:
            h = kata_to_hira(ch)
            h = _HIRA_FALLBACK.get(h, h) if h not in self.vocab else h
            if h not in self.vocab:
                return None
            ids.append(self.vocab[h])
        return ids

    @torch.inference_mode()
    def emission(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav)
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)  # do_normalize=True と同じ
        logits = self.model(x[None].to(self.device)).logits
        return torch.log_softmax(logits.float(), dim=-1).cpu()


# ---------------------------------------------------------------- 候補2: MMS_FA
_BASE_ROMA = {
    **dict(zip("アイウエオ", ["a", "i", "u", "e", "o"])),
    **dict(zip("カキクケコ", ["ka", "ki", "ku", "ke", "ko"])),
    **dict(zip("ガギグゲゴ", ["ga", "gi", "gu", "ge", "go"])),
    **dict(zip("サシスセソ", ["sa", "shi", "su", "se", "so"])),
    **dict(zip("ザジズゼゾ", ["za", "ji", "zu", "ze", "zo"])),
    **dict(zip("タチツテト", ["ta", "chi", "tsu", "te", "to"])),
    **dict(zip("ダヂヅデド", ["da", "ji", "zu", "de", "do"])),
    **dict(zip("ナニヌネノ", ["na", "ni", "nu", "ne", "no"])),
    **dict(zip("ハヒフヘホ", ["ha", "hi", "fu", "he", "ho"])),
    **dict(zip("バビブベボ", ["ba", "bi", "bu", "be", "bo"])),
    **dict(zip("パピプペポ", ["pa", "pi", "pu", "pe", "po"])),
    **dict(zip("マミムメモ", ["ma", "mi", "mu", "me", "mo"])),
    **dict(zip("ヤユヨ", ["ya", "yu", "yo"])),
    **dict(zip("ラリルレロ", ["ra", "ri", "ru", "re", "ro"])),
    **dict(zip("ワヰヱヲ", ["wa", "i", "e", "o"])),
    "ン": "n", "ヴ": "vu", "ヵ": "ka", "ヶ": "ke",
    **dict(zip("ァィゥェォャュョヮ", ["a", "i", "u", "e", "o", "ya", "yu", "yo", "wa"])),
}
_SMALL_VOWEL = {"ャ": "a", "ュ": "u", "ョ": "o", "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o", "ヮ": "a"}


def mora_to_roma(mora: str, prev_roma: str | None, next_mora: str | None) -> str | None:
    if mora == "ー":
        return prev_roma[-1] if prev_roma and prev_roma[-1] in "aiueo" else None
    if mora == "ッ":
        if next_mora is None:
            return None
        nxt = mora_to_roma(next_mora, None, None)
        return nxt[0] if nxt and nxt[0] not in "aiueon" else None
    if len(mora) == 1:
        return _BASE_ROMA.get(mora)
    head, small = mora[0], mora[1]
    base = _BASE_ROMA.get(head)
    if base is None:
        return None
    stem = base.rstrip("aiueo") or base
    if small in "ャュョ":
        if stem in ("sh", "ch", "j"):
            return stem + _SMALL_VOWEL[small]
        return stem + "y" + _SMALL_VOWEL[small]
    if small == "ヮ":
        return stem + "wa"
    return stem + _SMALL_VOWEL[small]


class MmsAligner:
    name = "mms"

    def __init__(self, device: str) -> None:
        self.bundle = torchaudio.pipelines.MMS_FA
        self.device = device
        self.model = self.bundle.get_model(with_star=False).to(device).eval()
        self.dict = self.bundle.get_dict(star=None)
        self.blank = self.dict["-"]
        self._romas: list[str | None] = []

    def prepare(self, moras: list[str]) -> list[str | None]:
        romas: list[str | None] = []
        for i, m in enumerate(moras):
            prev = romas[-1] if romas else None
            nxt = moras[i + 1] if i + 1 < len(moras) else None
            romas.append(mora_to_roma(m, prev, nxt))
        return romas

    def tokens_for_roma(self, roma: str | None) -> list[int] | None:
        if not roma or any(c not in self.dict for c in roma):
            return None
        return [self.dict[c] for c in roma]

    @torch.inference_mode()
    def emission(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav)[None].to(self.device)
        emission, _ = self.model(x)
        return torch.log_softmax(emission.float(), dim=-1).cpu()


# ---------------------------------------------------------------- 共通
def align(emission: torch.Tensor, token_groups: list[list[int]], blank: int, wav_len: int):
    """モーラごとのトークン列を連結して強制アライメントし、モーラの (開始秒, 終了秒) を返す。"""
    flat = [t for g in token_groups for t in g]
    targets = torch.tensor([flat], dtype=torch.int32)
    labels, _ = AF.forced_align(emission, targets, blank=blank)
    labels = labels[0].tolist()
    num_frames = emission.shape[1]
    sec_per_frame = wav_len / SAMPLE_RATE / num_frames
    # トークンごとの区間（連続するフレームを1区間に。同一トークンの反復は blank を挟む）
    spans: list[tuple[int, int]] = []
    prev = blank
    for f, lab in enumerate(labels):
        if lab != blank and (lab != prev or not spans):
            spans.append((f, f + 1))
        elif lab != blank and lab == prev:
            spans[-1] = (spans[-1][0], f + 1)
        prev = lab
    if len(spans) != len(flat):
        raise RuntimeError(f"トークン区間数 {len(spans)} != 目標 {len(flat)}")
    out = []
    k = 0
    for g in token_groups:
        s = spans[k][0]
        e = spans[k + len(g) - 1][1]
        out.append((s * sec_per_frame, e * sec_per_frame))
        k += len(g)
    return out


def pick_clips(clips_path: Path, dev_path: Path, num: int) -> list[dict]:
    dev_speakers = set(load_split(dev_path))
    bands: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    with clips_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["client_id"] not in dev_speakers:
                continue
            mps = r["mora_per_second"]
            band = 0 if mps < 4 else 1 if mps < 6 else 2 if mps < 8 else 3
            bands[band].append(r)
    rng = random.Random(SEED)
    per = num // 4
    chosen = []
    for b in range(4):
        chosen += rng.sample(bands[b], per)
    total = sum(len(v) for v in bands.values())
    return chosen, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=20)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--clips", default="data/processed/clips.jsonl")
    ap.add_argument("--dev", default="configs/splits/dev.json")
    ap.add_argument("--out", default="data/processed/alignment_trial")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips, dev_total = pick_clips(Path(args.clips), Path(args.dev), args.num_clips)
    print(f"dev 件数 {dev_total}、試行 {len(clips)} 件", flush=True)

    t0 = time.perf_counter()
    aligners = [HiraganaAligner(args.device), MmsAligner(args.device)]
    print(f"モデル読み込み {time.perf_counter() - t0:.1f}s", flush=True)

    # ウォームアップ（初回のMPSカーネル生成を計時から除く）
    warm = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    for a in aligners:
        a.emission(warm)

    results = []
    for r in clips:
        wav, _ = load_audio(r["audio_path"])
        moras = split_mora(r["kana"])
        rec = {
            "clip_id": r["clip_id"],
            "duration_sec": len(wav) / SAMPLE_RATE,
            "label_mora": r["mora"],
            "count_mora": count_mora(r["sentence"]),
            "split_mora": len(moras),
            "moras": moras,
        }
        for a in aligners:
            t = time.perf_counter()
            try:
                if isinstance(a, MmsAligner):
                    romas = a.prepare(moras)
                    groups = [a.tokens_for_roma(x) for x in romas]
                else:
                    groups = [a.tokens_for_mora(m) for m in moras]
                bad = [moras[i] for i, g in enumerate(groups) if g is None]
                if bad:
                    raise ValueError(f"トークンに変換できないモーラ {bad}")
                em = a.emission(wav)
                spans = align(em, groups, a.blank, len(wav))
                rec[a.name] = {"ok": True, "spans": spans, "num_mora": len(spans)}
            except Exception as e:  # noqa: BLE001 試行では失敗理由を記録して続ける
                rec[a.name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            rec[a.name]["sec"] = time.perf_counter() - t
        results.append(rec)
        print(
            r["clip_id"],
            f"{rec['duration_sec']:.2f}s",
            *(f"{a.name}={'ok' if rec[a.name]['ok'] else 'NG'}({rec[a.name]['sec']:.2f}s)" for a in aligners),
            flush=True,
        )

    with (out_dir / "trial.jsonl").open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 集計
    summary = {"dev_total": dev_total, "num_clips": len(results), "device": args.device}
    summary["label_eq_count_mora"] = sum(r["label_mora"] == r["count_mora"] == r["split_mora"] for r in results)
    total_audio = sum(r["duration_sec"] for r in results)
    for a in aligners:
        ok = [r for r in results if r[a.name]["ok"]]
        secs = [r[a.name]["sec"] for r in results]
        summary[a.name] = {
            "ok": len(ok),
            "mora_eq_label": sum(r[a.name]["num_mora"] == r["label_mora"] for r in ok),
            "mean_sec_per_clip": statistics.mean(secs),
            "rtf": sum(secs) / total_audio,
            "est_dev_hours": statistics.mean(secs) * dev_total / 3600,
            "errors": [r[a.name]["error"] for r in results if not r[a.name]["ok"]],
        }
    # 候補間のモーラ境界（開始・終了）の差
    diffs = []
    for r in results:
        if r["hiragana"]["ok"] and r["mms"]["ok"]:
            for (s1, e1), (s2, e2) in zip(r["hiragana"]["spans"], r["mms"]["spans"]):
                diffs += [abs(s1 - s2), abs(e1 - e2)]
    if diffs:
        d = np.array(diffs, dtype=np.float32)
        summary["boundary_absdiff_ms"] = {
            "median": float(np.median(d) * 1000),
            "p90": float(np.quantile(d, 0.9) * 1000),
            "mean": float(d.mean() * 1000),
            "n": int(d.size),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
