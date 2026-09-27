"""実環境評価の録音（docs/plan.md 第9段階、docs/directives/2026-09-27.md 系統C）。

scripts/record_eval.py の中身のうち、マイクに触れない部分をここに置き、単体テストで確かめる。

- 文の読み込み（sentences_labeled.tsv または sentences.txt）
- ファイル名の組み立て（条件と文の番号・IDを入れる）
- 16kHz・モノラル・16bit PCM での書き出し（録り直しは同じファイルへの上書き）
- 1文ずつの録音の進行（Enter で開始・終了、r で録り直し、q で終了）

実際の録音は Recorder（start と stop を持つもの）に任せ、テストでは偽物を渡す。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

TARGET_SR = 16000

# 条件の値（ファイル名に入れる）と画面に出す名前。
MICS: dict[str, str] = {
    "builtin": "内蔵マイク",
    "earphone": "イヤホンマイク",
    "far": "対面で1メートル以上離れた位置",
}
RATES: dict[str, str] = {
    "normal": "通常",
    "fast": "速め",
    "veryfast": "かなり速め",
}

# 録音後に知らせる目安（信号の値だけを見る。音声を聴く判断はしない）。
CLIP_PEAK = 0.999
MIN_SECONDS = 0.5


@dataclass(frozen=True)
class Sentence:
    index: int  # 1始まりの番号
    sentence_id: str
    text: str
    kana: str = ""


def load_sentences(path: Path) -> list[Sentence]:
    """読み上げる文を読む。

    - .tsv: 見出し行つきで sentence_id・text 列（kana 列があれば読みとして使う）
    - それ以外: 「文ID:文」を1行1文
    """
    path = Path(path)
    items: list[tuple[str, str, str]] = []
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            for col in ("sentence_id", "text"):
                if col not in fields:
                    raise ValueError(f"{path} に列 {col} がない")
            for row in reader:
                items.append((row["sentence_id"], row["text"], row.get("kana") or ""))
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                sentence_id, sep, text = line.partition(":")
                if not sep or not sentence_id or not text:
                    raise ValueError(f"「文ID:文」の形でない行: {line!r}")
                items.append((sentence_id, text, ""))
    if not items:
        raise ValueError(f"{path} に文がない")
    ids = [sid for sid, _, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} に同じ文IDが重複している")
    for sid in ids:
        _check_token(sid, "文ID")
    return [
        Sentence(index=i, sentence_id=sid, text=text, kana=kana)
        for i, (sid, text, kana) in enumerate(items, start=1)
    ]


def _check_token(value: str, what: str) -> None:
    """ファイル名に入れる値に、パスの区切りや空白が無いことを確かめる。"""
    if not value or any(ch in value for ch in "/\\ \t:"):
        raise ValueError(f"{what}に使えない文字を含む: {value!r}")


def validate_condition(mic: str, rate: str) -> None:
    if mic not in MICS:
        raise ValueError(f"マイクの種類は {list(MICS)} のどれか: {mic!r}")
    if rate not in RATES:
        raise ValueError(f"話速は {list(RATES)} のどれか: {rate!r}")


def build_filename(mic: str, rate: str, sentence: Sentence) -> str:
    """例: builtin_normal_01_BASIC5000_0036.wav"""
    validate_condition(mic, rate)
    _check_token(sentence.sentence_id, "文ID")
    return f"{mic}_{rate}_{sentence.index:02d}_{sentence.sentence_id}.wav"


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """(frames,) または (frames, channels) の音声を 16kHz・モノラルの float32 にする。"""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"音声の次元が不正: {audio.shape}")
    audio = audio.astype(np.float32, copy=False)
    if sr <= 0:
        raise ValueError(f"標本化周波数が不正: {sr}")
    if sr != TARGET_SR and audio.size > 0:
        from scipy.signal import resample_poly

        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return audio


def save_wav(path: Path, audio: np.ndarray, sr: int) -> np.ndarray:
    """16kHz・モノラル・16bit PCM で書き出す。既存のファイルは置き換える。

    途中で止まっても前の録音が壊れないよう、一時ファイルに書いてから置き換える。
    書き出した（変換後の）音声を返す。
    """
    import soundfile as sf

    path = Path(path)
    mono = np.clip(to_mono_16k(audio, sr), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    sf.write(tmp, mono, TARGET_SR, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)
    return mono


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop(self) -> tuple[np.ndarray, int]: ...


def run_session(
    sentences: list[Sentence],
    mic: str,
    rate: str,
    out_dir: Path,
    recorder: Recorder,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    start: int = 1,
) -> list[Path]:
    """1文ずつ表示して録音する。保存したファイルのパス（同じ文は1回）を返す。

    操作: Enter で録音開始、Enter で録音終了して保存。保存の後、Enter で次の文、
    r で同じ文を録り直し（同じファイルに上書き）、q で終了。
    """
    validate_condition(mic, rate)
    if not 1 <= start <= len(sentences):
        raise ValueError(f"開始番号は 1〜{len(sentences)}: {start}")
    out_dir = Path(out_dir)
    saved: list[Path] = []
    total = len(sentences)
    print_fn(f"条件: マイク={MICS[mic]}（{mic}）、話速={RATES[rate]}（{rate}）。保存先: {out_dir}")
    for sentence in sentences[start - 1 :]:
        path = out_dir / build_filename(mic, rate, sentence)
        while True:
            print_fn("")
            print_fn(f"[{sentence.index}/{total}] {sentence.sentence_id}（話速: {RATES[rate]}）")
            print_fn(f"  {sentence.text}")
            if sentence.kana:
                print_fn(f"  （読み: {sentence.kana}）")
            if path.exists():
                print_fn(f"  既に {path.name} がある。録音すると上書きする。")
            cmd = input_fn("Enter: 録音開始 / q: 終了 > ").strip().lower()
            if cmd == "q":
                return saved
            recorder.start()
            input_fn("● 録音中… Enter: 録音終了 > ")
            audio, sr = recorder.stop()
            mono = save_wav(path, audio, sr)
            if path not in saved:
                saved.append(path)
            seconds = mono.size / TARGET_SR
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            print_fn(f"  保存: {path.name}（{seconds:.1f}秒、最大振幅 {peak:.2f}）")
            if seconds < MIN_SECONDS:
                print_fn("  注意: 録音が短い。r で録り直せる。")
            if peak >= CLIP_PEAK:
                print_fn("  注意: 音が割れている可能性がある（最大振幅が上限）。r で録り直せる。")
            cmd = input_fn("Enter: 次の文へ / r: 録り直し / q: 終了 > ").strip().lower()
            if cmd == "r":
                continue
            if cmd == "q":
                return saved
            break
    print_fn("")
    print_fn(f"この条件の {total} 文が終わった。")
    return saved
