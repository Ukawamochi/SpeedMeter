"""モーラ単位の強制アライメント（docs/directives/2026-09-25.md タスク4-1）。

採用手段と規則は docs/decisions/008-forced-alignment.md による。

- 音響モデル: ひらがなCTC ``vumichien/wav2vec2-large-xlsr-japanese-hiragana``
  （コミット固定）。推論は mps、``torchaudio.functional.forced_align`` は CPU
- モーラ列: clips.jsonl の ``kana`` を ``count_mora_from_kana`` と同じ結合規則で分ける
- かな1文字＝語彙の1トークン（語彙に無い「ゎゕゖゐゑ」は「わかけいえ」に置き換える）
- モーラ区間: 開始＝先頭トークンが最初に出たフレームの開始、
  終了＝末尾トークンが最後に出たフレームの終了。フレーム長は「波形長 ÷ フレーム数」
- 発話区間: 最初のモーラの開始〜最後のモーラの終了
- 失敗（除外）の理由コード: ``mora_mismatch_label``・``unknown_token``・
  ``align_failed``・``mora_mismatch_alignment``

窓の正解モーラ数の数え方（008 4節の「中点で数える」推奨）はここでは扱わない。
"""

from __future__ import annotations

import json
import logging
import random
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from spkrate.labels.mora import _KANA_PATTERN, _SMALL_COMBINING, count_mora

logger = logging.getLogger(__name__)

__all__ = [
    "BLANK_TOKEN",
    "HIRAGANA_MODEL",
    "HIRAGANA_REVISION",
    "REASON_ALIGN_FAILED",
    "REASON_MORA_MISMATCH_ALIGNMENT",
    "REASON_MORA_MISMATCH_LABEL",
    "REASON_UNKNOWN_TOKEN",
    "REASONS",
    "AlignmentError",
    "HiraganaAligner",
    "MoraSpan",
    "align_clip",
    "align_clips_to_jsonl",
    "align_emission",
    "kata_to_hira",
    "min_frames_required",
    "select_dev_clips",
    "split_mora",
    "token_spans",
    "tokens_for_mora",
]

HIRAGANA_MODEL = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
HIRAGANA_REVISION = "017225bb128a6b1c6de9d58391f891908d69bc7b"
BLANK_TOKEN = "[PAD]"  # CTC の blank は pad_token（008 1.2節）
SAMPLE_RATE = 16000

REASON_MORA_MISMATCH_LABEL = "mora_mismatch_label"
REASON_UNKNOWN_TOKEN = "unknown_token"
REASON_ALIGN_FAILED = "align_failed"
REASON_MORA_MISMATCH_ALIGNMENT = "mora_mismatch_alignment"
REASONS = (
    REASON_MORA_MISMATCH_LABEL,
    REASON_UNKNOWN_TOKEN,
    REASON_ALIGN_FAILED,
    REASON_MORA_MISMATCH_ALIGNMENT,
)

# 語彙に無い文字の置き換え（008 3節）。
_HIRA_FALLBACK = {"ゎ": "わ", "ゕ": "か", "ゖ": "け", "ゐ": "い", "ゑ": "え"}


class AlignmentError(Exception):
    """アライメントの失敗。``reason`` は 008 の理由コード。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in REASONS:
            raise ValueError(f"未定義の理由コード: {reason}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class MoraSpan:
    kana: str
    start: float
    end: float


# ---------------------------------------------------------------- モーラ分割
def split_mora(kana: str) -> list[str]:
    """カタカナ読みをモーラに分ける。

    ``count_mora_from_kana`` と同じ規則: カタカナ・長音記号以外は数えずに捨てる。
    小書き「ャュョァィゥェォヮ」は直前の文字がかな（小書き結合文字以外）なら
    その文字と結合し、そうでなければ単独で1モーラ。ヵ・ヶ・ン・ッ・ーは各1モーラ。
    したがって ``len(split_mora(k)) == count_mora_from_kana(k)`` が常に成り立つ。
    """
    moras: list[str] = []
    for index, ch in enumerate(kana):
        if _KANA_PATTERN.fullmatch(ch) is None:
            continue
        if ch in _SMALL_COMBINING and index > 0:
            prev = kana[index - 1]
            if _KANA_PATTERN.fullmatch(prev) is not None and prev not in _SMALL_COMBINING:
                moras[-1] += ch
                continue
        moras.append(ch)
    return moras


# ---------------------------------------------------------------- トークン化
def kata_to_hira(ch: str) -> str:
    """カタカナ1文字をひらがなへ。対応の無い文字（「ー」など）はそのまま返す。"""
    code = ord(ch)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return ch


def tokens_for_mora(mora: str, vocab: dict[str, int]) -> list[int] | None:
    """モーラ（かな1〜2文字）を語彙のトークンID列にする。変換できなければ None。"""
    ids: list[int] = []
    for ch in mora:
        h = kata_to_hira(ch)
        if h not in vocab:
            h = _HIRA_FALLBACK.get(h, h)
        if h not in vocab:
            return None
        ids.append(vocab[h])
    return ids


def min_frames_required(flat_tokens: Sequence[int]) -> int:
    """forced_align に必要な最小フレーム数（トークン数＋連続する同一トークンの数）。"""
    repeats = sum(1 for a, b in zip(flat_tokens, flat_tokens[1:]) if a == b)
    return len(flat_tokens) + repeats


# ---------------------------------------------------------------- アライメント
def token_spans(frame_labels: Sequence[int], blank: int) -> list[tuple[int, int]]:
    """フレームごとのラベル列から、トークンごとの [開始フレーム, 終了フレーム) を得る。

    同一トークンが blank を挟まずに続くフレームは1区間にまとめる。
    blank を挟んだ同一トークンは別のトークン（反復）として扱う。
    """
    spans: list[tuple[int, int]] = []
    prev = blank
    for f, lab in enumerate(frame_labels):
        if lab == blank:
            pass
        elif lab == prev:
            spans[-1] = (spans[-1][0], f + 1)
        else:
            spans.append((f, f + 1))
        prev = lab
    return spans


def _forced_align(emission: torch.Tensor, flat: list[int], blank: int) -> list[int]:
    import torchaudio.functional as AF

    targets = torch.tensor([flat], dtype=torch.int32)
    labels, _ = AF.forced_align(emission, targets, blank=blank)
    return labels[0].tolist()


def align_emission(
    emission: torch.Tensor,
    token_groups: Sequence[Sequence[int]],
    blank: int,
    wav_len: int,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """放出確率（1, フレーム数, 語彙数; log_softmax 済み; CPU）からモーラ区間（秒）を得る。

    Raises:
        AlignmentError: ``align_failed`` または ``mora_mismatch_alignment``。
    """
    flat = [t for g in token_groups for t in g]
    if not flat or any(len(g) == 0 for g in token_groups):
        raise AlignmentError(REASON_ALIGN_FAILED, "空のトークン列")
    num_frames = int(emission.shape[1])
    need = min_frames_required(flat)
    if num_frames < need:
        raise AlignmentError(REASON_ALIGN_FAILED, f"フレーム数 {num_frames} < 必要数 {need}")
    emission = emission.detach().to("cpu", torch.float32)
    try:
        labels = _forced_align(emission, flat, blank)
    except Exception as e:  # noqa: BLE001 forced_align の失敗は理由コードにまとめる
        raise AlignmentError(REASON_ALIGN_FAILED, f"{type(e).__name__}: {e}") from e

    spans = token_spans(labels, blank)
    if len(spans) != len(flat):
        raise AlignmentError(
            REASON_MORA_MISMATCH_ALIGNMENT, f"トークン区間数 {len(spans)} != トークン数 {len(flat)}"
        )
    sec_per_frame = wav_len / sample_rate / num_frames
    out: list[tuple[float, float]] = []
    k = 0
    for g in token_groups:
        s = spans[k][0]
        e = spans[k + len(g) - 1][1]
        out.append((s * sec_per_frame, e * sec_per_frame))
        k += len(g)
    if len(out) != len(token_groups):
        raise AlignmentError(REASON_MORA_MISMATCH_ALIGNMENT, "モーラ区間数が一致しない")
    return out


def align_clip(
    record: dict,
    wav: np.ndarray,
    emission_fn: Callable[[np.ndarray], torch.Tensor],
    vocab: dict[str, int],
    blank: int,
    *,
    count_mora_fn: Callable[[str], int] = count_mora,
) -> dict:
    """1クリップを処理し、出力レコード（成功・失敗とも）を返す。

    record は clips.jsonl の1行（``clip_id``・``kana``・``mora``・``sentence``）。
    """
    clip_id = record["clip_id"]

    def fail(reason: str, detail: str) -> dict:
        return {"clip_id": clip_id, "ok": False, "reason": reason, "detail": detail}

    moras = split_mora(record["kana"])
    label = int(record["mora"])
    counted = int(count_mora_fn(record["sentence"]))
    if not (len(moras) == label == counted):
        return fail(
            REASON_MORA_MISMATCH_LABEL,
            f"かな分割 {len(moras)}, mora列 {label}, count_mora {counted}",
        )
    groups = [tokens_for_mora(m, vocab) for m in moras]
    bad = [m for m, g in zip(moras, groups) if g is None]
    if bad:
        return fail(REASON_UNKNOWN_TOKEN, f"変換できないモーラ {bad}")
    try:
        emission = emission_fn(wav)
        spans = align_emission(emission, groups, blank, len(wav))
    except AlignmentError as e:
        return fail(e.reason, e.detail)
    if len(spans) != label:
        return fail(REASON_MORA_MISMATCH_ALIGNMENT, f"モーラ区間数 {len(spans)} != mora {label}")
    return {
        "clip_id": clip_id,
        "ok": True,
        "duration_sec": len(wav) / SAMPLE_RATE,
        "moras": [{"kana": m, "start": s, "end": e} for m, (s, e) in zip(moras, spans)],
        "speech_start": spans[0][0],
        "speech_end": spans[-1][1],
    }


# ---------------------------------------------------------------- 音響モデル
class HiraganaAligner:
    """ひらがなCTCモデル（コミット固定）の放出確率を求める。

    ``device`` は mps・cuda・cpu。使えないデバイスを指定すると、モデルを読む前に
    ``spkrate.device.DeviceUnavailableError`` で止める（cpu に落とさない）。cuda では
    学習・評価と同じく TF32 を無効にする。実行した計算機の記録は ``environment`` に持つ。
    """

    def __init__(self, device: str = "mps") -> None:
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        from spkrate.device import configure_backends, environment_info, resolve_device

        self.device = resolve_device(device)
        self.environment = environment_info(self.device)
        self.environment["backends"] = configure_backends(self.device)
        self.model = (
            Wav2Vec2ForCTC.from_pretrained(HIRAGANA_MODEL, revision=HIRAGANA_REVISION)
            .to(device)
            .eval()
        )
        vocab_path = hf_hub_download(HIRAGANA_MODEL, "vocab.json", revision=HIRAGANA_REVISION)
        self.vocab: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.blank = self.vocab[BLANK_TOKEN]

    @torch.inference_mode()
    def emission(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)  # feature extractor の do_normalize と同じ
        logits = self.model(x[None].to(self.device)).logits
        return torch.log_softmax(logits.float(), dim=-1).cpu()

    def align(self, record: dict, wav: np.ndarray, **kwargs) -> dict:
        return align_clip(record, wav, self.emission, self.vocab, self.blank, **kwargs)


# ---------------------------------------------------------------- まとめて処理
def select_dev_clips(
    clips_path: str | Path,
    dev_split_path: str | Path,
    *,
    num: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """dev の話者に属するクリップを返す。num を指定すると seed で無作為に num 件選ぶ。"""
    from spkrate.data.splits import load_split

    dev_speakers = set(load_split(dev_split_path))
    records: list[dict] = []
    with Path(clips_path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["client_id"] in dev_speakers:
                records.append(r)
    if num is not None and num < len(records):
        records = random.Random(seed).sample(records, num)
    return records


def _read_done_ids(out_path: Path) -> set[str]:
    """既存の出力から処理済みの clip_id を読む。途中で切れた最終行は切り詰める。"""
    if not out_path.exists():
        return set()
    data = out_path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        logger.warning("出力の最終行が途中で切れているため切り詰める: %s", out_path)
        with out_path.open("r+b") as f:
            f.truncate(cut)
        data = data[:cut]
    done: set[str] = set()
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["clip_id"])
    return done


@dataclass
class BatchStats:
    processed: int = 0
    skipped: int = 0
    ok: int = 0
    failed: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def align_clips_to_jsonl(
    records: Iterable[dict],
    out_path: str | Path,
    align_fn: Callable[[dict, np.ndarray], dict],
    *,
    load_fn: Callable[[str], np.ndarray] | None = None,
    resume: bool = True,
    log_every: int = 100,
) -> BatchStats:
    """クリップをまとめて処理し JSONL に1行ずつ追記する。

    resume=True なら既存の出力にある clip_id を飛ばす（途中で止まっても再開できる）。
    処理中に出た警告（MPS の CPU フォールバックを含む）はログに記録する。
    """
    if load_fn is None:
        from spkrate.eval.audio import load_audio

        def load_fn(path: str) -> np.ndarray:
            return load_audio(path)[0]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_ids(out_path) if resume else set()
    stats = BatchStats()
    mode = "a" if resume else "w"
    t_start = time.perf_counter()
    with out_path.open(mode, encoding="utf-8") as f:
        for r in _iter_pending(records, done, stats):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wav = load_fn(r["audio_path"])
                rec = align_fn(r, wav)
            for w in caught:
                msg = str(w.message)
                tag = "MPS_CPU_FALLBACK" if "fall" in msg.lower() and "cpu" in msg.lower() else "WARNING"
                logger.warning("%s clip_id=%s %s: %s", tag, r["clip_id"], w.category.__name__, msg)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            stats.processed += 1
            if rec["ok"]:
                stats.ok += 1
            else:
                stats.failed[rec["reason"]] = stats.failed.get(rec["reason"], 0) + 1
            if log_every and stats.processed % log_every == 0:
                el = time.perf_counter() - t_start
                logger.info(
                    "処理 %d 件（%.3f 秒/件）失敗 %s", stats.processed, el / stats.processed, stats.failed
                )
    stats.seconds = time.perf_counter() - t_start
    return stats


def _iter_pending(records: Iterable[dict], done: set[str], stats: BatchStats) -> Iterator[dict]:
    for r in records:
        if r["clip_id"] in done:
            stats.skipped += 1
            continue
        yield r
