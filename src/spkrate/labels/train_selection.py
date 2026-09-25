"""train のアライメント付与・無発話の指標・学習に使うクリップの選別（指示書 2026-09-26 タスク2）。

dev と同じ手段・同じ定義を train に適用する。

- アライメント: ``spkrate.labels.alignment.align_clip``（docs/decisions/008-forced-alignment.md）。
  出力の1行は ``data/processed/alignments/dev.jsonl`` と同じ形式
- 無発話の指標: ``scripts/detect_no_speech.py`` と同じ（``energy_metrics`` と ``ctc_metrics``）。
  出力の1行は ``data/processed/no_speech/dev_metrics.jsonl`` と同じ形式
- ひらがなCTCの放出確率はアライメントと貪欲デコード（cer）で共通なので、1クリップにつき
  1回だけ推論し、両方の出力に使う（``measure_clip``）
- 2つの出力は別々に途中再開できる（``measure_clips_to_jsonl``。出力ごとに処理済みの
  clip_id を読み、欠けている側だけに書く）
- 選別: アライメントの失敗（ok=false）・モーラ数の不一致・端のモーラの孤立
  （``spkrate.eval.dev_window.isolation_flags``、閾値 1.0 秒、>）・無発話疑い
  （configs/eval/no_speech.yaml の規則）を除く（``exclusion_reasons``）
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spkrate.eval.dev_window import isolation_flags
from spkrate.eval.metrics import band_of
from spkrate.eval.no_speech import ctc_metrics, energy_metrics, normalize_reference
from spkrate.labels.alignment import SAMPLE_RATE, _read_done_ids, align_clip

logger = logging.getLogger(__name__)

__all__ = [
    "EXCLUSION_PRIORITY",
    "REASON_ALIGN_FAILED",
    "REASON_HEAD_ISOLATED",
    "REASON_MORA_MISMATCH",
    "REASON_NO_SPEECH",
    "REASON_TAIL_ISOLATED",
    "DualStats",
    "exclusion_reasons",
    "measure_clip",
    "measure_clips_to_jsonl",
    "no_speech_record",
    "primary_reason",
]

# 除外の理由（排他の件数ではこの順で最初に該当したものに数える）
REASON_ALIGN_FAILED = "align_failed"  # アライメントの失敗（ok=false。理由コードは問わない）
REASON_MORA_MISMATCH = "mora_mismatch"  # ok だが moras の要素数 ≠ clips.jsonl の mora
REASON_HEAD_ISOLATED = "head_isolated"  # 先頭2モーラの開始時刻の差 > 閾値
REASON_TAIL_ISOLATED = "tail_isolated"  # 末尾2モーラの開始時刻の差 > 閾値
REASON_NO_SPEECH = "no_speech_suspect"  # configs/eval/no_speech.yaml の規則に該当
EXCLUSION_PRIORITY: tuple[str, ...] = (
    REASON_ALIGN_FAILED,
    REASON_MORA_MISMATCH,
    REASON_HEAD_ISOLATED,
    REASON_TAIL_ISOLATED,
    REASON_NO_SPEECH,
)


# ---------------------------------------------------------------- 1クリップの計算
def no_speech_record(
    record: Mapping[str, Any],
    wav: np.ndarray,
    emission: Any,
    vocab: Mapping[str, int],
    blank: int,
) -> dict:
    """無発話判定の指標の1行（scripts/detect_no_speech.py の ``measure`` と同じ内容・順序）。"""
    ref = normalize_reference(record["kana"], vocab)
    mora = int(record["mora"])
    out = {
        "clip_id": record["clip_id"],
        "ok": True,
        "duration_sec": len(wav) / SAMPLE_RATE,
        "mora": mora,
        "mora_per_second": float(record["mora_per_second"]),
        "band": band_of(float(record["mora_per_second"])),
        "ref": ref,
    }
    out.update(energy_metrics(wav))
    out.update(ctc_metrics(emission, ref, vocab, blank, mora))
    return out


def measure_clip(
    record: dict,
    wav: np.ndarray,
    emission_fn: Callable[[np.ndarray], Any],
    vocab: dict[str, int],
    blank: int,
    *,
    need_alignment: bool = True,
    need_metrics: bool = True,
    **align_kwargs: Any,
) -> tuple[dict | None, dict | None]:
    """放出確率を1回だけ求め、(アライメントの行, 無発話の指標の行) を返す。不要な側は None。"""
    cache: list[Any] = []

    def once(w: np.ndarray) -> Any:
        if not cache:
            cache.append(emission_fn(w))
        return cache[0]

    align_row = align_clip(record, wav, once, vocab, blank, **align_kwargs) if need_alignment else None
    metrics_row = no_speech_record(record, wav, once(wav), vocab, blank) if need_metrics else None
    return align_row, metrics_row


# ---------------------------------------------------------------- まとめて処理
@dataclass
class DualStats:
    processed: int = 0
    skipped: int = 0
    ok: int = 0
    failed: dict[str, int] | None = None
    warnings: int = 0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.failed is None:
            self.failed = {}


def measure_clips_to_jsonl(
    records: Iterable[dict],
    align_out: str | Path,
    metrics_out: str | Path,
    measure_fn: Callable[..., tuple[dict | None, dict | None]],
    *,
    load_fn: Callable[[str], np.ndarray] | None = None,
    log_every: int = 1000,
) -> DualStats:
    """クリップごとに ``measure_fn(record, wav, need_alignment=, need_metrics=)`` を呼び、2つの JSONL に追記する。

    それぞれの出力にすでにある clip_id はその出力には書かない（両方にあれば音声も読まない）。
    途中で切れた最終行は切り詰める（``_read_done_ids``）。処理中の警告（MPS の CPU フォールバックを
    含む）はログに clip_id つきで記録する。
    """
    if load_fn is None:
        from spkrate.eval.audio import load_audio

        def load_fn(path: str) -> np.ndarray:
            return load_audio(path)[0]

    align_out, metrics_out = Path(align_out), Path(metrics_out)
    for p in (align_out, metrics_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    done_a = _read_done_ids(align_out)
    done_m = _read_done_ids(metrics_out)
    stats = DualStats()
    t_start = time.perf_counter()
    with align_out.open("a", encoding="utf-8") as fa, metrics_out.open("a", encoding="utf-8") as fm:
        for r in records:
            cid = r["clip_id"]
            need_a, need_m = cid not in done_a, cid not in done_m
            if not (need_a or need_m):
                stats.skipped += 1
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wav = load_fn(r["audio_path"])
                arow, mrow = measure_fn(r, wav, need_alignment=need_a, need_metrics=need_m)
            for w in caught:
                msg = str(w.message)
                tag = "MPS_CPU_FALLBACK" if "fall" in msg.lower() and "cpu" in msg.lower() else "WARNING"
                logger.warning("%s clip_id=%s %s: %s", tag, cid, w.category.__name__, msg)
                stats.warnings += 1
            # 指標を先に書く（アライメントの行があれば指標の行もある状態を保つ）
            if need_m:
                fm.write(json.dumps(mrow, ensure_ascii=False) + "\n")
                fm.flush()
            if need_a:
                fa.write(json.dumps(arow, ensure_ascii=False) + "\n")
                fa.flush()
                if arow["ok"]:
                    stats.ok += 1
                else:
                    stats.failed[arow["reason"]] = stats.failed.get(arow["reason"], 0) + 1
            stats.processed += 1
            if log_every and stats.processed % log_every == 0:
                el = time.perf_counter() - t_start
                logger.info(
                    "処理 %d 件（既存 %d 件を飛ばした）%.3f 秒/件、失敗 %s、警告 %d",
                    stats.processed, stats.skipped, el / stats.processed, stats.failed, stats.warnings,
                )
    stats.seconds = time.perf_counter() - t_start
    return stats


# ---------------------------------------------------------------- 選別
def exclusion_reasons(
    align_row: Mapping[str, Any] | None,
    label_mora: int,
    no_speech_suspect: bool,
    isolation_threshold_sec: float = 1.0,
) -> list[str]:
    """1クリップの除外理由（重複あり、``EXCLUSION_PRIORITY`` の順）。空なら学習に使う。

    ``align_row`` が None（アライメントの出力に無い）の場合は呼び出し側で扱う。
    失敗のクリップにはモーラ区間が無いので、不一致・孤立は判定しない。
    """
    reasons: list[str] = []
    if align_row is None or not align_row["ok"]:
        reasons.append(REASON_ALIGN_FAILED)
    else:
        moras: Sequence[Mapping[str, float]] = align_row["moras"]
        if len(moras) != int(label_mora):
            reasons.append(REASON_MORA_MISMATCH)
        head, tail = isolation_flags(moras, isolation_threshold_sec)
        if head:
            reasons.append(REASON_HEAD_ISOLATED)
        if tail:
            reasons.append(REASON_TAIL_ISOLATED)
    if no_speech_suspect:
        reasons.append(REASON_NO_SPEECH)
    return reasons


def primary_reason(reasons: Sequence[str]) -> str | None:
    """優先順（``EXCLUSION_PRIORITY``）で最初の理由。排他の件数に使う。"""
    for key in EXCLUSION_PRIORITY:
        if key in reasons:
            return key
    return None
