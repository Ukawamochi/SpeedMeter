"""src/spkrate/labels/train_selection.py の単体テスト（指示書 2026-09-26 タスク2）。

合成の語彙・放出確率で書く（実モデルは使わない）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from spkrate.labels.alignment import align_clip
from spkrate.labels.train_selection import (
    EXCLUSION_PRIORITY,
    REASON_ALIGN_FAILED,
    REASON_HEAD_ISOLATED,
    REASON_MORA_MISMATCH,
    REASON_NO_SPEECH,
    REASON_TAIL_ISOLATED,
    exclusion_reasons,
    measure_clip,
    measure_clips_to_jsonl,
    no_speech_record,
    primary_reason,
)

_CHARS = list("あいうえおかきくけこさしすせそたちつてとなにぬねのらりるれろんっー")
VOCAB = {c: i for i, c in enumerate(_CHARS)}
VOCAB["|"] = len(VOCAB)
BLANK = len(VOCAB)
VOCAB["[PAD]"] = BLANK
V = BLANK + 1


def _record(clip_id: str = "c1", kana: str = "サクラ", mora: int = 3) -> dict:
    return {
        "clip_id": clip_id,
        "audio_path": f"/nonexistent/{clip_id}.mp3",
        "sentence": "さくら",
        "kana": kana,
        "mora": mora,
        "mora_per_second": 3.0,
    }


def _emission_for(tokens: list[int], frames: int = 50) -> torch.Tensor:
    """指定のトークンを等間隔に1フレームずつ置き、残りを blank にした放出確率。"""
    lp = torch.full((1, frames, V), -20.0)
    lp[0, :, BLANK] = 0.0
    step = frames // (len(tokens) + 1)
    for i, t in enumerate(tokens):
        f = (i + 1) * step
        lp[0, f, :] = -20.0
        lp[0, f, t] = 0.0
    return torch.log_softmax(lp, dim=-1)


def _counter_emission(tokens: list[int]):
    calls = []

    def fn(wav: np.ndarray) -> torch.Tensor:
        calls.append(1)
        return _emission_for(tokens)

    return fn, calls


WAV = np.zeros(16000, dtype=np.float32)
WAV[4000:12000] = 0.1 * np.sin(np.arange(8000) / 5.0)
SAKURA = [VOCAB["さ"], VOCAB["く"], VOCAB["ら"]]


# ---------------------------------------------------------------- 1クリップ
def test_measure_clip_calls_model_once_and_matches_separate_paths():
    fn, calls = _counter_emission(SAKURA)
    arow, mrow = measure_clip(_record(), WAV, fn, VOCAB, BLANK, count_mora_fn=lambda s: 3)
    assert len(calls) == 1
    # 別々に計算した場合（dev の align_dev.py・detect_no_speech.py の手順）と同じ
    fn2, _ = _counter_emission(SAKURA)
    assert arow == align_clip(_record(), WAV, fn2, VOCAB, BLANK, count_mora_fn=lambda s: 3)
    assert json.dumps(mrow) == json.dumps(no_speech_record(_record(), WAV, _emission_for(SAKURA), VOCAB, BLANK))
    assert arow["ok"] and len(arow["moras"]) == 3
    assert mrow["cer"] == 0.0 and mrow["ref"] == "さくら"
    assert list(mrow)[:7] == ["clip_id", "ok", "duration_sec", "mora", "mora_per_second", "band", "ref"]


def test_measure_clip_metrics_even_when_alignment_fails_early():
    fn, calls = _counter_emission(SAKURA)
    arow, mrow = measure_clip(_record(mora=4), WAV, fn, VOCAB, BLANK, count_mora_fn=lambda s: 4)
    assert not arow["ok"] and arow["reason"] == "mora_mismatch_label"
    assert mrow is not None and len(calls) == 1


def test_measure_clip_only_requested_side():
    fn, calls = _counter_emission(SAKURA)
    arow, mrow = measure_clip(_record(), WAV, fn, VOCAB, BLANK, need_alignment=False, count_mora_fn=lambda s: 3)
    assert arow is None and mrow is not None and len(calls) == 1
    fn, calls = _counter_emission(SAKURA)
    arow, mrow = measure_clip(_record(), WAV, fn, VOCAB, BLANK, need_metrics=False, count_mora_fn=lambda s: 3)
    assert arow is not None and mrow is None


# ---------------------------------------------------------------- まとめて処理・再開
def _measure(r, w, **kw):
    fn, _ = _counter_emission(SAKURA)
    return measure_clip(r, w, fn, VOCAB, BLANK, count_mora_fn=lambda s: 3, **kw)


def test_measure_clips_to_jsonl_resume_each_output(tmp_path):
    recs = [_record(f"c{i}") for i in range(5)]
    a, m = tmp_path / "a.jsonl", tmp_path / "m.jsonl"
    loads = []

    def load(p):
        loads.append(p)
        return WAV

    st = measure_clips_to_jsonl(recs[:3], a, m, _measure, load_fn=load)
    assert st.processed == 3 and st.ok == 3
    # 指標だけ1行多い状態、アライメントの最終行が途中で切れた状態を作る
    m_lines = m.read_text().splitlines()
    a_lines = a.read_text().splitlines()
    m.write_text("\n".join(m_lines) + "\n" + json.dumps({"clip_id": "c3"}) + "\n")
    a.write_text("\n".join(a_lines[:2]) + "\n" + a_lines[2][:10])
    loads.clear()
    st = measure_clips_to_jsonl(recs, a, m, _measure, load_fn=load)
    assert st.skipped == 2 and st.processed == 3  # c2（アライメントのみ）、c3（アライメントのみ）、c4（両方）
    a_ids = [json.loads(x)["clip_id"] for x in a.read_text().splitlines()]
    m_ids = [json.loads(x)["clip_id"] for x in m.read_text().splitlines()]
    assert sorted(a_ids) == [f"c{i}" for i in range(5)] and len(a_ids) == 5
    assert sorted(m_ids) == [f"c{i}" for i in range(5)] and len(m_ids) == 5
    assert len(loads) == 3


def test_measure_clips_to_jsonl_logs_warnings(tmp_path, caplog):
    import warnings

    def warn_measure(r, w, **kw):
        warnings.warn("The operator 'aten::x' is not currently supported on the MPS backend and will fall back to run on the CPU.")
        return _measure(r, w, **kw)

    with caplog.at_level("WARNING"):
        st = measure_clips_to_jsonl([_record()], tmp_path / "a", tmp_path / "m", warn_measure, load_fn=lambda p: WAV)
    assert st.warnings == 1
    assert any("MPS_CPU_FALLBACK clip_id=c1" in r.message for r in caplog.records)


# ---------------------------------------------------------------- 選別
def _row(starts: list[float]) -> dict:
    return {"ok": True, "moras": [{"kana": "ア", "start": s, "end": s + 0.02} for s in starts]}


def test_exclusion_reasons_each_and_overlap():
    assert exclusion_reasons(_row([0.5, 0.6, 0.7]), 3, False) == []
    assert exclusion_reasons({"ok": False, "reason": "align_failed"}, 3, False) == [REASON_ALIGN_FAILED]
    assert exclusion_reasons(None, 3, True) == [REASON_ALIGN_FAILED, REASON_NO_SPEECH]
    assert exclusion_reasons(_row([0.5, 0.6, 0.7]), 4, False) == [REASON_MORA_MISMATCH]
    # 閾値ちょうど（1.0）は孤立としない（>）
    assert exclusion_reasons(_row([0.0, 1.0, 1.1]), 3, False) == []
    assert exclusion_reasons(_row([0.0, 1.01, 1.1]), 3, False) == [REASON_HEAD_ISOLATED]
    assert exclusion_reasons(_row([0.0, 0.1, 1.2]), 3, False) == [REASON_TAIL_ISOLATED]
    both = exclusion_reasons(_row([0.0, 1.5, 3.0]), 3, True)
    assert both == [REASON_HEAD_ISOLATED, REASON_TAIL_ISOLATED, REASON_NO_SPEECH]
    assert exclusion_reasons(_row([0.0]), 1, False) == []  # 1モーラは孤立を判定しない


def test_primary_reason_follows_priority():
    assert primary_reason([]) is None
    assert primary_reason([REASON_NO_SPEECH, REASON_TAIL_ISOLATED]) == REASON_TAIL_ISOLATED
    assert primary_reason(list(reversed(EXCLUSION_PRIORITY))) == REASON_ALIGN_FAILED


@pytest.mark.parametrize("threshold", [0.5, 2.0])
def test_exclusion_threshold_parameter(threshold):
    got = exclusion_reasons(_row([0.0, 1.0, 1.1]), 3, False, isolation_threshold_sec=threshold)
    assert got == ([REASON_HEAD_ISOLATED] if threshold < 1.0 else [])
