"""src/spkrate/labels/alignment.py の単体テスト（タスク4-1）。

モデルを使わないテストは合成の放出確率・トークン列で書く。実モデルを使うテストは
重みが Hugging Face のキャッシュに無い場合、または dev のクリップが無い場合に skip する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spkrate.labels.alignment import (
    HIRAGANA_MODEL,
    HIRAGANA_REVISION,
    REASON_ALIGN_FAILED,
    REASON_MORA_MISMATCH_ALIGNMENT,
    REASON_MORA_MISMATCH_LABEL,
    REASON_UNKNOWN_TOKEN,
    AlignmentError,
    align_clip,
    align_clips_to_jsonl,
    align_emission,
    min_frames_required,
    split_mora,
    token_spans,
    tokens_for_mora,
)
from spkrate.labels.mora import count_mora_from_kana

REPO = Path(__file__).resolve().parents[1]

# 合成の語彙（本物の語彙の部分集合を模したもの）。blank は末尾。
_CHARS = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんっーゃゅょぁぃぅぇぉきしちにひみりぎじびぴゔ")
VOCAB = {c: i for i, c in enumerate(dict.fromkeys(_CHARS))}
BLANK = len(VOCAB)
VOCAB["[PAD]"] = BLANK
V = BLANK + 1


# ---------------------------------------------------------------- モーラ分割
@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("サクラ", ["サ", "ク", "ラ"]),
        ("キョウ", ["キョ", "ウ"]),  # 拗音
        ("シュッパツ", ["シュ", "ッ", "パ", "ツ"]),  # 拗音＋促音
        ("クヮシ", ["クヮ", "シ"]),  # 合拗音
        ("シヮ", ["シヮ"]),
        ("ヵヶ", ["ヵ", "ヶ"]),  # ヵヶは単独
        ("カヵ", ["カ", "ヵ"]),
        ("センセー", ["セ", "ン", "セ", "ー"]),  # ン・ー
        ("ッッ", ["ッ", "ッ"]),
        ("ァイ", ["ァ", "イ"]),  # 先頭の小書きは単独
        ("キャャ", ["キャ", "ャ"]),  # 小書きの後の小書きは単独
        ("ティー", ["ティ", "ー"]),
        ("ッャ", ["ッャ"]),
        ("ア、ャ", ["ア", "ャ"]),  # 非かなの後の小書きは単独、非かなは数えない
        ("", []),
    ],
)
def test_split_mora_rules(kana, expected):
    assert split_mora(kana) == expected
    assert len(split_mora(kana)) == count_mora_from_kana(kana)


def test_split_mora_matches_count_mora_random():
    rng = np.random.default_rng(0)
    alphabet = list("アカキシチニヒミリギジンッーャュョァィゥェォヮヵヶヴ、A")
    for _ in range(2000):
        kana = "".join(rng.choice(alphabet, size=int(rng.integers(0, 12))))
        moras = split_mora(kana)
        assert len(moras) == count_mora_from_kana(kana), kana
        assert "".join(moras) == "".join(c for c in kana if c not in "、A")


def test_split_mora_matches_count_mora_on_clips():
    clips = REPO / "data/processed/clips.jsonl"
    if not clips.exists():
        pytest.skip("clips.jsonl が無い")
    with clips.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3000:
                break
            r = json.loads(line)
            assert len(split_mora(r["kana"])) == count_mora_from_kana(r["kana"])


# ---------------------------------------------------------------- トークン化
def test_tokens_for_mora_basic_and_fallback():
    assert tokens_for_mora("キョ", VOCAB) == [VOCAB["き"], VOCAB["ょ"]]
    assert tokens_for_mora("ー", VOCAB) == [VOCAB["ー"]]
    assert tokens_for_mora("ヴ", VOCAB) == [VOCAB["ゔ"]]
    # 語彙に無い文字は 008 の置き換え表で置き換える
    assert tokens_for_mora("クヮ", VOCAB) == [VOCAB["く"], VOCAB["わ"]]
    assert tokens_for_mora("ヵ", VOCAB) == [VOCAB["か"]]
    assert tokens_for_mora("ヶ", VOCAB) == [VOCAB["け"]]
    assert tokens_for_mora("ヰ", VOCAB) == [VOCAB["い"]]
    assert tokens_for_mora("ヱ", VOCAB) == [VOCAB["え"]]
    # それでも無ければ None
    assert tokens_for_mora("ヷ", VOCAB) is None
    assert tokens_for_mora("ゼ", VOCAB) is None  # 合成語彙に「ぜ」は無い


def test_min_frames_required():
    assert min_frames_required([1, 2, 3]) == 3
    assert min_frames_required([1, 1, 2, 2, 2]) == 5 + 3


# ---------------------------------------------------------------- 区間化
def test_token_spans_merges_runs_and_splits_repeats():
    B = 9
    labels = [B, 1, 1, B, 1, 2, 2, B, B, 3]
    assert token_spans(labels, B) == [(1, 3), (4, 5), (5, 7), (9, 10)]


def _emission_for(frame_labels: list[int]) -> torch.Tensor:
    """各フレームで指定のラベルが圧倒的に高い log_softmax 放出確率を作る。"""
    logits = torch.full((1, len(frame_labels), V), -10.0)
    for f, lab in enumerate(frame_labels):
        logits[0, f, lab] = 10.0
    return torch.log_softmax(logits, dim=-1)


def test_align_emission_mora_boundaries():
    # モーラ: キョ / ウ / ッ / ー 。トークン: き ょ う っ ー
    groups = [tokens_for_mora(m, VOCAB) for m in ["キョ", "ウ", "ッ", "ー"]]
    ki, yo, u, tsu, bar = (VOCAB[c] for c in "きょうっー")
    B = BLANK
    frames = [B, B, ki, yo, yo, B, u, B, tsu, tsu, tsu, bar, B, B]
    em = _emission_for(frames)
    wav_len = len(frames) * 320  # 20ms/フレーム
    spans = align_emission(em, groups, B, wav_len)
    fl = 0.02
    expected = [(2, 5), (6, 7), (8, 11), (11, 12)]
    assert len(spans) == 4
    for (s, e), (es, ee) in zip(spans, expected):
        assert s == pytest.approx(es * fl)
        assert e == pytest.approx(ee * fl)


def test_align_emission_uses_wav_len_over_frames():
    groups = [[VOCAB["あ"]]]
    em = _emission_for([BLANK, VOCAB["あ"], BLANK, BLANK])
    spans = align_emission(em, groups, BLANK, wav_len=16000)  # 1秒 / 4フレーム
    assert spans == [pytest.approx((0.25, 0.5))]


def test_align_emission_too_short_is_align_failed():
    a = VOCAB["あ"]
    groups = [[a], [a], [a]]  # 必要フレーム数 3+2=5
    em = _emission_for([a, BLANK, a, BLANK])
    with pytest.raises(AlignmentError) as ei:
        align_emission(em, groups, BLANK, 4 * 320)
    assert ei.value.reason == REASON_ALIGN_FAILED


def test_align_emission_forced_align_exception_is_align_failed(monkeypatch):
    import spkrate.labels.alignment as al

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(al, "_forced_align", boom)
    em = _emission_for([BLANK, VOCAB["あ"]])
    with pytest.raises(AlignmentError) as ei:
        align_emission(em, [[VOCAB["あ"]]], BLANK, 640)
    assert ei.value.reason == REASON_ALIGN_FAILED


def test_align_emission_span_count_mismatch(monkeypatch):
    import spkrate.labels.alignment as al

    a, i = VOCAB["あ"], VOCAB["い"]
    # forced_align が「あ い」ではなく「あ」だけを返したと仮定
    monkeypatch.setattr(al, "_forced_align", lambda em, flat, blank: [a, a, BLANK, BLANK])
    em = _emission_for([a, BLANK, i, BLANK])
    with pytest.raises(AlignmentError) as ei:
        align_emission(em, [[a], [i]], BLANK, 4 * 320)
    assert ei.value.reason == REASON_MORA_MISMATCH_ALIGNMENT


def test_alignment_error_rejects_unknown_reason():
    with pytest.raises(ValueError):
        AlignmentError("other")


# ---------------------------------------------------------------- 1クリップの処理と除外
def _record(kana, mora, sentence="dummy", clip_id="c1"):
    return {"clip_id": clip_id, "kana": kana, "mora": mora, "sentence": sentence, "audio_path": clip_id}


def _emission_fn_for(kana: str):
    """かなを1トークン1フレームずつ blank で挟んで並べた放出確率を返す関数。"""
    toks = [t for m in split_mora(kana) for t in tokens_for_mora(m, VOCAB)]
    frames = [BLANK]
    for t in toks:
        frames += [t, BLANK]
    em = _emission_for(frames)
    return (lambda wav: em), len(frames) * 320


def test_align_clip_ok_output_format():
    kana = "キョーワ"  # キョ ー ワ → 3モーラ（「わ」は語彙にある）
    fn, wav_len = _emission_fn_for(kana)
    rec = align_clip(_record(kana, 3), np.zeros(wav_len, np.float32), fn, VOCAB, BLANK, count_mora_fn=lambda s: 3)
    assert rec["ok"] is True
    assert rec["clip_id"] == "c1"
    assert [m["kana"] for m in rec["moras"]] == ["キョ", "ー", "ワ"]
    assert rec["speech_start"] == rec["moras"][0]["start"]
    assert rec["speech_end"] == rec["moras"][-1]["end"]
    starts = [m["start"] for m in rec["moras"]]
    assert starts == sorted(starts)
    assert all(m["end"] > m["start"] for m in rec["moras"])
    json.dumps(rec)  # JSON に書ける


@pytest.mark.parametrize(
    ("label", "counted"),
    [(4, 3), (3, 4), (4, 4)],  # mora列・count_mora のどちらか／両方がかな分割数と違う
)
def test_align_clip_mora_mismatch_label(label, counted):
    kana = "キョーワ"
    fn, wav_len = _emission_fn_for(kana)
    rec = align_clip(
        _record(kana, label), np.zeros(wav_len, np.float32), fn, VOCAB, BLANK, count_mora_fn=lambda s: counted
    )
    assert rec == {"clip_id": "c1", "ok": False, "reason": REASON_MORA_MISMATCH_LABEL, "detail": rec["detail"]}


def test_align_clip_unknown_token_does_not_run_model():
    def fn(wav):
        raise AssertionError("呼ばれてはいけない")

    rec = align_clip(_record("ヷア", 2), np.zeros(3200, np.float32), fn, VOCAB, BLANK, count_mora_fn=lambda s: 2)
    assert rec["ok"] is False and rec["reason"] == REASON_UNKNOWN_TOKEN


def test_align_clip_align_failed_short_clip():
    em = _emission_for([BLANK, VOCAB["あ"]])  # 2フレームしかない
    rec = align_clip(
        _record("アイウ", 3), np.zeros(640, np.float32), lambda w: em, VOCAB, BLANK, count_mora_fn=lambda s: 3
    )
    assert rec["ok"] is False and rec["reason"] == REASON_ALIGN_FAILED


def test_align_clip_mora_mismatch_alignment(monkeypatch):
    import spkrate.labels.alignment as al

    monkeypatch.setattr(al, "_forced_align", lambda em, flat, blank: [BLANK] * em.shape[1])
    fn, wav_len = _emission_fn_for("アイ")
    rec = align_clip(_record("アイ", 2), np.zeros(wav_len, np.float32), fn, VOCAB, BLANK, count_mora_fn=lambda s: 2)
    assert rec["ok"] is False and rec["reason"] == REASON_MORA_MISMATCH_ALIGNMENT


# ---------------------------------------------------------------- まとめて処理・再開
def test_align_clips_to_jsonl_resume(tmp_path):
    kanas = {"a": "アイ", "b": "カキ", "c": "ヷ"}
    records = [_record(k, len(split_mora(k)), clip_id=cid) for cid, k in kanas.items()]

    def align_fn(r, wav):
        fn, _ = _emission_fn_for(r["kana"]) if r["kana"] != "ヷ" else (lambda w: None, 0)
        return align_clip(r, wav, fn, VOCAB, BLANK, count_mora_fn=lambda s, r=r: r["mora"])

    out = tmp_path / "x" / "dev.jsonl"
    load = lambda p: np.zeros(16000, np.float32)  # noqa: E731
    s1 = align_clips_to_jsonl(records[:2], out, align_fn, load_fn=load)
    assert s1.processed == 2 and s1.ok == 2
    # 途中で切れた行を模す
    with out.open("a", encoding="utf-8") as f:
        f.write('{"clip_id": "c", "ok"')
    s2 = align_clips_to_jsonl(records, out, align_fn, load_fn=load)
    assert s2.skipped == 2 and s2.processed == 1
    assert s2.failed == {REASON_UNKNOWN_TOKEN: 1}
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x["clip_id"] for x in lines] == ["a", "b", "c"]


# ---------------------------------------------------------------- 実モデル（少数）
def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    p = try_to_load_from_cache(HIRAGANA_MODEL, "model.safetensors", revision=HIRAGANA_REVISION)
    v = try_to_load_from_cache(HIRAGANA_MODEL, "vocab.json", revision=HIRAGANA_REVISION)
    return isinstance(p, str) and isinstance(v, str)


@pytest.mark.skipif(not _model_cached(), reason="ひらがなCTCモデルの重みが未取得")
def test_real_model_aligns_dev_clips(tmp_path):
    from spkrate.labels.alignment import HiraganaAligner, select_dev_clips

    clips, dev = REPO / "data/processed/clips.jsonl", REPO / "configs/splits/dev.json"
    if not clips.exists() or not dev.exists():
        pytest.skip("clips.jsonl または dev 分割が無い")
    records = select_dev_clips(clips, dev, num=2, seed=1)
    if not all((REPO / r["audio_path"]).exists() for r in records):
        pytest.skip("音声ファイルが無い")
    for r in records:
        r["audio_path"] = str(REPO / r["audio_path"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    aligner = HiraganaAligner(device)
    assert set(tokens_for_mora(m, aligner.vocab) is not None for m in ["キャ", "ヮ", "ヵ", "ヶ", "ン", "ッ", "ー"]) == {True}
    stats = align_clips_to_jsonl(records, tmp_path / "dev.jsonl", aligner.align)
    assert stats.processed == 2
    for line in (tmp_path / "dev.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert rec["ok"], rec
        r = next(x for x in records if x["clip_id"] == rec["clip_id"])
        assert len(rec["moras"]) == r["mora"]
        assert 0.0 <= rec["speech_start"] < rec["speech_end"] <= rec["duration_sec"] + 1e-6
