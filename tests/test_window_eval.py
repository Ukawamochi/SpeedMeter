"""窓単位の評価の集計（src/spkrate/eval/window_eval.py）の検証。実データは使わない。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spkrate.eval.dev_window import (
    KIND_CONCAT,
    KIND_SINGLE,
    DevWindowSet,
    WindowSource,
    source_moras,
    window_mora,
    window_starts,
)
from spkrate.eval.metrics import compute_metrics
from spkrate.eval.window_eval import (
    band_codes,
    rate_summary,
    shifted_window_labels,
    short_clip_window_mask,
    source_window_ranges,
    zero_window_summary,
)

SR = 16000
W = 32000
HOP = 4000


def _alignment(times: list[float], width: float = 0.02) -> dict:
    return {"ok": True, "moras": [{"kana": "ア", "start": t, "end": t + width} for t in times]}


def _window_set(sources: list[WindowSource], alignments: dict) -> DevWindowSet:
    idx, st, mora, kind = [], [], [], []
    for number, source in enumerate(sources):
        starts = window_starts(source.num_samples, W, HOP)
        if starts.size == 0:
            continue
        s, e = source_moras(source, alignments, SR)
        idx.append(np.full(starts.size, number, dtype=np.int32))
        st.append(starts)
        mora.append(window_mora(s, e, starts.astype(np.float32) / SR, 2.0))
        kind.append(np.full(starts.size, 0 if source.kind == KIND_SINGLE else 1, dtype=np.int8))
    return DevWindowSet(
        sources=sources,
        source_index=np.concatenate(idx),
        start_sample=np.concatenate(st),
        mora=np.concatenate(mora).astype(np.float32),
        kind=np.concatenate(kind),
        known_no_speech=np.zeros(sum(a.size for a in idx), dtype=bool),
    )


def _fixture():
    alignments = {
        "long": _alignment([0.1 * k for k in range(1, 30)]),   # 3.2 秒
        "short": _alignment([0.2, 0.5, 0.8, 1.1]),            # 1.5 秒
        "mid": _alignment([0.3, 0.9, 1.8, 2.4]),               # 2.75 秒
        "tiny": _alignment([0.05]),                             # 1.0 秒
    }
    sources = [
        WindowSource("long", KIND_SINGLE, "a", ("long",), (51200,), ()),
        WindowSource("short", KIND_SINGLE, "a", ("short",), (24000,), ()),  # 窓なし
        WindowSource("mid", KIND_SINGLE, "a", ("mid",), (44000,), ()),
        WindowSource(
            "g", KIND_CONCAT, "a", ("long", "short", "mid"), (51200, 24000, 44000), (8000, 16000)
        ),
        WindowSource("h", KIND_CONCAT, "a", ("tiny", "mid", "tiny"), (16000, 44000, 16000), (8000, 8000)),
    ]
    return sources, alignments


# ---------------------------------------------------------------- 指標


def test_rate_summary_matches_compute_metrics():
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 22, size=500).astype(np.float32)
    true[:40] = 0.0
    pred = (true + rng.normal(0, 1.5, size=500)).astype(np.float32)
    summary = rate_summary(true, pred, 2.0)
    ref = compute_metrics(true.tolist(), pred.tolist(), [2.0] * 500)
    assert summary["n"] == ref.num_segments
    assert summary["mae"] == pytest.approx(ref.mae_moras_per_sec, rel=1e-5)
    assert summary["correlation"] == pytest.approx(ref.correlation, rel=1e-5)
    for key, ref_mae, ref_n in (
        ("under4", ref.mae_band_under4, ref.n_band_under4),
        ("4to6", ref.mae_band_4to6, ref.n_band_4to6),
        ("6to8", ref.mae_band_6to8, ref.n_band_6to8),
        ("over8", ref.mae_band_over8, ref.n_band_over8),
    ):
        assert summary["bands"][key]["n"] == ref_n
        assert summary["bands"][key]["mae"] == pytest.approx(ref_mae, rel=1e-5)


def test_rate_summary_bias_sign_and_empty_band():
    true = np.array([0.0, 2.0, 4.0], dtype=np.float32)  # 毎秒 0, 1, 2 → すべて4未満
    pred = true + np.float32(1.0)  # 毎秒 +0.5
    summary = rate_summary(true, pred, 2.0)
    assert summary["bias"] == pytest.approx(0.5)
    assert summary["mae"] == pytest.approx(0.5)
    assert summary["bands"]["under4"]["bias"] == pytest.approx(0.5)
    assert summary["bands"]["over8"]["n"] == 0
    assert math.isnan(summary["bands"]["over8"]["mae"])


def test_band_codes_boundaries():
    codes = band_codes(np.array([0.0, 3.999, 4.0, 6.0, 7.99, 8.0, 20.0], dtype=np.float32))
    assert codes.tolist() == [0, 0, 1, 2, 2, 3, 3]


def test_zero_window_summary():
    pred = np.array([0.0, 0.2, 0.5, 1.0, 3.0], dtype=np.float32)
    out = zero_window_summary(pred, 2.0)
    assert out["n"] == 5
    assert out["mean_mora"] == pytest.approx(0.94)
    assert out["mean_rate"] == pytest.approx(0.47)
    assert out["frac_ge"]["0.5"] == pytest.approx(0.6)
    assert out["frac_ge"]["1"] == pytest.approx(0.4)
    assert out["frac_ge"]["2"] == pytest.approx(0.2)
    assert out["max_mora"] == pytest.approx(3.0)
    assert zero_window_summary(np.zeros(0, dtype=np.float32))["n"] == 0


# ---------------------------------------------------------------- 窓と音源の対応


def test_source_window_ranges():
    sources, alignments = _fixture()
    ws = _window_set(sources, alignments)
    ranges = source_window_ranges(ws)
    assert ranges[1, 0] == ranges[1, 1]  # 2.0 秒未満の単一クリップは窓なし
    for number, (lo, hi) in enumerate(ranges):
        assert np.all(ws.source_index[lo:hi] == number)
    assert ranges[-1, 1] == len(ws)


def test_short_clip_window_mask_is_overlap_with_short_clip():
    sources, alignments = _fixture()
    ws = _window_set(sources, alignments)
    mask = short_clip_window_mask(ws, W)
    assert not mask[ws.kind == 0].any()
    for i in np.flatnonzero(ws.kind == 1):
        source = ws.sources[ws.source_index[i]]
        s = int(ws.start_sample[i])
        expected = any(
            length < W and s < off + length and off < s + W
            for off, length in zip(source.offsets, source.clip_samples, strict=True)
        )
        assert mask[i] == expected
    # 組 g: short は [59200, 83200)。窓 [24000, 56000) は重ならず、[28000, 60000) は重なる
    g = [i for i in range(len(ws)) if ws.sources[ws.source_index[i]].source_id == "g"]
    by_start = {int(ws.start_sample[i]): bool(mask[i]) for i in g}
    assert by_start[24000] is False
    assert by_start[28000] is True
    assert by_start[80000] is True
    # short の終わり 83200 以降に始まる窓は重ならない
    assert by_start[84000] is False


# ---------------------------------------------------------------- 正解の不確かさ


def test_shift_zero_reproduces_labels_exactly():
    sources, alignments = _fixture()
    ws = _window_set(sources, alignments)
    labels, clamped = shifted_window_labels(ws, alignments, 0.0)
    np.testing.assert_array_equal(labels, ws.mora)
    assert clamped == 0


def test_shift_moves_moras_within_clip_and_keeps_offsets():
    alignments = {"p": _alignment([1.99], 0.0), "q": _alignment([0.03], 0.0)}  # 点のモーラ
    # p: 2.5 秒、無音 0.5 秒、q: 2.0 秒。q のモーラは連結音声で 3.03 秒
    source = WindowSource("g", KIND_CONCAT, "s", ("p", "q"), (40000, 32000), (8000,))
    ws = _window_set([source], alignments)
    base = dict(zip(ws.start_sample.tolist(), ws.mora.tolist(), strict=True))
    assert base[0] == pytest.approx(1.0)  # 1.99 は [0, 2) に入る
    plus, _ = shifted_window_labels(ws, alignments, 0.05)
    shifted = dict(zip(ws.start_sample.tolist(), plus.tolist(), strict=True))
    assert shifted[0] == pytest.approx(0.0)  # 2.04 は窓 [0, 2) の外へ
    minus, clamped = shifted_window_labels(ws, alignments, -0.05)
    # q のモーラ 0.03 − 0.05 < 0 はクリップの先頭 0 に収める（連結音声で 3.0 秒。無音へは出ない）
    assert clamped == 1
    back = dict(zip(ws.start_sample.tolist(), minus.tolist(), strict=True))
    assert back[int(1.25 * SR)] == pytest.approx(2.0)  # [1.25, 3.25) に 1.94 と 3.0
    assert back[int(1.0 * SR)] == pytest.approx(1.0)  # [1.0, 3.0) には 1.94 だけ
