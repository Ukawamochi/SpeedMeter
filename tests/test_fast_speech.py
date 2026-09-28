"""高速度域の評価（src/spkrate/eval/fast_speech.py）の検証。実データは使わない。"""

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
from spkrate.eval.fast_speech import (
    RATE_BIN_KEYS,
    REPORTED_BIN_KEYS,
    binned_metrics_rows,
    binned_summary,
    build_fast_windows,
    condition_name,
    rate_bin_codes,
    select_sources,
    stretched_num_samples,
    stretched_source_moras,
    wsola_speed_up,
)

SR = 16000
W = 32000
HOP = 4000


def _alignment(times: list[float], width: float = 0.02) -> dict:
    return {"ok": True, "moras": [{"kana": "ア", "start": t, "end": t + width} for t in times]}


def _single(clip_id: str, seconds: float) -> WindowSource:
    return WindowSource(source_id=clip_id, kind=KIND_SINGLE, client_id="c", clip_ids=(clip_id,),
                        clip_samples=(int(round(seconds * SR)),), gap_samples=())


def _window_set(sources: list[WindowSource]) -> DevWindowSet:
    # 窓の配列は使わない（build_fast_windows・select_sources は音源の定義だけを読む）
    return DevWindowSet(sources=sources, source_index=np.zeros(0, np.int32),
                        start_sample=np.zeros(0, np.int64), mora=np.zeros(0, np.float32),
                        kind=np.zeros(0, np.int8), known_no_speech=np.zeros(0, bool))


# ------------------------------------------------------------------ 区間


def test_bin_keys() -> None:
    assert RATE_BIN_KEYS == ("lt6", "6to7", "7to8", "8to9", "9to10", "10to11", "11to12",
                             "12to13", "13to14", "ge14")
    assert REPORTED_BIN_KEYS[0] == "6to7" and REPORTED_BIN_KEYS[-1] == "ge14"
    assert condition_name(1.5) == "x1.5" and condition_name(2.0) == "x2"


def test_rate_bin_codes_boundaries() -> None:
    rates = np.array([0.0, 5.999, 6.0, 6.5, 6.999, 7.0, 12.0, 12.99, 13.0, 13.999, 14.0, 30.0], np.float32)
    keys = [RATE_BIN_KEYS[c] for c in rate_bin_codes(rates)]
    assert keys == ["lt6", "lt6", "6to7", "6to7", "6to7", "7to8", "12to13", "12to13", "13to14",
                    "13to14", "ge14", "ge14"]
    with pytest.raises(ValueError):
        rate_bin_codes(np.array([np.nan], np.float32))


def test_binned_summary_values() -> None:
    # 正解の毎秒モーラ数: 12.2, 12.8（12to13）, 13.5（13to14）。窓内のモーラ数は ×2
    true = np.array([12.2, 12.8, 13.5], np.float32) * 2
    pred = np.array([11.0, 12.0, 12.5], np.float32) * 2
    out = binned_summary(true, pred, 2.0)
    assert set(out) == set(RATE_BIN_KEYS)
    b = out["12to13"]
    assert b["n"] == 2
    assert b["true_mean"] == pytest.approx(12.5, abs=1e-5)
    assert b["pred_mean"] == pytest.approx(11.5, abs=1e-5)
    assert b["bias"] == pytest.approx(-1.0, abs=1e-5)
    assert b["mae"] == pytest.approx(1.0, abs=1e-5)
    assert b["pred_p10"] == pytest.approx(11.1, abs=1e-4)
    assert b["pred_p90"] == pytest.approx(11.9, abs=1e-4)
    assert out["13to14"]["n"] == 1 and out["13to14"]["bias"] == pytest.approx(-1.0, abs=1e-5)
    empty = out["ge14"]
    assert empty["n"] == 0 and math.isnan(empty["pred_mean"]) and math.isnan(empty["pred_p90"])


def test_binned_metrics_rows_split_names_and_values() -> None:
    true = np.array([6.5, 6.6, 12.5, 12.4, 3.0], np.float32) * 2
    pred = np.array([6.0, 7.0, 11.5, 11.4, 3.0], np.float32) * 2
    rows = binned_metrics_rows("dev_fast_x2", true, pred, include_overall=True)
    names = [name for name, _ in rows]
    assert names == ["dev_fast_x2", "dev_fast_x2_bin_6to7", "dev_fast_x2_bin_12to13"]
    overall = rows[0][1]
    assert overall.num_segments == 5
    bin12 = dict(rows)["dev_fast_x2_bin_12to13"]
    assert bin12.num_segments == 2
    assert bin12.mae_moras_per_sec == pytest.approx(1.0, abs=1e-5)
    assert bin12.n_band_over8 == 2
    # lt6 の区間は書かない。全体の行を省ける
    rows2 = binned_metrics_rows("dev_window", true, pred, include_overall=False)
    assert [n for n, _ in rows2] == ["dev_window_bin_6to7", "dev_window_bin_12to13"]


# ------------------------------------------------------------------ 時刻の変換と按分


def test_stretched_num_samples() -> None:
    assert stretched_num_samples(48000, 1.5) == 32000
    assert stretched_num_samples(64000, 2.0) == 32000
    assert stretched_num_samples(100001, 2.0) == 50000  # round(50000.5) は偶数丸めで 50000
    with pytest.raises(ValueError):
        stretched_num_samples(100, 0.0)


def test_stretched_source_moras_speed1_matches_source_moras() -> None:
    source = WindowSource(source_id="concat-0", kind=KIND_CONCAT, client_id="c", clip_ids=("a", "b"),
                          clip_samples=(40000, 24000), gap_samples=(8000,))
    alignments = {"a": _alignment([0.1, 0.9, 2.3]), "b": _alignment([0.2, 1.1])}
    s1, e1 = stretched_source_moras(source, alignments, SR, 1.0)
    s0, e0 = source_moras(source, alignments, SR)
    assert np.array_equal(s1, s0) and np.array_equal(e1, e0)


def test_stretched_source_moras_concat_speed2() -> None:
    # a: 2.5秒、無音0.5秒、b は 3.0秒から始まる。2倍速では時刻がすべて半分になる
    source = WindowSource(source_id="concat-0", kind=KIND_CONCAT, client_id="c", clip_ids=("a", "b"),
                          clip_samples=(40000, 24000), gap_samples=(8000,))
    alignments = {"a": _alignment([0.1, 2.3], width=0.1), "b": _alignment([0.2], width=0.1)}
    s, e = stretched_source_moras(source, alignments, SR, 2.0)
    np.testing.assert_allclose(s, [0.05, 1.15, 1.6], atol=1e-6)
    np.testing.assert_allclose(e, [0.1, 1.2, 1.65], atol=1e-6)


def test_build_fast_windows_synthetic_labels() -> None:
    # 8秒の単一クリップに 0.5秒ごと（毎秒2モーラ）の点でないモーラ（幅0.1秒）を16個置く。
    # 1.5倍速で長さ 5.333秒・毎秒3モーラ、2倍速で 4秒・毎秒4モーラ
    times = [0.2 + 0.5 * i for i in range(16)]
    alignments = {"a": _alignment(times, width=0.1), "short": _alignment([0.5, 1.0])}
    sources = [_single("a", 8.0), _single("short", 3.0)]
    ws = _window_set(sources)
    for speed, rate in ((1.5, 3.0), (2.0, 4.0)):
        fast, lengths = build_fast_windows(ws, [0, 1], alignments, speed=speed)
        assert lengths.tolist() == [int(round(8.0 * SR / speed)), int(round(3.0 * SR / speed))]
        starts = window_starts(int(lengths[0]), W, HOP)
        # 2倍速の3秒の音源（1.5秒）は窓を持たない。1.5倍速では2.0秒で窓1つ
        n_short = len(window_starts(int(lengths[1]), W, HOP))
        assert len(fast) == starts.size + n_short
        mask = fast.source_index == 0
        assert np.array_equal(fast.start_sample[mask], starts)
        expected = window_mora(np.asarray(times, np.float32) / np.float32(speed),
                               (np.asarray(times, np.float32) + np.float32(0.1)) / np.float32(speed),
                               starts.astype(np.float32) / SR, 2.0)
        np.testing.assert_allclose(fast.mora[mask], expected, atol=1e-5)
        # 中ほどの窓（端の影響が無い）の毎秒モーラ数は速さに比例する
        middle = fast.mora[mask][1:-1] / 2.0
        np.testing.assert_allclose(middle, rate, atol=0.35)
        assert np.mean(fast.mora[mask] / 2.0) == pytest.approx(rate, abs=0.35)


def test_build_fast_windows_hand_computed() -> None:
    # 2倍速: モーラ [2.0, 2.4] 秒 → [1.0, 1.2] 秒。窓 [0, 2) に全部、窓 [0.25, 2.25) にも全部、
    # モーラ [4.2, 4.6] → [2.1, 2.3]: 窓 [0.25, 2.25) に 0.15/0.2 = 0.75
    alignments = {"a": {"ok": True, "moras": [{"start": 2.0, "end": 2.4}, {"start": 4.2, "end": 4.6}]}}
    ws = _window_set([_single("a", 4.5)])
    fast, lengths = build_fast_windows(ws, [0], alignments, speed=2.0)
    assert lengths.tolist() == [36000]
    assert fast.start_sample.tolist() == [0, 4000]
    np.testing.assert_allclose(fast.mora, [1.0, 1.75], atol=1e-5)


# ------------------------------------------------------------------ 選択


def _many_sources() -> DevWindowSet:
    sources = [_single(f"s{i:03d}", 2.0 + 0.1 * i) for i in range(60)]
    sources += [WindowSource(source_id=f"concat-{i:05d}", kind=KIND_CONCAT, client_id="c",
                             clip_ids=(f"x{i}", f"y{i}", f"z{i}"), clip_samples=(30000, 30000, 30000),
                             gap_samples=(8000, 8000)) for i in range(20)]
    return _window_set(sources)


def test_select_sources_fixed_by_seed() -> None:
    ws = _many_sources()
    a = select_sources(ws, seed=202609281, num_sources=25, max_speed=2.0, window_samples=W)
    b = select_sources(ws, seed=202609281, num_sources=25, max_speed=2.0, window_samples=W)
    c = select_sources(ws, seed=1, num_sources=25, max_speed=2.0, window_samples=W)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.diff(a) > 0) and a.size == 25
    # 候補は 2倍速でも窓を持つ音源（元の長さ 4.0秒以上）だけ
    for n in a:
        assert ws.sources[int(n)].num_samples >= 2 * W
    # 期待値を直接計算して一致を確かめる
    eligible = np.array([i for i, s in enumerate(ws.sources) if round(s.num_samples / 2.0) >= W])
    expected = np.sort(eligible[np.random.default_rng(202609281).permutation(eligible.size)[:25]])
    assert np.array_equal(a, expected)
    assert any(ws.sources[int(n)].kind == KIND_CONCAT for n in a)
    with pytest.raises(ValueError):
        select_sources(ws, seed=0, num_sources=10_000, max_speed=2.0, window_samples=W)


# ------------------------------------------------------------------ WSOLA


def test_wsola_speed_up_length_and_timing() -> None:
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 6, np.float32)
    onsets = [0.3, 1.7, 2.9, 4.1, 5.5]
    for t in onsets:
        i = int(t * SR)
        x[i:i + 800] = rng.standard_normal(800).astype(np.float32) * 0.5
    for speed in (1.5, 2.0):
        y = wsola_speed_up(x, speed)
        assert y.dtype == np.float32 and y.size == stretched_num_samples(x.size, speed)
        assert np.array_equal(y, wsola_speed_up(x, speed))  # 決定的
        env = np.convolve(np.abs(y), np.ones(80) / 80, "same")
        active = np.flatnonzero(env > 0.05)
        # 0.2秒以上あいた最初の標本を立ち上がりとする
        found = active[np.r_[True, np.diff(active) > 0.2 * SR]] / SR
        assert len(found) == len(onsets)
        # 出力の時刻は t / speed に 20ms 以内で対応する
        np.testing.assert_allclose(found, np.asarray(onsets) / speed, atol=0.02)
    with pytest.raises(ValueError):
        wsola_speed_up(x, 1.3)  # 512 × 1.3 は整数でない
