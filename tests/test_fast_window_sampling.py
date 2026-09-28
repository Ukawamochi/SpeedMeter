"""高速度域の学習（docs/directives/2026-09-28-fast-speech.md タスク2）の単体テスト。

- exp014: 時間伸縮の範囲 0.5〜1.5倍（spec の注）でも、既存の設定の乱数の系列が変わらないこと、
  方式Bの下限 max(下限, 2.0 ÷ 音源長) が保たれること
- exp015: 速い窓の選び直し（windows.fast_window_redraws。docs/decisions/011-fast-window-sampling.md）が
  既定で無効で、有効にしても主の乱数の系列・窓数を変えず、速い窓だけを増やすこと

音声は合成で、実データは読まない。
"""

from __future__ import annotations

import importlib.util
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spkrate.data.augment import AugmentConfig
from spkrate.train.method_b import (
    FAST_REDRAW_TAG,
    TrainClip,
    WindowEpochStats,
    WindowSettings,
    WindowTrainDataset,
    draw_window_plan,
    plan_for_stretch,
    redraw_stretch_keep_position,
    stretch_lower_bound,
)

SR = 16000
W = 32000
WIDE = (0.5, 1.5)


def _moras(duration: float, step: float = 0.13) -> list[tuple[float, float]]:
    """毎秒約7.7モーラの等間隔のモーラ（伸縮率0.77以下で毎秒10以上になる）。"""
    return [(float(t), float(t) + 0.02) for t in np.arange(0.05, duration - 0.02, step)]


class _Corpus:
    def __init__(self, num_speakers: int = 4, clips_per_speaker: int = 5) -> None:
        rng = np.random.default_rng(0)
        self.clips: list[TrainClip] = []
        starts: list[float] = []
        ends: list[float] = []
        self.waves: dict[str, np.ndarray] = {}
        for s in range(num_speakers):
            for c in range(clips_per_speaker):
                dur = float(rng.uniform(2.0, 7.0))
                n = int(round(dur * SR))
                moras = _moras(n / SR)
                path = f"audio/s{s}c{c}.wav"
                self.clips.append(TrainClip(f"s{s}c{c}", f"spk{s}", path, n, len(starts), len(moras)))
                starts.extend(a for a, _ in moras)
                ends.extend(b for _, b in moras)
                wave_rng = np.random.default_rng(zlib.crc32(path.encode()))
                self.waves[path] = (wave_rng.standard_normal(n) * 0.1).astype(np.float32)
        self.starts = np.asarray(starts, dtype=np.float32)
        self.ends = np.asarray(ends, dtype=np.float32)

    def dataset(self, *, settings: WindowSettings = WindowSettings(), augment=None, seed: int = 7):
        return WindowTrainDataset(
            self.clips,
            self.starts,
            self.ends,
            settings=settings,
            augment=augment,
            seed=seed,
            loader=lambda p: self.waves[p].copy(),
        )


def _wide_log_uniform(prob: float = 0.5) -> AugmentConfig:
    """exp014 の伸縮（0.5〜1.5倍・対数一様）。他の拡張は外す。"""
    return replace(
        AugmentConfig().disabled(),
        time_stretch_prob=prob,
        time_stretch_range=WIDE,
        time_stretch_distribution="log_uniform",
    )


# --------------------------------------------------------------------------------------
# exp014: 範囲 0.5〜1.5


def test_defaults_are_unchanged() -> None:
    """既定の伸縮の範囲は0.7〜1.5倍のまま、速い窓の選び直しは無効。"""
    assert AugmentConfig().time_stretch_range == (0.7, 1.5)
    assert AugmentConfig().time_stretch_distribution == "uniform"
    settings = WindowSettings()
    assert settings.fast_window_redraws == 0
    assert settings.fast_window_min_rate == 10.0
    assert WindowSettings.from_mapping(None) == settings


def test_wide_range_keeps_random_stream_and_lower_bound() -> None:
    """0.5〜1.5倍でも抽選の当たり外れと後続の乱数は既定と同じで、伸縮率は [max(0.5, W/N), 1.5]。"""
    wide = replace(AugmentConfig(), time_stretch_range=WIDE, time_stretch_distribution="log_uniform")
    below_07 = 0
    drawn = 0
    for i in range(600):
        n = [40000, 64000, 120000][i % 3]
        r0, r1 = np.random.default_rng((5, 1, i)), np.random.default_rng((5, 1, i))
        p0 = draw_window_plan(r0, n, window_samples=W, margin_samples=8000, augment=AugmentConfig())
        p1 = draw_window_plan(r1, n, window_samples=W, margin_samples=8000, augment=wide)
        assert p0.stretch_drawn == p1.stretch_drawn
        assert r0.random() == r1.random()
        if p1.stretch_drawn:
            drawn += 1
            low = stretch_lower_bound(0.5, n, W)
            assert low == max(0.5, W / n)
            assert low <= p1.stretch <= 1.5
            assert p1.stretch * n >= W - 0.5  # 伸縮後も窓が切れる
            below_07 += int(p1.stretch < 0.7)
    assert drawn > 200
    assert below_07 > 0  # 120000標本（7.5秒）の音源では0.7未満が出る


def test_short_source_cannot_reach_half_speed() -> None:
    """4秒（64000標本）未満の音源では、下限 W/N が0.5より上がり2倍速に届かない。"""
    assert stretch_lower_bound(0.5, 40000, W) == pytest.approx(0.8)
    assert stretch_lower_bound(0.5, 64000, W) == pytest.approx(0.5)
    assert stretch_lower_bound(0.5, 128000, W) == 0.5


# --------------------------------------------------------------------------------------
# exp015: 速い窓の選び直し


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"fast_window_redraws": -1}, "0以上の整数"),
        ({"fast_window_redraws": 1.5}, "0以上の整数"),
        ({"fast_window_redraws": True}, "0以上の整数"),
        ({"fast_window_min_rate": 0.0}, "正の有限値"),
        ({"fast_window_min_rate": float("nan")}, "正の有限値"),
    ],
)
def test_settings_validation(mapping: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WindowSettings.from_mapping(mapping)


def test_redraws_require_time_stretch() -> None:
    corpus = _Corpus()
    settings = WindowSettings(fast_window_redraws=2)
    with pytest.raises(ValueError, match="時間伸縮が有効"):
        corpus.dataset(settings=settings, augment=None)
    with pytest.raises(ValueError, match="時間伸縮が有効"):
        corpus.dataset(settings=settings, augment=replace(_wide_log_uniform(), time_stretch_enabled=False))


@pytest.mark.parametrize("augment", [AugmentConfig(), _wide_log_uniform()])
def test_disabled_redraws_match_plain_plan(augment: AugmentConfig) -> None:
    """既定（K=0）の窓の取り方と後続の乱数は ``draw_window_plan`` をそのまま使った場合と同じ。"""
    corpus = _Corpus()
    ds = corpus.dataset(augment=augment)
    ds.set_epoch(1)
    for index in range(len(ds)):
        source = ds.source(index)
        rng, plan = ds._draw_plan(index, source)
        ref_rng = np.random.default_rng((ds.seed, 1, index))
        ref = draw_window_plan(
            ref_rng, source.num_samples, window_samples=W, margin_samples=8000, augment=augment
        )
        assert plan == ref
        assert plan.fast_redraws == 0
        assert rng.random() == ref_rng.random()


def test_redraws_keep_main_stream_count_and_only_add_fast_windows() -> None:
    """有効（K=3）でも主の乱数の系列と窓数は同じ。変わる窓は伸縮した遅い窓だけで、変わった窓は速い。"""
    corpus = _Corpus()
    augment = _wide_log_uniform()
    base = corpus.dataset(augment=augment)
    fast = corpus.dataset(settings=WindowSettings(fast_window_redraws=3), augment=augment)
    base.set_epoch(1)
    fast.set_epoch(1)
    assert len(base) == len(fast)
    threshold = 10.0 * 2.0
    changed = 0
    n_fast_base = n_fast = 0
    for index in range(len(base)):
        lb, pb, sb = base.window_label(index)
        lf, pf, sf = fast.window_label(index)
        assert sb == sf
        rb, _ = base._draw_plan(index, sb)
        rf, _ = fast._draw_plan(index, sf)
        assert rb.random() == rf.random()  # 残りの拡張・周波数マスクの系列は同じ
        n_fast_base += int(lb >= threshold)
        n_fast += int(lf >= threshold)
        if pf == pb:
            assert pf.fast_redraws == 0
            if pb.stretch_drawn and lb < threshold:
                pass  # 引き直しても速くならなかった窓は元のまま
            continue
        changed += 1
        assert pb.stretch_drawn and lb < threshold  # 引き直すのは伸縮した遅い窓だけ
        assert 1 <= pf.fast_redraws <= 3
        assert lf >= threshold
        n = sb.num_samples
        assert stretch_lower_bound(0.5, n, W) <= pf.stretch <= 1.5
        assert 0 <= pf.start <= pf.stretched_length - W
        # 窓の相対位置を保つ
        old_span = pb.stretched_length - W
        new_span = pf.stretched_length - W
        if old_span > 0 and new_span > 0:
            assert abs(pf.start / new_span - pb.start / old_span) <= 1.0 / new_span + 1e-9
    assert changed > 0
    assert n_fast == n_fast_base + changed


def test_redraw_uses_separate_generator() -> None:
    """引き直しの伸縮率は種 (seed, epoch, index, FAST_REDRAW_TAG) の生成器から順に引く。"""
    corpus = _Corpus()
    augment = _wide_log_uniform()
    ds = corpus.dataset(settings=WindowSettings(fast_window_redraws=3), augment=augment)
    ds.set_epoch(2)
    checked = 0
    for index in range(len(ds)):
        _, plan, source = ds.window_label(index)
        if plan.fast_redraws == 0:
            continue
        base = draw_window_plan(
            np.random.default_rng((ds.seed, 2, index)),
            source.num_samples,
            window_samples=W,
            margin_samples=8000,
            augment=augment,
        )
        sub = np.random.default_rng((ds.seed, 2, index, FAST_REDRAW_TAG))
        for attempt in range(1, plan.fast_redraws + 1):
            candidate = redraw_stretch_keep_position(
                base, sub, source.num_samples, window_samples=W, margin_samples=8000,
                augment=augment, attempt=attempt,
            )
        assert candidate == plan
        checked += 1
    assert checked > 0


def test_redraw_window_matches_label_without_audio() -> None:
    """選び直した窓でも ``window``（音声を伸縮して切る）と ``window_label`` の正解・取り方が一致する。"""
    corpus = _Corpus(num_speakers=3, clips_per_speaker=4)
    ds = corpus.dataset(settings=WindowSettings(fast_window_redraws=3), augment=_wide_log_uniform())
    ds.set_epoch(1)
    redrawn = 0
    for index in range(len(ds)):
        label, plan, source = ds.window_label(index)
        info = ds.window(index)
        assert plan == info["plan"]
        assert label == float(info["mora"])
        assert info["waveform"].shape == (W,)
        redrawn += int(plan.fast_redraws > 0)
    assert redrawn > 0


def test_plan_for_stretch_matches_draw_window_plan() -> None:
    for i in range(200):
        n = [40000, 64000, 120000][i % 3]
        plan = draw_window_plan(
            np.random.default_rng((3, 0, i)), n, window_samples=W, margin_samples=8000,
            augment=_wide_log_uniform(),
        )
        again = plan_for_stretch(
            n, stretch=plan.stretch, stretch_drawn=plan.stretch_drawn, start=plan.start,
            window_samples=W, margin_samples=8000,
        )
        assert again == plan


def test_window_epoch_stats_counts_high_rates() -> None:
    stats = WindowEpochStats()
    # 毎秒 9.5・10・12.5・14・0 と、無音サンプル（種類が空）
    stats.update(
        ["single", "concat", "single", "concat", "single", ""],
        [1.0] * 6,
        [19.0, 20.0, 25.0, 28.0, 0.0, 40.0],
        [2.0] * 6,
    )
    d = stats.as_dict()
    assert d["window_total"] == 5
    assert (d["window_rate_ge10"], d["window_rate_ge12"], d["window_rate_ge14"]) == (3, 2, 1)
    assert "毎秒10/12/14以上=3/2/1" in stats.describe(1)


# --------------------------------------------------------------------------------------
# 集計スクリプト（scripts/epoch_window_label_distribution.py）の伸縮率の分布


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "epoch_window_label_distribution.py"
    spec = importlib.util.spec_from_file_location("epoch_window_label_distribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_stretch_summary() -> None:
    mod = _load_script()
    assert mod.nominal_stretch_cdf(1.0, 0.5, 1.5, "log_uniform") == pytest.approx(np.log(2) / np.log(3))
    assert mod.nominal_stretch_cdf(1.0, 0.7, 1.5, "uniform") == pytest.approx(0.375)
    stretches = np.asarray([0.55, 0.65, 0.95, 1.2, 1.0, 1.5], dtype=np.float32)
    drawn = np.asarray([True, True, True, True, False, True])
    cut = np.asarray([False, True, False, False, True, False])
    out = mod.summarize_stretch(stretches, drawn, cut, 0.5, 1.5, "log_uniform")
    assert out["stretched"] == 5
    assert out["below1"] == pytest.approx(3 / 5)
    assert out["cut_by_lower_bound"] == pytest.approx(1 / 5)
    assert out["bins"]["0.5-0.6"]["actual"] == pytest.approx(1 / 5)
    assert out["bins"]["1.4-1.5"]["actual"] == pytest.approx(1 / 5)  # 上端を含む
    assert sum(v["nominal"] for v in out["bins"].values()) == pytest.approx(1.0)
    labels = np.asarray([19.9, 20.0, 24.0, 28.0], dtype=np.float32)
    summary = mod.summarize(labels, ["single"] * 4, np.ones(4, np.float32), 2.0)
    assert (summary["rate_ge10"], summary["rate_ge12"], summary["rate_ge14"]) == (0.75, 0.5, 0.25)
