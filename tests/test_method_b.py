"""方式Bの学習データの単体テスト（docs/decisions/009-method-b.md 4節）。

4.1 窓の正解の計算、4.2 時刻の変換、4.3 連結の生成、4.4 その他。
音声は合成（``loader`` で clip の audio_path から決定的な波形を返す）で、実データは読まない。
"""

from __future__ import annotations

import json
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spkrate.data.augment import AugmentConfig, ArrayNoiseSource, time_stretch
from spkrate.eval.dev_window import (
    KIND_CONCAT,
    KIND_SINGLE,
    ConcatSpec,
    WindowSource,
    build_concat_groups,
    window_mora,
)
from spkrate.eval.dev_window_val import (
    DevWindowValDataset,
    main_indicator_mask,
    select_val_windows,
)
from spkrate.train.method_b import (
    TrainClip,
    WindowEpochStats,
    WindowSettings,
    WindowTrainDataset,
    assemble_excerpt,
    build_epoch_concat_groups,
    draw_window_plan,
    load_train_clips,
    read_clip_list,
    stretch_lower_bound,
    stretched_window_label,
    window_start_in_excerpt,
)
from spkrate.train.silence import SilenceSettings, build_silence_dataset

SR = 16000
W = 32000
SPEC = ConcatSpec()


# --------------------------------------------------------------------------------------
# 合成データ


def _waveform_for(path: str, num_samples: int) -> np.ndarray:
    """clip ごとに違う決定的な波形（ビット一致の確認用）。"""
    seed = zlib.crc32(path.encode("utf-8"))
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(num_samples) * 0.1).astype(np.float32)


class Clips:
    """(clip_id, client_id, 標本数, モーラ区間の列) から TrainClip 一式と loader を作る。"""

    def __init__(self, specs: list[tuple[str, str, int, list[tuple[float, float]]]]) -> None:
        self.clips: list[TrainClip] = []
        starts: list[float] = []
        ends: list[float] = []
        self.waves: dict[str, np.ndarray] = {}
        for clip_id, client_id, n, moras in specs:
            path = f"audio/{clip_id}.wav"
            self.clips.append(
                TrainClip(clip_id, client_id, path, n, len(starts), len(moras))
            )
            starts.extend(s for s, _ in moras)
            ends.extend(e for _, e in moras)
            self.waves[path] = _waveform_for(path, n)
        self.starts = np.asarray(starts, dtype=np.float32)
        self.ends = np.asarray(ends, dtype=np.float32)

    def loader(self, path: str) -> np.ndarray:
        return self.waves[path].copy()

    def dataset(self, **kwargs) -> WindowTrainDataset:
        kwargs.setdefault("seed", 7)
        return WindowTrainDataset(
            self.clips, self.starts, self.ends, loader=self.loader, **kwargs
        )


def _regular_moras(duration: float, step: float = 0.13, length: float = 0.02) -> list[tuple[float, float]]:
    return [(t, t + length) for t in np.arange(0.05, duration - length, step)]


def _corpus(num_speakers: int = 4, clips_per_speaker: int = 5, *, with_short: bool = True) -> Clips:
    specs = []
    rng = np.random.default_rng(0)
    for s in range(num_speakers):
        for c in range(clips_per_speaker):
            if with_short and c == 0:
                dur = 1.4  # 2.0秒未満（連結の中でだけ使う）
            else:
                dur = float(rng.uniform(2.0, 5.0))
            n = int(round(dur * SR))
            specs.append((f"s{s}c{c}", f"spk{s}", n, _regular_moras(n / SR)))
    # 2件しか無い話者（連結に使えない）
    for c in range(2):
        n = int(round(3.0 * SR))
        specs.append((f"lonely{c}", "spk_lonely", n, _regular_moras(3.0)))
    return Clips(specs)


NO_STRETCH_ONLY = AugmentConfig().disabled()  # 全拡張の確率0


def _stretch_only(prob: float = 1.0) -> AugmentConfig:
    return replace(AugmentConfig().disabled(), time_stretch_prob=prob)


# ======================================================================================
# 4.1 窓の正解の計算


def test_single_window_label_matches_dev_window_function() -> None:
    """伸縮なしの単一クリップ: 正解は window_mora(開始, 終了, [w / sr], 2.0) と一致し、波形も一致する。"""
    corpus = _corpus()
    ds = corpus.dataset(augment=None)
    for index in range(ds.num_single):
        info = ds.window(index)
        source: WindowSource = info["source"]
        assert source.kind == KIND_SINGLE
        clip = next(c for c in corpus.clips if c.clip_id == source.clip_ids[0])
        sl = slice(clip.mora_offset, clip.mora_offset + clip.mora_count)
        w = info["plan"].start
        expected = window_mora(corpus.starts[sl], corpus.ends[sl], np.array([w / SR], np.float32), 2.0)[0]
        assert info["mora"] == pytest.approx(float(expected), abs=1e-5)
        wave = corpus.waves[clip.audio_path]
        np.testing.assert_array_equal(info["waveform"], wave[w : w + W])


@pytest.mark.parametrize("stretch", [0.7, 1.0, 1.3, 1.5])
def test_conservation_windows_tiling_sum_to_total(stretch: float) -> None:
    """重なりなく並べた窓の正解の和が、覆った範囲の全モーラ数（按分）と一致する。"""
    rng = np.random.default_rng(1)
    starts = np.sort(rng.uniform(0.0, 9.0, 60)).astype(np.float32)
    ends = (starts + rng.uniform(0.0, 0.2, 60)).astype(np.float32)  # 点のモーラも含む
    a = 8000  # 抜粋の開始（標本）
    total = 0.0
    tiles = 3
    for i in range(tiles):
        total += float(
            stretched_window_label(starts, ends, excerpt_start=a, stretch=stretch, window_start=i * W, window_sec=2.0)
        )
    # 伸縮後の [0, 3W) は元の [a/sr, a/sr + 3·2.0/s)
    t0 = a / SR
    expected = window_mora(starts, ends, np.array([t0], np.float32), tiles * 2.0 / stretch)[0]
    assert total == pytest.approx(float(expected), abs=1e-3)


def test_point_mora_inside_counts_one_and_right_edge_zero() -> None:
    starts = np.array([1.0, 2.0], np.float32)
    ends = np.array([1.0, 2.0], np.float32)  # 点（end <= start）
    inside = stretched_window_label(starts[:1], ends[:1], excerpt_start=0, stretch=1.0, window_start=0, window_sec=2.0)
    edge = stretched_window_label(starts[1:], ends[1:], excerpt_start=0, stretch=1.0, window_start=0, window_sec=2.0)
    assert inside == 1.0
    assert edge == 0.0


def test_window_start_range_covers_both_ends() -> None:
    """開始は0以上 N' − W 以下で両端が選ばれうる。32,000標本ちょうどは開始0だけ。"""
    for n, expected in ((W + 3, {0, 1, 2, 3}), (W, {0})):
        seen = set()
        for seed in range(400):
            plan = draw_window_plan(
                np.random.default_rng(seed), n, window_samples=W, margin_samples=8000, augment=None
            )
            assert 0 <= plan.start <= plan.stretched_length - W
            seen.add(plan.start)
        assert seen == expected
    with pytest.raises(ValueError):
        draw_window_plan(np.random.default_rng(0), W - 1, window_samples=W, margin_samples=0, augment=None)


def test_item_output_shapes_and_label_range() -> None:
    corpus = _corpus()
    noise = ArrayNoiseSource([np.random.default_rng(3).standard_normal(SR).astype(np.float32)])
    ds = corpus.dataset(augment=AugmentConfig(), noise_source=noise)
    ds.set_epoch(1)
    for index in list(range(0, len(ds), 3)):
        item = ds[index]
        assert item.features.shape == (201, 80)
        assert item.features.dtype == np.float32
        assert isinstance(item.mora, float)
        assert np.float32(item.mora) == item.mora  # float32 で計算した値
        source = ds.source(index)
        total = sum(c.mora_count for c in corpus.clips if c.clip_id in source.clip_ids)
        assert 0.0 <= item.mora <= total + 1e-4
        assert item.duration_sec == 2.0
        assert item.kind in (KIND_SINGLE, KIND_CONCAT)


def test_time_preserving_augmentations_do_not_change_label() -> None:
    """残響・雑音・帯域制限・音量・周波数マスクの有無で、同じ乱数の窓の正解が変わらない。"""
    corpus = _corpus()
    noise = ArrayNoiseSource([np.random.default_rng(3).standard_normal(SR).astype(np.float32)])
    full = replace(
        AugmentConfig(),
        reverb_prob=1.0,
        noise_prob=1.0,
        band_limit_prob=1.0,
        volume_prob=1.0,
        freq_mask_prob=1.0,
    )
    only_stretch = replace(AugmentConfig().disabled(), time_stretch_prob=full.time_stretch_prob)
    ds_full = corpus.dataset(augment=full, noise_source=noise)
    ds_plain = corpus.dataset(augment=only_stretch)
    changed = 0
    for index in range(len(ds_full)):
        a, b = ds_full.window(index), ds_plain.window(index)
        assert a["plan"] == b["plan"]
        assert a["mora"] == b["mora"]
        changed += int(not np.array_equal(a["waveform"], b["waveform"]))
    assert changed > 0  # 拡張は実際に掛かっている


# ======================================================================================
# 4.2 時刻の変換


def test_no_stretch_is_identity_and_bit_exact_for_concat_too() -> None:
    corpus = _corpus()
    ds = corpus.dataset(augment=_stretch_only(prob=0.0))
    for index in range(len(ds)):
        info = ds.window(index)
        plan, source = info["plan"], info["source"]
        assert plan.stretch == 1.0
        assert info["excerpt_window_start"] == plan.start - plan.excerpt_start
        full = _full_source_waveform(source, corpus)
        np.testing.assert_array_equal(info["waveform"], full[plan.start : plan.start + W])


@pytest.mark.parametrize("stretch", [0.7, 1.0, 1.5])
def test_label_linearity_under_time_mapping(stretch: float) -> None:
    rng = np.random.default_rng(5)
    starts = np.sort(rng.uniform(0.0, 8.0, 50)).astype(np.float32)
    ends = (starts + rng.uniform(0.005, 0.1, 50)).astype(np.float32)
    a, k = 12345, 7777
    got = stretched_window_label(starts, ends, excerpt_start=a, stretch=stretch, window_start=k, window_sec=2.0)
    t0 = a / SR + k / (SR * stretch)
    expected = window_mora(starts, ends, np.array([t0], np.float32), 2.0 / stretch)[0]
    assert float(got) == pytest.approx(float(expected), abs=1e-3)


@pytest.mark.parametrize("stretch", [0.7, 1.5])
def test_click_positions_follow_time_mapping(stretch: float) -> None:
    """既知の時刻のクリックが、抜粋の切り出しと time_stretch の後に変換後の時刻へ来る。

    位相ボコーダ（窓1024点＝64ミリ秒、刻み256点＝16ミリ秒）は短いバーストをフレーム単位で
    ずらしてにじませるので、1件ごとの許容は刻み2つ分（32ミリ秒）とする。ずれは偏りのない
    ばらつきであることを、全クリックの平均のずれ（8ミリ秒以内）で確かめる（系統的なずれ＝
    時刻の変換の誤りなら平均に現れる）。
    """
    from spkrate.data.augment import STFT_HOP_LENGTH

    n = 10 * SR
    signal = np.zeros(n, np.float32)
    clicks = np.round(np.arange(1.1, 8.6, 0.61), 3)
    burst = (np.hanning(160) * np.sin(2 * np.pi * 1000 * np.arange(160) / SR)).astype(np.float32)
    for t in clicks:
        i = int(t * SR)
        signal[i : i + 160] += burst
    a, b = int(0.8 * SR), int(9.2 * SR)
    stretched = time_stretch(signal[a:b], stretch)
    assert stretched.size == int(round((b - a) * stretch))
    energy = stretched.astype(np.float32) ** 2
    errors = []
    for t in clicks:
        predicted = ((t + 0.005) - a / SR) * stretch  # バーストの中心を変換した時刻
        centre = int(predicted * SR)
        lo, hi = max(0, centre - int(0.1 * SR)), min(stretched.size, centre + int(0.1 * SR))
        seg = energy[lo:hi]
        observed = (lo + float(np.sum(np.arange(seg.size) * seg) / np.sum(seg))) / SR
        errors.append(observed - predicted)
    errors = np.asarray(errors)
    assert np.max(np.abs(errors)) < 2 * STFT_HOP_LENGTH / SR
    assert abs(float(np.mean(errors))) < 0.008


def test_stretch_lower_bound_keeps_window_possible() -> None:
    aug = _stretch_only(prob=1.0)
    for n in (W, 33000, 40000, 45714, 45715, 60000):
        low = stretch_lower_bound(0.7, n, W)
        assert low == max(0.7, W / n)
        for seed in range(50):
            plan = draw_window_plan(np.random.default_rng(seed), n, window_samples=W, margin_samples=8000, augment=aug)
            assert plan.stretch_drawn
            assert low <= plan.stretch <= 1.5
            assert plan.stretched_length >= W
            assert 0 <= plan.start <= plan.stretched_length - W


def test_excerpt_window_start_clamped_and_label_uses_actual_k() -> None:
    assert window_start_in_excerpt(100, 200, 1.0, 40000, W) == 0
    assert window_start_in_excerpt(50000, 0, 1.0, 40000, W) == 40000 - W
    assert window_start_in_excerpt(1000, 100, 1.5, 40000, W) == 1000 - 150
    corpus = _corpus()
    ds = corpus.dataset(augment=_stretch_only(prob=1.0))
    differs = 0
    for index in range(len(ds)):
        info = ds.window(index)
        plan, k = info["plan"], info["excerpt_window_start"]
        excerpt_len = int(round((plan.excerpt_end - plan.excerpt_start) * plan.stretch))
        assert 0 <= k <= excerpt_len - W
        starts, ends = ds.source_moras(info["source"])
        kwargs = dict(excerpt_start=plan.excerpt_start, stretch=plan.stretch, window_sec=2.0)
        assert info["mora"] == stretched_window_label(starts, ends, window_start=k, **kwargs)
        shifted = stretched_window_label(starts, ends, window_start=k + 1, **kwargs)
        differs += int(shifted != info["mora"])
    assert differs > 0  # k をずらすと正解も変わる（実際に使った k で計算している）


# ======================================================================================
# 4.3 連結の生成


def _full_source_waveform(source: WindowSource, corpus: Clips) -> np.ndarray:
    pieces = []
    for i, clip_id in enumerate(source.clip_ids):
        clip = next(c for c in corpus.clips if c.clip_id == clip_id)
        pieces.append(corpus.waves[clip.audio_path])
        if i < len(source.gap_samples):
            pieces.append(np.zeros(source.gap_samples[i], np.float32))
    return np.concatenate(pieces)


def test_groups_same_speaker_sizes_gaps_and_unique_within_pass() -> None:
    corpus = _corpus(num_speakers=5, clips_per_speaker=9)
    speaker_of = {c.clip_id: c.client_id for c in corpus.clips}
    by_speaker: dict[str, list[str]] = {}
    for c in corpus.clips:
        by_speaker.setdefault(c.client_id, []).append(c.clip_id)
    groups, _ = build_concat_groups(by_speaker, seed=(7, 1, 0xC0CA7, 0), spec=SPEC, sample_rate=SR)
    used = [c for _, members, _ in groups for c in members]
    assert len(used) == len(set(used))  # 1回の作成の中で高々1回
    for client_id, members, gaps in groups:
        assert 3 <= len(members) <= 5
        assert {speaker_of[m] for m in members} == {client_id}
        assert len(gaps) == len(members) - 1
        for g in gaps:
            assert 0.3 * SR <= g <= 1.5 * SR
            assert g % 160 == 0  # 0.01秒の倍数
    epoch_groups, stats = build_epoch_concat_groups(by_speaker, seed=7, epoch=1, count=100, spec=SPEC)
    assert len(epoch_groups) == 100 and stats["passes"] >= 2
    for client_id, members, _ in epoch_groups:
        assert {speaker_of[m] for m in members} == {client_id}


def test_reproducible_per_epoch_and_changes_across_epochs() -> None:
    corpus = _corpus()
    ds1 = corpus.dataset(augment=_stretch_only(prob=0.5))
    ds2 = corpus.dataset(augment=_stretch_only(prob=0.5))
    ds1.set_epoch(3)
    ds2.set_epoch(3)
    assert ds1.groups == ds2.groups
    for index in range(len(ds1)):
        a, b = ds1.window(index), ds2.window(index)
        assert a["plan"] == b["plan"] and a["mora"] == b["mora"]
        np.testing.assert_array_equal(a["waveform"], b["waveform"])
    before = list(ds1.groups)
    ds1.set_epoch(4)
    assert ds1.groups != before


def test_ratio_one_to_one_and_constant_length() -> None:
    corpus = _corpus()
    ds = corpus.dataset(augment=None)
    lengths = set()
    for epoch in range(1, 5):
        ds.set_epoch(epoch)
        assert len(ds.groups) == ds.num_single
        kinds = [ds.source(i).kind for i in range(len(ds))]
        assert kinds.count(KIND_SINGLE) == kinds.count(KIND_CONCAT) == ds.num_single
        lengths.add(len(ds))
    assert lengths == {2 * ds.num_single}


def test_short_clips_only_in_concat_and_small_speakers_excluded() -> None:
    corpus = _corpus()
    ds = corpus.dataset(augment=None)
    short = {c.clip_id for c in corpus.clips if c.num_samples < W}
    assert short
    singles = {ds.source(i).clip_ids[0] for i in range(ds.num_single)}
    assert not singles & short
    in_groups = set()
    for epoch in range(1, 4):
        ds.set_epoch(epoch)
        in_groups |= {c for _, members, _ in ds.groups for c in members}
    assert short & in_groups
    assert not {"lonely0", "lonely1"} & in_groups
    assert ds.stats["speakers_too_few_clips"] == 1


def test_concat_label_uses_offsets_and_boundaries() -> None:
    # 1件目: 2.5秒、モーラは 0.5〜2.3 に等間隔。2件目: 3.0秒。無音 1.0 秒
    m1 = [(0.5 + 0.2 * i, 0.52 + 0.2 * i) for i in range(10)]
    m2 = [(0.1 + 0.25 * i, 0.12 + 0.25 * i) for i in range(10)]
    corpus = Clips([
        ("a", "s", int(2.5 * SR), m1),
        ("b", "s", int(3.0 * SR), m2),
        ("c", "s", int(3.0 * SR), m2),  # 組が作れるように3件にする（この試験では使わない）
    ])
    ds = corpus.dataset(augment=None)
    kw = dict(excerpt_start=0, stretch=1.0, window_sec=2.0)
    # 無音 2.0 秒の組: 窓 [2.4, 4.4) は1件目の末尾の無音（2.32秒以降）・挟んだ無音（2.5〜4.5）だけ → 0
    wide = WindowSource("w", KIND_CONCAT, "s", ("a", "b"), (int(2.5 * SR), int(3.0 * SR)), (2 * SR,))
    starts, ends = ds.source_moras(wide)
    assert stretched_window_label(starts, ends, window_start=int(2.4 * SR), **kw) == 0.0
    # 無音 1.0 秒の組: 2件目の時刻は 3.5 秒ずれる
    source = WindowSource("g", KIND_CONCAT, "s", ("a", "b"), (int(2.5 * SR), int(3.0 * SR)), (SR,))
    starts, ends = ds.source_moras(source)
    shift = 3.5
    np.testing.assert_allclose(starts[10:], np.array([s for s, _ in m2], np.float32) + shift, atol=1e-6)
    # 境界をまたぐ窓 [2.0, 4.0): 1件目の 2.1, 2.3（2件）+ 2件目の 3.6, 3.85（2件）
    both = stretched_window_label(starts, ends, window_start=2 * SR, **kw)
    first = window_mora(np.array([s for s, _ in m1], np.float32), np.array([e for _, e in m1], np.float32), np.array([2.0], np.float32), 2.0)[0]
    second = window_mora(np.array([s for s, _ in m2], np.float32), np.array([e for _, e in m2], np.float32), np.array([2.0 - shift], np.float32), 2.0)[0]
    assert float(both) == pytest.approx(float(first + second), abs=1e-5)
    assert float(first) == pytest.approx(2.0, abs=1e-5) and float(second) == pytest.approx(2.0, abs=1e-5)


def test_partial_assembly_matches_full_concatenation() -> None:
    corpus = _corpus()
    ds = corpus.dataset(augment=None)
    calls: list[str] = []

    def counting_loader(path: str) -> np.ndarray:
        calls.append(path)
        return corpus.loader(path)

    for index in range(ds.num_single, len(ds)):
        source = ds.source(index)
        full = _full_source_waveform(source, corpus)
        for a, b in ((0, source.num_samples), (5000, 5000 + 40000), (source.num_samples - 30000, source.num_samples)):
            a = max(0, a)
            calls.clear()
            part = assemble_excerpt(source, a, b, lambda cid: counting_loader(f"audio/{cid}.wav"))
            np.testing.assert_array_equal(part, full[a:b])
            overlapping = sum(
                1 for off, n in zip(source.offsets, source.clip_samples) if max(a, off) < min(b, off + n)
            )
            assert len(calls) == overlapping  # 重なるクリップだけを読む


def test_build_concat_groups_int_seed_unchanged() -> None:
    by_speaker = {"A": list("abcdefg"), "B": list("hijk")}
    as_int, s1 = build_concat_groups(by_speaker, seed=5, spec=SPEC, sample_rate=SR)
    as_list, s2 = build_concat_groups(by_speaker, seed=[5], spec=SPEC, sample_rate=SR)
    assert as_int == as_list and s1 == s2
    other, _ = build_concat_groups(by_speaker, seed=(5, 1), spec=SPEC, sample_rate=SR)
    assert other != as_int


def test_no_groups_possible_raises() -> None:
    with pytest.raises(ValueError, match="連結の組"):
        build_epoch_concat_groups({"A": ["a", "b"]}, seed=0, epoch=0, count=3, spec=SPEC)


# ======================================================================================
# 4.4 その他


def test_silence_samples_are_two_seconds_and_three_percent() -> None:
    corpus = _corpus(num_speakers=20, clips_per_speaker=6)
    ds = corpus.dataset(augment=None)
    settings = SilenceSettings(enabled=True, mix={"digital_silence": 1.0, "musan_noise": 0.0, "quiet_noise": 0.0})
    silence = build_silence_dataset(settings, ds, normalizer=None, seed=1, match_feature_storage=False)
    assert len(silence) == int(np.floor(0.03 * 2 * ds.num_single + 0.5))
    assert len(silence) > 0
    assert set(silence.num_samples) == {W}
    assert silence[0].features.shape == (201, 80)


def _fake_window_set() -> SimpleNamespace:
    sources = [
        SimpleNamespace(clip_ids=("a",)),
        SimpleNamespace(clip_ids=("b",)),
        SimpleNamespace(clip_ids=("c", "d", "e")),
        SimpleNamespace(clip_ids=("f", "g", "h")),
    ]
    source_index = np.repeat(np.arange(4), [30, 20, 40, 10]).astype(np.int32)
    known = np.isin(source_index, [1])
    return SimpleNamespace(sources=sources, source_index=source_index, known_no_speech=known)


def test_val_selection_main_indicator_and_deterministic() -> None:
    ws = _fake_window_set()
    mask = main_indicator_mask(ws, {"b", "d"})  # b（known）と d（suspect、連結の組ごと除く）
    assert mask.sum() == 30 + 10
    picked = select_val_windows(mask, 25, seed=20260928)
    assert picked.size == 25
    assert np.all(np.diff(picked) > 0)
    assert np.all(mask[picked])
    np.testing.assert_array_equal(picked, select_val_windows(mask, 25, seed=20260928))
    assert not np.array_equal(picked, select_val_windows(mask, 25, seed=1))
    np.testing.assert_array_equal(select_val_windows(mask, 1000, seed=0), np.flatnonzero(mask))
    with pytest.raises(ValueError, match="known_no_speech"):
        main_indicator_mask(ws, {"d"})  # known の窓が除外に含まれない


def test_dev_window_val_dataset_reads_and_normalizes(tmp_path: Path) -> None:
    from spkrate.train.data import Normalizer

    n = 6
    feats = np.lib.format.open_memmap(tmp_path / "features.npy", mode="w+", dtype=np.float16, shape=(n, 201, 80))
    feats[:] = np.arange(n, dtype=np.float16)[:, None, None]
    del feats
    np.savez(tmp_path / "windows.npz", window_index=np.arange(n) * 3, mora=np.arange(n, dtype=np.float32),
             kind=np.zeros(n, np.int8), source_index=np.zeros(n, np.int32))
    meta = {"frames": 201, "computed": 4, "complete": False}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="済んでいない"):
        DevWindowValDataset(tmp_path)
    ds = DevWindowValDataset(tmp_path, limit=4, normalizer=Normalizer(1.0, 2.0, mode="global"))
    assert len(ds) == 4
    item = ds[3]
    assert item.features.shape == (201, 80)
    assert float(item.features[0, 0]) == pytest.approx((3.0 - 1.0) / 2.0)
    assert item.mora == 3.0 and item.duration_sec == 2.0 and item.clip_id == "dev_window:9"


def test_length_mismatch_stops() -> None:
    corpus = _corpus()
    ds = corpus.dataset(augment=None)
    ds.loader = lambda path: corpus.loader(path)[:-10]
    with pytest.raises(ValueError, match="長さがアライメント時と違う"):
        ds.window(0)


def _write_inputs(tmp_path: Path, *, truncated_last_line: bool = True) -> dict[str, Path]:
    clips = [
        {"clip_id": f"c{i}", "audio_path": f"a/c{i}.wav", "client_id": "spkA" if i < 4 else "spkB",
         "sentence": "x", "kana": "ア", "mora": 2, "duration_sec": 2.5, "mora_per_second": 0.8}
        for i in range(6)
    ]
    (tmp_path / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips), encoding="utf-8")
    (tmp_path / "train.json").write_text(json.dumps({"split": "train", "client_ids": ["spkA"]}), encoding="utf-8")
    (tmp_path / "test.json").write_text(json.dumps({"split": "test", "client_ids": ["spkA"]}), encoding="utf-8")
    rows = [
        {"clip_id": f"c{i}", "ok": i != 3, "duration_sec": 2.5004,
         "moras": [{"kana": "ア", "start": 0.1, "end": 0.12}, {"kana": "イ", "start": 1.0, "end": 1.02}]}
        for i in range(6)
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if truncated_last_line:
        text += '{"clip_id": "c9", "ok": tr'
    (tmp_path / "align.jsonl").write_text(text, encoding="utf-8")
    return {"clips": tmp_path / "clips.jsonl", "train": tmp_path / "train.json",
            "test": tmp_path / "test.json", "align": tmp_path / "align.jsonl"}


def test_load_train_clips_checks(tmp_path: Path) -> None:
    p = _write_inputs(tmp_path)
    clips, starts, ends = load_train_clips(
        ["c0", "c2", "c1"], clips_jsonl=p["clips"], train_split=p["train"], alignments_path=p["align"]
    )
    assert [c.clip_id for c in clips] == ["c0", "c2", "c1"]
    assert clips[0].num_samples == 40006  # round(2.5004 × 16000)
    assert starts.dtype == np.float32 and starts.size == 6
    limited, _, _ = load_train_clips(
        ["c2", "c0", "c1"], clips_jsonl=p["clips"], train_split=p["train"], alignments_path=p["align"], limit=2
    )
    assert [c.clip_id for c in limited] == ["c0", "c1"]
    with pytest.raises(ValueError, match="テストセット"):
        load_train_clips(["c0"], clips_jsonl=p["clips"], train_split=p["test"], alignments_path=p["align"])
    with pytest.raises(ValueError, match="話者に属さない"):
        load_train_clips(["c0", "c4"], clips_jsonl=p["clips"], train_split=p["train"], alignments_path=p["align"])
    with pytest.raises(ValueError, match="失敗"):
        load_train_clips(["c3"], clips_jsonl=p["clips"], train_split=p["train"], alignments_path=p["align"])
    (tmp_path / "list.txt").write_text("c0\nc1\n\n", encoding="utf-8")
    assert read_clip_list(tmp_path / "list.txt") == ["c0", "c1"]


def test_window_epoch_stats_counts() -> None:
    stats = WindowEpochStats()
    stats.update(["single", "concat", "", "concat"], [1.0, 1.2, 1.0, 0.8], [0.0, 10.0, 0.0, 17.0], [2.0] * 4)
    d = stats.as_dict()
    assert d["window_total"] == 3 and d["window_single"] == 1 and d["window_concat"] == 2
    assert d["window_zero"] == 1
    assert d["window_stretch_rate"] == pytest.approx(2 / 3)
    assert d["window_band_under4"] == 1 and d["window_band_4to6"] == 1 and d["window_band_over8"] == 1


def test_run_training_with_window_dataset(tmp_path: Path) -> None:
    """方式Bのデータセットと検証窓で学習ループが最後まで回り、窓の集計が metrics.jsonl に残る。"""
    from spkrate.models.cnn import CnnConfig
    from spkrate.train.train import TrainConfig, run_training
    from spkrate.train.silence import TrainWithSilence

    corpus = _corpus()
    train = corpus.dataset(augment=_stretch_only(prob=0.5))
    silence = build_silence_dataset(
        SilenceSettings(enabled=True, ratio=0.2, mix={"digital_silence": 1.0, "musan_noise": 0.0, "quiet_noise": 0.0}),
        train, normalizer=None, seed=0, match_feature_storage=False,
    )
    n = 8
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    feats = np.lib.format.open_memmap(val_dir / "features.npy", mode="w+", dtype=np.float16, shape=(n, 201, 80))
    feats[:] = np.random.default_rng(0).standard_normal((n, 201, 80)).astype(np.float16)
    del feats
    np.savez(val_dir / "windows.npz", window_index=np.arange(n), mora=np.full(n, 10.0, np.float32),
             kind=np.zeros(n, np.int8), source_index=np.zeros(n, np.int32))
    (val_dir / "meta.json").write_text(json.dumps({"frames": 201, "computed": n, "complete": True}), encoding="utf-8")
    small = CnnConfig(n_mels=80, freq_channels=(4, 4), freq_strides=(4, 4), freq_kernel=(3, 3),
                      temporal_channels=(8, 8), dilations=(1, 2), temporal_kernel=3)
    config = TrainConfig.from_mapping({
        "experiment_id": "mb", "device": "cpu", "runs_dir": str(tmp_path / "runs"),
        "model_overrides": small.as_dict(),
        "data": {"source": "windows", "dev_source": "dev_window_val", "dev_window_val_dir": str(val_dir)},
        "train": {"epochs": 1, "batch_size": 8, "num_workers": 0, "log_interval": 0, "bucketing": False},
    })
    outcome = run_training(
        config, train_dataset=TrainWithSilence(train, silence), dev_dataset=DevWindowValDataset(val_dir)
    )
    row = outcome.history[0]
    assert row["window_total"] == len(train)
    assert row["window_single"] == row["window_concat"] == train.num_single
    assert row["train_clips"] == len(train) + len(silence)
    assert row["val_clips"] == n
    assert "窓の集計" in (outcome.run_dir / "log.txt").read_text(encoding="utf-8")


# ======================================================================================
# 伸縮率の分布（docs/experiments/011-search-plan.md 4.2節、exp010）

# 変更前のコード（f094f03）で記録した値: 種 (20260921, 1, i)、音源長 [40000, 64000, 120000][i % 3]、
# AugmentConfig() の既定。(伸縮率, 窓の開始, 抽選に当たったか, 次の random())
_GOLDEN_PLANS = [
    (1.2405802597381568, 9828, True, 0.894982455686084),
    (1.0011578649612467, 31163, True, 0.49547072682173343),
    (1.0, 12398, False, 0.4625963125916397),
    (1.2761195538457317, 5161, True, 0.7420798609669378),
    (1.2895112747272055, 24367, True, 0.6310544948034217),
    (1.0, 44265, False, 0.18734388622039455),
    (1.0, 808, False, 0.9941267353078993),
    (1.3430738967633693, 42630, True, 0.9044851436160809),
    (0.9741894201201099, 73523, True, 0.5981704169268661),
    (0.9814316482052319, 6360, True, 0.3884370561077113),
    (1.0, 7893, False, 0.45370945018777153),
    (0.9745679867867129, 80006, True, 0.2802268519742359),
]


def test_default_stretch_plan_sequence_is_unchanged() -> None:
    """既定（uniform）の設定で、窓の取り方と後続の乱数が変更前と同じ。"""
    for i, (stretch, start, drawn, following) in enumerate(_GOLDEN_PLANS):
        rng = np.random.default_rng((20260921, 1, i))
        plan = draw_window_plan(
            rng, [40000, 64000, 120000][i % 3], window_samples=W, margin_samples=8000, augment=AugmentConfig()
        )
        assert (plan.stretch, plan.start, plan.stretch_drawn) == (stretch, start, drawn)
        assert rng.random() == following


def test_log_uniform_plan_keeps_random_stream_and_range() -> None:
    """対数一様でも抽選の当たり外れと後続の乱数は一様と同じで、伸縮率は [max(0.7, W/N), 1.5]。"""
    log_cfg = replace(AugmentConfig(), time_stretch_distribution="log_uniform")
    below = 0
    drawn_total = 0
    for i in range(400):
        n = [40000, 64000, 120000][i % 3]
        ru, rl = np.random.default_rng((5, 1, i)), np.random.default_rng((5, 1, i))
        pu = draw_window_plan(ru, n, window_samples=W, margin_samples=8000, augment=AugmentConfig())
        pl = draw_window_plan(rl, n, window_samples=W, margin_samples=8000, augment=log_cfg)
        assert pu.stretch_drawn == pl.stretch_drawn
        assert ru.random() == rl.random()
        if pl.stretch_drawn:
            drawn_total += 1
            assert stretch_lower_bound(0.7, n, W) <= pl.stretch <= 1.5
            assert pl.stretch <= pu.stretch + 1e-12  # 同じ一様乱数なら対数一様の方が小さいか等しい
            below += int(pl.stretch < 1.0)
    assert drawn_total > 100 and below > 0


@pytest.mark.parametrize("distribution", ["uniform", "log_uniform"])
def test_window_label_without_audio_matches_window(distribution: str) -> None:
    """``window_label``（音声を読まない）の正解・取り方が ``window`` と一致する。"""
    corpus = _corpus()
    augment = replace(_stretch_only(prob=0.5), time_stretch_distribution=distribution)
    ds = corpus.dataset(augment=augment)
    ds.set_epoch(1)
    stretched = 0
    for index in range(len(ds)):
        label, plan, source = ds.window_label(index)
        info = ds.window(index)
        assert plan == info["plan"]
        assert source == info["source"]
        assert label == float(info["mora"])
        stretched += int(plan.stretch != 1.0)
    assert stretched > 0
