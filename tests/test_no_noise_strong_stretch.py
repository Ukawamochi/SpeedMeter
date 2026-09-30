"""強く縮めた窓に雑音を重ねない規則（docs/decisions/012-no-noise-on-strong-stretch.md。exp019）の単体テスト。

- 既定（``noise_skip_stretch_below`` が None）では拡張と窓の系列が変わらない
- 有効にすると、伸縮率が閾値未満で雑音重畳の抽選に当たった音声・窓だけ雑音が省かれ、
  他の音声・窓と、後続の乱数の系列は既定と同じになる
- 方式Bのエポックの集計（雑音を重ねた窓・省いた窓）が正しい

音声は合成で、実データは読まない。
"""

from __future__ import annotations

import json
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spkrate.data.augment import (
    AugmentConfig,
    ArrayNoiseSource,
    augment_waveform,
    describe_augment_config,
    estimate_application_rates,
)
from spkrate.eval.dev_window_val import DevWindowValDataset
from spkrate.train.method_b import (
    NOISE_SKIPPED,
    TrainClip,
    WindowEpochStats,
    WindowSettings,
    WindowTrainDataset,
)
from spkrate.train.silence import SilenceSettings, TrainWithSilence, build_silence_dataset

SR = 16000
THRESHOLD = 0.7

# 雑音の長さは抜粋より短いもの（繰り返して埋める）と長いもの（切り出し位置を引く）を混ぜる
_NOISE_LENGTHS = (SR, 6 * SR)


def _noise_source() -> ArrayNoiseSource:
    rng = np.random.default_rng(3)
    return ArrayNoiseSource([rng.standard_normal(n).astype(np.float32) for n in _NOISE_LENGTHS])


def _zero_noise_source() -> ArrayNoiseSource:
    """同じ長さの全0の雑音。乱数の消費は ``_noise_source`` と同じで、足しても波形を変えない。"""
    return ArrayNoiseSource([np.zeros(n, dtype=np.float32) for n in _NOISE_LENGTHS])


def _exp016_like(**overrides) -> AugmentConfig:
    """exp016 の拡張（伸縮 0.5〜1.5・対数一様。他は既定）。"""
    return replace(
        AugmentConfig(),
        time_stretch_range=(0.5, 1.5),
        time_stretch_distribution="log_uniform",
        **overrides,
    )


# --------------------------------------------------------------------------------------
# 設定


def test_default_is_disabled() -> None:
    assert AugmentConfig().noise_skip_stretch_below is None
    assert AugmentConfig.from_mapping(None).noise_skip_stretch_below is None
    assert AugmentConfig.from_mapping({"noise_skip_stretch_below": None}).noise_skip_stretch_below is None
    assert not AugmentConfig().skips_noise(0.5)


def test_from_mapping_accepts_threshold_and_rejects_bad_values() -> None:
    config = AugmentConfig.from_mapping({"noise_skip_stretch_below": 0.7})
    assert config.noise_skip_stretch_below == 0.7
    assert config.skips_noise(0.6999) and not config.skips_noise(0.7) and not config.skips_noise(1.0)
    assert AugmentConfig.from_mapping({"noise_skip_stretch_below": 1}).noise_skip_stretch_below == 1.0
    for bad in (0, 0.0, -0.1, 1.2, float("nan"), float("inf"), True, "0.7"):
        with pytest.raises(ValueError, match="noise_skip_stretch_below"):
            AugmentConfig.from_mapping({"noise_skip_stretch_below": bad})


def test_describe_mentions_rule_only_when_enabled() -> None:
    default = describe_augment_config(AugmentConfig(), noise_description="x")
    enabled = describe_augment_config(
        replace(AugmentConfig(), noise_skip_stretch_below=0.7), noise_description="x"
    )
    assert not any("noise_skip_stretch_below" in line for line in default)
    noise_line = [line for line in enabled if line.startswith("拡張: 雑音重畳")][0]
    assert "伸縮率0.7未満の音声には重ねない" in noise_line
    # 雑音以外の行は同じ
    assert [line for line in default if not line.startswith("拡張: 雑音重畳")] == [
        line for line in enabled if not line.startswith("拡張: 雑音重畳")
    ]


def test_estimate_application_rates_counts_skip_but_keeps_other_rates() -> None:
    base = estimate_application_rates(_exp016_like())
    skip = estimate_application_rates(_exp016_like(noise_skip_stretch_below=THRESHOLD))
    for a, b in zip(base, skip, strict=True):
        assert a.selected == b.selected
        if a.name == "noise":
            assert b.applied < a.applied
        else:
            assert a.applied == b.applied


# --------------------------------------------------------------------------------------
# augment_waveform（方式Aの経路と、方式Bの残りの拡張）


def _augment_pair(index: int, samples: np.ndarray, *, applied_stretch: float = 1.0, time_stretch: bool = True):
    base_cfg = _exp016_like(noise_prob=0.5)
    skip_cfg = replace(base_cfg, noise_skip_stretch_below=THRESHOLD)
    if not time_stretch:
        base_cfg = replace(base_cfg, time_stretch_enabled=False)
        skip_cfg = replace(skip_cfg, time_stretch_enabled=False)
    out = []
    for cfg, source in ((base_cfg, _noise_source()), (skip_cfg, _noise_source()), (base_cfg, _zero_noise_source())):
        rng = np.random.default_rng((11, index))
        result = augment_waveform(
            samples, rng, config=cfg, noise_source=source, applied_stretch=applied_stretch
        )
        out.append((result, rng.random()))
    return out


def test_augment_waveform_skips_only_strong_stretch_and_keeps_stream() -> None:
    samples = (np.random.default_rng(0).standard_normal(SR // 2) * 0.1).astype(np.float32)
    skipped = kept_noise = 0
    for index in range(80):
        (base, after_base), (skip, after_skip), (zero, after_zero) = _augment_pair(index, samples)
        # 後続の乱数は3通りとも同じ（抽選・SNR・雑音の選択を同じだけ消費する）
        assert after_base == after_skip == after_zero
        assert base.stretch == skip.stretch
        if "noise" in base.applied and base.stretch < THRESHOLD:
            skipped += 1
            assert skip.skipped == ("noise",)
            assert "noise" not in skip.applied and "noise" not in skip.effective
            assert "snr_db" not in skip.params
            # 雑音だけを除いた波形（全0の雑音を足した場合＝足さない場合と同じ）
            np.testing.assert_array_equal(skip.samples, zero.samples)
            assert skip.applied == tuple(n for n in base.applied if n != "noise")
        else:
            kept_noise += int("noise" in base.applied)
            assert skip.skipped == ()
            np.testing.assert_array_equal(skip.samples, base.samples)
            assert skip.applied == base.applied and skip.effective == base.effective
            assert skip.params == base.params
    assert skipped > 0 and kept_noise > 0


def test_applied_stretch_is_used_when_time_stretch_is_done_by_caller() -> None:
    """方式Bのように伸縮を呼び出し側で済ませた場合、``applied_stretch`` で判定する。"""
    samples = (np.random.default_rng(1).standard_normal(SR // 2) * 0.1).astype(np.float32)
    hits = 0
    for index in range(30):
        for applied_stretch in (0.6, 0.7, 1.2):
            (base, a0), (skip, a1), _ = _augment_pair(
                index, samples, applied_stretch=applied_stretch, time_stretch=False
            )
            assert a0 == a1
            if "noise" in base.applied and applied_stretch < THRESHOLD:
                hits += 1
                assert skip.skipped == ("noise",)
            else:
                assert skip.skipped == ()
                np.testing.assert_array_equal(skip.samples, base.samples)
    assert hits > 0


# --------------------------------------------------------------------------------------
# 方式Bの窓


def _moras(duration: float, step: float = 0.13) -> list[tuple[float, float]]:
    return [(float(t), float(t) + 0.02) for t in np.arange(0.05, duration - 0.02, step)]


class _Corpus:
    def __init__(self, num_speakers: int = 3, clips_per_speaker: int = 4) -> None:
        rng = np.random.default_rng(0)
        self.clips: list[TrainClip] = []
        starts: list[float] = []
        ends: list[float] = []
        self.waves: dict[str, np.ndarray] = {}
        for s in range(num_speakers):
            for c in range(clips_per_speaker):
                n = int(round(float(rng.uniform(2.0, 4.5)) * SR))
                moras = _moras(n / SR)
                path = f"audio/s{s}c{c}.wav"
                self.clips.append(TrainClip(f"s{s}c{c}", f"spk{s}", path, n, len(starts), len(moras)))
                starts.extend(a for a, _ in moras)
                ends.extend(b for _, b in moras)
                wave_rng = np.random.default_rng(zlib.crc32(path.encode()))
                self.waves[path] = (wave_rng.standard_normal(n) * 0.1).astype(np.float32)
        self.starts = np.asarray(starts, dtype=np.float32)
        self.ends = np.asarray(ends, dtype=np.float32)

    def dataset(self, augment: AugmentConfig, noise_source, *, redraws: int = 1) -> WindowTrainDataset:
        return WindowTrainDataset(
            self.clips,
            self.starts,
            self.ends,
            settings=WindowSettings(fast_window_redraws=redraws),
            augment=augment,
            noise_source=noise_source,
            seed=7,
            loader=lambda p: self.waves[p].copy(),
        )


def _light(config: AugmentConfig) -> AugmentConfig:
    """残響を外して速くした拡張（残響は雑音より前なので、外しても規則の検査には関係しない）。"""
    return replace(config, reverb_enabled=False)


def test_window_dataset_default_unchanged_and_skip_only_strong_stretch() -> None:
    corpus = _Corpus()
    base_cfg = _light(_exp016_like())
    base = corpus.dataset(base_cfg, _noise_source())
    same = corpus.dataset(replace(base_cfg, noise_skip_stretch_below=None), _noise_source())
    skip = corpus.dataset(replace(base_cfg, noise_skip_stretch_below=THRESHOLD), _noise_source())
    zero = corpus.dataset(base_cfg, _zero_noise_source())
    skipped = kept = 0
    for epoch in (1, 2):
        for ds in (base, same, skip, zero):
            ds.set_epoch(epoch)
        for index in range(len(base)):
            b, s = base.window(index), skip.window(index)
            assert b["plan"] == s["plan"] and b["mora"] == s["mora"]
            # 既定（None を明示しても）は完全に同じ
            ib, i_same, i_skip = base[index], same[index], skip[index]
            np.testing.assert_array_equal(ib.features, i_same.features)
            assert ib.augment_applied == i_same.augment_applied
            if b["plan"].stretch < THRESHOLD and "noise" in ib.augment_applied:
                skipped += 1
                assert NOISE_SKIPPED in i_skip.augment_applied
                assert "noise" not in i_skip.augment_applied
                assert tuple(n for n in i_skip.augment_applied if n != NOISE_SKIPPED) == tuple(
                    n for n in ib.augment_applied if n != "noise"
                )
                # 雑音だけを除いた窓（後続の帯域制限・音量・周波数マスクの乱数は同じ）
                np.testing.assert_array_equal(i_skip.features, zero[index].features)
            else:
                kept += int("noise" in ib.augment_applied)
                assert NOISE_SKIPPED not in i_skip.augment_applied
                np.testing.assert_array_equal(i_skip.features, ib.features)
                assert i_skip.augment_applied == ib.augment_applied
            assert b["rng"].random() == s["rng"].random()
    assert skipped > 0 and kept > 0


# --------------------------------------------------------------------------------------
# 集計


def test_window_epoch_stats_counts_noise_and_skipped() -> None:
    stats = WindowEpochStats()
    stats.update(
        ["single", "concat", "", "concat"],
        [1.0, 0.6, 1.0, 0.8],
        [6.0, 20.0, 0.0, 17.0],
        [2.0] * 4,
        applied=[("noise", "volume"), (NOISE_SKIPPED,), ("noise",), ()],
    )
    d = stats.as_dict()
    assert d["window_total"] == 3  # 無音サンプル（種類が空）は数えない
    assert d["window_noise"] == 1 and d["window_noise_rate"] == pytest.approx(1 / 3)
    assert d["window_noise_skipped"] == 1 and d["window_noise_skipped_rate"] == pytest.approx(1 / 3)
    # 既存の項目の値は applied の有無で変わらない
    plain = WindowEpochStats()
    plain.update(["single", "concat", "", "concat"], [1.0, 0.6, 1.0, 0.8], [6.0, 20.0, 0.0, 17.0], [2.0] * 4)
    p = plain.as_dict()
    assert p["window_noise"] == 0 and p["window_noise_skipped"] == 0
    for key, value in p.items():
        if not key.startswith("window_noise"):
            assert d[key] == value
    assert "雑音を重ねた窓=1(0.3333) 雑音を省いた窓=1(0.3333)" in stats.describe(1)


def test_run_training_records_noise_window_counts(tmp_path: Path) -> None:
    """学習ループが雑音を重ねた窓・省いた窓を metrics.jsonl と log.txt に書き、値がデータセットと一致する。"""
    from spkrate.models.cnn import CnnConfig
    from spkrate.train.train import TrainConfig, run_training

    corpus = _Corpus()
    augment = _light(_exp016_like(noise_prob=0.8, noise_skip_stretch_below=THRESHOLD))
    train = corpus.dataset(augment, _noise_source())
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
        "experiment_id": "nn", "device": "cpu", "runs_dir": str(tmp_path / "runs"),
        "model_overrides": small.as_dict(),
        "data": {"source": "windows", "dev_source": "dev_window_val", "dev_window_val_dir": str(val_dir)},
        "train": {"epochs": 1, "batch_size": 8, "num_workers": 0, "log_interval": 0, "bucketing": False},
    })
    outcome = run_training(
        config, train_dataset=TrainWithSilence(train, silence), dev_dataset=DevWindowValDataset(val_dir)
    )
    row = outcome.history[0]
    # 学習後のデータセットはエポック1の状態。全窓を数え直して一致を確かめる
    applied = [train[i].augment_applied for i in range(len(train))]
    expected_noise = sum("noise" in a for a in applied)
    expected_skipped = sum(NOISE_SKIPPED in a for a in applied)
    assert expected_skipped > 0
    assert row["window_noise"] == expected_noise
    assert row["window_noise_skipped"] == expected_skipped
    assert row["window_noise_rate"] == pytest.approx(expected_noise / len(train))
    assert row["window_noise_skipped_rate"] == pytest.approx(expected_skipped / len(train))
    metrics = [json.loads(line) for line in (outcome.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(m.get("window_noise_skipped") == expected_skipped for m in metrics)
    log = (outcome.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "雑音を省いた窓=" in log and "雑音を重ねた窓=" in log
