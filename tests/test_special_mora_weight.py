"""特殊拍の多い窓の損失の重み（docs/decisions/013-special-mora-loss-weight.md。exp020）の単体テスト。

- 特殊拍の判定と窓の特殊拍の割合が scripts/special_mora_analysis.py の数え方と一致する
- 重みはバッチ内で平均1、生の重みに上限が掛かる。α = 0（既定）は無効
- 既定では損失の値と窓・拡張の乱数の系列が変わらない
- 方式Bのエポックの集計（割合の平均・0.1刻みの分布、重みの平均・最大）

音声は合成で、実データは読まない。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spkrate.data.augment import AugmentConfig
from spkrate.eval.dev_window_val import DevWindowValDataset
from spkrate.train.data import collate_clips
from spkrate.train.method_b import (
    TrainClip,
    WindowEpochStats,
    WindowSettings,
    WindowTrainDataset,
    load_train_clips,
    stretched_window_label,
    stretched_window_special,
)
from spkrate.train.silence import SilenceSettings, TrainWithSilence, build_silence_dataset
from spkrate.train.special_mora import (
    MIN_LABEL_MORA,
    is_special_mora,
    ratio_bin,
    special_mora_flags,
    special_mora_weights,
    special_ratio,
    weighted_mse_loss,
)

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def _analysis_module():
    """scripts/special_mora_analysis.py（分析の数え方の正）を読み込む。"""
    name = "special_mora_analysis_for_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "special_mora_analysis.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _analysis_ratio(starts, ends, kanas, window_start: float) -> tuple[float, float]:
    """分析スクリプトの数え方（mora_fractions と classify_mora）で (正解, 特殊拍の割合)。"""
    mod = _analysis_module()
    frac = mod.mora_fractions(np.asarray(starts, np.float32), np.asarray(ends, np.float32),
                              np.asarray([window_start], np.float32), mod.WINDOW_SEC)
    mask = np.array([mod.classify_mora(k)["special"] for k in kanas], dtype=bool)
    total = float(frac.sum(axis=1, dtype=np.float32)[0])
    return total, float(frac[:, mask].sum(axis=1, dtype=np.float32)[0]) / total


# --------------------------------------------------------------------------------------
# 特殊拍の判定と割合


def test_is_special_mora_matches_analysis_classification() -> None:
    mod = _analysis_module()
    for kana in ["ー", "ン", "ッ", "ア", "キャ", "ティ", "クヮ", "ヵ", "ヶ", "ァ", "ンー"]:
        assert is_special_mora(kana) == mod.classify_mora(kana)["special"], kana
    np.testing.assert_array_equal(special_mora_flags(["カ", "ー", "ン", "シャ", "ッ"]), [0, 1, 1, 0, 1])


def test_special_ratio_hand_example() -> None:
    # 窓 [0, 2): カ(全部) ー(全部) ン(半分) ト(窓の外)。点のモーラ ッ(窓内) も1つ
    starts = np.array([0.0, 0.2, 1.9, 2.5, 1.0], np.float32)
    ends = np.array([0.2, 0.4, 2.1, 2.6, 1.0], np.float32)
    special = np.array([False, True, True, False, True])
    label = stretched_window_label(starts, ends, excerpt_start=0, stretch=1.0, window_start=0, window_sec=2.0)
    count = stretched_window_special(starts, ends, special, excerpt_start=0, stretch=1.0,
                                     window_start=0, window_sec=2.0)
    assert float(label) == pytest.approx(3.5)
    assert float(count) == pytest.approx(2.5)
    assert special_ratio(float(count), float(label)) == pytest.approx(2.5 / 3.5)
    # 正解が1.0モーラ未満（0を含む）なら割合0
    assert MIN_LABEL_MORA == 1.0
    assert special_ratio(0.5, 0.9) == 0.0 and special_ratio(0.0, 0.0) == 0.0
    assert special_ratio(1.0, 1.0) == 1.0


def test_ratio_bin_edges() -> None:
    assert [ratio_bin(x) for x in (0.0, 0.0999, 0.1, 0.35, 0.9, 1.0)] == [0, 0, 1, 3, 9, 9]


class _Corpus:
    """かな付きの合成コーパス（特殊拍を混ぜる）。"""

    KANAS = ("カ", "ー", "ン", "シャ", "ッ", "ア", "イ")

    def __init__(self, num_speakers: int = 3, clips_per_speaker: int = 4) -> None:
        rng = np.random.default_rng(0)
        self.clips: list[TrainClip] = []
        starts: list[float] = []
        ends: list[float] = []
        kanas: list[str] = []
        self.waves: dict[str, np.ndarray] = {}
        for s in range(num_speakers):
            for c in range(clips_per_speaker):
                n = int(round(float(rng.uniform(2.0, 4.5)) * SR))
                times = np.arange(0.05, n / SR - 0.1, float(rng.uniform(0.09, 0.2)))
                path = f"audio/s{s}c{c}.wav"
                self.clips.append(TrainClip(f"s{s}c{c}", f"spk{s}", path, n, len(starts), len(times)))
                starts.extend(float(t) for t in times)
                ends.extend(float(t) + float(rng.uniform(0.0, 0.08)) for t in times)
                kanas.extend(str(k) for k in rng.choice(self.KANAS, size=len(times), p=[.3, .15, .1, .1, .1, .15, .1]))
                wave_rng = np.random.default_rng(zlib.crc32(path.encode()))
                self.waves[path] = (wave_rng.standard_normal(n) * 0.1).astype(np.float32)
        self.starts = np.asarray(starts, dtype=np.float32)
        self.ends = np.asarray(ends, dtype=np.float32)
        self.kanas = kanas
        self.special = special_mora_flags(kanas)

    def dataset(self, augment: AugmentConfig | None, *, special: bool = True, redraws: int = 0) -> WindowTrainDataset:
        return WindowTrainDataset(
            self.clips, self.starts, self.ends,
            settings=WindowSettings(fast_window_redraws=redraws),
            augment=augment, seed=7,
            loader=lambda p: self.waves[p].copy(),
            mora_special=self.special if special else None,
        )


def _augment() -> AugmentConfig:
    """伸縮（0.5〜1.5、対数一様）と音量・帯域制限・周波数マスク。残響・雑音は外して速くする。"""
    return replace(AugmentConfig(), time_stretch_range=(0.5, 1.5), time_stretch_distribution="log_uniform",
                   reverb_enabled=False, noise_enabled=False)


def test_window_ratio_matches_analysis_counting() -> None:
    """窓の割合が、伸縮後の時刻に分析スクリプトの数え方を当てた値と一致する（単一・連結、伸縮あり）。"""
    corpus = _Corpus()
    ds = corpus.dataset(_augment(), redraws=1)
    ds.set_epoch(1)
    checked = stretched = 0
    for index in range(len(ds)):
        info = ds.window(index)
        plan, source, k = info["plan"], info["source"], info["excerpt_window_start"]
        starts, ends = ds.source_moras(source)
        kanas = [kana for clip_id in source.clip_ids for kana in
                 corpus.kanas[ds.clips[ds._index[clip_id]].mora_offset:
                              ds.clips[ds._index[clip_id]].mora_offset + ds.clips[ds._index[clip_id]].mora_count]]
        shift = np.float32(plan.excerpt_start / SR)
        s = np.float32(plan.stretch)
        total, expected = _analysis_ratio((starts - shift) * s, (ends - shift) * s, kanas, k / SR)
        assert float(info["mora"]) == pytest.approx(total, abs=1e-4)
        if total >= MIN_LABEL_MORA:
            assert info["special_ratio"] == pytest.approx(expected, abs=1e-5)
            checked += 1
            stretched += int(plan.stretch != 1.0)
        else:
            assert info["special_ratio"] == 0.0
        # 音声を読まない経路と同じ値
        label, ratio, plan2 = ds.window_special_ratio(index)
        assert plan2 == plan and label == pytest.approx(float(info["mora"])) and ratio == info["special_ratio"]
        assert ds[index].special_ratio == info["special_ratio"]
    assert checked > 10 and stretched > 0


def test_load_train_clips_with_special(tmp_path: Path) -> None:
    clips = [{"clip_id": f"c{i}", "audio_path": f"a/c{i}.wav", "client_id": "spkA", "sentence": "x",
              "kana": "ア", "mora": 3, "duration_sec": 2.5, "mora_per_second": 1.2} for i in range(2)]
    (tmp_path / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips), encoding="utf-8")
    (tmp_path / "train.json").write_text(json.dumps({"split": "train", "client_ids": ["spkA"]}), encoding="utf-8")
    kanas = [["カ", "ー", "シャ"], ["ッ", "ン", "ア"]]
    rows = [{"clip_id": f"c{i}", "ok": True, "duration_sec": 2.5,
             "moras": [{"kana": k, "start": 0.1 * j, "end": 0.1 * j + 0.05} for j, k in enumerate(kanas[i])]}
            for i in range(2)]
    (tmp_path / "align.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    kwargs = dict(clips_jsonl=tmp_path / "clips.jsonl", train_split=tmp_path / "train.json",
                  alignments_path=tmp_path / "align.jsonl")
    plain = load_train_clips(["c1", "c0"], **kwargs)
    assert len(plain) == 3  # 既定は従来と同じ3つ
    clips_out, starts, ends, special = load_train_clips(["c1", "c0"], with_special=True, **kwargs)
    np.testing.assert_array_equal(starts, plain[1])
    np.testing.assert_array_equal(special, [True, True, False, False, True, False])


# --------------------------------------------------------------------------------------
# 重み


def test_weights_mean_one_and_cap() -> None:
    ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    w = special_mora_weights(ratios, 5.0, 3.0)
    assert w.dtype == np.float32
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-6)
    raw = np.minimum(1.0 + 5.0 * np.asarray(ratios), 3.0)
    np.testing.assert_allclose(w, raw / raw.mean(), rtol=1e-6)
    # 上限は生の重みに掛かる: 0.4 以上は同じ重み
    assert w[4] == w[5] and np.all(np.diff(w) >= 0)
    assert float(w.max() / w.min()) == pytest.approx(3.0)
    # すべて同じ割合なら全件1
    np.testing.assert_array_equal(special_mora_weights([0.2] * 4, 5.0, 3.0), np.ones(4, np.float32))
    np.testing.assert_array_equal(special_mora_weights([0.0, 0.5], 0.0, 3.0), np.ones(2, np.float32))
    with pytest.raises(ValueError):
        special_mora_weights([0.1], -1.0, 3.0)
    with pytest.raises(ValueError):
        special_mora_weights([0.1], 1.0, 0.5)
    with pytest.raises(ValueError):
        special_mora_weights([0.1], float("nan"), 3.0)


def test_weighted_mse_with_unit_weights_equals_mse() -> None:
    pred = torch.tensor([1.0, 2.5, 7.0])
    target = torch.tensor([1.5, 2.0, 9.0])
    ones = torch.ones(3)
    assert float(weighted_mse_loss(pred, target, ones)) == pytest.approx(
        float(torch.nn.functional.mse_loss(pred, target)), rel=1e-6
    )
    w = torch.tensor([0.5, 0.5, 2.0])
    assert float(weighted_mse_loss(pred, target, w)) == pytest.approx((0.5 * 0.25 + 0.5 * 0.25 + 2.0 * 4.0) / 3)
    with pytest.raises(ValueError):
        weighted_mse_loss(pred, target, torch.ones(2))


def test_settings_default_disabled_and_validation() -> None:
    from spkrate.train.train import TrainSettings

    default = TrainSettings()
    assert default.special_mora_weight_alpha == 0.0 and not default.special_mora_weighting
    on = TrainSettings.from_mapping({"special_mora_weight_alpha": 5.0, "special_mora_weight_max": 3.0})
    assert on.special_mora_weighting
    with pytest.raises(ValueError, match="mse"):
        TrainSettings.from_mapping({"special_mora_weight_alpha": 5.0, "loss": "poisson"})
    with pytest.raises(ValueError):
        TrainSettings.from_mapping({"special_mora_weight_alpha": -0.1})
    with pytest.raises(ValueError):
        TrainSettings.from_mapping({"special_mora_weight_alpha": 1.0, "special_mora_weight_max": 0.9})


# --------------------------------------------------------------------------------------
# 既定で系列・損失が変わらない


def test_special_flags_do_not_change_window_series() -> None:
    """特殊拍の印を渡しても、窓・正解・拡張・後続の乱数の系列は印なしと同じ。"""
    corpus = _Corpus()
    with_flags = corpus.dataset(_augment(), redraws=1)
    without = corpus.dataset(_augment(), special=False, redraws=1)
    for epoch in (1, 2):
        with_flags.set_epoch(epoch)
        without.set_epoch(epoch)
        for index in range(len(with_flags)):
            a, b = with_flags.window(index), without.window(index)
            assert a["plan"] == b["plan"] and a["mora"] == b["mora"]
            np.testing.assert_array_equal(a["waveform"], b["waveform"])
            assert a["rng"].random() == b["rng"].random()
            assert b["special_ratio"] == 0.0
            ia, ib = with_flags[index], without[index]
            np.testing.assert_array_equal(ia.features, ib.features)
            assert ia.augment_applied == ib.augment_applied and ia.mora == ib.mora


def _val_dir(tmp_path: Path, n: int = 8) -> Path:
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    feats = np.lib.format.open_memmap(val_dir / "features.npy", mode="w+", dtype=np.float16, shape=(n, 201, 80))
    feats[:] = np.random.default_rng(0).standard_normal((n, 201, 80)).astype(np.float16)
    del feats
    np.savez(val_dir / "windows.npz", window_index=np.arange(n), mora=np.full(n, 10.0, np.float32),
             kind=np.zeros(n, np.int8), source_index=np.zeros(n, np.int32))
    (val_dir / "meta.json").write_text(json.dumps({"frames": 201, "computed": n, "complete": True}), encoding="utf-8")
    return val_dir


def _run(tmp_path: Path, name: str, train_dataset, val_dir: Path, **train_overrides):
    from spkrate.models.cnn import CnnConfig
    from spkrate.train.train import TrainConfig, run_training

    small = CnnConfig(n_mels=80, freq_channels=(4, 4), freq_strides=(4, 4), freq_kernel=(3, 3),
                      temporal_channels=(8, 8), dilations=(1, 2), temporal_kernel=3)
    config = TrainConfig.from_mapping({
        "experiment_id": name, "device": "cpu", "runs_dir": str(tmp_path / "runs"),
        "model_overrides": small.as_dict(),
        "data": {"source": "windows", "dev_source": "dev_window_val", "dev_window_val_dir": str(val_dir)},
        "train": {"epochs": 2, "batch_size": 8, "num_workers": 0, "log_interval": 0, "bucketing": False,
                  **train_overrides},
    })
    return run_training(config, train_dataset=train_dataset, dev_dataset=DevWindowValDataset(val_dir))


def _with_silence(train: WindowTrainDataset) -> TrainWithSilence:
    silence = build_silence_dataset(
        SilenceSettings(enabled=True, ratio=0.2, mix={"digital_silence": 1.0, "musan_noise": 0.0, "quiet_noise": 0.0}),
        train, normalizer=None, seed=0, match_feature_storage=False,
    )
    return TrainWithSilence(train, silence)


def _metric_rows(outcome) -> list[dict]:
    return [json.loads(line) for line in (outcome.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def test_default_training_loss_unchanged_and_weighting_changes_it(tmp_path: Path) -> None:
    """既定（α = 0）では、特殊拍の印の有無で学習の損失・検証の指標が完全に同じ。α > 0 では変わる。"""
    corpus = _Corpus()
    val_dir = _val_dir(tmp_path)
    keys = ("train_loss", "val_loss", "mae_moras_per_sec", "train_mae_moras_per_sec")
    base = _run(tmp_path, "base", _with_silence(corpus.dataset(_augment(), special=False)), val_dir)
    flags = _run(tmp_path, "flags", _with_silence(corpus.dataset(_augment())), val_dir)
    for rb, rf in zip(base.history, flags.history, strict=True):
        for key in keys:
            assert rb[key] == rf[key], key
        assert rf["train_loss_weight_mean"] == 1.0 and rf["train_loss_weight_max"] == 1.0
        assert rf["window_loss_weight_mean"] == 1.0
    # 印ありでは特殊拍の割合が集計される（印なしは0）
    assert flags.history[0]["window_special_ratio_mean"] > 0.05
    assert base.history[0]["window_special_ratio_mean"] == 0.0
    weighted = _run(tmp_path, "weighted", _with_silence(corpus.dataset(_augment())), val_dir,
                    special_mora_weight_alpha=5.0, special_mora_weight_max=3.0)
    assert weighted.history[0]["train_loss"] != flags.history[0]["train_loss"]
    for row in weighted.history:
        # 無音サンプルを含む全件の重みの平均は1（バッチ内で正規化）
        assert row["train_loss_weight_mean"] == pytest.approx(1.0, abs=1e-5)
        assert row["train_loss_weight_max"] > 1.0
        assert row["window_loss_weight_max"] == row["train_loss_weight_max"]
    log = (weighted.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "特殊拍の多い窓の損失の重み: 有効" in log
    assert "特殊拍の割合の分布(0.1刻み)=" in log and "損失の重み(平均/最大)=" in log
    assert "特殊拍の多い窓の損失の重み: なし" in (flags.run_dir / "log.txt").read_text(encoding="utf-8")
    assert _metric_rows(weighted)[0]["window_special_ratio_bin00"] == weighted.history[0]["window_special_ratio_bin00"]


def test_weighting_requires_window_source(tmp_path: Path) -> None:
    from spkrate.train.train import TrainConfig, run_training

    config = TrainConfig.from_mapping({
        "experiment_id": "x", "device": "cpu", "runs_dir": str(tmp_path / "runs"),
        "data": {"source": "features"},
        "train": {"epochs": 1, "special_mora_weight_alpha": 5.0},
    })
    with pytest.raises(ValueError, match="windows"):
        run_training(config, train_dataset=[], dev_dataset=[])


# --------------------------------------------------------------------------------------
# 集計


def test_window_epoch_stats_special_and_weights() -> None:
    stats = WindowEpochStats()
    stats.update(
        ["single", "concat", "", "concat"],
        [1.0, 0.6, 1.0, 0.8],
        [6.0, 20.0, 0.0, 17.0],
        [2.0] * 4,
        special_ratios=[0.05, 0.25, 0.0, 1.0],
        weights=[0.8, 1.2, 0.7, 1.9],
    )
    d = stats.as_dict()
    assert d["window_special_ratio_mean"] == pytest.approx((0.05 + 0.25 + 1.0) / 3)  # 無音サンプルは除く
    assert d["window_special_ratio_bin00"] == pytest.approx(1 / 3)
    assert d["window_special_ratio_bin02"] == pytest.approx(1 / 3)
    assert d["window_special_ratio_bin09"] == pytest.approx(1 / 3)
    assert sum(d[f"window_special_ratio_bin{b:02d}"] for b in range(10)) == pytest.approx(1.0)
    assert d["window_loss_weight_mean"] == pytest.approx((0.8 + 1.2 + 1.9) / 3)
    assert d["window_loss_weight_max"] == pytest.approx(1.9)
    text = stats.describe(1)
    assert "特殊拍の割合の平均=0.4333" in text and "損失の重み(平均/最大)=1.3000/1.9000" in text
    # 割合も重みも渡さなければ nan と 1
    plain = WindowEpochStats()
    plain.update(["single"], [1.0], [6.0], [2.0])
    p = plain.as_dict()
    assert math.isnan(p["window_special_ratio_mean"]) and p["window_loss_weight_mean"] == 1.0


def test_collate_carries_special_ratios() -> None:
    corpus = _Corpus()
    ds = corpus.dataset(None)
    items = [ds[i] for i in range(4)]
    batch = collate_clips(items)
    assert batch.special_ratios == tuple(item.special_ratio for item in items)
    assert batch.to("cpu").special_ratios == batch.special_ratios
