"""無音・雑音のみのサンプル（正解モーラ数0）の単体テスト（spkrate.train.silence）。

実データには依存しない。MUSAN は一時ディレクトリに小さな wav を書いて模す。

固定する性質。

- 3種（デジタル無音・MUSAN雑音のみ・極小音量の雑音）のどれも正解モーラ数が0で、
  バッチの ``moras`` と損失関数に渡る目標値も0になる
- 種類ごとの波形: デジタル無音は全標本0、MUSAN雑音は元の振幅、極小音量は指定の dBFS
- 追加件数は ``ratio`` × 学習件数で、内訳は ``mix`` に従う
- 無効なら学習データに何も加えず、検証には有効でも加えない
- 特徴量・正規化は既存の ``LogMelSpectrogram`` / ``Normalizer`` と同一
- MUSAN が見つからない等の設定不備は学習開始前（データ読み込みより前）に止まる
- log.txt に追加の有無・割合・3種の件数が出る
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from spkrate.data.augment import ArrayNoiseSource
from spkrate.features.melspec import LogMelSpectrogram
from spkrate.train.data import Normalizer, collate_clips
from spkrate.train.silence import (
    SilenceDataset,
    SilenceSettings,
    SilenceSetupError,
    TrainWithSilence,
    allocate_counts,
    build_silence_dataset,
    check_silence_setup,
    is_silence_clip,
)
from spkrate.train.train import (
    TrainConfig,
    _make_loader,
    _train_one_epoch,
    build_datasets,
    run_training,
)
from spkrate.models.cnn import SpeechRateCNN

from test_train import (
    N_MELS,
    SyntheticClipDataset,
    _small_model_config,
    _write_feature_shard,
    _write_normalization,
)

KINDS = ("digital_silence", "musan_noise", "quiet_noise")


def _write_musan(root: Path, *, num_files: int = 3, amplitude: float = 0.1) -> Path:
    noise_dir = root / "noise" / "free-sound"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(num_files):
        samples = (amplitude * rng.standard_normal(16000)).astype(np.float32)
        sf.write(noise_dir / f"noise-{index:04d}.wav", samples, 16000)
    return root


def _noise_source(amplitude: float = 0.1) -> ArrayNoiseSource:
    rng = np.random.default_rng(3)
    return ArrayNoiseSource([(amplitude * rng.standard_normal(16000)).astype(np.float32)])


def _normalizer() -> Normalizer:
    return Normalizer(np.full(N_MELS, -7.0), np.full(N_MELS, 4.0), mode="per_mel")


def _dataset(kind: str, *, count: int = 4, **kwargs: object) -> SilenceDataset:
    return SilenceDataset(
        [kind] * count,
        [1.5] * count,
        normalizer=_normalizer(),
        noise_source=_noise_source(),
        seed=11,
        **kwargs,  # type: ignore[arg-type]
    )


def _rms_dbfs(samples: np.ndarray) -> float:
    return 20.0 * float(np.log10(np.sqrt(np.mean(np.square(samples.astype(np.float64))))))


# --------------------------------------------------------------------------------------
# 正解が0であること（ラベル・バッチ・損失へ渡る目標値）


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_has_zero_mora_label(kind: str) -> None:
    dataset = _dataset(kind)
    for index in range(len(dataset)):
        item = dataset[index]
        assert item.mora == 0.0
        assert is_silence_clip(item.clip_id) and kind in item.clip_id
        assert item.features.dtype == np.float32
        assert item.features.shape == (dataset.frame_counts[index], N_MELS)
        assert item.augment_applied == ()
    batch = collate_clips([dataset[i] for i in range(len(dataset))])
    assert torch.equal(batch.moras, torch.zeros(len(dataset)))


@pytest.mark.parametrize("kind", KINDS)
def test_loss_receives_zero_target_for_silence(kind: str) -> None:
    """学習ループで損失関数に渡る目標値が、無音サンプルでは0であること。"""
    base = SyntheticClipDataset(6, seed=0)
    combined = TrainWithSilence(base, _dataset(kind, count=5))
    seen_targets: list[torch.Tensor] = []

    def spy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seen_targets.append(target.detach().cpu().clone())
        return torch.nn.functional.mse_loss(prediction, target)

    config = TrainConfig.from_mapping(
        {"experiment_id": "spy", "train": {"batch_size": 4, "num_workers": 0, "log_interval": 0}}
    )
    loader = _make_loader(combined, batch_size=4, shuffle=True, settings=config.train, seed=0)
    model = SpeechRateCNN(_small_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ids: list[str] = []
    original_iter = loader.__iter__

    class Recording:
        def __iter__(self):  # noqa: ANN204
            for batch in original_iter():
                ids.extend(batch.clip_ids)
                yield batch

    stats = _train_one_epoch(
        model,
        Recording(),  # type: ignore[arg-type]
        spy_loss,
        optimizer,
        torch.device("cpu"),
        settings=config.train,
        epoch=1,
        logger=logging.getLogger("test"),
    )
    targets = torch.cat(seen_targets).tolist()
    assert len(ids) == len(targets) == len(combined)
    seen = list(zip(ids, targets, strict=True))
    silence = [value for clip_id, value in seen if is_silence_clip(clip_id)]
    speech = [value for clip_id, value in seen if not is_silence_clip(clip_id)]
    assert silence == [0.0] * 5
    assert all(value > 0 for value in speech)
    assert stats["train_silence_clips"] == 5


# --------------------------------------------------------------------------------------
# 3種の中身


def test_digital_silence_is_all_zero() -> None:
    dataset = _dataset("digital_silence")
    samples = dataset.waveform(0)
    assert samples.size == 24000 and not np.any(samples)
    # 特徴量はデジタル無音の対数メル log(1e-6) を正規化したもの
    expected = (np.log(1e-6) - (-7.0)) / 4.0
    np.testing.assert_allclose(dataset[0].features, expected, rtol=1e-5)


def test_musan_noise_keeps_original_amplitude() -> None:
    dataset = _dataset("musan_noise")
    level = _rms_dbfs(dataset.waveform(0))
    assert level == pytest.approx(20.0 * np.log10(0.1), abs=0.5)


def test_quiet_noise_level_within_range() -> None:
    dataset = _dataset("quiet_noise", count=20, quiet_noise_dbfs_range=(-85.0, -65.0))
    levels = [_rms_dbfs(dataset.waveform(i)) for i in range(len(dataset))]
    assert min(levels) >= -85.0 - 1e-3 and max(levels) <= -65.0 + 1e-3
    assert max(levels) - min(levels) > 1.0  # 件ごとに音量が変わる


def test_content_changes_with_epoch_but_length_is_fixed() -> None:
    dataset = _dataset("musan_noise")
    first = dataset.waveform(0)
    dataset.set_epoch(2)
    second = dataset.waveform(0)
    assert first.size == second.size
    assert not np.array_equal(first, second)


def test_features_match_existing_melspec_and_normalizer() -> None:
    dataset = _dataset("quiet_noise")
    samples = dataset.waveform(1)
    expected = _normalizer()(LogMelSpectrogram()(samples, 16000))
    np.testing.assert_array_equal(dataset[1].features, expected)


def test_feature_storage_rounding_matches_float16_path() -> None:
    dataset = _dataset("quiet_noise", match_feature_storage=True)
    samples = dataset.waveform(0)
    stored = LogMelSpectrogram()(samples, 16000).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(dataset[0].features, _normalizer()(stored))


# --------------------------------------------------------------------------------------
# 割合と内訳


def test_allocate_counts_default_ratio_and_equal_mix() -> None:
    counts = allocate_counts(1000, 0.03, SilenceSettings().mix)
    assert sum(counts.values()) == 30
    assert counts == {"digital_silence": 10, "musan_noise": 10, "quiet_noise": 10}


def test_allocate_counts_follows_ratio_and_mix() -> None:
    counts = allocate_counts(1000, 0.1, {"digital_silence": 1, "musan_noise": 0, "quiet_noise": 3})
    assert counts == {"digital_silence": 25, "musan_noise": 0, "quiet_noise": 75}
    uneven = allocate_counts(100, 0.05, SilenceSettings().mix)
    assert sum(uneven.values()) == 5


def test_default_settings() -> None:
    settings = SilenceSettings()
    assert settings.enabled is False
    assert settings.ratio == 0.03
    assert settings.mix == {"digital_silence": 1.0, "musan_noise": 1.0, "quiet_noise": 1.0}
    assert settings.quiet_noise_dbfs_range == (-90.0, -60.0)
    assert settings.duration_range_sec is None


def test_partial_mix_replaces_default() -> None:
    settings = SilenceSettings.from_mapping({"mix": {"digital_silence": 1.0}})
    assert settings.mix == {"digital_silence": 1.0, "musan_noise": 0.0, "quiet_noise": 0.0}
    assert not settings.needs_musan


def test_build_uses_ratio_and_train_durations() -> None:
    base = SyntheticClipDataset(200, seed=0)
    settings = SilenceSettings.from_mapping({"enabled": True, "ratio": 0.1})
    dataset = build_silence_dataset(
        settings,
        base,
        normalizer=_normalizer(),
        seed=1,
        match_feature_storage=False,
        noise_source=_noise_source(),
    )
    assert len(dataset) == 20
    assert sum(dataset.counts.values()) == 20
    base_frames = set(base.frame_counts)
    # 長さは学習データの長さの中から選ばれる（フレーム数は 1 + 標本数 // 160）
    for frames in dataset.frame_counts:
        assert frames - 1 in base_frames


# --------------------------------------------------------------------------------------
# 設定からの組み立て（features 経路）と無効時


def _features_config(tmp_path: Path, silence: dict | None) -> TrainConfig:
    features = tmp_path / "features"
    _write_feature_shard(features / "train", num_clips=40)
    _write_feature_shard(features / "dev", num_clips=6)
    split = tmp_path / "train.json"
    split.write_text(
        json.dumps({"split": "train", "client_ids": ["speaker_a", "speaker_b"]}), encoding="utf-8"
    )
    dev_split = tmp_path / "dev.json"
    dev_split.write_text(
        json.dumps({"split": "dev", "client_ids": ["speaker_a", "speaker_b"]}), encoding="utf-8"
    )
    mapping: dict = {
        "experiment_id": "silence_features",
        "device": "cpu",
        "runs_dir": str(tmp_path / "runs"),
        "model_overrides": _small_model_config().as_dict(),
        "data": {
            "source": "features",
            "features_dir": str(features),
            "train_split": str(split),
            "dev_split": str(dev_split),
            "normalization": str(_write_normalization(tmp_path / "norm.yaml")),
        },
        "augment": {"enabled": False},
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "log_interval": 0},
    }
    if silence is not None:
        mapping["silence_samples"] = silence
    return TrainConfig.from_mapping(mapping)


def test_features_path_adds_silence_to_train_only(tmp_path: Path) -> None:
    musan = _write_musan(tmp_path / "musan")
    config = _features_config(
        tmp_path, {"enabled": True, "ratio": 0.3, "musan_root": str(musan)}
    )
    train, dev, _ = build_datasets(config)
    assert isinstance(train, TrainWithSilence)
    assert len(train) == 40 + 12
    assert train.silence_counts == {"digital_silence": 4, "musan_noise": 4, "quiet_noise": 4}
    assert len(train.frame_counts) == len(train)  # type: ignore[arg-type]
    silence_items = [train[i] for i in range(40, 52)]
    assert all(item.mora == 0.0 for item in silence_items)
    assert all(not is_silence_clip(dev[i].clip_id) for i in range(len(dev)))
    assert len(dev) == 6


def test_disabled_adds_nothing(tmp_path: Path) -> None:
    config = _features_config(tmp_path, None)
    assert config.silence_samples.enabled is False
    train, _, _ = build_datasets(config)
    assert not isinstance(train, TrainWithSilence)
    assert len(train) == 40
    assert check_silence_setup(config.silence_samples) == ["無音サンプル=なし"]


def test_disabled_explicitly_adds_nothing(tmp_path: Path) -> None:
    config = _features_config(tmp_path, {"enabled": False, "ratio": 0.5})
    train, _, _ = build_datasets(config)
    assert len(train) == 40


def test_enabled_without_augment_is_independent(tmp_path: Path) -> None:
    """拡張なし（features 経路）でも有効化でき、学習が完走して log.txt に件数が出る。"""
    musan = _write_musan(tmp_path / "musan")
    config = _features_config(
        tmp_path, {"enabled": True, "ratio": 0.3, "musan_root": str(musan)}
    )
    outcome = run_training(config)
    log_text = (outcome.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "無音サンプル=あり 割合=0.3" in log_text
    assert "デジタル無音=4 MUSAN雑音のみ=4 極小音量の雑音=4" in log_text
    rows = [json.loads(line) for line in (outcome.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["train_clips"] == 52
    assert rows[0]["train_silence_clips"] == 12
    assert rows[0]["val_clips"] == 6


# --------------------------------------------------------------------------------------
# 学習開始前の検査


def test_missing_musan_stops_before_data_loading(tmp_path: Path) -> None:
    config = _features_config(
        tmp_path, {"enabled": True, "musan_root": str(tmp_path / "no_musan")}
    )
    with pytest.raises(SilenceSetupError, match="musan_root"):
        run_training(config)
    log_text = (config.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "無音サンプルの設定の誤りで停止する" in log_text
    assert "データ:" not in log_text  # データ読み込みより前に止まっている


def test_musan_without_wav_stops(tmp_path: Path) -> None:
    (tmp_path / "musan" / "noise").mkdir(parents=True)
    with pytest.raises(SilenceSetupError, match="wav"):
        check_silence_setup(
            SilenceSettings.from_mapping({"enabled": True, "musan_root": str(tmp_path / "musan")})
        )


def test_digital_only_does_not_need_musan(tmp_path: Path) -> None:
    settings = SilenceSettings.from_mapping(
        {"enabled": True, "musan_root": str(tmp_path / "none"), "mix": {"digital_silence": 1}}
    )
    lines = check_silence_setup(settings)
    assert "雑音源=（使わない）" in lines[1]


@pytest.mark.parametrize(
    "mapping",
    [
        {"ratio": 0.0},
        {"ratio": 1.5},
        {"mix": {"digital_silence": 0, "musan_noise": 0, "quiet_noise": 0}},
        {"mix": {"digital_silence": -1}},
        {"quiet_noise_dbfs_range": [-60, -90]},
        {"quiet_noise_dbfs_range": [-10, 5]},
        {"duration_range_sec": [0.0, 2.0]},
    ],
)
def test_invalid_settings_stop(tmp_path: Path, mapping: dict) -> None:
    _write_musan(tmp_path / "musan")
    with pytest.raises(SilenceSetupError):
        settings = SilenceSettings.from_mapping(
            {"enabled": True, "musan_root": str(tmp_path / "musan"), **mapping}
        )
        check_silence_setup(settings)


def test_unknown_key_and_kind_rejected() -> None:
    with pytest.raises(ValueError):
        SilenceSettings.from_mapping({"proportion": 0.1})
    with pytest.raises(SilenceSetupError):
        SilenceSettings.from_mapping({"mix": {"speech": 1.0}})
