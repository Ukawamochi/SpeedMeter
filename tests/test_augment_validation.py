"""拡張設定の学習開始前検査の単体テスト（results/augment_validation.md）。

実データには依存しない。MUSAN は一時ディレクトリに小さな wav を書いて模す。

固定する性質。

- 拡張が有効なのに、ある拡張が黙って実行されない設定は ``AugmentSetupError`` で止まる
  （musan_root の欠落、features 経路、確率0、無変化の範囲、依存の欠落など）
- 止まるのはデータ読み込み・モデル構築より前で、理由が log.txt に残る
- 拡張ごとの ``*_enabled: false`` はエラーにならず、その拡張の確率・範囲は検査しない
- 正しい設定では止まらず、log.txt の学習ループより前に拡張の一覧が出る
- 拡張が無効なら「拡張=なし」と出る。無効なのに params や musan_root が書かれていれば
  止めずに警告を出し、log.txt にも残す
"""

from __future__ import annotations

import json

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from spkrate.data.augment import AugmentConfig, AugmentSetupError, validate_augment_config
from spkrate.train.train import TrainConfig, check_augment_setup, run_training

from test_train import SyntheticClipDataset, _small_model_config


def _write_musan(root: Path, *, num_files: int = 3) -> Path:
    noise_dir = root / "noise" / "free-sound"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(num_files):
        samples = (0.1 * rng.standard_normal(16000)).astype(np.float32)
        sf.write(noise_dir / f"noise-{index:04d}.wav", samples, 16000)
    return root


def _write_split(path: Path, num_files: int = 3) -> Path:
    """模擬 MUSAN（``noise/free-sound/noise-XXXX.wav``）の分割ファイル。最後の1件を評価用にする。"""
    names = [f"noise/free-sound/noise-{index:04d}.wav" for index in range(num_files)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": names[:-1], "eval": names[-1:]}), encoding="utf-8")
    return path


def _config(tmp_path: Path, *, augment: dict[str, Any] | None = None, **data: Any) -> TrainConfig:
    augment_section: dict[str, Any] = {
        "enabled": True,
        "musan_root": str(tmp_path / "musan"),
        "musan_noise_split": str(_write_split(tmp_path / "musan_noise.json")),
    }
    augment_section.update(augment or {})
    return TrainConfig.from_mapping(
        {
            "experiment_id": "augment_check",
            "device": "cpu",
            "runs_dir": str(tmp_path / "runs"),
            "data": {"source": "waveform", **data},
            "augment": augment_section,
            "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "log_interval": 0},
        }
    )


@pytest.fixture()
def musan(tmp_path: Path) -> Path:
    return _write_musan(tmp_path / "musan")


# --------------------------------------------------------------------------------------
# 雑音源と経路


def test_missing_musan_root_stops(tmp_path: Path) -> None:
    config = _config(tmp_path, augment={"musan_root": None})
    with pytest.raises(AugmentSetupError, match="musan_root が未設定"):
        check_augment_setup(config)


def test_nonexistent_musan_root_stops(tmp_path: Path) -> None:
    config = _config(tmp_path, augment={"musan_root": str(tmp_path / "no_such_dir")})
    with pytest.raises(AugmentSetupError, match="存在しない"):
        check_augment_setup(config)


def test_musan_without_noise_subset_stops(tmp_path: Path) -> None:
    (tmp_path / "musan" / "music").mkdir(parents=True)
    with pytest.raises(AugmentSetupError, match="雑音ファイルが無く"):
        check_augment_setup(_config(tmp_path))


def test_musan_with_zero_noise_files_stops(tmp_path: Path) -> None:
    (tmp_path / "musan" / "noise").mkdir(parents=True)
    with pytest.raises(AugmentSetupError, match="雑音ファイルが無く"):
        check_augment_setup(_config(tmp_path))


def test_features_source_with_augment_stops(tmp_path: Path, musan: Path) -> None:
    config = _config(tmp_path, source="features")
    with pytest.raises(AugmentSetupError, match="data.source=features"):
        check_augment_setup(config)


# --------------------------------------------------------------------------------------
# 個別の拡張が実行されない設定


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"time_stretch_prob": 0.0}, "時間伸縮の確率"),
        ({"reverb_prob": 0.0}, "残響の確率"),
        ({"noise_prob": 0.0}, "雑音重畳の確率"),
        ({"band_limit_prob": 0.0}, "帯域制限の確率"),
        ({"volume_prob": 0.0}, "音量変化の確率"),
        ({"freq_mask_prob": 0.0}, "周波数マスクの確率"),
        ({"volume_prob": 1.5}, "1以下"),
        ({"snr_db_range": [20.0, 0.0]}, "下限が上限を超えている"),
        ({"time_stretch_range": [1.0, 1.0]}, "時間伸縮が常に無変化"),
        ({"time_stretch_range": [0.0, 1.5]}, "正の値"),
        ({"rt60_range": [0.0, 0.7]}, "正の値"),
        ({"room_z_range": [0.5, 4.0]}, "音源とマイクを置けない"),
        ({"reverb_max_order": 0}, "残響が付かない"),
        (
            {"rt60_range": [0.01, 0.02], "room_x_range": [9.0, 10.0]},
            "常に無響",
        ),
        (
            {"low_hz_range": [0.0, 0.0], "high_hz_range": [8000.0, 9000.0]},
            "帯域制限が常に無変化",
        ),
        ({"low_hz_range": [0.0, 4000.0]}, "通過帯域が空"),
        ({"band_limit_order": 0}, "band_limit_order"),
        ({"gain_db_range": [0.0, 0.0]}, "音量変化が常に無変化"),
        ({"freq_mask_num": 0}, "周波数マスクが常に無変化"),
        ({"freq_mask_max_width": 0}, "周波数マスクが常に無変化"),
        ({"sample_rate": 22050}, "16kHz"),
    ],
)
def test_ineffective_augment_params_stop(
    tmp_path: Path, musan: Path, params: dict[str, Any], message: str
) -> None:
    config = _config(tmp_path, augment={"params": params})
    with pytest.raises(AugmentSetupError, match=message):
        check_augment_setup(config)


@pytest.mark.parametrize("module", ["pyroomacoustics", "librosa"])
def test_missing_dependency_stops(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)  # import すると ImportError になる
    with pytest.raises(AugmentSetupError, match=module):
        validate_augment_config(AugmentConfig())


def test_default_augment_config_is_valid() -> None:
    validate_augment_config(AugmentConfig())


# --------------------------------------------------------------------------------------
# ログ


def test_valid_config_lists_augmentations(tmp_path: Path, musan: Path) -> None:
    lines = check_augment_setup(_config(tmp_path))
    assert lines[0].startswith("拡張=あり")
    joined = "\n".join(lines)
    for name in ("時間伸縮", "残響", "雑音重畳", "帯域制限", "音量変化", "周波数マスク"):
        assert f"拡張: {name} 確率=" in joined
    assert str(musan / "noise") in joined
    assert "の train、2ファイル" in joined  # 学習用だけ（3件中2件）


def _read_log(config: TrainConfig) -> list[str]:
    return (config.run_dir / "log.txt").read_text(encoding="utf-8").splitlines()


def test_training_log_lists_augmentations_before_loop(tmp_path: Path, musan: Path) -> None:
    config = _config(tmp_path).with_changes(model_overrides=_small_model_config().as_dict())
    run_training(
        config,
        train_dataset=SyntheticClipDataset(8, seed=0),
        dev_dataset=SyntheticClipDataset(4, seed=1),
    )
    lines = _read_log(config)
    assert "コミット=" in lines[0]
    noise_line = next(i for i, line in enumerate(lines) if "拡張: 雑音重畳" in line)
    first_epoch = next(i for i, line in enumerate(lines) if "エポック1" in line)
    model_line = next(i for i, line in enumerate(lines) if "モデル:" in line)
    assert noise_line < model_line < first_epoch
    assert "の train、2ファイル" in lines[noise_line]


def test_training_log_says_no_augment_when_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path, augment={"enabled": False}, source="features")
    config = config.with_changes(model_overrides=_small_model_config().as_dict())
    run_training(
        config,
        train_dataset=SyntheticClipDataset(8, seed=0),
        dev_dataset=SyntheticClipDataset(4, seed=1),
    )
    assert any(line.endswith("拡張=なし") for line in _read_log(config))


def test_training_stops_before_model_and_logs_reason(tmp_path: Path) -> None:
    config = _config(tmp_path, augment={"musan_root": None})
    with pytest.raises(AugmentSetupError):
        run_training(
            config,
            train_dataset=SyntheticClipDataset(8, seed=0),
            dev_dataset=SyntheticClipDataset(4, seed=1),
        )
    text = "\n".join(_read_log(config))
    assert "拡張の設定の誤りで停止する" in text
    assert "モデル:" not in text
    assert not (config.run_dir / "metrics.jsonl").exists()


# --------------------------------------------------------------------------------------
# 拡張が無効なのに設定が書かれている場合（docs/questions.md 2026-09-24 回答1）

_TEST_LOGGER = "test_augment_validation"


@pytest.mark.parametrize(
    ("augment", "expected"),
    [
        ({"params": {"noise_prob": 0.2}}, "augment.params"),
        ({}, "augment.musan_root"),  # _config は musan_root を書く
    ],
)
def test_disabled_augment_with_settings_warns_but_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, augment: dict[str, Any], expected: str
) -> None:
    config = _config(tmp_path, augment={"enabled": False, **augment}, source="features")
    # spkrate.train は setup_logger で propagate=False になりうるので、専用のロガーを渡す。
    with caplog.at_level("WARNING", logger=_TEST_LOGGER):
        lines = check_augment_setup(config, logging.getLogger(_TEST_LOGGER))
    assert lines == ["拡張=なし"]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert expected in warnings[0].getMessage()
    assert "使われない" in warnings[0].getMessage()


def test_disabled_augment_without_settings_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(
        tmp_path,
        augment={"enabled": False, "musan_root": None, "musan_noise_split": None},
        source="features",
    )
    with caplog.at_level("WARNING", logger=_TEST_LOGGER):
        assert check_augment_setup(config, logging.getLogger(_TEST_LOGGER)) == ["拡張=なし"]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_disabled_augment_warning_is_written_to_log_txt(tmp_path: Path) -> None:
    config = _config(
        tmp_path, augment={"enabled": False, "params": {"noise_prob": 0.2}}, source="features"
    )
    config = config.with_changes(model_overrides=_small_model_config().as_dict())
    run_training(
        config,
        train_dataset=SyntheticClipDataset(8, seed=0),
        dev_dataset=SyntheticClipDataset(4, seed=1),
    )
    lines = _read_log(config)
    warning = next(line for line in lines if "WARNING" in line and "augment.params" in line)
    assert "augment.musan_root" in warning
    assert any("エポック1" in line for line in lines)  # 止まらずに学習した


# --------------------------------------------------------------------------------------
# 拡張ごとの有効フラグ（docs/questions.md 2026-09-24 回答2）

_NAMES = ("time_stretch", "reverb", "noise", "band_limit", "volume", "freq_mask")
_LABELS = ("時間伸縮", "残響", "雑音重畳", "帯域制限", "音量変化", "周波数マスク")


def test_individual_flags_default_to_enabled() -> None:
    config = AugmentConfig()
    assert all(getattr(config, f"{name}_enabled") is True for name in _NAMES)


@pytest.mark.parametrize(("name", "label"), list(zip(_NAMES, _LABELS)))
def test_individually_disabled_augment_does_not_stop(
    tmp_path: Path, musan: Path, name: str, label: str
) -> None:
    # 確率0や無効な範囲が書かれていても、個別に無効なら検査しない
    params: dict[str, Any] = {f"{name}_enabled": False, f"{name}_prob": 0.0}
    lines = check_augment_setup(_config(tmp_path, augment={"params": params}))
    assert f"拡張: {label} 無効（{name}_enabled=false）" in lines
    others = [other for other in _LABELS if other != label]
    for other in others:
        assert any(line.startswith(f"拡張: {other} 確率=") for line in lines)


def test_all_individually_disabled_does_not_stop(tmp_path: Path, musan: Path) -> None:
    params = {f"{name}_enabled": False for name in _NAMES}
    lines = check_augment_setup(_config(tmp_path, augment={"params": params}))
    assert sum("無効（" in line for line in lines) == len(_NAMES)


def test_noise_disabled_does_not_require_musan(tmp_path: Path) -> None:
    config = _config(tmp_path, augment={"musan_root": None, "params": {"noise_enabled": False}})
    lines = check_augment_setup(config)
    assert "拡張: 雑音重畳 無効（noise_enabled=false）" in lines


def test_enabled_with_zero_probability_still_stops(tmp_path: Path, musan: Path) -> None:
    config = _config(
        tmp_path, augment={"params": {"reverb_enabled": True, "reverb_prob": 0.0}}
    )
    with pytest.raises(AugmentSetupError, match="reverb_enabled: false"):
        check_augment_setup(config)


def test_non_bool_flag_stops() -> None:
    with pytest.raises(AugmentSetupError, match="true か false"):
        validate_augment_config(AugmentConfig.from_mapping({"reverb_enabled": "false"}))


# --------------------------------------------------------------------------------------
# docs/spec.md の範囲（docs/questions.md 2026-09-24 回答3）


@pytest.mark.parametrize(
    "params",
    [
        {"time_stretch_range": [0.4, 1.5]},  # spec の許す最大（0.5〜1.5倍）の外
        {"time_stretch_range": [0.7, 1.6]},
        {"time_stretch_range": [0.5, 2.0]},
        {"snr_db_range": [-5.0, 20.0]},
        {"snr_db_range": [0.0, 30.0]},
    ],
)
def test_out_of_spec_range_stops(tmp_path: Path, musan: Path, params: dict[str, Any]) -> None:
    config = _config(tmp_path, augment={"params": params})
    with pytest.raises(AugmentSetupError, match="docs/spec.md の範囲"):
        check_augment_setup(config)


@pytest.mark.parametrize(
    "params",
    [
        {"time_stretch_range": [0.7, 1.5]},  # 両端を含む
        {"time_stretch_range": [0.5, 1.5]},  # 新しい実験の設定で使ってよい範囲（両端を含む）
        {"time_stretch_range": [0.6, 1.5]},
        {"time_stretch_range": [0.8, 1.2]},  # 内側への狭めは許す
        {"snr_db_range": [0.0, 20.0]},
        {"snr_db_range": [5.0, 15.0]},
        # spec.md に範囲が無い項目には新たな制約を設けない
        {"gain_db_range": [-30.0, 20.0]},
        {"rt60_range": [0.2, 1.2]},
        {"high_hz_range": [2000.0, 7900.0]},
    ],
)
def test_within_spec_range_or_unspecified_passes(
    tmp_path: Path, musan: Path, params: dict[str, Any]
) -> None:
    check_augment_setup(_config(tmp_path, augment={"params": params}))


def test_out_of_spec_range_is_ignored_when_individually_disabled(
    tmp_path: Path, musan: Path
) -> None:
    params = {"time_stretch_enabled": False, "time_stretch_range": [0.5, 2.0]}
    check_augment_setup(_config(tmp_path, augment={"params": params}))


# --------------------------------------------------------------------------------------
# 設計上の確率と実際の適用率（docs/questions.md 2026-09-24 回答4）


def test_application_rates_are_deterministic_and_fast() -> None:
    import time

    from spkrate.data.augment import APPLICATION_RATE_TRIALS, estimate_application_rates

    started = time.perf_counter()
    first = estimate_application_rates(AugmentConfig())
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0
    assert first == estimate_application_rates(AugmentConfig())
    assert [rate.label for rate in first] == list(_LABELS)
    for rate in first:
        assert rate.trials == APPLICATION_RATE_TRIALS
        assert rate.applied <= rate.selected <= rate.trials
        assert rate.designed == getattr(AugmentConfig(), f"{rate.name}_prob")
        # 2000回なら抽選の揺らぎは設計上の確率から0.05以内に収まる
        assert abs(rate.selected / rate.trials - rate.designed) < 0.05


def test_anechoic_reverb_is_counted_as_not_applied() -> None:
    from spkrate.data.augment import estimate_application_rates

    rates = {rate.name: rate for rate in estimate_application_rates(AugmentConfig())}
    reverb = rates["reverb"]
    assert reverb.applied < reverb.selected  # 既定の範囲では一部の組が無響になる
    assert 0.0 < 1.0 - reverb.applied / reverb.selected < 0.15
    for name in ("time_stretch", "noise", "band_limit", "volume"):
        assert rates[name].applied == rates[name].selected

    # 大きな部屋と短い RT60 だけなら、ほぼすべての組が実現できない
    hard = AugmentConfig(rt60_range=(0.1, 0.12), room_x_range=(9.0, 10.0),
                         room_y_range=(9.0, 10.0), room_z_range=(3.5, 4.0))
    reverb = {rate.name: rate for rate in estimate_application_rates(hard)}["reverb"]
    assert reverb.selected > 0 and reverb.applied < reverb.selected // 2


def test_application_rates_for_disabled_and_missing_noise() -> None:
    from spkrate.data.augment import estimate_application_rates

    config = AugmentConfig(volume_enabled=False)
    rates = {rate.name: rate for rate in estimate_application_rates(config, noise_available=False)}
    assert rates["volume"].designed == 0.0 and rates["volume"].applied == 0
    assert rates["noise"].designed == 0.0 and rates["noise"].applied == 0
    assert rates["reverb"].designed == 0.3


def test_training_log_records_designed_and_actual_rates(tmp_path: Path, musan: Path) -> None:
    config = _config(tmp_path, augment={"params": {"band_limit_enabled": False}})
    config = config.with_changes(model_overrides=_small_model_config().as_dict())
    run_training(
        config,
        train_dataset=SyntheticClipDataset(8, seed=0),
        dev_dataset=SyntheticClipDataset(4, seed=1),
    )
    lines = _read_log(config)
    model_line = next(i for i, line in enumerate(lines) if "モデル:" in line)
    rate_lines = [i for i, line in enumerate(lines) if "適用率: " in line]
    assert len(rate_lines) == len(_LABELS)
    assert max(rate_lines) < model_line
    header = next(line for line in lines if "拡張の適用率（" in line)
    assert "固定シード0で2000回" in header
    reverb = next(line for line in lines if "適用率: 残響" in line)
    assert "設計=0.300 実際=" in reverb
    band = next(line for line in lines if "適用率: 帯域制限" in line)
    assert "設計=0.000 実際=0.000" in band


# --------------------------------------------------------------------------------------
# 学習中の実適用回数（エポック終了時のログ）


def test_augment_result_effective_excludes_anechoic_and_silent() -> None:
    from spkrate.data.augment import ArrayNoiseSource, augment_waveform

    always = {f"{name}_prob": 1.0 for name in _NAMES}
    noise = ArrayNoiseSource([np.full(1600, 0.1, dtype=np.float32)])
    signal = (0.1 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    result = augment_waveform(
        signal,
        np.random.default_rng(0),
        config=AugmentConfig.from_mapping(always),
        noise_source=noise,
    )
    assert set(result.effective) == set(result.applied)

    # 実現できない残響（大きな部屋に短い RT60）と、無音への雑音は effective に入らない
    hard = AugmentConfig.from_mapping(
        always
        | {
            "time_stretch_enabled": False,
            "rt60_range": [0.1, 0.1],
            "room_x_range": [10.0, 10.0],
            "room_y_range": [10.0, 10.0],
            "room_z_range": [4.0, 4.0],
        }
    )
    result = augment_waveform(
        np.zeros(16000, dtype=np.float32), np.random.default_rng(0), config=hard,
        noise_source=noise,
    )
    assert "reverb" in result.applied and "reverb" not in result.effective
    assert "noise" in result.applied and "noise" not in result.effective


def test_augment_feature_with_status_matches_augment_feature() -> None:
    from spkrate.data.augment import augment_feature, augment_feature_with_status

    feature = np.random.default_rng(0).standard_normal((30, 80)).astype(np.float32)
    config = AugmentConfig(freq_mask_prob=1.0)
    plain = augment_feature(feature, np.random.default_rng(3), config=config)
    masked, applied = augment_feature_with_status(feature, np.random.default_rng(3), config=config)
    np.testing.assert_array_equal(plain, masked)
    assert applied == (not np.array_equal(masked, feature))


def test_collate_carries_augment_applied() -> None:
    from spkrate.train.data import ClipItem, collate_clips

    items = [
        ClipItem(np.zeros((5, 80), np.float32), 1.0, 0.05, "a", ("reverb", "freq_mask")),
        ClipItem(np.zeros((3, 80), np.float32), 1.0, 0.03, "b"),
    ]
    batch = collate_clips(items)
    assert batch.augment_applied == (("reverb", "freq_mask"), ())
    assert batch.to("cpu").augment_applied == batch.augment_applied


class _TaggedDataset(SyntheticClipDataset):
    """実際に掛かった拡張の名前を付けた合成データ（学習中の数え上げの試験用）。"""

    def __getitem__(self, index: int):  # type: ignore[override]
        from dataclasses import replace

        tags = ("reverb", "freq_mask") if index % 2 == 0 else ("volume",)
        return replace(self.items[index], augment_applied=tags)


def test_training_log_records_augment_counts_per_epoch(tmp_path: Path, musan: Path) -> None:
    config = _config(tmp_path).with_changes(model_overrides=_small_model_config().as_dict())
    config = config.with_changes(train=dataclasses.replace(config.train, epochs=2))
    run_training(
        config,
        train_dataset=_TaggedDataset(8, seed=0),
        dev_dataset=SyntheticClipDataset(4, seed=1),
    )
    lines = _read_log(config)
    for epoch in (1, 2):
        line = next(x for x in lines if f"エポック{epoch} 拡張の実適用回数" in x)
        assert "学習8件中" in line
        assert "残響=4(0.500)" in line and "周波数マスク=4(0.500)" in line
        assert "音量変化=4(0.500)" in line and "時間伸縮=0(0.000)" in line
    metrics = (config.run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert "augment_counts" not in metrics



def test_waveform_dataset_reports_effective_augmentations(tmp_path: Path, musan: Path) -> None:
    from spkrate.data.augment import MusanNoiseSource
    from spkrate.data.common_voice import ClipRecord
    from spkrate.train.data import WaveformClipDataset

    t = np.arange(16000, dtype=np.float32) / 16000
    sf.write(tmp_path / "clip.wav", (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 16000)
    record = ClipRecord(
        clip_id="c0", audio_path="clip.wav", client_id="x", sentence="テスト", kana="テスト",
        mora=3, duration_sec=1.0, mora_per_second=3.0,
    )
    config = AugmentConfig.from_mapping(
        {f"{name}_prob": 1.0 for name in _NAMES} | {"reverb_enabled": False}
    )
    dataset = WaveformClipDataset(
        [record], tmp_path, augment=config, noise_source=MusanNoiseSource(musan), seed=0
    )
    item = dataset[0]
    assert "reverb" not in item.augment_applied
    assert {"time_stretch", "noise", "band_limit", "volume"} <= set(item.augment_applied)
