"""拡張設定の学習開始前検査の単体テスト（results/augment_validation.md）。

実データには依存しない。MUSAN は一時ディレクトリに小さな wav を書いて模す。

固定する性質。

- 拡張が有効なのに、ある拡張が黙って実行されない設定は ``AugmentSetupError`` で止まる
  （musan_root の欠落、features 経路、確率0、無変化の範囲、依存の欠落など）
- 止まるのはデータ読み込み・モデル構築より前で、理由が log.txt に残る
- 正しい設定では止まらず、log.txt の学習ループより前に拡張の一覧が出る
- 拡張が無効なら「拡張=なし」と出る。無効なのに params や musan_root が書かれていれば
  止めずに警告を出し、log.txt にも残す
"""

from __future__ import annotations

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


def _config(tmp_path: Path, *, augment: dict[str, Any] | None = None, **data: Any) -> TrainConfig:
    augment_section: dict[str, Any] = {"enabled": True, "musan_root": str(tmp_path / "musan")}
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
    assert "3ファイル" in joined


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
    assert "3ファイル" in lines[noise_line]


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
    config = _config(tmp_path, augment={"enabled": False, "musan_root": None}, source="features")
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
