"""MUSAN noise の学習用・評価用の分割（src/spkrate/data/musan_split.py）の試験。

- 固定した分割ファイル configs/splits/musan_noise.json の学習用と評価用が重ならず、
  区分ごとの比率が8対2であること
- 分割の生成が決定的で、区分ごとに比率を保つこと
- 読み出しは指定した側だけを返し、重なった分割ファイルや指定の欠落は止まること
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spkrate.data.musan_split import (
    MUSAN_SPLIT_SEED,
    MusanSplitError,
    build_musan_noise_split,
    load_musan_noise_split,
    require_split_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = REPO_ROOT / "configs" / "splits" / "musan_noise.json"


def _groups() -> dict[str, list[str]]:
    return {
        "a": [f"noise/a/a-{i:03d}.wav" for i in range(50)],
        "b": [f"noise/b/b-{i:03d}.wav" for i in range(11)],
    }


def test_fixed_split_file_is_disjoint_and_stratified() -> None:
    payload = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    train, evaluation = set(payload["train"]), set(payload["eval"])
    assert not train & evaluation
    assert len(train) == len(payload["train"]) and len(evaluation) == len(payload["eval"])
    assert payload["seed"] == MUSAN_SPLIT_SEED
    assert payload["num_train"] + payload["num_eval"] == 930
    for group, count in payload["counts"].items():
        in_train = [p for p in train if p.startswith(f"noise/{group}/")]
        in_eval = [p for p in evaluation if p.startswith(f"noise/{group}/")]
        assert (len(in_train), len(in_eval)) == (count["train"], count["eval"])
        assert count["eval"] == round(count["total"] * 0.2)
    assert set(payload["counts"]) == {"free-sound", "sound-bible"}


def test_build_split_is_deterministic_and_keeps_ratio_per_group() -> None:
    first = build_musan_noise_split(_groups(), seed=7)
    second = build_musan_noise_split(_groups(), seed=7)
    assert first == second
    assert first["counts"] == {
        "a": {"train": 40, "eval": 10, "total": 50},
        "b": {"train": 9, "eval": 2, "total": 11},
    }
    assert not set(first["train"]) & set(first["eval"])
    everything = sorted(p for files in _groups().values() for p in files)
    assert sorted(first["train"] + first["eval"]) == everything
    assert build_musan_noise_split(_groups(), seed=8)["eval"] != first["eval"]


def test_load_returns_only_requested_part(tmp_path: Path) -> None:
    payload = build_musan_noise_split(_groups(), seed=7)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_musan_noise_split(path, "train") == payload["train"]
    assert load_musan_noise_split(path, "eval") == payload["eval"]
    with pytest.raises(ValueError):
        load_musan_noise_split(path, "all")


def test_load_rejects_overlapping_split(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps({"train": ["noise/a/x.wav", "noise/a/y.wav"], "eval": ["noise/a/x.wav"]}),
        encoding="utf-8",
    )
    with pytest.raises(MusanSplitError, match="重なっている"):
        load_musan_noise_split(path, "train")


def test_missing_split_file_or_setting_stops(tmp_path: Path) -> None:
    with pytest.raises(MusanSplitError, match="分割ファイルが無い"):
        load_musan_noise_split(tmp_path / "none.json", "eval")
    for value in (None, ""):
        with pytest.raises(MusanSplitError, match="未設定"):
            require_split_path(value, setting="augment.musan_noise_split")


# --------------------------------------------------------------------------------------
# 各経路が指定された側だけを読むこと・指定が無ければ止まること
#
# 模擬 MUSAN: noise/free-sound に5ファイル、noise/sound-bible に5ファイル。


def _write_mock_musan(tmp_path: Path) -> tuple[Path, Path, dict]:
    import numpy as np
    import soundfile as sf

    root = tmp_path / "musan"
    rng = np.random.default_rng(0)
    for group in ("free-sound", "sound-bible"):
        directory = root / "noise" / group
        directory.mkdir(parents=True)
        for index in range(5):
            samples = (0.1 * rng.standard_normal(8000)).astype(np.float32)
            sf.write(directory / f"{group}-{index}.wav", samples, 16000)
    from spkrate.data.musan_split import list_noise_files

    payload = build_musan_noise_split(list_noise_files(root), seed=3)
    split = tmp_path / "musan_noise.json"
    split.write_text(json.dumps(payload), encoding="utf-8")
    return root, split, payload


def _relative(root: Path, paths) -> set[str]:
    return {Path(p).relative_to(root).as_posix() for p in paths}


def test_source_from_split_reads_only_requested_part(tmp_path: Path) -> None:
    import numpy as np

    from spkrate.data.augment import MusanNoiseSource

    root, split, payload = _write_mock_musan(tmp_path)
    for part in ("train", "eval"):
        source = MusanNoiseSource.from_split(root, split, part)
        assert _relative(root, source.paths) == set(payload[part])
        loaded: list[Path] = []
        original = source.load
        source.load = lambda path, _orig=original: (loaded.append(path), _orig(path))[1]
        rng = np.random.default_rng(0)
        for _ in range(200):
            source.sample(4000, rng)
        assert _relative(root, loaded) <= set(payload[part])
    with pytest.raises(MusanSplitError, match="未設定"):
        MusanNoiseSource.from_split(root, None, "train")


def _train_config(tmp_path: Path, root: Path, split: Path | None):
    from spkrate.train.train import TrainConfig

    return TrainConfig.from_mapping(
        {
            "experiment_id": "musan_split",
            "device": "cpu",
            "runs_dir": str(tmp_path / "runs"),
            "data": {"source": "waveform"},
            "augment": {
                "enabled": True,
                "musan_root": str(root),
                "musan_noise_split": None if split is None else str(split),
            },
            "silence_samples": {
                "enabled": True,
                "musan_root": str(root),
                "musan_noise_split": None if split is None else str(split),
            },
            "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "log_interval": 0},
        }
    )


def test_augment_noise_uses_train_part_only(tmp_path: Path) -> None:
    import logging

    from spkrate.train.train import _build_noise_source, check_augment_setup

    root, split, payload = _write_mock_musan(tmp_path)
    config = _train_config(tmp_path, root, split)
    source = _build_noise_source(config, logging.getLogger("test"))
    assert _relative(root, source.paths) == set(payload["train"])
    assert any(f"train、{payload['num_train']}ファイル" in line
               for line in check_augment_setup(config))


def test_silence_noise_uses_train_part_only(tmp_path: Path) -> None:
    from spkrate.train.silence import check_silence_setup

    root, split, payload = _write_mock_musan(tmp_path)
    settings = _train_config(tmp_path, root, split).silence_samples
    assert f"train、{payload['num_train']}ファイル" in check_silence_setup(settings)[1]

    from spkrate.train.silence import build_silence_dataset

    class _Base:
        durations_sec = [1.0] * 20

        def __len__(self) -> int:
            return 20

    import dataclasses

    settings = dataclasses.replace(settings, ratio=0.5)  # 10件（3種）で雑音の種類を含める
    dataset = build_silence_dataset(
        settings, _Base(), normalizer=None, seed=0, match_feature_storage=False
    )
    assert _relative(root, dataset.noise_source.paths) == set(payload["train"])
    settings_without_split = dataclasses.replace(settings, musan_noise_split=None)
    with pytest.raises(MusanSplitError, match="未設定"):
        build_silence_dataset(
            settings_without_split, _Base(), normalizer=None, seed=0, match_feature_storage=False
        )


def test_training_without_split_stops_before_start(tmp_path: Path) -> None:
    from spkrate.data.augment import AugmentSetupError
    from spkrate.train.silence import SilenceSetupError, check_silence_setup
    from spkrate.train.train import check_augment_setup

    root, _, _ = _write_mock_musan(tmp_path)
    config = _train_config(tmp_path, root, None)
    with pytest.raises(AugmentSetupError, match="augment.musan_noise_split が未設定"):
        check_augment_setup(config)
    with pytest.raises(SilenceSetupError, match="silence_samples.musan_noise_split が未設定"):
        check_silence_setup(config.silence_samples)


def test_past_experiment_config_stops(tmp_path: Path) -> None:
    """exp004 など分割導入前の設定は記録として残し、そのまま学習（再開）すると止まる。"""
    from spkrate.data.augment import AugmentSetupError
    from spkrate.train.silence import SilenceSetupError, check_silence_setup
    from spkrate.train.train import check_augment_setup, load_train_config

    config = load_train_config(REPO_ROOT / "configs" / "exp004.yaml")
    assert config.augment.musan_noise_split is None
    with pytest.raises(AugmentSetupError):
        check_augment_setup(config)
    with pytest.raises(SilenceSetupError):
        check_silence_setup(config.silence_samples)


def test_dev_noisy_uses_eval_part_only(tmp_path: Path) -> None:
    import yaml

    from spkrate.eval.noisy import load_noisy_config, make_noise_source

    root, split, payload = _write_mock_musan(tmp_path)
    mapping = yaml.safe_load(
        (REPO_ROOT / "configs" / "eval" / "dev_noisy.yaml").read_text(encoding="utf-8")
    )
    assert mapping["musan_noise_split"] == "configs/splits/musan_noise.json"
    mapping.update(musan_root=str(root), musan_noise_split=str(split))
    path = tmp_path / "dev_noisy.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    source = make_noise_source(load_noisy_config(path))
    assert _relative(root, source.paths) == set(payload["eval"])


def test_dev_noisy_without_split_stops(tmp_path: Path) -> None:
    import yaml

    from spkrate.eval.noisy import NoisyDevConfig, load_noisy_config, make_noise_source

    mapping = yaml.safe_load(
        (REPO_ROOT / "configs" / "eval" / "dev_noisy.yaml").read_text(encoding="utf-8")
    )
    mapping.pop("musan_noise_split")
    path = tmp_path / "dev_noisy.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    with pytest.raises(MusanSplitError, match="未設定"):
        load_noisy_config(path)
    config = load_noisy_config(REPO_ROOT / "configs" / "eval" / "dev_noisy.yaml")
    bare = NoisyDevConfig(seed=config.seed, snr_db=config.snr_db, reverb=config.reverb)
    with pytest.raises(MusanSplitError, match="未設定"):
        make_noise_source(bare)


def test_d2_noise_uses_eval_part_only(tmp_path: Path) -> None:
    from spkrate.eval.window_diag import make_d2_noise_source

    root, split, payload = _write_mock_musan(tmp_path)
    source = make_d2_noise_source(str(split), str(root))
    assert _relative(root, source.paths) == set(payload["eval"])
    with pytest.raises(MusanSplitError, match="未設定"):
        make_d2_noise_source(None, str(root))


def test_window_diagnostics_without_split_stops_before_model_load(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "window_diagnostics", REPO_ROOT / "scripts" / "window_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    with pytest.raises(MusanSplitError, match="未設定"):
        module.main(["--checkpoint", str(tmp_path / "no_such_checkpoint.pt")])
