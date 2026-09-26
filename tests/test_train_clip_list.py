"""方式A（data.source が features・waveform）の学習クリップの絞り込み（data.train_clip_list）の単体テスト。

実データには依存しない。特徴量・clips.jsonl・MUSAN は一時ディレクトリに小さく作る。

固定する性質。

- 一覧を与えると、その clip_id のクリップだけで学習する（features・waveform の両経路）
- 件数の上限（max_train_clips）は絞り込みの後の件数に掛かる
- 無音サンプルの長さは絞った後のクリップの長さから復元抽出する（docs/spec.md「学習データ」）
- 一覧を与えなければ従来と同じ（全クリップ）
- clips.jsonl に無い clip_id、分割の話者に属さない clip_id、test.json は学習開始前に止まる
- 方式Aの拡張と無音サンプルの MUSAN は分割の学習用（train）だけを使う
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spkrate.train.data import (
    FeatureClipDataset,
    WaveformClipDataset,
    check_clip_list,
    select_clip_records,
)
from spkrate.train.silence import TrainWithSilence
from spkrate.train.train import TrainConfig, build_datasets

from test_silence import _write_musan
from test_train import _small_model_config, _write_feature_shard, _write_normalization


def _write_split(path: Path, split: str, client_ids: list[str]) -> Path:
    path.write_text(json.dumps({"split": split, "client_ids": client_ids}), encoding="utf-8")
    return path


def _write_clips_jsonl(path: Path, num_clips: int = 40) -> Path:
    """_write_feature_shard と同じ clip_id・話者の clips.jsonl（長さはクリップごとに違う）。"""
    rows = []
    for index in range(num_clips):
        duration = 1.0 + 0.1 * index
        rows.append(
            {
                "clip_id": f"clip_{index}",
                "audio_path": f"audio/clip_{index}.wav",
                "client_id": "speaker_a" if index % 2 == 0 else "speaker_b",
                "sentence": "テスト",
                "kana": "テスト",
                "mora": 3,
                "duration_sec": duration,
                "mora_per_second": 3.0 / duration,
            }
        )
    rows.append(
        {
            "clip_id": "other_0",
            "audio_path": "audio/other_0.wav",
            "client_id": "speaker_z",
            "sentence": "テスト",
            "kana": "テスト",
            "mora": 3,
            "duration_sec": 2.0,
            "mora_per_second": 1.5,
        }
    )
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_list(path: Path, clip_ids: list[str]) -> Path:
    path.write_text("# 学習に使う clip_id\n" + "".join(f"{c}\n" for c in clip_ids), encoding="utf-8")
    return path


def _config(
    tmp_path: Path,
    *,
    source: str,
    clip_list: Path | None = None,
    silence: dict | None = None,
    augment: dict | None = None,
    max_train_clips: int | None = None,
) -> TrainConfig:
    features = tmp_path / "features"
    if not (features / "train" / "manifest.json").is_file():
        _write_feature_shard(features / "train", num_clips=40)
        _write_feature_shard(features / "dev", num_clips=6)
    data: dict = {
        "source": source,
        "features_dir": str(features),
        "clips_jsonl": str(_write_clips_jsonl(tmp_path / "clips.jsonl")),
        "audio_root": str(tmp_path),
        "train_split": str(_write_split(tmp_path / "train.json", "train", ["speaker_a", "speaker_b"])),
        "dev_split": str(_write_split(tmp_path / "dev.json", "dev", ["speaker_a", "speaker_b"])),
        "normalization": str(_write_normalization(tmp_path / "norm.yaml")),
        "max_train_clips": max_train_clips,
    }
    if clip_list is not None:
        data["train_clip_list"] = str(clip_list)
    mapping: dict = {
        "experiment_id": "clip_list",
        "device": "cpu",
        "runs_dir": str(tmp_path / "runs"),
        "model_overrides": _small_model_config().as_dict(),
        "data": data,
        "augment": augment or {"enabled": False},
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "log_interval": 0},
    }
    if silence is not None:
        mapping["silence_samples"] = silence
    return TrainConfig.from_mapping(mapping)


def _speech(dataset):  # noqa: ANN001, ANN202
    return dataset.base if isinstance(dataset, TrainWithSilence) else dataset


# --------------------------------------------------------------------------------------
# 下位の関数


def test_select_clip_records_filters_by_clip_ids(tmp_path: Path) -> None:
    clips = _write_clips_jsonl(tmp_path / "clips.jsonl")
    split = _write_split(tmp_path / "train.json", "train", ["speaker_a", "speaker_b"])
    everything = select_clip_records(clips, split)
    assert len(everything) == 40
    assert select_clip_records(clips, split, clip_ids=None) == everything
    chosen = select_clip_records(clips, split, clip_ids={"clip_3", "clip_7", "clip_9"})
    assert [r.clip_id for r in chosen] == ["clip_3", "clip_7", "clip_9"]
    # 上限は絞り込みの後の件数に掛かる（clips.jsonl の順の先頭）
    limited = select_clip_records(clips, split, clip_ids={"clip_3", "clip_7", "clip_9"}, limit=2)
    assert [r.clip_id for r in limited] == ["clip_3", "clip_7"]


def test_feature_dataset_filters_by_clip_ids(tmp_path: Path) -> None:
    directory = _write_feature_shard(tmp_path / "train", num_clips=10)
    assert len(FeatureClipDataset(directory)) == 10
    assert len(FeatureClipDataset(directory, clip_ids=None)) == 10
    dataset = FeatureClipDataset(directory, clip_ids=["clip_1", "clip_4", "clip_8"])
    assert [dataset[i].clip_id for i in range(len(dataset))] == ["clip_1", "clip_4", "clip_8"]
    limited = FeatureClipDataset(directory, clip_ids=["clip_1", "clip_4", "clip_8"], limit=2)
    assert len(limited) == 2


def test_check_clip_list_stops_on_bad_ids(tmp_path: Path) -> None:
    clips = _write_clips_jsonl(tmp_path / "clips.jsonl")
    split = _write_split(tmp_path / "train.json", "train", ["speaker_a", "speaker_b"])
    check_clip_list(["clip_0", "clip_1"], clips, split)
    with pytest.raises(ValueError, match="clips.jsonl に無い"):
        check_clip_list(["clip_0", "nope"], clips, split)
    with pytest.raises(ValueError, match="話者に属さない"):
        check_clip_list(["clip_0", "other_0"], clips, split)
    test_split = _write_split(tmp_path / "test.json", "test", ["speaker_a"])
    with pytest.raises(ValueError, match="テストセット"):
        check_clip_list(["clip_0"], clips, test_split)


# --------------------------------------------------------------------------------------
# 設定からの組み立て


@pytest.mark.parametrize("source", ["features", "waveform"])
def test_without_list_uses_all_clips(tmp_path: Path, source: str) -> None:
    train, _, _ = build_datasets(_config(tmp_path, source=source))
    assert len(train) == 40


@pytest.mark.parametrize("source", ["features", "waveform"])
def test_list_restricts_training_clips(tmp_path: Path, source: str) -> None:
    wanted = [f"clip_{i}" for i in range(0, 40, 3)]
    clip_list = _write_list(tmp_path / "usable.txt", wanted)
    train, dev, _ = build_datasets(_config(tmp_path, source=source, clip_list=clip_list))
    assert len(train) == len(wanted)
    if source == "features":
        ids = [train[i].clip_id for i in range(len(train))]
    else:
        assert isinstance(train, WaveformClipDataset)
        ids = [r.clip_id for r in train.records]
    assert ids == wanted
    assert len(dev) == 6  # 検証は絞らない


@pytest.mark.parametrize("source", ["features", "waveform"])
def test_limit_applies_after_list(tmp_path: Path, source: str) -> None:
    clip_list = _write_list(tmp_path / "usable.txt", ["clip_5", "clip_6", "clip_7", "clip_8"])
    train, _, _ = build_datasets(
        _config(tmp_path, source=source, clip_list=clip_list, max_train_clips=3)
    )
    assert len(train) == 3


@pytest.mark.parametrize("source", ["features", "waveform"])
def test_silence_durations_come_from_listed_clips(tmp_path: Path, source: str) -> None:
    wanted = [f"clip_{i}" for i in range(1, 11)]
    clip_list = _write_list(tmp_path / "usable.txt", wanted)
    musan = _write_musan(tmp_path / "musan")
    silence = {
        "enabled": True,
        "ratio": 1.0,  # 10件 → 10件の無音サンプル（長さの抽出を多めに見る）
        "mix": {"digital_silence": 1.0},
        "musan_root": str(musan),
        "musan_noise_split": str(musan / "musan_noise.json"),
    }
    config = _config(tmp_path, source=source, clip_list=clip_list, silence=silence)
    train, _, _ = build_datasets(config)
    assert isinstance(train, TrainWithSilence)
    speech = _speech(train)
    assert len(speech) == 10
    listed = {round(d * 16000) for d in speech.durations_sec}
    assert len(train.silence) == 10
    assert set(train.silence.num_samples) <= listed
    if source == "waveform":  # 長さが全クリップで異なるので、一覧外の長さが無いことまで言える
        unlisted = {round((1.0 + 0.1 * i) * 16000) for i in range(11, 40)}
        assert not set(train.silence.num_samples) & unlisted


def test_invalid_list_stops_before_training(tmp_path: Path) -> None:
    clip_list = _write_list(tmp_path / "usable.txt", ["clip_0", "other_0"])
    with pytest.raises(ValueError, match="話者に属さない"):
        build_datasets(_config(tmp_path, source="waveform", clip_list=clip_list))


def test_method_a_musan_uses_train_part_of_split(tmp_path: Path) -> None:
    """方式A（waveform）の拡張の雑音源と無音サンプルの雑音源は、分割の学習用だけを使う。"""
    musan = _write_musan(tmp_path / "musan", num_files=4)
    split = str(musan / "musan_noise.json")
    config = _config(
        tmp_path,
        source="waveform",
        clip_list=_write_list(tmp_path / "usable.txt", ["clip_0", "clip_1"]),
        augment={"enabled": True, "musan_root": str(musan), "musan_noise_split": split},
        silence={"enabled": True, "ratio": 1.0, "musan_root": str(musan), "musan_noise_split": split},
    )
    train, _, _ = build_datasets(config)
    eval_name = "noise-0003.wav"
    speech = _speech(train)
    augment_paths = [Path(p).name for p in speech.noise_source.paths]
    assert len(augment_paths) == 3 and eval_name not in augment_paths
    silence_paths = [Path(p).name for p in train.silence.noise_source.paths]
    assert len(silence_paths) == 3 and eval_name not in silence_paths
