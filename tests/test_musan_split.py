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
