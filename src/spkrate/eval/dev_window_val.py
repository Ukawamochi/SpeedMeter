"""方式Bの学習の検証に使う dev_window の抽出50,000窓（docs/decisions/009-method-b.md 1.7節）。

設定は ``configs/eval/dev_window_val.yaml``、入口は ``scripts/build_dev_window_val.py``。

- 母集団: ``data/processed/dev_window/`` の窓のうち、主指標の除外（known_no_speech:
  ``results/error_cases/to_listen.tsv``、no_speech_suspect: ``results/no_speech_suspect_dev.tsv``）の
  クリップを由来に持たない窓（``spkrate.eval.no_speech.windows_from_clips``。
  ``scripts/eval_dev_window.py`` と同じ）
- 抽出: ``default_rng(seed).choice(主指標の窓の番号, num_windows, replace=False)`` を昇順に並べる。
  単一と連結で層に分けない（``select_val_windows``）
- 特徴量: ``DevWindowAudio.window_at``（clean）で波形を再生成し、対数メル（正規化前）を float16 で
  ``output_dir/features.npy``（形 (件数, 201, 80)）に保存する。窓の番号・正解・単一/連結の別は
  ``windows.npz``、設定と件数は ``meta.json``（``complete`` が真なら全件の計算が済んでいる）
- 学習中は ``DevWindowValDataset`` が memmap で読み、正規化してから返す。正解はモーラ数、
  長さは2.0秒（``evaluate_dev`` がそのまま毎秒モーラ数の指標を出す）

雑音下の版は使わない（検証は clean。雑音下は学習後の評価で測る）。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

__all__ = [
    "DEFAULT_VAL_CONFIG",
    "DevWindowValConfig",
    "DevWindowValDataset",
    "load_suspect_ids",
    "load_val_config",
    "main_indicator_mask",
    "select_val_windows",
]

DEFAULT_VAL_CONFIG = Path("configs/eval/dev_window_val.yaml")


@dataclass(frozen=True)
class DevWindowValConfig:
    seed: int
    num_windows: int
    dev_window_config: str = "configs/eval/dev_window.yaml"
    suspect_list: str = "results/no_speech_suspect_dev.tsv"
    output_dir: str = "data/processed/dev_window_val"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DevWindowValConfig":
        values = dict(mapping)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(f"DevWindowValConfig に未知の設定項目: {unknown}")
        config = cls(**values)
        if config.num_windows <= 0:
            raise ValueError(f"num_windows は正: {config.num_windows}")
        return config


def load_val_config(path: str | Path = DEFAULT_VAL_CONFIG) -> DevWindowValConfig:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return DevWindowValConfig.from_mapping(yaml.safe_load(handle))


def load_suspect_ids(path: str | Path) -> set[str]:
    """no_speech_suspect の clip_id（TSV の ``clip_id`` 列。``#`` の行は注記として飛ばす）。"""
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return {row["clip_id"] for row in csv.DictReader(lines, delimiter="\t")}


def main_indicator_mask(window_set: Any, excluded_clip_ids: Iterable[str]) -> np.ndarray:
    """主指標の窓（除外のクリップを由来に持たない窓）の真偽配列。"""
    from spkrate.eval.no_speech import windows_from_clips

    flagged = windows_from_clips(window_set, excluded_clip_ids)
    known = np.asarray(window_set.known_no_speech, dtype=bool)
    if not np.array_equal(known, known & flagged):
        raise ValueError("known_no_speech の窓が除外に含まれていない")
    return ~flagged


def select_val_windows(mask: np.ndarray, num_windows: int, seed: int) -> np.ndarray:
    """主指標の窓から ``num_windows`` 件を非復元抽出し、昇順の窓の番号（int64）を返す。

    母集団が ``num_windows`` より少なければ全件を返す。
    """
    population = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
    size = min(int(num_windows), population.size)
    picked = np.random.default_rng(int(seed)).choice(population, size=size, replace=False)
    return np.sort(picked).astype(np.int64)


class DevWindowValDataset(Dataset):
    """事前計算した検証窓の特徴量を memmap で読む（拡張なし）。"""

    def __init__(
        self,
        directory: str | Path,
        *,
        normalizer: Any = None,
        limit: int | None = None,
        window_sec: float = 2.0,
    ) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"検証窓の特徴量が無い: {meta_path}（scripts/build_dev_window_val.py で作る）"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(self.directory / "windows.npz") as arrays:
            self.window_index = arrays["window_index"].astype(np.int64)
            self.mora = arrays["mora"].astype(np.float32)
            self.kind = arrays["kind"].astype(np.int8)
        done = int(self.meta.get("computed", 0))
        count = self.window_index.size if limit is None else min(int(limit), self.window_index.size)
        if not self.meta.get("complete", False) and count > done:
            raise ValueError(
                f"検証窓の特徴量の計算が済んでいない（{done}/{self.window_index.size} 件）: {self.directory}"
            )
        self.count = count
        self.normalizer = normalizer
        self.window_sec = float(window_sec)
        self._features: np.ndarray | None = None

    def __len__(self) -> int:
        return self.count

    @property
    def frame_counts(self) -> list[int]:
        return [int(self.meta["frames"])] * self.count

    @property
    def durations_sec(self) -> list[float]:
        return [self.window_sec] * self.count

    def _array(self) -> np.ndarray:
        if self._features is None:
            self._features = np.load(self.directory / "features.npy", mmap_mode="r")
        return self._features

    def __getitem__(self, index: int):  # noqa: ANN204
        from spkrate.train.data import ClipItem

        if not 0 <= index < self.count:
            raise IndexError(index)
        feature = np.asarray(self._array()[index], dtype=np.float32)
        if self.normalizer is not None:
            feature = self.normalizer(feature)
        return ClipItem(
            features=feature,
            mora=float(self.mora[index]),
            duration_sec=self.window_sec,
            clip_id=f"dev_window:{int(self.window_index[index])}",
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_features"] = None  # memmap をワーカーへ持ち越さない
        return state


def config_as_dict(config: DevWindowValConfig) -> dict[str, Any]:
    return asdict(config)
