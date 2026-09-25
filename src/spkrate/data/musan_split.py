"""MUSAN noise のファイル単位の学習用・評価用の分割（docs/spec.md「学習データ」「dev_noisy」）。

学習時の雑音（拡張の雑音重畳・無音サンプル）と評価時の雑音（dev_noisy・診断 D2 の雑音窓）に
同じ雑音ファイルが入ると、評価が「学習で聞いた雑音」での値になり、未知の雑音への頑健さを
測れない。そこで ``data/musan/noise`` 以下の wav をファイル単位で学習用（``train``）と
評価用（``eval``）に分け、``configs/splits/musan_noise.json`` に固定する。

分割の性質

- 単位は wav ファイル。比率は学習8対評価2（``DEFAULT_EVAL_RATIO``）
- 取得元の区分（``noise`` 直下のサブディレクトリ。free-sound・sound-bible）ごとに比率を保つ。
  区分ごとに評価用を round(件数 × 0.2) 件とる
- 乱択は固定の種（``MUSAN_SPLIT_SEED``）だけを使う。区分ごとにパスで整列してから
  ``numpy.random.default_rng([seed, 区分の番号])`` で並べ替えるので、何度実行しても同じ分割になる
- ファイルは ``musan_root`` からの相対パス（例 ``noise/free-sound/noise-free-sound-0000.wav``）で記録する

使う側は ``load_musan_noise_split(path, "train" | "eval")`` で片側だけを読む
（``MusanNoiseSource.from_split``）。分割ファイルのパスは各設定で指定し、指定が無いまま
MUSAN noise を使おうとすると学習・評価の開始前に ``MusanSplitError`` で止める。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_EVAL_RATIO",
    "MUSAN_SPLIT_SEED",
    "SPLIT_PARTS",
    "MusanSplitError",
    "build_musan_noise_split",
    "list_noise_files",
    "load_musan_noise_split",
    "require_split_path",
]

# 分割を固定した日付（2026-09-25）。この値を変えない限り分割は再現される。
MUSAN_SPLIT_SEED = 20260925
DEFAULT_EVAL_RATIO = 0.2
SPLIT_PARTS = ("train", "eval")


class MusanSplitError(RuntimeError):
    """MUSAN noise の分割の指定が無い・壊れている。学習・評価の開始前に止めるための例外。"""


def list_noise_files(musan_root: str | Path) -> dict[str, list[str]]:
    """``<musan_root>/noise`` 以下の wav を区分（直下のサブディレクトリ）ごとに返す。

    値は ``musan_root`` からの相対パス（``/`` 区切り）で、パス順に整列する。
    ``noise`` 直下に置かれた wav は区分 ``"."`` とする。
    """
    root = Path(musan_root)
    noise_dir = root / "noise"
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"MUSAN の noise が見つからない: {noise_dir}")
    groups: dict[str, list[str]] = {}
    for path in sorted(noise_dir.rglob("*.wav")):
        relative = path.relative_to(noise_dir)
        group = relative.parts[0] if len(relative.parts) > 1 else "."
        groups.setdefault(group, []).append(path.relative_to(root).as_posix())
    if not groups:
        raise FileNotFoundError(f"MUSAN noise の wav が1つも無い: {noise_dir}")
    return {group: sorted(files) for group, files in sorted(groups.items())}


def build_musan_noise_split(
    groups: Mapping[str, Sequence[str]],
    *,
    seed: int = MUSAN_SPLIT_SEED,
    eval_ratio: float = DEFAULT_EVAL_RATIO,
) -> dict[str, Any]:
    """区分ごとに比率を保って学習用・評価用に分け、分割ファイルの内容を返す。"""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio は0より大きく1未満: {eval_ratio}")
    train: list[str] = []
    evaluation: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for index, group in enumerate(sorted(groups)):
        files = sorted(groups[group])
        if len(set(files)) != len(files):
            raise ValueError(f"区分 {group} に重複したパスがある")
        order = np.random.default_rng([int(seed), index]).permutation(len(files))
        num_eval = int(round(len(files) * eval_ratio))
        eval_part = sorted(files[i] for i in order[:num_eval])
        train_part = sorted(files[i] for i in order[num_eval:])
        evaluation.extend(eval_part)
        train.extend(train_part)
        counts[group] = {"train": len(train_part), "eval": len(eval_part), "total": len(files)}
    return {
        "unit": "file",
        "seed": int(seed),
        "eval_ratio": float(eval_ratio),
        "method": (
            "取得元の区分（noise 直下のサブディレクトリ）ごとにパスで整列し、"
            "numpy.random.default_rng([seed, 区分の番号（区分名の整列順、0始まり）]).permutation で"
            "並べ替えて先頭 round(件数 × eval_ratio) 件を評価用、残りを学習用とする"
        ),
        "paths_relative_to": "musan_root（通常 data/musan）",
        "generated_by": "scripts/build_musan_noise_split.py",
        "implementation": "src/spkrate/data/musan_split.py",
        "usage": {
            "train": "データ拡張の雑音重畳・無音サンプル（学習時のみ）",
            "eval": "dev_noisy・診断 D2 の雑音窓（評価時のみ）",
        },
        "counts": counts,
        "num_train": len(train),
        "num_eval": len(evaluation),
        "train": sorted(train),
        "eval": sorted(evaluation),
    }


def require_split_path(split_path: str | Path | None, *, setting: str) -> Path:
    """分割ファイルのパスが指定されていることを確かめる。無ければ ``MusanSplitError``。"""
    if split_path is None or str(split_path).strip() == "":
        raise MusanSplitError(
            f"{setting} が未設定。MUSAN noise を使う経路では学習用・評価用の分割ファイル"
            "（通常 configs/splits/musan_noise.json）の指定が必要（docs/spec.md「学習データ」）"
        )
    return Path(split_path)


def load_musan_noise_split(split_path: str | Path, part: str) -> list[str]:
    """分割ファイルから片側（``"train"`` か ``"eval"``）のファイル一覧だけを返す。

    学習用と評価用が重なっている、または空の分割ファイルは ``MusanSplitError`` で拒否する。
    """
    if part not in SPLIT_PARTS:
        raise ValueError(f"part は {SPLIT_PARTS} のいずれか: {part}")
    path = Path(split_path)
    if not path.is_file():
        raise MusanSplitError(f"MUSAN noise の分割ファイルが無い: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        train = [str(p) for p in payload["train"]]
        evaluation = [str(p) for p in payload["eval"]]
    except (KeyError, TypeError) as error:
        raise MusanSplitError(f"分割ファイルに train / eval が無い: {path}") from error
    overlap = sorted(set(train) & set(evaluation))
    if overlap:
        raise MusanSplitError(
            f"分割ファイルの学習用と評価用が重なっている（{len(overlap)}件。例 {overlap[0]}）: {path}"
        )
    files = train if part == "train" else evaluation
    if not files:
        raise MusanSplitError(f"分割ファイルの {part} が空: {path}")
    return list(files)
