"""評価時の音声読み込み（docs/PLAN.md 第3段階 3-1）。

docs/spec.md の入力は「16kHzモノラル音声」である。一方 Common Voice の mp3 は
クリップごとに標本化周波数が異なる（32kHz や 48kHz が混在する）ため、評価器へ渡す
前にここでモノラル化と16kHzへの再標本化を行う。

data/ 以下は読み取りのみで、変換結果をファイルへ書き戻すことはしない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = ["TARGET_SAMPLE_RATE", "load_audio", "resample", "to_mono"]

# docs/spec.md「入力: 16kHzモノラル音声」。
TARGET_SAMPLE_RATE = 16000


def to_mono(samples: np.ndarray) -> np.ndarray:
    """(フレーム数, チャンネル数) または1次元の配列をモノラルの1次元にする。

    多チャンネルはチャンネル方向の平均を取る。dtype は float32 に揃える
    （CLAUDE.md「float64は使わない」）。
    """
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=1, dtype=np.float32)
    raise ValueError(f"想定外の次元数: {array.ndim}")


def resample(
    samples: np.ndarray, orig_sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """モノラル波形を再標本化する。dtype は float32 を保つ。

    標本化周波数が同じ場合は変換せずそのまま返す。
    """
    array = to_mono(samples)
    if orig_sample_rate <= 0:
        raise ValueError(f"標本化周波数は正の値である必要がある: {orig_sample_rate}")
    if orig_sample_rate == target_sample_rate:
        return array
    # librosa は依存に含まれており、既定で高品質な soxr を使う。
    import librosa

    converted = librosa.resample(
        array, orig_sr=orig_sample_rate, target_sr=target_sample_rate
    )
    return np.asarray(converted, dtype=np.float32)


def load_audio(
    path: str | Path, *, target_sample_rate: int = TARGET_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """音声ファイルを16kHzモノラルの float32 配列として読む。

    Returns:
        (波形, 標本化周波数)。標本化周波数は ``target_sample_rate`` そのもの。
    """
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = to_mono(samples)
    return resample(mono, int(sample_rate), target_sample_rate), target_sample_rate


def duration_sec(samples: np.ndarray, sample_rate: int) -> float:
    """波形の長さ（秒）。"""
    if sample_rate <= 0:
        raise ValueError(f"標本化周波数は正の値である必要がある: {sample_rate}")
    return float(len(to_mono(samples))) / float(sample_rate)
