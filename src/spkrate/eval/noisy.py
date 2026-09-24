"""雑音下評価セット dev_noisy の生成（設定は ``configs/eval/dev_noisy.yaml``）。

dev（``configs/splits/dev.json``）の各クリップに、固定のインパルス応答による残響と
MUSAN noise を SNR 5 / 10 / 15 dB で掛けたものを dev_noisy とする。構成の詳細は
設定ファイルの冒頭のコメントに書いた。要点は次のとおり。

- dev 全件 × 3条件（SNR ごとに dev 全件）。正解のモーラ数と区間長は dev と同じ
  （雑音も残響も発話を消さず、``apply_reverb`` は長さと時刻を変えない）
- 適用順は「残響 → 雑音」（``spkrate.data.augment.augment_waveform`` と同じ）
- 雑音ファイルと切り出し位置はクリップごとの乱数生成器
  ``default_rng([seed, clip_key(clip_id)])`` で決まり、SNR には依存しない
- 残響は乱数を使わない固定のインパルス応答1本（``fixed_rir``）

## 保存形式: 評価のたびに波形から決定的に生成する

事前計算した特徴量を ``data/processed/`` に置く案と比べ、次の理由で都度生成を採った。

- 所要時間: exp001 の最良チェックポイントで dev 先頭3,000件を評価すると、mp3の復号・
  残響・雑音・対数メル・mps での推論まで含めて1条件あたり約11秒（約260件/秒、
  DataLoader のワーカー0。ワーカーを増やすと遅くなった。``scripts/eval_dev_full.py`` の
  ``--noisy-num-workers``）だった。1クリップあたりの内訳は復号1.6ミリ秒・残響1.7ミリ秒・
  雑音0.7ミリ秒・対数メル0.3ミリ秒。dev 全件29,518件 × 3条件で約6分（mp3 がディスク
  キャッシュに無い初回はこれより長い）と見積もられ、事前計算で省けるのはこの数分である
- 容量: dev の特徴量は float16 で約2.0GiB（``data/processed/features/dev``）。3条件で
  約6GiB を追加で持つことになる
- 一貫性: 生成規則（本モジュール・設定・シード）が変わったときに古い特徴量が残る
  危険がない。決定性は ``tests/test_noisy.py`` で確認する（同じシードでビット一致、
  別シードで異なる）

## 実行環境への依存

生成は numpy の乱数生成器（PCG64）と pyroomacoustics・scipy・librosa（再標本化）の
決定的な計算だけからなる。同じ環境なら毎回ビット一致する。ライブラリの版が変わると
浮動小数点の丸めが変わり得るので、比較する評価は同じロックファイル（uv.lock）で行う。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from spkrate.data.augment import (
    SAMPLE_RATE,
    MusanNoiseSource,
    NoiseSource,
    add_noise,
    apply_reverb,
)
from spkrate.data.common_voice import ClipRecord
from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram
from spkrate.train.data import ClipItem

__all__ = [
    "DEFAULT_NOISY_CONFIG",
    "NoisyClipDataset",
    "NoisyDevConfig",
    "ReverbSpec",
    "clip_key",
    "clip_rng",
    "degrade_waveform",
    "fixed_rir",
    "load_noisy_config",
    "make_noise_source",
    "snr_label",
]

DEFAULT_NOISY_CONFIG = Path("configs/eval/dev_noisy.yaml")


@dataclass(frozen=True)
class ReverbSpec:
    """固定のインパルス応答を作る部屋の値（すべてメートル・秒）。"""

    room_dim_m: tuple[float, float, float]
    rt60_sec: float
    source_pos_m: tuple[float, float, float]
    mic_pos_m: tuple[float, float, float]
    max_order: int

    def __post_init__(self) -> None:
        for name in ("room_dim_m", "source_pos_m", "mic_pos_m"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"reverb.{name} は3要素で書くこと: {getattr(self, name)}")
        for axis in range(3):
            size = self.room_dim_m[axis]
            for name in ("source_pos_m", "mic_pos_m"):
                position = getattr(self, name)[axis]
                if not 0.0 < position < size:
                    raise ValueError(f"reverb.{name} が部屋の外にある: {getattr(self, name)}")
        if self.rt60_sec <= 0.0:
            raise ValueError(f"reverb.rt60_sec は正の値: {self.rt60_sec}")
        if self.max_order < 0:
            raise ValueError(f"reverb.max_order は0以上: {self.max_order}")


@dataclass(frozen=True)
class NoisyDevConfig:
    """``configs/eval/dev_noisy.yaml`` の内容。"""

    seed: int
    snr_db: tuple[float, ...]
    reverb: ReverbSpec
    musan_root: str = "data/musan"
    musan_subsets: tuple[str, ...] = ("noise",)

    def __post_init__(self) -> None:
        if not self.snr_db:
            raise ValueError("snr_db を1つ以上書くこと")
        if len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError(f"snr_db に重複がある: {self.snr_db}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "NoisyDevConfig":
        reverb = mapping["reverb"]
        return cls(
            seed=int(mapping["seed"]),
            snr_db=tuple(float(value) for value in mapping["snr_db"]),
            reverb=ReverbSpec(
                room_dim_m=tuple(float(v) for v in reverb["room_dim_m"]),
                rt60_sec=float(reverb["rt60_sec"]),
                source_pos_m=tuple(float(v) for v in reverb["source_pos_m"]),
                mic_pos_m=tuple(float(v) for v in reverb["mic_pos_m"]),
                max_order=int(reverb["max_order"]),
            ),
            musan_root=str(mapping.get("musan_root", "data/musan")),
            musan_subsets=tuple(str(s) for s in mapping.get("musan_subsets", ("noise",))),
        )


def load_noisy_config(path: str | Path = DEFAULT_NOISY_CONFIG) -> NoisyDevConfig:
    """設定ファイルを読む。"""
    import yaml

    with open(path, encoding="utf-8") as handle:
        return NoisyDevConfig.from_mapping(yaml.safe_load(handle))


def snr_label(snr_db: float) -> str:
    """SNR の表記（5.0 → "5"、7.5 → "7.5"）。split 名やファイル名に使う。"""
    return f"{float(snr_db):g}"


def make_noise_source(
    config: NoisyDevConfig, repo_root: str | Path | None = None
) -> MusanNoiseSource:
    """設定の MUSAN を読む雑音源（読み取りのみ）。相対パスは ``repo_root`` から解決する。"""
    root = Path(config.musan_root)
    if not root.is_absolute() and repo_root is not None:
        root = Path(repo_root) / root
    return MusanNoiseSource(root, subsets=config.musan_subsets)


@lru_cache(maxsize=8)
def _fixed_rir_cached(spec: ReverbSpec, sample_rate: int) -> np.ndarray:
    import pyroomacoustics as pra

    absorption, _ = pra.inverse_sabine(spec.rt60_sec, list(spec.room_dim_m))
    room = pra.ShoeBox(
        list(spec.room_dim_m),
        fs=int(sample_rate),
        materials=pra.Material(absorption),
        max_order=int(spec.max_order),
    )
    room.add_source(list(spec.source_pos_m))
    room.add_microphone(list(spec.mic_pos_m))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    peak = float(np.max(np.abs(rir)))
    if not peak > 0.0:
        raise ValueError(f"インパルス応答が全て0になった: {spec}")
    rir = np.ascontiguousarray(rir / np.float32(peak))
    rir.setflags(write=False)
    return rir


def fixed_rir(spec: ReverbSpec, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """設定の固定値からインパルス応答を1本作る（乱数を使わない。最大値の絶対値で1に正規化）。

    Sabine の式の逆算で吸音率を決める。実現できない組（吸音率が1を超える）は
    ``ValueError``（拡張の ``generate_rir`` と違い、無響に黙って置き換えない）。
    同じ値での2回目以降はキャッシュを返す（読み取り専用の配列）。
    """
    return _fixed_rir_cached(spec, int(sample_rate))


def clip_key(clip_id: str) -> int:
    """clip_id から決まる64ビットの非負整数（SHA-256 の先頭8バイト）。

    Python の ``hash`` はプロセスごとに変わるので使わない。
    """
    digest = hashlib.sha256(clip_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def clip_rng(seed: int, clip_id: str) -> np.random.Generator:
    """そのクリップの雑音の選択に使う乱数生成器。評価順や件数に依存しない。"""
    return np.random.default_rng([int(seed), clip_key(clip_id)])


def degrade_waveform(
    samples: np.ndarray,
    clip_id: str,
    snr_db: float,
    *,
    config: NoisyDevConfig,
    noise_source: NoiseSource,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """1クリップの波形に dev_noisy の加工（固定の残響 → 雑音）を掛ける。

    雑音ファイルと切り出し位置は ``clip_rng(config.seed, clip_id)`` で決まり、
    ``snr_db`` には依存しない。戻り値は入力と同じ長さの float32。
    """
    reverberant = apply_reverb(samples, fixed_rir(config.reverb, sample_rate))
    noise = noise_source.sample(reverberant.size, clip_rng(config.seed, clip_id))
    return add_noise(reverberant, noise, float(snr_db))


class NoisyClipDataset(Dataset):
    """dev_noisy の1条件（1つの SNR）を ``ClipItem`` として返す。

    ``spkrate.train.train.evaluate_dev`` にそのまま渡せる（``collate_clips`` と
    ``LengthBucketBatchSampler`` を使う）。区間長は ``ClipRecord.duration_sec``
    （clips.jsonl の値。事前計算特徴量の dev と同じ）を使う。
    """

    def __init__(
        self,
        records: Sequence[ClipRecord],
        audio_root: str | Path,
        snr_db: float,
        *,
        config: NoisyDevConfig,
        noise_source: NoiseSource,
        normalizer: Any = None,
        mel_config: Any = MEL_DEFAULTS,
    ) -> None:
        if not records:
            raise ValueError("クリップが1件も無い")
        self.records = list(records)
        self.audio_root = Path(audio_root)
        self.snr_db = float(snr_db)
        self.config = config
        self.noise_source = noise_source
        self.normalizer = normalizer
        self.mel_config = mel_config
        self._transform: LogMelSpectrogram | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def frame_counts(self) -> list[int]:
        """各件のおおよそのフレーム数（バッチ分け用。clips.jsonl の長さから見積る）。"""
        return [max(1, int(record.duration_sec * 100.0)) for record in self.records]

    def waveform(self, index: int) -> np.ndarray:
        """加工後の16kHz波形。"""
        from spkrate.eval.audio import load_audio

        record = self.records[index]
        samples, sample_rate = load_audio(self.audio_root / record.audio_path)
        return degrade_waveform(
            samples,
            record.clip_id,
            self.snr_db,
            config=self.config,
            noise_source=self.noise_source,
            sample_rate=sample_rate,
        )

    def __getitem__(self, index: int) -> ClipItem:
        record = self.records[index]
        if self._transform is None:
            self._transform = LogMelSpectrogram(self.mel_config)
        feature = self._transform(self.waveform(index), SAMPLE_RATE)
        if self.normalizer is not None:
            feature = self.normalizer(feature)
        return ClipItem(
            features=np.asarray(feature, dtype=np.float32),
            mora=float(record.mora),
            duration_sec=float(record.duration_sec),
            clip_id=record.clip_id,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_transform"] = None  # torch のモジュールをワーカーへ持ち越さない
        return state
