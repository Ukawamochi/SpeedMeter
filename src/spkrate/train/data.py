"""学習・検証のデータ供給（docs/PLAN.md 第5段階 5-2）。

docs/decisions/005-window-strategy.md の方式Aに従う。1件の入力は**クリップ全体**で、
1件の正解は**そのクリップのモーラ数**（毎秒モーラ数ではない）である。毎秒モーラ数への
換算（モーラ数 ÷ クリップ長）は評価のときだけ行う。

## 2つの経路（docs/decisions/006-augmentation.md 1節）

| 経路 | 使う場面 | 中身 |
| --- | --- | --- |
| ``features`` | 拡張なしの学習、検証・評価 | ``data/processed/features/<split>`` の事前計算済み float16 シャードを memmap で読む |
| ``waveform`` | 拡張ありの学習 | mp3 を16kHzで読む → ``augment_waveform`` → ``log_mel_spectrogram`` → 正規化 → ``augment_feature``（周波数マスクのみ） |

006 の決定により拡張は**波形**に掛ける。したがって拡張ありのときに事前計算特徴量を
使うことはできない（``build_dataset`` はその組み合わせを ``ValueError`` で拒否する）。
検証はどちらの設定でも拡張なし・``features`` 経路である。

## 詰め物とバッチ化

クリップ長は1.1〜13.3秒（最大12倍）とばらつく。``collate_clips`` は最長のクリップに
合わせて0で詰め、有効フレーム数を ``lengths`` として返す。``SpeechRateCNN`` は
``lengths`` で詰め物のフレームを除いてから総和をとるので、詰め物の量は出力に影響しない
（``tests/test_cnn.py::test_output_does_not_depend_on_padding``）。

## 長さでまとめる（bucketing）を入れた理由

docs/decisions/005-window-strategy.md 4.2節が「長さの近いものを集めてバッチにし、
詰め物の総量を減らす」を実装上の必須事項に挙げている。既定で有効にする
（``LengthBucketBatchSampler``）。無作為に組むと、平均4.47秒に対し最長13.3秒の
クリップが1件混ざるだけでバッチ全体がその長さに引き伸ばされ、計算の大半が0の領域に
費やされる。完全に長さ順に並べると標本の組み合わせがエポック間で固定されるため、
**「無作為に並べる → ``batch_size × pool_batches`` 件の塊ごとに長さで整列 → 切る →
バッチの順序を無作為化」**という折衷にした。塊の大きさで無作為性と詰め物の量を
調整できる（``pool_batches=1`` は実質的に無作為、大きくするほど長さが揃う）。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from spkrate.data.augment import AugmentConfig, NoiseSource, augment_feature, augment_waveform
from spkrate.data.common_voice import ClipRecord
from spkrate.data.splits import load_clip_records, load_split
from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram

__all__ = [
    "Batch",
    "ClipItem",
    "FeatureClipDataset",
    "LengthBucketBatchSampler",
    "Normalizer",
    "WaveformClipDataset",
    "collate_clips",
    "load_normalization",
    "select_clip_records",
]

# 1秒あたりのフレーム数（hop_length=160 / 16000Hz）。長さの見積りに使う。
FRAMES_PER_SECOND = 100.0


# --------------------------------------------------------------------------------------
# 正規化


class Normalizer:
    """configs/normalization.yaml の固定値で対数メルを正規化する。

    docs/spec.md「学習データ全体から算出した平均と標準偏差を固定値として使う。
    発話ごとの正規化は行わない」。``mode`` は ``per_mel``（メル次元ごとに80組。yaml の
    既定）か ``global``（全体で1組）。
    """

    def __init__(self, mean: np.ndarray | float, std: np.ndarray | float, *, mode: str) -> None:
        self.mode = mode
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if np.any(self.std <= 0.0):
            raise ValueError("標準偏差に0以下の値がある")

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        """``(フレーム数, n_mels)`` の対数メルを正規化する。float32 を返す。"""
        array = np.asarray(feature, dtype=np.float32)
        return (array - self.mean) / self.std

    def as_dict(self) -> dict[str, Any]:
        """config_snapshot.yaml に残す要約（80組の値そのものは書かない）。"""
        return {
            "mode": self.mode,
            "mean_first": float(np.ravel(self.mean)[0]),
            "std_first": float(np.ravel(self.std)[0]),
            "num_values": int(np.size(self.mean)),
        }


def load_normalization(path: str | Path, *, mode: str | None = None) -> Normalizer:
    """configs/normalization.yaml を読んで ``Normalizer`` を作る。

    ``mode`` を省略した場合は yaml の ``mode``（既定 ``per_mel``）に従う。
    """
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    resolved = mode or payload.get("mode") or "per_mel"
    if resolved not in {"per_mel", "global"}:
        raise ValueError(f"正規化の mode は per_mel か global: {resolved}")
    section = payload[resolved]
    return Normalizer(section["mean"], section["std"], mode=resolved)


# --------------------------------------------------------------------------------------
# 1件と1バッチ


@dataclass(frozen=True)
class ClipItem:
    """1クリップ分の学習用の組。

    Attributes:
        features: 正規化済みの対数メル ``(フレーム数, n_mels)`` float32。
        mora: そのクリップの正解モーラ数（毎秒モーラ数ではない）。
        duration_sec: クリップ長（秒）。評価で毎秒モーラ数へ直すときに使う。
        clip_id: クリップの識別子。
    """

    features: np.ndarray
    mora: float
    duration_sec: float
    clip_id: str


@dataclass(frozen=True)
class Batch:
    """詰め物を入れた1バッチ。``lengths`` が有効フレーム数である。"""

    features: torch.Tensor  # (バッチ, フレーム, n_mels) float32
    lengths: torch.Tensor  # (バッチ,) long
    moras: torch.Tensor  # (バッチ,) float32  正解モーラ数
    durations: torch.Tensor  # (バッチ,) float32  クリップ長（秒）
    clip_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            features=self.features.to(device),
            lengths=self.lengths.to(device),
            moras=self.moras.to(device),
            durations=self.durations.to(device),
            clip_ids=self.clip_ids,
        )


def collate_clips(items: Sequence[ClipItem]) -> Batch:
    """``ClipItem`` の列を詰め物付きのバッチにする。

    詰め物の値は0である（``SpeechRateCNN`` は入口でも ``lengths`` でマスクするので、
    ここで入れる値は結果に影響しない）。float64 は使わない（CLAUDE.md「実行環境」）。
    """
    if not items:
        raise ValueError("空のバッチは作れない")
    n_mels = int(items[0].features.shape[1])
    lengths = [int(item.features.shape[0]) for item in items]
    max_frames = max(lengths)
    features = np.zeros((len(items), max_frames, n_mels), dtype=np.float32)
    for index, item in enumerate(items):
        array = np.asarray(item.features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != n_mels:
            raise ValueError(
                f"特徴量の形が揃っていない: {tuple(array.shape)}（n_mels={n_mels} が必要）"
            )
        features[index, : array.shape[0]] = array
    return Batch(
        features=torch.from_numpy(features),
        lengths=torch.tensor(lengths, dtype=torch.long),
        moras=torch.tensor([float(item.mora) for item in items], dtype=torch.float32),
        durations=torch.tensor(
            [float(item.duration_sec) for item in items], dtype=torch.float32
        ),
        clip_ids=tuple(item.clip_id for item in items),
    )


# --------------------------------------------------------------------------------------
# 経路1: 事前計算済み特徴量（拡張なし）


class FeatureClipDataset(Dataset):
    """``data/processed/features/<split>`` の事前計算済み特徴量を読む。

    形式は docs/decisions/004-feature-storage.md のとおり、1シャード＝1000クリップ分を
    連結した ``(フレーム数, 80)`` の float16 ``.npy`` と、``offset`` / ``n_frames`` を
    持つ索引 json である。memmap で開き、クリップの行スライスだけを float32 にする。

    memmap のハンドルはプロセスごとに遅延して開く（``DataLoader`` のワーカーに
    ファイルハンドルを引き継がせないため）。
    """

    def __init__(
        self,
        features_dir: str | Path,
        *,
        client_ids: Sequence[str] | None = None,
        normalizer: Normalizer | None = None,
        limit: int | None = None,
        max_frames: int | None = None,
        min_frames: int = 1,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.normalizer = normalizer
        manifest = json.loads(
            (self.features_dir / "manifest.json").read_text(encoding="utf-8")
        )
        allowed = set(client_ids) if client_ids is not None else None

        entries: list[dict[str, Any]] = []
        for shard in manifest["shards"]:
            index = json.loads(
                (self.features_dir / shard["index"]).read_text(encoding="utf-8")
            )
            for clip in index["clips"]:
                if allowed is not None and clip["client_id"] not in allowed:
                    continue
                n_frames = int(clip["n_frames"])
                if n_frames < min_frames:
                    continue
                if max_frames is not None and n_frames > max_frames:
                    continue
                entries.append(
                    {
                        "array": shard["array"],
                        "offset": int(clip["offset"]),
                        "n_frames": n_frames,
                        "mora": float(clip["mora"]),
                        "duration_sec": float(clip["duration_sec"]),
                        "clip_id": clip["clip_id"],
                    }
                )
                if limit is not None and len(entries) >= limit:
                    break
            if limit is not None and len(entries) >= limit:
                break
        if not entries:
            raise ValueError(
                f"読み出せるクリップが無い: {self.features_dir}"
                "（分割の client_id と特徴量の突き合わせを確認すること）"
            )
        self.entries = entries
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def frame_counts(self) -> list[int]:
        """各件のフレーム数（長さでまとめるバッチ分けに使う）。"""
        return [int(entry["n_frames"]) for entry in self.entries]

    def _array(self, name: str) -> np.ndarray:
        array = self._arrays.get(name)
        if array is None:
            array = np.load(self.features_dir / name, mmap_mode="r")
            self._arrays[name] = array
        return array

    def __getitem__(self, index: int) -> ClipItem:
        entry = self.entries[index]
        array = self._array(entry["array"])
        offset = entry["offset"]
        feature = np.asarray(
            array[offset : offset + entry["n_frames"]], dtype=np.float32
        )
        if self.normalizer is not None:
            feature = self.normalizer(feature)
        return ClipItem(
            features=feature,
            mora=entry["mora"],
            duration_sec=entry["duration_sec"],
            clip_id=entry["clip_id"],
        )

    def __getstate__(self) -> dict[str, Any]:
        # memmap をワーカープロセスへ持ち越さない。
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state


# --------------------------------------------------------------------------------------
# 経路2: 波形から都度計算（拡張あり）


def select_clip_records(
    clips_jsonl: str | Path,
    split_path: str | Path,
    *,
    limit: int | None = None,
    min_duration_sec: float = 0.0,
    max_duration_sec: float | None = None,
) -> list[ClipRecord]:
    """clips.jsonl から、指定した分割に属するクリップを取り出す。

    ``split_path`` に ``configs/splits/test.json`` を渡すと
    ``spkrate.data.splits.load_split`` が拒否する（docs/PLAN.md 禁止事項）。
    """
    client_ids = set(load_split(split_path))
    records: list[ClipRecord] = []
    for record in load_clip_records(clips_jsonl):
        if record.client_id not in client_ids:
            continue
        if record.duration_sec < min_duration_sec:
            continue
        if max_duration_sec is not None and record.duration_sec > max_duration_sec:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


class WaveformClipDataset(Dataset):
    """波形を読んで拡張し、その場で対数メルを計算する（拡張ありの学習用）。

    docs/decisions/006-augmentation.md の経路そのままである。

        mp3を16kHzで読む → augment_waveform → log_mel_spectrogram → 正規化
        → augment_feature（周波数方向のマスクのみ。時間方向のマスクは禁止）

    正解のモーラ数は**どの拡張でも不変**である（``AugmentResult.mora_count``）。
    時間伸縮で変わるのはクリップ長だけなので、``duration_sec`` は拡張後の波形長から
    取り直す。毎秒モーラ数はモーラ数 ÷ その長さになり、自動的に整合する。

    乱数はエポックと件の番号から作る（``seed``, ``epoch``, ``index`` の3つ組）。
    ``set_epoch`` をエポックごとに呼ぶと、同じ件でもエポックごとに違う拡張が掛かり、
    かつ実行を再現できる。
    """

    def __init__(
        self,
        records: Sequence[ClipRecord],
        audio_root: str | Path,
        *,
        normalizer: Normalizer | None = None,
        augment: AugmentConfig | None = None,
        noise_source: NoiseSource | None = None,
        seed: int = 0,
        mel_config: Any = MEL_DEFAULTS,
    ) -> None:
        if not records:
            raise ValueError("クリップが1件も無い")
        self.records = list(records)
        self.audio_root = Path(audio_root)
        self.normalizer = normalizer
        self.augment = augment
        self.noise_source = noise_source
        self.seed = int(seed)
        self.mel_config = mel_config
        self.epoch = 0
        self._transform: LogMelSpectrogram | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def frame_counts(self) -> list[int]:
        """各件のおおよそのフレーム数（clips.jsonl の長さから見積る）。

        時間伸縮を掛けると実際の長さは変わるが、これはバッチの組み方を決めるだけの
        値なので見積りで足りる。
        """
        return [
            max(1, int(record.duration_sec * FRAMES_PER_SECOND)) for record in self.records
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _mel(self) -> LogMelSpectrogram:
        if self._transform is None:
            self._transform = LogMelSpectrogram(self.mel_config)
        return self._transform

    def __getitem__(self, index: int) -> ClipItem:
        from spkrate.eval.audio import load_audio

        record = self.records[index]
        samples, sample_rate = load_audio(self.audio_root / record.audio_path)
        rng = np.random.default_rng((self.seed, self.epoch, index))
        if self.augment is not None:
            result = augment_waveform(
                samples, rng, config=self.augment, noise_source=self.noise_source
            )
            samples = result.samples
        duration_sec = float(len(samples)) / float(sample_rate)
        feature = self._mel()(samples, sample_rate)
        if self.normalizer is not None:
            feature = self.normalizer(feature)
        if self.augment is not None:
            # 周波数方向のマスクは正規化後の特徴量に掛ける（006の2節）。
            feature = augment_feature(feature, rng, config=self.augment)
        return ClipItem(
            features=np.asarray(feature, dtype=np.float32),
            mora=float(record.mora),
            duration_sec=duration_sec,
            clip_id=record.clip_id,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_transform"] = None  # torch のモジュールをワーカーへ持ち越さない
        return state


# --------------------------------------------------------------------------------------
# 長さでまとめるバッチ分け


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """長さの近いものを同じバッチに集めるバッチ分け（モジュール docstring を参照）。

    手順は「無作為に並べる → ``batch_size × pool_batches`` 件の塊ごとに長さで整列 →
    ``batch_size`` ずつ切る → バッチの順序を無作為化」である。``shuffle=False`` の
    ときは長さ順に整列して切るだけになる（検証のように順序が結果に影響しない場合に使う）。

    どの件もちょうど1回だけ現れる（``drop_last=True`` のときは端数のバッチを捨てる）。
    """

    def __init__(
        self,
        frame_counts: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        pool_batches: int = 20,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size は1以上: {batch_size}")
        if pool_batches < 1:
            raise ValueError(f"pool_batches は1以上: {pool_batches}")
        self.frame_counts = [int(value) for value in frame_counts]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.pool_batches = int(pool_batches)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """エポックごとに並びを変えるための種。学習ループから呼ぶ。"""
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        order = list(range(len(self.frame_counts)))
        if self.shuffle:
            np.random.default_rng((self.seed, self.epoch)).shuffle(order)
        pool_size = self.batch_size * self.pool_batches
        batches: list[list[int]] = []
        for start in range(0, len(order), pool_size):
            pool = order[start : start + pool_size]
            pool.sort(key=lambda index: self.frame_counts[index])
            for offset in range(0, len(pool), self.batch_size):
                batches.append(pool[offset : offset + self.batch_size])
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        if self.shuffle:
            rng = np.random.default_rng((self.seed, self.epoch, 1))
            order_of_batches = list(range(len(batches)))
            rng.shuffle(order_of_batches)
            batches = [batches[index] for index in order_of_batches]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        total = len(self.frame_counts)
        if self.drop_last:
            # 塊ごとに端数が出るため、単純な割り算では数えられない。
            return len(self._batches())
        return (total + self.batch_size - 1) // self.batch_size
