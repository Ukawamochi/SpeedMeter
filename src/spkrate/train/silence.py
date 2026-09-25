"""無音・雑音のみのサンプル（正解モーラ数0）を学習データに加える。

docs/spec.md「学習データ」節の実装である。

## 目的

学習データ（Common Voice のクリップ）はすべて発話を含み、正解モーラ数が0のクリップが
無い。そのため無音付近でのモデルの出力が学習で制約されておらず、仕様の「無音は0モーラ」を
満たす保証が無い（診断 D2 がこの状態を測る）。そこで正解0のサンプルを学習データに
加える。検証（dev）には加えない（評価の母集団を変えないため）。

## 3つの種類

| 名前 | 中身 |
| --- | --- |
| ``digital_silence`` | 全標本0の波形（デジタル無音） |
| ``musan_noise`` | MUSAN noise（発話を含まない。``<musan_root>/noise``）を元の振幅のまま切り出したもの。診断 D2 の雑音窓と同じ作り方（``MusanNoiseSource.sample``） |
| ``quiet_noise`` | MUSAN noise の切り出しを、実効値が ``quiet_noise_dbfs_range`` の一様乱数（dBFS、満振幅1.0に対する実効値）になるよう縮めたもの |

いずれも正解モーラ数は0.0である（``ClipItem.mora``）。

## 既定値とその理由（``SilenceSettings``）

- ``ratio=0.03``: 学習データのクリップ数の3%を**追加**する（学習データ N 件に対し
  ``round(0.03 N)`` 件を足し、合計は約 1.03 N 件）。指示された既定値。少量にとどめるのは、
  正解0のサンプルが多すぎると損失の平均が「全体に小さく出す」方向へ引かれ、発話クリップの
  推定（exp001 で見られた低出力の傾向）を悪化させうるためである。
- ``mix`` は3種を等分（各1/3）: どの種類が D2 の改善に効くかの事前知識が無く、D2 が
  デジタル無音と MUSAN noise の両方を測るため、偏らせる根拠が無い。
- 長さ（``duration_range_sec=None``）: 学習クリップの長さの分布から復元抽出する。
  方式A（クリップ全体入力）では出力がフレームの総和なので、無音サンプルだけ長さの分布が
  違うと「長さ」そのものが正解0の手掛かりになりうる。発話クリップと同じ分布にすれば
  長さからは区別できない。長さは件ごとに構築時に1回だけ決め、エポック間で固定する
  （長さでまとめるバッチ分けに正確なフレーム数を渡すため）。中身（雑音の選択・位置・音量）は
  エポックごとに変える。
- ``quiet_noise_dbfs_range=(-90, -60)``: 白色雑音で実測すると、実効値 −90dBFS の対数メルは
  平均 −13.5 でデジタル無音（log(1e-6) = −13.8）とほぼ同じ、−60dBFS で −8.4 である。
  学習データ全体の対数メルの平均は −7.3（configs/normalization.yaml の global）で、
  この範囲は「デジタル無音」と「学習データの典型的な水準」の間の空白を埋める。
  −60dBFS より大きい音量は ``musan_noise``（元の振幅）が受け持つ。
- ``musan_root="data/musan"``: data/DATASETS.md の配置先。

## 拡張を掛けない（判断と理由）

拡張（``augment.enabled``）が有効でも、無音・雑音サンプルには波形の拡張も周波数マスクも
掛けない。

1. 時間伸縮・雑音重畳・残響・帯域制限はモーラ0を変えないが、元が無音や雑音なので
   「雑音のみの入力」の多様性はほとんど増えない（デジタル無音への雑音重畳は SN比が
   定義できず何もしない、``_add_noise``）。
2. 音量変化（−12〜+6dB）は ``quiet_noise`` の音量を設定した範囲の外へ動かし、
   ログに記録した「極小音量の範囲」と実際の入力が食い違う。
3. 拡張の実適用回数のログ（エポックごと）は発話クリップに対する率として読むものであり、
   無音サンプルを混ぜるとその率の意味が変わる。
4. 拡張の有無とは独立に同じ無音サンプルが入るので、拡張の有無の比較に無音サンプルの
   違いが混ざらない。

## 特徴量と正規化

特徴量は ``LogMelSpectrogram``（docs/spec.md の特徴量、src/spkrate/features/melspec.py）、
正規化は学習データと同じ ``Normalizer``（configs/normalization.yaml）で、その場で計算する。
学習データが事前計算特徴量（``data.source=features``）のときは、対数メルを一度 float16 に
丸めてから float32 に戻す（``match_feature_storage=True``）。事前計算特徴量は float16 で
保存されている（scripts/precompute_features.py）ので、丸めの有無で無音サンプルだけが
区別されないようにするためである。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from spkrate.data.augment import SAMPLE_RATE, MusanNoiseSource, NoiseSource
from spkrate.data.musan_split import MusanSplitError
from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram, num_frames

__all__ = [
    "DEFAULT_MIX",
    "SILENCE_CLIP_PREFIX",
    "SILENCE_KINDS",
    "SilenceDataset",
    "SilenceSettings",
    "SilenceSetupError",
    "TrainWithSilence",
    "allocate_counts",
    "build_silence_dataset",
    "check_silence_setup",
    "describe_silence_counts",
    "is_silence_clip",
]

#: 種類の名前とログに書く日本語名。
SILENCE_KINDS: tuple[tuple[str, str], ...] = (
    ("digital_silence", "デジタル無音"),
    ("musan_noise", "MUSAN雑音のみ"),
    ("quiet_noise", "極小音量の雑音"),
)
_KIND_NAMES = tuple(name for name, _ in SILENCE_KINDS)
_KIND_LABELS = dict(SILENCE_KINDS)

#: 3種を等分する（理由はモジュール docstring）。
DEFAULT_MIX: dict[str, float] = {name: 1.0 for name in _KIND_NAMES}

#: 無音サンプルの clip_id の接頭辞。学習ループで発話クリップと区別するのに使う。
SILENCE_CLIP_PREFIX = "silence/"

# ``quiet_noise`` で雑音の切り出しがほぼ無音だったとみなす実効値の閾値（白色雑音で代える）。
_EPSILON = 1e-12


def is_silence_clip(clip_id: str) -> bool:
    """無音サンプルの clip_id か。"""
    return str(clip_id).startswith(SILENCE_CLIP_PREFIX)


class SilenceSetupError(ValueError):
    """無音サンプルの設定の誤り（学習開始前に止める）。"""


# --------------------------------------------------------------------------------------
# 設定


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SilenceSetupError(f"silence_samples.{name} は2要素の [下限, 上限]: {value}")
    low, high = float(value[0]), float(value[1])
    if not (math.isfinite(low) and math.isfinite(high)):
        raise SilenceSetupError(f"silence_samples.{name} は有限値: {value}")
    if low > high:
        raise SilenceSetupError(f"silence_samples.{name} の下限が上限を超えている: {value}")
    return low, high


@dataclass(frozen=True)
class SilenceSettings:
    """無音・雑音のみのサンプルの設定（yaml の ``silence_samples`` 節）。

    既定値の理由はモジュール docstring を参照。

    Attributes:
        enabled: 追加するか。拡張（``augment.enabled``）とは独立である。
        ratio: 学習データのクリップ数に対する追加件数の割合（0より大きく1以下）。
        mix: 3種の内訳の重み。書いた場合は既定を置き換える（書かなかった種類は0）。
        musan_root: MUSAN の配置先。``musan_noise`` か ``quiet_noise`` の重みが正なら必要。
        musan_noise_split: MUSAN noise の学習用・評価用の分割ファイル（通常
            ``configs/splits/musan_noise.json``）。学習用（train）だけを使う。``musan_noise`` か
            ``quiet_noise`` の重みが正なら必要で、無ければ学習開始前に止める。
        quiet_noise_dbfs_range: ``quiet_noise`` の実効値の範囲（dBFS）。
        duration_range_sec: 長さの範囲（秒、一様）。``None`` なら学習クリップの長さの
            分布から復元抽出する。
    """

    enabled: bool = False
    ratio: float = 0.03
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    musan_root: str = "data/musan"
    musan_noise_split: str | None = None
    quiet_noise_dbfs_range: tuple[float, float] = (-90.0, -60.0)
    duration_range_sec: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "SilenceSettings":
        values = dict(mapping or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"SilenceSettings に未知の設定項目: {unknown}")
        if "mix" in values:
            mix = values["mix"]
            if not isinstance(mix, Mapping):
                raise SilenceSetupError(f"silence_samples.mix は辞書: {mix}")
            bad = sorted(set(mix) - set(_KIND_NAMES))
            if bad:
                raise SilenceSetupError(
                    f"silence_samples.mix に未知の種類: {bad}（使えるのは {_KIND_NAMES}）"
                )
            values["mix"] = {name: float(mix.get(name, 0.0)) for name in _KIND_NAMES}
        if "quiet_noise_dbfs_range" in values:
            values["quiet_noise_dbfs_range"] = _pair(
                values["quiet_noise_dbfs_range"], "quiet_noise_dbfs_range"
            )
        if values.get("duration_range_sec") is not None:
            values["duration_range_sec"] = _pair(
                values["duration_range_sec"], "duration_range_sec"
            )
        return cls(**values)

    @property
    def needs_musan(self) -> bool:
        return self.mix.get("musan_noise", 0.0) > 0.0 or self.mix.get("quiet_noise", 0.0) > 0.0

    def validate(self) -> None:
        """値の範囲を検査する。誤りなら ``SilenceSetupError``。"""
        if not (math.isfinite(self.ratio) and 0.0 < self.ratio <= 1.0):
            raise SilenceSetupError(
                f"silence_samples.ratio は 0 より大きく 1 以下: {self.ratio}"
                "（追加しないなら enabled: false にする）"
            )
        weights = [float(self.mix.get(name, 0.0)) for name in _KIND_NAMES]
        if any(not math.isfinite(w) or w < 0.0 for w in weights):
            raise SilenceSetupError(f"silence_samples.mix の重みは0以上の有限値: {self.mix}")
        if sum(weights) <= 0.0:
            raise SilenceSetupError(f"silence_samples.mix の重みがすべて0: {self.mix}")
        low, high = _pair(self.quiet_noise_dbfs_range, "quiet_noise_dbfs_range")
        if high > 0.0:
            raise SilenceSetupError(
                f"silence_samples.quiet_noise_dbfs_range は0dBFS以下: {self.quiet_noise_dbfs_range}"
            )
        if self.duration_range_sec is not None:
            low, _ = _pair(self.duration_range_sec, "duration_range_sec")
            if low <= 0.0:
                raise SilenceSetupError(
                    f"silence_samples.duration_range_sec の下限は正: {self.duration_range_sec}"
                )


def allocate_counts(num_train: int, ratio: float, mix: Mapping[str, float]) -> dict[str, int]:
    """学習データ ``num_train`` 件に対する追加件数を3種に割り振る。

    合計は ``round(ratio × num_train)``（0.5は切り上げ）。内訳は重みに比例させ、
    端数は最大剰余法で配る（合計が必ず一致する）。
    """
    total = int(math.floor(float(ratio) * int(num_train) + 0.5))
    weights = np.array([max(0.0, float(mix.get(name, 0.0))) for name in _KIND_NAMES])
    if total <= 0 or weights.sum() <= 0.0:
        return {name: 0 for name in _KIND_NAMES}
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return {name: int(count) for name, count in zip(_KIND_NAMES, counts, strict=True)}


# --------------------------------------------------------------------------------------
# データセット


class SilenceDataset(Dataset):
    """正解モーラ数0のサンプルを都度生成する（モジュール docstring を参照）。

    ``kinds[i]`` と ``durations_sec[i]`` は構築時に固定し、中身の乱数は
    ``(seed, epoch, index, 1)`` から作る（``WaveformClipDataset`` の
    ``(seed, epoch, index)`` とは別の系列）。
    """

    def __init__(
        self,
        kinds: Sequence[str],
        durations_sec: Sequence[float],
        *,
        normalizer: Any = None,
        noise_source: NoiseSource | None = None,
        quiet_noise_dbfs_range: tuple[float, float] = (-90.0, -60.0),
        seed: int = 0,
        match_feature_storage: bool = False,
        mel_config: Any = MEL_DEFAULTS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if len(kinds) != len(durations_sec):
            raise ValueError("kinds と durations_sec の長さが違う")
        for kind in kinds:
            if kind not in _KIND_NAMES:
                raise ValueError(f"未知の種類: {kind}")
        if noise_source is None and any(kind != "digital_silence" for kind in kinds):
            raise ValueError("雑音の種類には noise_source が必要")
        self.kinds = list(kinds)
        self.num_samples = [max(1, int(round(float(d) * sample_rate))) for d in durations_sec]
        self.normalizer = normalizer
        self.noise_source = noise_source
        self.quiet_noise_dbfs_range = tuple(float(v) for v in quiet_noise_dbfs_range)
        self.seed = int(seed)
        self.match_feature_storage = bool(match_feature_storage)
        self.mel_config = mel_config
        self.sample_rate = int(sample_rate)
        self.epoch = 0
        self._transform: LogMelSpectrogram | None = None

    def __len__(self) -> int:
        return len(self.kinds)

    @property
    def frame_counts(self) -> list[int]:
        return [num_frames(n, self.mel_config) for n in self.num_samples]

    @property
    def counts(self) -> dict[str, int]:
        return {name: self.kinds.count(name) for name in _KIND_NAMES}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _mel(self) -> LogMelSpectrogram:
        if self._transform is None:
            self._transform = LogMelSpectrogram(self.mel_config)
        return self._transform

    def waveform(self, index: int) -> np.ndarray:
        """``index`` 件目の波形（16kHz float32）。"""
        kind = self.kinds[index]
        size = self.num_samples[index]
        if kind == "digital_silence":
            return np.zeros(size, dtype=np.float32)
        rng = np.random.default_rng((self.seed, self.epoch, index, 1))
        assert self.noise_source is not None
        noise = np.asarray(self.noise_source.sample(size, rng), dtype=np.float32)
        if kind == "musan_noise":
            return noise  # 元の振幅のまま（診断 D2 と同じ）
        low, high = self.quiet_noise_dbfs_range
        level_db = float(rng.uniform(low, high)) if high > low else low
        rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float32), dtype=np.float32)))
        if rms <= _EPSILON:
            # 切り出しが無音だった場合は白色雑音で代える（音量を指定どおりにするため）。
            noise = rng.standard_normal(size).astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(noise))))
        target = 10.0 ** (level_db / 20.0)
        return np.ascontiguousarray(noise * np.float32(target / rms), dtype=np.float32)

    def __getitem__(self, index: int):  # noqa: ANN204
        from spkrate.train.data import ClipItem

        samples = self.waveform(index)
        feature = self._mel()(samples, self.sample_rate)
        if self.match_feature_storage:
            feature = feature.astype(np.float16).astype(np.float32)
        if self.normalizer is not None:
            feature = self.normalizer(feature)
        return ClipItem(
            features=np.asarray(feature, dtype=np.float32),
            mora=0.0,
            duration_sec=float(samples.size) / float(self.sample_rate),
            clip_id=f"{SILENCE_CLIP_PREFIX}{self.kinds[index]}/{index:06d}",
            augment_applied=(),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_transform"] = None  # torch のモジュールをワーカーへ持ち越さない
        return state


class TrainWithSilence(Dataset):
    """学習データの後ろに無音サンプルを連結したもの。

    ``frame_counts`` と ``set_epoch`` を両方へ渡すので、長さでまとめるバッチ分けと
    エポックごとの乱数がそのまま働く。
    """

    def __init__(self, base: Dataset, silence: SilenceDataset) -> None:
        self.base = base
        self.silence = silence
        self._base_len = len(base)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self._base_len + len(self.silence)

    @property
    def frame_counts(self) -> list[int] | None:
        counts = getattr(self.base, "frame_counts", None)
        if counts is None:
            return None
        return [int(v) for v in counts] + self.silence.frame_counts

    @property
    def silence_counts(self) -> dict[str, int]:
        return self.silence.counts

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)  # type: ignore[attr-defined]
        self.silence.set_epoch(epoch)

    def __getitem__(self, index: int):  # noqa: ANN204
        if index < 0:
            index += len(self)
        if index < self._base_len:
            return self.base[index]
        return self.silence[index - self._base_len]


def _base_durations(base: Dataset) -> list[float]:
    durations = getattr(base, "durations_sec", None)
    if durations is None:
        counts = getattr(base, "frame_counts", None)
        if counts is None:
            raise ValueError("学習データの長さが分からない（durations_sec も frame_counts も無い）")
        durations = [int(c) / 100.0 for c in counts]
    return [float(d) for d in durations]


def build_silence_dataset(
    settings: SilenceSettings,
    base: Dataset,
    *,
    normalizer: Any,
    seed: int,
    match_feature_storage: bool,
    noise_source: NoiseSource | None = None,
) -> SilenceDataset:
    """設定と学習データから無音サンプルのデータセットを作る。

    ``noise_source`` を省略し、雑音の種類が必要なら ``settings.musan_root`` の
    MUSAN noise のうち ``settings.musan_noise_split`` の学習用（train）だけを使う。
    """
    settings.validate()
    counts = allocate_counts(len(base), settings.ratio, settings.mix)  # type: ignore[arg-type]
    kinds = [name for name in _KIND_NAMES for _ in range(counts[name])]
    rng = np.random.default_rng((int(seed), 0x5113))
    if settings.duration_range_sec is None:
        pool = np.asarray(_base_durations(base), dtype=np.float32)
        durations = rng.choice(pool, size=len(kinds), replace=True).tolist() if kinds else []
    else:
        low, high = settings.duration_range_sec
        durations = rng.uniform(low, high, size=len(kinds)).tolist()
    needs_noise = counts["musan_noise"] + counts["quiet_noise"] > 0
    if needs_noise and noise_source is None:
        noise_source = MusanNoiseSource.from_split(
            settings.musan_root,
            settings.musan_noise_split,
            "train",
            setting="silence_samples.musan_noise_split",
        )
    return SilenceDataset(
        kinds,
        durations,
        normalizer=normalizer,
        noise_source=noise_source if needs_noise else None,
        quiet_noise_dbfs_range=settings.quiet_noise_dbfs_range,
        seed=seed,
        match_feature_storage=match_feature_storage,
    )


# --------------------------------------------------------------------------------------
# 学習開始前の検査とログ


def check_silence_setup(settings: SilenceSettings) -> list[str]:
    """学習開始前に設定を検査し、log.txt に書く行を返す。

    MUSAN が必要なのに見つからない等は ``SilenceSetupError`` で止める。
    """
    if not settings.enabled:
        return ["無音サンプル=なし"]
    settings.validate()
    total = sum(settings.mix.values())
    mix_text = " ".join(
        f"{_KIND_LABELS[name]}={settings.mix.get(name, 0.0) / total:.3f}" for name in _KIND_NAMES
    )
    musan_text = "（使わない）"
    if settings.needs_musan:
        root = Path(settings.musan_root)
        if not root.is_dir():
            raise SilenceSetupError(
                f"silence_samples.musan_root={settings.musan_root} がディレクトリとして存在しない。"
                "配置先は data/DATASETS.md を参照する（雑音を使わないなら mix で"
                " musan_noise と quiet_noise を0にする）"
            )
        try:
            source = MusanNoiseSource.from_split(
                root,
                settings.musan_noise_split,
                "train",
                setting="silence_samples.musan_noise_split",
            )
        except MusanSplitError as error:
            raise SilenceSetupError(str(error)) from error
        try:
            paths = source.paths
        except FileNotFoundError as error:
            raise SilenceSetupError(
                f"silence_samples.musan_root={settings.musan_root} に MUSAN noise の wav が無い（{error}）"
            ) from error
        musan_text = (
            f"{root / 'noise'} の学習用（{settings.musan_noise_split} の train、{len(paths)}ファイル）"
        )
    low, high = settings.quiet_noise_dbfs_range
    duration_text = (
        "学習クリップの長さ分布から復元抽出"
        if settings.duration_range_sec is None
        else f"一様 {settings.duration_range_sec[0]:g}〜{settings.duration_range_sec[1]:g}秒"
    )
    return [
        f"無音サンプル=あり 割合={settings.ratio:g}（学習データの件数に対する追加件数）"
        f" 内訳比: {mix_text}",
        f"無音サンプル 極小音量={low:g}〜{high:g}dBFS 長さ={duration_text} 雑音源={musan_text}"
        " 拡張=掛けない 検証=追加しない",
    ]


def describe_silence_counts(num_train: int, counts: Mapping[str, int]) -> str:
    """実際に追加した件数（log.txt に書く1行）。"""
    parts = " ".join(f"{_KIND_LABELS[name]}={int(counts.get(name, 0))}" for name in _KIND_NAMES)
    total = sum(int(v) for v in counts.values())
    return f"無音サンプル件数: 学習{num_train}件に計{total}件を追加（{parts}）"
