"""データ拡張（docs/PLAN.md 第4段階 4-2、docs/spec.md「データ拡張」）。

docs/spec.md が許す拡張は次の5種である。

    時間伸縮0.7〜1.5倍（音程保持、モーラ数は不変）、雑音重畳SNR0〜20dB、
    合成残響、音量変化、帯域制限。**時間方向のマスクは正解と矛盾するため禁止**

この規則から、本モジュールの設計は次のようになる。

## 拡張は波形に適用する

時間伸縮・雑音重畳・残響・帯域制限はいずれも波形の演算であり、対数メルスペクトログラム
からは復元できない。したがって拡張は16kHzモノラルの波形に適用し、メルは拡張後に計算し直す
（実測コストは ``docs/decisions/006-augmentation.md``）。事前計算済みの
``data/processed/features/``（float16、18.4GiB）は**拡張しない経路**（検証・評価、
および拡張なしの対照実験）に使う。唯一の例外は周波数方向のマスクで、これは特徴量に
直接かける（``frequency_mask``）。

## ラベルとの関係

モーラ数は「その区間で何モーラ発話されたか」であって、音量・雑音・残響・帯域で**変わらない**。
時間伸縮でも**発話されたモーラの数は変わらない**。変わるのは音声長であり、したがって
毎秒モーラ数（表示値）だけが伸縮率に応じて変わる。

    伸縮率 ``stretch``（= 長さの倍率）に対して
        音声長     : duration * stretch
        モーラ数   : 不変
        毎秒モーラ数: mora_per_second / stretch

``stretch = 0.7`` なら音声は0.7倍の長さに縮み（＝速口になり）、毎秒モーラ数は
``1/0.7 = 1.43`` 倍になる。この関係は ``mora_count_after_time_stretch`` と
``mora_per_second_after_time_stretch`` として関数に書き、
``tests/test_augment.py`` で検証する。

## 時間方向のマスクを実装しない理由

ある時間区間を潰すと、その区間で発話されたモーラが聞こえなくなるのに正解のモーラ数は
そのままになり、入力と正解が矛盾する。周波数方向のマスクは発話区間を消さないので
モーラ数に影響しない。本モジュールは時間方向のマスクを**提供しない**。

## 使い方

>>> rng = np.random.default_rng(0)
>>> noise = MusanNoiseSource("data/musan")            # 読み取りのみ
>>> result = augment_waveform(samples, rng=rng, noise_source=noise)
>>> mora_per_second = result.mora_per_second(original_mora_per_second)
>>> mora_count = result.mora_count(original_mora_count)   # 不変
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

__all__ = [
    "SAMPLE_RATE",
    "STFT_HOP_LENGTH",
    "STFT_N_FFT",
    "AugmentConfig",
    "AugmentResult",
    "ArrayNoiseSource",
    "MusanNoiseSource",
    "NoiseSource",
    "add_noise",
    "augment_feature",
    "augment_waveform",
    "band_limit",
    "change_volume",
    "fit_noise",
    "frequency_mask",
    "generate_rir",
    "apply_reverb",
    "mora_count_after_time_stretch",
    "mora_per_second_after_time_stretch",
    "time_stretch",
]

# docs/spec.md「入力: 16kHzモノラル音声」。
SAMPLE_RATE = 16000

# 時間伸縮（位相ボコーダ）の分析窓。16kHz で 64ミリ秒・移動量16ミリ秒。
# librosa の既定（2048 = 128ミリ秒）は音節の立ち上がりを鈍らせるため短くしてある。
# これは伸縮の内部パラメータであり、docs/spec.md の特徴量（n_fft=400、hop_length=160）
# とは無関係である。
STFT_N_FFT = 1024
STFT_HOP_LENGTH = 256

_EPSILON = 1e-12


def _as_waveform(samples: np.ndarray) -> np.ndarray:
    """1次元 float32 の波形に揃える（CLAUDE.md「float64は使わない」）。"""
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"1次元のモノラル波形を渡すこと: ndim={array.ndim}")
    if array.size == 0:
        raise ValueError("空の波形は扱えない")
    return np.ascontiguousarray(array)


def _uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    if low > high:
        raise ValueError(f"範囲の下限が上限を超えている: {bounds}")
    if low == high:
        return low
    return float(rng.uniform(low, high))


# --------------------------------------------------------------------------------------
# 時間伸縮


def time_stretch(
    samples: np.ndarray, stretch: float, *, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """音程を保ったまま音声長を ``stretch`` 倍にする。

    位相ボコーダ（``librosa.stft`` → ``librosa.phase_vocoder`` → ``librosa.istft``。
    STFTのフレームを読み替えながら位相を進める方式）を使う。標本化周波数を変えないので
    **音程は保たれる**。再標本化で長さを変える素朴な方法は音程が ``1/stretch`` 倍に
    ずれるため使わない（``tests/test_augment.py`` で両者を対比している）。

    Args:
        samples: 16kHzモノラルの1次元波形。
        stretch: 長さの倍率。``0.7`` なら0.7倍の長さ（速口）、``1.5`` なら1.5倍（ゆっくり）。

    Returns:
        長さがちょうど ``round(len(samples) * stretch)`` の float32 波形。

    Note:
        **モーラ数は変わらない**。変わるのは毎秒モーラ数であり、
        ``mora_per_second_after_time_stretch`` を使って ``stretch`` で割ること。
    """
    waveform = _as_waveform(samples)
    if not math.isfinite(stretch) or stretch <= 0.0:
        raise ValueError(f"伸縮率は正の有限値である必要がある: {stretch}")
    if stretch == 1.0:
        return waveform
    del sample_rate  # 位相ボコーダは標本化周波数に依存しない（音程保持のため変えない）。

    import warnings

    import librosa

    # librosa の rate は「速さ」であり、長さは 1/rate 倍になる。
    rate = 1.0 / float(stretch)
    target_length = int(round(waveform.size * float(stretch)))

    with warnings.catch_warnings():
        # 位相の計算で numba が出す "invalid value encountered in cast" の RuntimeWarning。
        # librosa 内部の事情で呼び出し側からは直せない。出力が有限であることは
        # tests/test_augment.py で確認している。
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
        spectrum = librosa.stft(waveform, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)
        # ``librosa.effects.time_stretch`` は内部で ``np.arange(0, n_frames, rate)`` を
        # 使うが、浮動小数の累積誤差で最後の要素が n_frames に届いてしまう長さがあり、
        # ParameterError になる（librosa 1.0）。読み出し位置を自分で作って弾いておく。
        frame_positions = np.arange(0.0, spectrum.shape[-1], rate)
        frame_positions = frame_positions[frame_positions < spectrum.shape[-1]]
        stretched_spectrum = librosa.phase_vocoder(spectrum, t_out=frame_positions)
        stretched = librosa.istft(
            stretched_spectrum,
            n_fft=STFT_N_FFT,
            hop_length=STFT_HOP_LENGTH,
            length=target_length,
            dtype=np.float32,
        )
    return np.ascontiguousarray(np.asarray(stretched, dtype=np.float32))


def mora_count_after_time_stretch(mora_count: float, stretch: float) -> float:
    """時間伸縮後のモーラ数。**伸縮率によらず不変**（docs/spec.md「モーラ数は不変」）。

    伸縮しても発話された音の並びは変わらないので、数えられるモーラの数も変わらない。
    引数 ``stretch`` は不変であることを呼び出し側に明示するためだけに受け取る。
    """
    if not math.isfinite(stretch) or stretch <= 0.0:
        raise ValueError(f"伸縮率は正の有限値である必要がある: {stretch}")
    return float(mora_count)


def mora_per_second_after_time_stretch(mora_per_second: float, stretch: float) -> float:
    """時間伸縮後の毎秒モーラ数 ``mora_per_second / stretch``。

    モーラ数は不変で音声長が ``stretch`` 倍になるので、その商である毎秒モーラ数は
    ``stretch`` で割った値になる。``stretch=0.7``（縮む＝速口）なら1.43倍に増える。
    """
    if not math.isfinite(stretch) or stretch <= 0.0:
        raise ValueError(f"伸縮率は正の有限値である必要がある: {stretch}")
    return float(mora_per_second) / float(stretch)


# --------------------------------------------------------------------------------------
# 雑音重畳


class NoiseSource(Protocol):
    """雑音波形を供給するもの。``MusanNoiseSource`` と試験用の実装が従う。"""

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """``num_samples`` 標本の16kHzモノラル雑音を1つ返す。"""


def fit_noise(
    noise: np.ndarray, num_samples: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """雑音を目的の長さに合わせる。長ければ無作為な位置を切り出し、短ければ繰り返す。"""
    array = _as_waveform(noise)
    if num_samples <= 0:
        raise ValueError(f"長さは正の値である必要がある: {num_samples}")
    if array.size < num_samples:
        repeats = -(-num_samples // array.size)  # 切り上げ
        array = np.tile(array, repeats)
    if array.size == num_samples:
        return np.ascontiguousarray(array)
    if rng is None:
        offset = 0
    else:
        offset = int(rng.integers(0, array.size - num_samples + 1))
    return np.ascontiguousarray(array[offset : offset + num_samples])


def add_noise(samples: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """雑音を指定のSN比（dB）で重畳する。長さと時刻は変えない。

    SN比は全区間の実効値（二乗平均平方根）で定義する。
    ``snr_db=0`` なら信号と雑音の電力が等しく、``snr_db=20`` なら雑音は20dB小さい。

    雑音は音声に**重なるだけ**で発話を消さないので、モーラ数は変わらない。
    """
    waveform = _as_waveform(samples)
    if not math.isfinite(snr_db):
        raise ValueError(f"SN比は有限値である必要がある: {snr_db}")
    fitted = fit_noise(noise, waveform.size)

    signal_power = float(np.mean(np.square(waveform, dtype=np.float32), dtype=np.float32))
    noise_power = float(np.mean(np.square(fitted, dtype=np.float32), dtype=np.float32))
    if signal_power <= _EPSILON:
        # 無音に雑音を足すとSN比が定義できない。そのまま返す。
        return waveform
    if noise_power <= _EPSILON:
        return waveform
    scale = math.sqrt(signal_power / (noise_power * (10.0 ** (float(snr_db) / 10.0))))
    return np.ascontiguousarray((waveform + np.float32(scale) * fitted).astype(np.float32))


class ArrayNoiseSource:
    """あらかじめ用意した波形の配列から雑音を選ぶ（試験と小規模実験用）。"""

    def __init__(self, waveforms: Sequence[np.ndarray]) -> None:
        if len(waveforms) == 0:
            raise ValueError("雑音が1つもない")
        self._waveforms = [_as_waveform(waveform) for waveform in waveforms]

    def __len__(self) -> int:
        return len(self._waveforms)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        index = int(rng.integers(0, len(self._waveforms)))
        return fit_noise(self._waveforms[index], num_samples, rng)


class MusanNoiseSource:
    """MUSAN の雑音を読み出す（``data/musan``。**読み取りのみ**）。

    既定では ``noise`` サブセットだけを使う。``speech`` を混ぜると別話者の発話が
    入り、その話者のモーラが正解のモーラ数に入っていないため**ラベルと矛盾する**。
    したがって ``speech`` は指定してもここで拒否する（``docs/decisions/006-augmentation.md``）。

    ファイル一覧は最初の使用時に作る。波形はファイル単位でメモリに保持し
    （既定で最大64ファイル）、同じファイルを何度も復号しないようにする。
    """

    #: 使ってよいサブセット。``speech`` は上の理由で含めない。
    ALLOWED_SUBSETS = ("noise", "music")
    DEFAULT_SUBSETS = ("noise",)

    def __init__(
        self,
        root: str | Path = "data/musan",
        *,
        subsets: Sequence[str] = DEFAULT_SUBSETS,
        sample_rate: int = SAMPLE_RATE,
        cache_size: int = 64,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        self.cache_size = int(cache_size)
        requested = tuple(str(subset) for subset in subsets)
        if not requested:
            raise ValueError("サブセットを1つ以上指定すること")
        for subset in requested:
            if subset == "speech":
                raise ValueError(
                    "MUSAN の speech は使わない。別話者の発話が混ざるとモーラ数のラベルと"
                    "矛盾するため（docs/decisions/006-augmentation.md）"
                )
            if subset not in self.ALLOWED_SUBSETS:
                raise ValueError(
                    f"未知のサブセット: {subset}（使えるのは {self.ALLOWED_SUBSETS}）"
                )
        self.subsets = requested
        self._paths: list[Path] | None = None
        self._cache: dict[Path, np.ndarray] = {}

    @property
    def paths(self) -> list[Path]:
        """対象の wav ファイル一覧（並びは固定）。"""
        if self._paths is None:
            paths: list[Path] = []
            for subset in self.subsets:
                directory = self.root / subset
                if not directory.is_dir():
                    raise FileNotFoundError(
                        f"MUSAN の {subset} が見つからない: {directory}。"
                        "配置先は data/DATASETS.md を参照する"
                    )
                paths.extend(sorted(directory.rglob("*.wav")))
            if not paths:
                raise FileNotFoundError(f"MUSAN の wav が1つも無い: {self.root}")
            self._paths = paths
        return self._paths

    def load(self, path: Path) -> np.ndarray:
        """1ファイルを16kHzモノラル float32 で読む（読み取りのみ）。"""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        from spkrate.eval.audio import load_audio

        waveform, _ = load_audio(path, target_sample_rate=self.sample_rate)
        waveform = _as_waveform(waveform)
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = waveform
        return waveform

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        paths = self.paths
        path = paths[int(rng.integers(0, len(paths)))]
        return fit_noise(self.load(path), num_samples, rng)


# --------------------------------------------------------------------------------------
# 残響


def generate_rir(
    rng: np.random.Generator,
    *,
    sample_rate: int = SAMPLE_RATE,
    room_x_range: Sequence[float] = (3.0, 10.0),
    room_y_range: Sequence[float] = (3.0, 10.0),
    room_z_range: Sequence[float] = (2.3, 4.0),
    rt60_range: Sequence[float] = (0.1, 0.7),
    margin: float = 0.5,
    max_order: int = 12,
) -> np.ndarray:
    """pyroomacoustics で直方体の部屋のインパルス応答を1つ作る。

    残響時間 ``rt60`` を無作為に選び、Sabine の式の逆算（``pra.inverse_sabine``）で
    壁の吸音率を決める。音源とマイクは壁から ``margin`` メートル以上離して無作為に置く。

    Returns:
        直流を含む1次元 float32 のインパルス応答。最大値の絶対値が1になるよう正規化する
        （残響を足しても全体の音量が大きく変わらないようにするため）。
    """
    import pyroomacoustics as pra

    room_dim = [
        _uniform(rng, room_x_range),
        _uniform(rng, room_y_range),
        _uniform(rng, room_z_range),
    ]
    rt60 = _uniform(rng, rt60_range)
    try:
        absorption, sabine_order = pra.inverse_sabine(rt60, room_dim)
    except ValueError:
        # その寸法では実現できない残響時間（吸音率が1を超える）。無響に近い応答を返す。
        return np.array([1.0], dtype=np.float32)

    room = pra.ShoeBox(
        room_dim,
        fs=int(sample_rate),
        materials=pra.Material(absorption),
        max_order=int(min(sabine_order, max_order)),
    )
    limit = [max(margin, 0.0) for _ in room_dim]
    source = [_uniform(rng, (limit[axis], room_dim[axis] - limit[axis])) for axis in range(3)]
    mic = [_uniform(rng, (limit[axis], room_dim[axis] - limit[axis])) for axis in range(3)]
    room.add_source(source)
    room.add_microphone(mic)
    room.compute_rir()

    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    peak = float(np.max(np.abs(rir))) if rir.size else 0.0
    if peak <= _EPSILON:
        return np.array([1.0], dtype=np.float32)
    return np.ascontiguousarray(rir / np.float32(peak))


def apply_reverb(samples: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """インパルス応答を畳み込む。**長さと発話の時刻は変えない**。

    素朴に畳み込むと (1) 直接音がインパルス応答の先頭からの遅れ分だけ後ろにずれ、
    (2) 末尾に残響の尾が伸びて音声長が変わる。どちらも「窓に含まれるモーラ数」という
    正解とずれる原因になるので、直接音（インパルス応答の絶対値が最大の位置）が
    元の時刻に来るよう先頭を切り、元の長さに合わせて末尾も切る。
    """
    from scipy import signal as scipy_signal

    waveform = _as_waveform(samples)
    impulse = _as_waveform(rir)
    convolved = scipy_signal.fftconvolve(waveform, impulse, mode="full")
    direct = int(np.argmax(np.abs(impulse)))
    aligned = convolved[direct : direct + waveform.size]
    if aligned.size < waveform.size:  # 実際には起こらないが念のため
        aligned = np.pad(aligned, (0, waveform.size - aligned.size))
    return np.ascontiguousarray(aligned.astype(np.float32))


# --------------------------------------------------------------------------------------
# 音量変化


def change_volume(
    samples: np.ndarray, gain_db: float, *, prevent_clipping: bool = True, peak_limit: float = 0.99
) -> np.ndarray:
    """利得（dB）をかけて音量を変える。長さも波形の形も変えないのでモーラ数は不変。

    ``prevent_clipping`` が真のとき、振幅が ``peak_limit`` を超えないよう利得を抑える
    （波形を切り詰めると歪みが入り、元の音声とは別の性質の劣化になるため）。
    """
    waveform = _as_waveform(samples)
    if not math.isfinite(gain_db):
        raise ValueError(f"利得は有限値である必要がある: {gain_db}")
    gain = 10.0 ** (float(gain_db) / 20.0)
    if prevent_clipping:
        peak = float(np.max(np.abs(waveform)))
        if peak > _EPSILON and peak * gain > peak_limit:
            gain = peak_limit / peak
    return np.ascontiguousarray((waveform * np.float32(gain)).astype(np.float32))


# --------------------------------------------------------------------------------------
# 帯域制限


def band_limit(
    samples: np.ndarray,
    *,
    low_hz: float = 0.0,
    high_hz: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """バターワース型の帯域通過で高域・低域を削る（電話帯域などの再現）。

    零位相の ``sosfiltfilt`` を使うので群遅延が無く、発話の時刻がずれない。
    帯域を削っても発話の区切りは残るのでモーラ数は不変である。

    Args:
        low_hz: これより下を削る。``0`` 以下なら低域は削らない。
        high_hz: これより上を削る。``None`` かナイキスト周波数以上なら高域は削らない。
    """
    from scipy import signal as scipy_signal

    waveform = _as_waveform(samples)
    nyquist = float(sample_rate) / 2.0
    low = float(low_hz)
    high = nyquist if high_hz is None else float(high_hz)
    use_high_pass = low > 0.0
    use_low_pass = high < nyquist
    if use_high_pass and use_low_pass and low >= high:
        raise ValueError(f"通過帯域が空: low_hz={low}, high_hz={high}")
    if not use_high_pass and not use_low_pass:
        return waveform

    if use_high_pass and use_low_pass:
        sos = scipy_signal.butter(
            order, [low / nyquist, high / nyquist], btype="bandpass", output="sos"
        )
    elif use_high_pass:
        sos = scipy_signal.butter(order, low / nyquist, btype="highpass", output="sos")
    else:
        sos = scipy_signal.butter(order, high / nyquist, btype="lowpass", output="sos")

    padlen = 3 * (sos.shape[0] * 2)
    if waveform.size <= padlen:  # 短すぎて零位相フィルタが使えない
        return waveform
    filtered = scipy_signal.sosfiltfilt(sos, waveform.astype(np.float64))
    return np.ascontiguousarray(filtered.astype(np.float32))


# --------------------------------------------------------------------------------------
# 周波数方向のマスク（特徴量に対する唯一の拡張）


def frequency_mask(
    feature: np.ndarray,
    rng: np.random.Generator,
    *,
    num_masks: int = 2,
    max_width: int = 12,
    mask_value: float = 0.0,
) -> np.ndarray:
    """対数メルスペクトログラムの**周波数方向**を帯状に潰す。

    docs/spec.md は時間方向のマスクを禁じている（潰した区間のモーラが聞こえないのに
    正解のモーラ数は変わらず、入力と正解が矛盾するため）。本関数は周波数方向のみを扱い、
    どのフレームも消さないのでモーラ数に影響しない。

    Args:
        feature: ``(フレーム数, メル次元数)``。時間方向が第0軸。
        mask_value: 埋める値。正規化後の特徴量に適用する想定なので既定は0（平均）。

    Returns:
        入力とは別の配列（入力は書き換えない）。
    """
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"(フレーム数, メル次元数) の2次元配列を渡すこと: ndim={array.ndim}")
    if num_masks < 0 or max_width < 0:
        raise ValueError(f"負の値は指定できない: num_masks={num_masks}, max_width={max_width}")
    masked = array.copy()
    n_mels = masked.shape[1]
    if n_mels == 0 or max_width == 0:
        return masked
    for _ in range(int(num_masks)):
        width = int(rng.integers(0, min(int(max_width), n_mels) + 1))
        if width == 0:
            continue
        start = int(rng.integers(0, n_mels - width + 1))
        masked[:, start : start + width] = np.float32(mask_value)
    return masked


# --------------------------------------------------------------------------------------
# 設定とパイプライン


@dataclass(frozen=True)
class AugmentConfig:
    """各拡張の適用確率とパラメータ範囲（既定値は docs/decisions/006-augmentation.md）。"""

    sample_rate: int = SAMPLE_RATE

    # 時間伸縮（docs/spec.md「0.7〜1.5倍」）
    time_stretch_prob: float = 0.5
    time_stretch_range: tuple[float, float] = (0.7, 1.5)

    # 残響（pyroomacoustics）
    reverb_prob: float = 0.3
    rt60_range: tuple[float, float] = (0.1, 0.7)
    room_x_range: tuple[float, float] = (3.0, 10.0)
    room_y_range: tuple[float, float] = (3.0, 10.0)
    room_z_range: tuple[float, float] = (2.3, 4.0)
    reverb_max_order: int = 12

    # 雑音重畳（docs/spec.md「SNR0〜20dB」）
    noise_prob: float = 0.5
    snr_db_range: tuple[float, float] = (0.0, 20.0)

    # 帯域制限
    band_limit_prob: float = 0.25
    low_hz_range: tuple[float, float] = (0.0, 300.0)
    high_hz_range: tuple[float, float] = (3400.0, 7800.0)
    band_limit_order: int = 4

    # 音量変化
    volume_prob: float = 0.5
    gain_db_range: tuple[float, float] = (-12.0, 6.0)

    # 周波数方向のマスク（特徴量に適用。時間方向のマスクは禁止）
    freq_mask_prob: float = 0.5
    freq_mask_num: int = 2
    freq_mask_max_width: int = 12
    freq_mask_value: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "AugmentConfig":
        """設定ファイル由来の辞書から作る。未知のキーは誤りとして弾く。"""
        if not mapping:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"未知の設定項目: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for key, value in mapping.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                values[key] = (float(value[0]), float(value[1]))
            else:
                values[key] = value
        return replace(cls(), **values)

    def disabled(self) -> "AugmentConfig":
        """すべての適用確率を0にした設定（対照実験用）。"""
        return replace(
            self,
            time_stretch_prob=0.0,
            reverb_prob=0.0,
            noise_prob=0.0,
            band_limit_prob=0.0,
            volume_prob=0.0,
            freq_mask_prob=0.0,
        )


@dataclass(frozen=True)
class AugmentResult:
    """拡張の結果と、ラベルをどう読み替えるか。

    ``stretch`` 以外の拡張はラベルに影響しない。``stretch`` が影響するのは
    毎秒モーラ数だけで、モーラ数そのものは常に不変である。
    """

    samples: np.ndarray
    stretch: float = 1.0
    applied: tuple[str, ...] = field(default_factory=tuple)
    params: dict[str, float] = field(default_factory=dict)

    def mora_count(self, mora_count: float) -> float:
        """拡張後のモーラ数。**どの拡張でも不変**。"""
        return mora_count_after_time_stretch(mora_count, self.stretch)

    def mora_per_second(self, mora_per_second: float) -> float:
        """拡張後の毎秒モーラ数。時間伸縮の分だけ ``stretch`` で割る。"""
        return mora_per_second_after_time_stretch(mora_per_second, self.stretch)

    def duration_sec(self, sample_rate: int = SAMPLE_RATE) -> float:
        return float(self.samples.size) / float(sample_rate)


def augment_waveform(
    samples: np.ndarray,
    rng: np.random.Generator,
    *,
    config: AugmentConfig = AugmentConfig(),
    noise_source: NoiseSource | None = None,
) -> AugmentResult:
    """波形に拡張を適用する。

    適用順は「時間伸縮 → 残響 → 雑音重畳 → 帯域制限 → 音量変化」。
    収録の経路（部屋で響いた音に周囲の雑音が混ざり、伝送路で帯域が制限され、
    最後に録音利得がかかる）に合わせてある。

    ``noise_source`` を渡さない場合、雑音重畳は行わない（MUSAN が無い環境でも動く）。

    Returns:
        ``AugmentResult``。ラベルの読み替えは ``mora_count`` と ``mora_per_second`` を使う。
    """
    waveform = _as_waveform(samples)
    applied: list[str] = []
    params: dict[str, float] = {}
    stretch = 1.0

    if rng.random() < config.time_stretch_prob:
        stretch = _uniform(rng, config.time_stretch_range)
        waveform = time_stretch(waveform, stretch, sample_rate=config.sample_rate)
        applied.append("time_stretch")
        params["stretch"] = stretch

    if rng.random() < config.reverb_prob:
        rir = generate_rir(
            rng,
            sample_rate=config.sample_rate,
            room_x_range=config.room_x_range,
            room_y_range=config.room_y_range,
            room_z_range=config.room_z_range,
            rt60_range=config.rt60_range,
            max_order=config.reverb_max_order,
        )
        waveform = apply_reverb(waveform, rir)
        applied.append("reverb")

    if noise_source is not None and rng.random() < config.noise_prob:
        snr_db = _uniform(rng, config.snr_db_range)
        noise = noise_source.sample(waveform.size, rng)
        waveform = add_noise(waveform, noise, snr_db)
        applied.append("noise")
        params["snr_db"] = snr_db

    if rng.random() < config.band_limit_prob:
        low_hz = _uniform(rng, config.low_hz_range)
        high_hz = _uniform(rng, config.high_hz_range)
        waveform = band_limit(
            waveform,
            low_hz=low_hz,
            high_hz=high_hz,
            sample_rate=config.sample_rate,
            order=config.band_limit_order,
        )
        applied.append("band_limit")
        params["low_hz"] = low_hz
        params["high_hz"] = high_hz

    if rng.random() < config.volume_prob:
        gain_db = _uniform(rng, config.gain_db_range)
        waveform = change_volume(waveform, gain_db)
        applied.append("volume")
        params["gain_db"] = gain_db

    return AugmentResult(
        samples=waveform, stretch=stretch, applied=tuple(applied), params=params
    )


def augment_feature(
    feature: np.ndarray,
    rng: np.random.Generator,
    *,
    config: AugmentConfig = AugmentConfig(),
) -> np.ndarray:
    """特徴量に対する拡張（周波数方向のマスクのみ）。時間方向のマスクは行わない。"""
    if rng.random() < config.freq_mask_prob:
        return frequency_mask(
            feature,
            rng,
            num_masks=config.freq_mask_num,
            max_width=config.freq_mask_max_width,
            mask_value=config.freq_mask_value,
        )
    return np.asarray(feature, dtype=np.float32)
