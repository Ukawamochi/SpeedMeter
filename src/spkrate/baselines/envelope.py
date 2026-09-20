"""信号処理による話速ベースライン（docs/PLAN.md 第3段階 3-2）。

## 手順

1. 波形を短時間の二乗平均平方根（RMS）で包絡にする（既定は窓25ミリ秒・移動10ミリ秒。
   docs/spec.md の特徴量と同じ刻みに合わせてあり、包絡の標本化周波数は100Hzになる）
2. 包絡をデシベルへ変換する。**以降の処理は全てデシベル領域で行う。**
   音量の定数倍はデシベル領域では定数の加算になり、後段の帯域通過で完全に除かれるため、
   録音レベルの違いに影響されない（``peak_prominence`` の単位もデシベルになる）
3. 平滑化（移動平均）で細かい変動を落とす
4. 2Hzから10Hzの帯域通過フィルタ（Butterworth、零位相）で音節に対応する変動成分を残す
5. ピーク（または正方向のゼロ交差）を数え、これを音節数とみなす
6. 音節数に換算係数 ``mora_per_syllable`` を掛けてモーラ数とし、区間長で割って毎秒モーラ数にする

## 換算係数について

日本語のモーラ数は音節数と一致しない（撥音・促音・長音は独立したモーラだが、直前の音節に
含まれて1つの音量の山にしかならないことが多い）。そのため音節数からモーラ数への換算係数を
パラメータとして持ち、検証セット（dev）で調整する。調整の記録は
``results/envelope_baseline_tuning.md``、調整後の値は ``configs/baselines/envelope.yaml``。

## 無音の扱い

デシベル包絡が「最大値 + ``silence_floor_db``」以下のフレームは無音とみなし、そこに現れた
ピークは数えない。帯域通過後の信号は無音区間でも微小な振動を持つため、これを数えると
無音の長いクリップで過大評価になる。

一方、毎秒モーラ数を求める際の区間長には**無音を含む区間全体の長さ**を使う。正解の
モーラ数がクリップ全体に対して与えられているため、揃える必要があるからである。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

__all__ = [
    "COUNT_METHODS",
    "EnvelopeParams",
    "EnvelopeSpeedEstimator",
    "bandpass_envelope",
    "count_syllables_from_log_envelope",
    "frame_rms",
    "load_params",
    "log_envelope",
    "smooth",
    "syllable_rate",
]

# 音節の数え方。docs/PLAN.md 3-2「ピークまたはゼロ交差を数えて音節数とする」。
COUNT_METHODS: tuple[str, ...] = ("peak", "zero_cross")


@dataclass(frozen=True)
class EnvelopeParams:
    """信号処理ベースラインのパラメータ。

    Attributes:
        frame_length_sec: 包絡を求める窓長（秒）。
        hop_sec: 包絡の移動幅（秒）。包絡の標本化周波数は ``1 / hop_sec`` になる。
        floor_db: デシベル包絡の下限（最大値からの相対値）。無音での -inf を避ける。
        smoothing_sec: 平滑化の時定数（移動平均の窓長、秒）。0 なら平滑化しない。
        band_low_hz: 帯域通過の下限周波数。
        band_high_hz: 帯域通過の上限周波数。
        filter_order: Butterworth フィルタの次数。
        count_method: "peak"（極大を数える）または "zero_cross"（正方向のゼロ交差を数える）。
        peak_prominence: ピークとみなす突出量（デシベル）。"zero_cross" では使わない。
        min_peak_distance_sec: 隣り合う音節の最小間隔（秒）。
        silence_floor_db: 無音とみなす閾値（デシベル包絡の最大値からの相対値）。
        mora_per_syllable: 音節数からモーラ数への換算係数。
    """

    frame_length_sec: float = 0.025
    hop_sec: float = 0.010
    floor_db: float = -60.0
    smoothing_sec: float = 0.0
    band_low_hz: float = 2.0
    band_high_hz: float = 10.0
    filter_order: int = 3
    count_method: str = "peak"
    peak_prominence: float = 1.0
    min_peak_distance_sec: float = 0.06
    silence_floor_db: float = -40.0
    mora_per_syllable: float = 1.5

    def __post_init__(self) -> None:
        if self.count_method not in COUNT_METHODS:
            raise ValueError(f"未知の数え方: {self.count_method}（{COUNT_METHODS} のいずれか）")
        if not 0.0 < self.band_low_hz < self.band_high_hz:
            raise ValueError(
                f"帯域が不正: low={self.band_low_hz}, high={self.band_high_hz}"
            )
        if self.hop_sec <= 0.0 or self.frame_length_sec <= 0.0:
            raise ValueError("窓長と移動幅は正の値である必要がある")
        if self.band_high_hz >= 0.5 / self.hop_sec:
            raise ValueError(
                f"帯域の上限が包絡のナイキスト周波数以上: {self.band_high_hz} >= {0.5 / self.hop_sec}"
            )

    @property
    def envelope_rate(self) -> float:
        """包絡の標本化周波数（Hz）。"""
        return 1.0 / self.hop_sec

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def replace(self, **changes: object) -> "EnvelopeParams":
        """一部の値だけを差し替えた新しいパラメータを返す（探索で使う）。"""
        return EnvelopeParams(**{**asdict(self), **changes})


def load_params(path: str | Path) -> EnvelopeParams:
    """yaml からパラメータを読む（configs/baselines/envelope.yaml）。

    ``params`` キーの下にある値だけを見る。未知のキーがあれば ``ValueError``。
    """
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    values = payload.get("params", payload)
    known = {field.name for field in fields(EnvelopeParams)}
    unknown = [key for key in values if key not in known]
    if unknown:
        raise ValueError(f"未知のパラメータ: {unknown}")
    return EnvelopeParams(**values)


def frame_rms(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_length_sec: float = 0.025,
    hop_sec: float = 0.010,
) -> np.ndarray:
    """短時間の二乗平均平方根で音量包絡を求める。

    窓の中心が 0, hop, 2*hop, ... 秒に来るよう前後を0で埋める。戻り値の長さは
    ``ceil(len(samples) / hop)`` で、標本化周波数は ``sample_rate / hop`` になる。
    """
    if sample_rate <= 0:
        raise ValueError(f"標本化周波数は正の値である必要がある: {sample_rate}")
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame_length = max(1, int(round(frame_length_sec * sample_rate)))
    hop = max(1, int(round(hop_sec * sample_rate)))
    if signal.size == 0:
        return np.zeros(0, dtype=np.float32)

    num_frames = int(np.ceil(signal.size / hop))
    pad_left = frame_length // 2
    needed = (num_frames - 1) * hop + frame_length
    pad_right = max(0, needed - (signal.size + pad_left))
    padded = np.pad(signal, (pad_left, pad_right))
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop]
    windows = windows[:num_frames]
    power = np.mean(np.square(windows, dtype=np.float32), axis=1, dtype=np.float32)
    return np.sqrt(power, dtype=np.float32)


def log_envelope(rms: np.ndarray, *, floor_db: float = -60.0) -> np.ndarray:
    """RMS包絡をデシベルにする。最大値から ``floor_db`` だけ下で打ち切る。

    打ち切りは、無音区間の極端に小さい値が帯域通過後に大きな振動を生むのを防ぐため。
    """
    values = np.asarray(rms, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    decibels = 20.0 * np.log10(np.maximum(values, 1e-10, dtype=np.float32))
    return np.maximum(decibels, decibels.max() + np.float32(floor_db)).astype(np.float32)


def smooth(envelope: np.ndarray, envelope_rate: float, smoothing_sec: float) -> np.ndarray:
    """移動平均で平滑化する。``smoothing_sec`` が0以下、または窓が1点なら何もしない。"""
    values = np.asarray(envelope, dtype=np.float32).reshape(-1)
    if smoothing_sec <= 0.0 or values.size == 0:
        return values
    width = int(round(smoothing_sec * envelope_rate))
    if width <= 1:
        return values
    width = min(width, values.size)
    kernel = np.full(width, 1.0 / width, dtype=np.float32)
    # 端は同じ値で延長する（0で埋めると端に偽のピークができるため）。
    pad = width // 2
    extended = np.pad(values, (pad, pad), mode="edge")
    convolved = np.convolve(extended, kernel, mode="same")[pad : pad + values.size]
    return convolved.astype(np.float32)


def _sos(band_low_hz: float, band_high_hz: float, order: int, envelope_rate: float) -> np.ndarray:
    from scipy.signal import butter

    return butter(
        order,
        [band_low_hz, band_high_hz],
        btype="bandpass",
        fs=envelope_rate,
        output="sos",
    )


def bandpass_envelope(
    envelope: np.ndarray,
    envelope_rate: float,
    *,
    band_low_hz: float = 2.0,
    band_high_hz: float = 10.0,
    order: int = 3,
) -> np.ndarray:
    """包絡から帯域内の変動成分だけを取り出す（零位相のButterworth帯域通過）。

    零位相にするのはピーク位置を時間方向にずらさないため。フィルタの折り返し用に
    必要な長さに満たない短い包絡は、全て0の配列を返す（音節を数えない）。
    """
    values = np.asarray(envelope, dtype=np.float32).reshape(-1)
    sos = _sos(band_low_hz, band_high_hz, order, envelope_rate)
    padlen = 3 * (2 * sos.shape[0] + 1)
    if values.size <= padlen:
        return np.zeros_like(values)

    from scipy.signal import sosfiltfilt

    return np.asarray(sosfiltfilt(sos, values), dtype=np.float32)


def _count_peaks(
    filtered: np.ndarray,
    envelope_rate: float,
    *,
    prominence: float,
    min_distance_sec: float,
    active: np.ndarray | None,
) -> int:
    from scipy.signal import find_peaks

    distance = max(1, int(round(min_distance_sec * envelope_rate)))
    peaks, _ = find_peaks(filtered, prominence=prominence, distance=distance)
    if active is not None and peaks.size:
        peaks = peaks[active[peaks]]
    return int(peaks.size)


def _count_zero_crossings(
    filtered: np.ndarray,
    envelope_rate: float,
    *,
    min_distance_sec: float,
    active: np.ndarray | None,
) -> int:
    """負から正へ変わる点（正方向のゼロ交差）を数える。

    帯域通過後の信号は平均0の振動なので、正方向のゼロ交差1つが振動1周期に対応する。
    """
    if filtered.size < 2:
        return 0
    positive = filtered > 0.0
    crossings = np.flatnonzero(~positive[:-1] & positive[1:]) + 1
    if active is not None and crossings.size:
        crossings = crossings[active[crossings]]
    if crossings.size < 2:
        return int(crossings.size)
    min_distance = max(1, int(round(min_distance_sec * envelope_rate)))
    kept = [int(crossings[0])]
    for index in crossings[1:]:
        if int(index) - kept[-1] >= min_distance:
            kept.append(int(index))
    return len(kept)


def count_syllables_from_log_envelope(
    log_env: np.ndarray, envelope_rate: float, params: EnvelopeParams
) -> int:
    """デシベル包絡から音節数を数える。

    デシベル包絡を引数に取るのは、パラメータ探索で包絡の計算（音声の復号を含む）を
    一度だけ行い、平滑化から先だけを繰り返せるようにするためである。
    """
    values = np.asarray(log_env, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0
    smoothed = smooth(values, envelope_rate, params.smoothing_sec)
    active = smoothed > (smoothed.max() + np.float32(params.silence_floor_db))
    filtered = bandpass_envelope(
        smoothed,
        envelope_rate,
        band_low_hz=params.band_low_hz,
        band_high_hz=params.band_high_hz,
        order=params.filter_order,
    )
    if params.count_method == "peak":
        return _count_peaks(
            filtered,
            envelope_rate,
            prominence=params.peak_prominence,
            min_distance_sec=params.min_peak_distance_sec,
            active=active,
        )
    return _count_zero_crossings(
        filtered,
        envelope_rate,
        min_distance_sec=params.min_peak_distance_sec,
        active=active,
    )


def syllable_rate(count: int, duration_sec: float) -> float:
    """音節数と区間長から毎秒音節数を求める。区間長が0以下なら0を返す。"""
    if duration_sec <= 0.0:
        return 0.0
    return float(count) / float(duration_sec)


@dataclass(frozen=True)
class EnvelopeSpeedEstimator:
    """音声を受け取り毎秒モーラ数を返す推定器（``spkrate.eval.runner.Estimator``）。"""

    params: EnvelopeParams = EnvelopeParams()

    def log_envelope_of(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """波形からデシベル包絡を求める（パラメータ探索でのキャッシュ作成にも使う）。"""
        rms = frame_rms(
            samples,
            sample_rate,
            frame_length_sec=self.params.frame_length_sec,
            hop_sec=self.params.hop_sec,
        )
        return log_envelope(rms, floor_db=self.params.floor_db)

    def count_syllables(self, samples: np.ndarray, sample_rate: int) -> int:
        """波形から音節数を数える。"""
        return count_syllables_from_log_envelope(
            self.log_envelope_of(samples, sample_rate),
            self.params.envelope_rate,
            self.params,
        )

    def __call__(self, samples: np.ndarray, sample_rate: int) -> float:
        """毎秒モーラ数を返す。区間長は無音を含む波形全体の長さを使う。"""
        length = int(np.asarray(samples).reshape(-1).size)
        if length == 0 or sample_rate <= 0:
            return 0.0
        duration = float(length) / float(sample_rate)
        count = self.count_syllables(samples, sample_rate)
        return self.params.mora_per_syllable * syllable_rate(count, duration)
