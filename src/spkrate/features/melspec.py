"""対数メルスペクトログラム（docs/PLAN.md 第4段階 4-1、docs/spec.md「特徴量」）。

docs/spec.md が定める特徴量は次のとおりである。

    対数メルスペクトログラム。n_fft=400（25ミリ秒）、hop_length=160（10ミリ秒）、
    n_mels=80、f_min=0、f_max=8000、power=2.0、対数は log(mel + 1e-6)

入力は16kHzモノラル float32 の1次元波形（``spkrate.eval.audio.load_audio`` の戻り値）
である。出力は ``(フレーム数, 80)`` の float32 配列で、時間方向が第0軸である。

## 仕様に書かれていないパラメータ（torchaudio の既定に従う）

第8段階8-2でブラウザ実装と突き合わせるため、**採用した値をすべてここに書き残す**。
下表は ``torchaudio.transforms.MelSpectrogram`` の既定値であり、本実装はこれを
そのまま使う（``MEL_DEFAULTS`` に同じ値を定数として持つ）。

| 項目 | 値 | 補足 |
| --- | --- | --- |
| 窓関数 | ``torch.hann_window`` | torchaudio の既定（``window_fn`` の既定値） |
| 窓の周期性 | ``periodic=True`` | ``torch.hann_window`` の既定。``w[n] = 0.5 - 0.5*cos(2*pi*n/N)``（N=400、n=0..399） |
| ``win_length`` | 400 | 既定は ``n_fft`` と同じ。25ミリ秒 |
| ``center`` | ``True`` | 波形の両端を ``n_fft // 2`` = 200 標本ぶん詰め物してから切り出す |
| ``pad_mode`` | ``"reflect"`` | ``center=True`` のときの詰め方 |
| ``pad`` | 0 | 波形の前後への追加の零詰めはしない |
| ``onesided`` | ``True`` | 実数入力なので片側スペクトル（201ビン）。torchaudio 2.11 では引数自体が廃止予定で常に片側なので渡していない |
| ``normalized`` | ``False`` | STFT の結果を窓で割る正規化はしない |
| ``power`` | 2.0 | docs/spec.md の指定（パワースペクトル） |
| メル尺度 | ``mel_scale="htk"`` | torchaudio の既定。``m = 2595 * log10(1 + f / 700)``。librosa の既定（slaney）とは**異なる** |
| フィルタの正規化 | ``norm=None`` | torchaudio の既定。三角フィルタの頂点が1になる形。librosa の既定（``norm="slaney"``、帯域幅で割って面積を1にする）とは**異なる** |
| ``f_min`` / ``f_max`` | 0 / 8000 | docs/spec.md の指定 |
| ``n_mels`` | 80 | docs/spec.md の指定 |
| 対数 | ``log(mel + 1e-6)`` | 自然対数。デシベル変換（10*log10）ではない |
| dtype | float32 | CLAUDE.md「float64は使わない」 |

フレーム数は ``center=True`` により ``1 + 波形長 // 160`` になる。
2.0秒（32000標本）の窓なら 201 フレームである。

## 使い方

>>> from spkrate.features.melspec import log_mel_spectrogram
>>> feature = log_mel_spectrogram(samples, sample_rate=16000)  # (フレーム数, 80)

同じ設定を何度も使う場合は ``LogMelSpectrogram`` を1つ作り回して使う
（メルフィルタバンクと窓の作り直しを避けるため）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

__all__ = [
    "MEL_DEFAULTS",
    "MelSpecConfig",
    "LogMelSpectrogram",
    "log_mel_spectrogram",
    "num_frames",
]


@dataclass(frozen=True)
class MelSpecConfig:
    """対数メルスペクトログラムの設定。既定値は docs/spec.md の指定そのもの。"""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 2.0
    log_offset: float = 1e-6
    # 以下は docs/spec.md に明記が無く、torchaudio の既定に合わせた値。
    window: str = "hann"
    window_periodic: bool = True
    center: bool = True
    pad_mode: str = "reflect"
    pad: int = 0
    onesided: bool = True
    normalized: bool = False
    mel_scale: str = "htk"
    mel_norm: str | None = None


# モジュール docstring の表と同じ内容を、機械可読な形でも残す（8-2の突き合わせ用）。
MEL_DEFAULTS = MelSpecConfig()


def num_frames(num_samples: int, config: MelSpecConfig = MEL_DEFAULTS) -> int:
    """波形長（標本数）から出力フレーム数を求める。

    ``center=True`` のとき ``1 + num_samples // hop_length`` である。
    """
    if num_samples < 0:
        raise ValueError(f"標本数は0以上である必要がある: {num_samples}")
    if not config.center:
        raise NotImplementedError("center=False のフレーム数はこの実装では扱わない")
    return 1 + num_samples // config.hop_length


class LogMelSpectrogram:
    """対数メルスペクトログラムの計算器。

    メルフィルタバンクと窓を1度だけ作って使い回す。呼び出しごとの戻り値は
    ``(フレーム数, n_mels)`` の float32 ndarray である。
    """

    def __init__(self, config: MelSpecConfig = MEL_DEFAULTS) -> None:
        if config.window != "hann" or not config.window_periodic:
            raise NotImplementedError(
                "この実装は periodic な hann 窓のみを扱う（torchaudio の既定）"
            )
        self.config = config
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            pad=config.pad,
            n_mels=config.n_mels,
            window_fn=torch.hann_window,
            power=config.power,
            normalized=config.normalized,
            center=config.center,
            pad_mode=config.pad_mode,
            norm=config.mel_norm,
            mel_scale=config.mel_scale,
        )
        # 学習しない固定の変換なので勾配は持たせない。
        self._transform.eval()
        for parameter in self._transform.parameters():
            parameter.requires_grad_(False)

    @property
    def mel_filterbank(self) -> np.ndarray:
        """メルフィルタバンク ``(周波数ビン数, n_mels)``。8-2 の突き合わせ用。"""
        return self._transform.mel_scale.fb.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def __call__(self, samples: np.ndarray | torch.Tensor, sample_rate: int | None = None) -> np.ndarray:
        """波形から対数メルスペクトログラム ``(フレーム数, n_mels)`` を計算する。

        Args:
            samples: 16kHzモノラルの1次元波形。float32 へ変換して使う。
            sample_rate: 与えた場合は設定と一致するか検査する（16000）。
        """
        config = self.config
        if sample_rate is not None and int(sample_rate) != config.sample_rate:
            raise ValueError(
                f"標本化周波数が設定と異なる: {sample_rate} != {config.sample_rate}。"
                "spkrate.eval.audio.load_audio で16kHzへ変換してから渡すこと"
            )
        if isinstance(samples, torch.Tensor):
            waveform = samples.detach().to(torch.float32)
        else:
            waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        if waveform.ndim != 1:
            raise ValueError(f"1次元のモノラル波形を渡すこと: ndim={waveform.ndim}")
        if waveform.numel() == 0:
            raise ValueError("空の波形は扱えない")

        mel = self._transform(waveform)  # (n_mels, フレーム数)
        log_mel = torch.log(mel + config.log_offset)
        return log_mel.transpose(0, 1).contiguous().to(torch.float32).numpy()


_DEFAULT_TRANSFORM: LogMelSpectrogram | None = None


def log_mel_spectrogram(
    samples: np.ndarray | torch.Tensor,
    sample_rate: int | None = None,
    config: MelSpecConfig = MEL_DEFAULTS,
) -> np.ndarray:
    """docs/spec.md の設定で対数メルスペクトログラムを計算する（簡便関数）。

    既定の設定を使う場合は計算器を1つだけ作って使い回す。
    """
    global _DEFAULT_TRANSFORM
    if config == MEL_DEFAULTS:
        if _DEFAULT_TRANSFORM is None:
            _DEFAULT_TRANSFORM = LogMelSpectrogram(MEL_DEFAULTS)
        transform = _DEFAULT_TRANSFORM
    else:
        transform = LogMelSpectrogram(config)
    return transform(samples, sample_rate)
