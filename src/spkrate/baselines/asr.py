"""書き起こしによる話速ベースライン（docs/PLAN.md 第3段階 3-3）。

## 手順

1. 音声認識モデルの小型版で日本語の書き起こしを得る
2. 書き起こしに ``spkrate.labels.mora.count_mora`` を適用してモーラ数を数える
   （学習ラベルの作り方と同じ経路。NFKC正規化 → pyopenjtalk で読み変換 → モーラ計数）
3. モーラ数を区間長で割って毎秒モーラ数にする

使うモデルと設定の選定理由は ``docs/decisions/003-asr-baseline.md``、
値は ``configs/baselines/asr.yaml`` に記録する。

## 構造

音声認識そのものは ``Transcriber``（音声 → 文字列）として切り出してあり、
``AsrSpeedEstimator`` はそれを受け取るだけである。こうすることで、

- 書き起こしからモーラ数・毎秒モーラ数を出す部分を、モデルの取得や推論なしで試験できる
- 音声認識モデルを差し替えても、後段の数え方は変わらない

``FasterWhisperTranscriber`` が実際の実装で、モデルの読み込みは最初の呼び出しまで
遅延させてある（試験や設定の読み書きだけならモデルを取りに行かない）。

## 区間長の扱い

毎秒モーラ数の分母には、書き起こされた区間ではなく**渡された音声全体の長さ**を使う。
正解のモーラ数がクリップ全体に対して与えられているため、揃える必要があるからである
（信号処理ベースラインの無音の扱いと同じ考え方）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = [
    "AsrParams",
    "AsrSpeedEstimator",
    "FasterWhisperTranscriber",
    "Transcriber",
    "load_params",
    "mora_rate_from_transcript",
]


class Transcriber(Protocol):
    """音声を受け取り書き起こし文字列を返す関数。"""

    def __call__(self, samples: np.ndarray, sample_rate: int) -> str: ...


@dataclass(frozen=True)
class AsrParams:
    """書き起こしベースラインの設定。

    Attributes:
        model_name: CTranslate2形式のモデル名（Hugging Face のリポジトリID）。
        device: 音声認識を動かすデバイス（"cpu" または "cuda"）。
            CTranslate2 は mps に対応しないため cpu を使う。理由は
            docs/decisions/003-asr-baseline.md を参照。
        compute_type: 計算精度（"float32"、"int8" など）。
        cpu_threads: CPU で使う演算スレッド数。0 なら CTranslate2 の既定。
        language: 認識する言語。日本語固定で言語判定を省く。
        beam_size: ビーム幅。1 なら貪欲復号。
        condition_on_previous_text: 直前の書き起こしを文脈として渡すか。
        without_timestamps: 時刻トークンを出さないか。
        vad_filter: 無音除去（VAD）を前処理として掛けるか。
    """

    model_name: str = "Systran/faster-whisper-small"
    device: str = "cpu"
    compute_type: str = "float32"
    cpu_threads: int = 0
    language: str = "ja"
    beam_size: int = 1
    condition_on_previous_text: bool = False
    without_timestamps: bool = True
    vad_filter: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name が空である")
        if self.beam_size < 1:
            raise ValueError(f"beam_size は1以上である必要がある: {self.beam_size}")
        if self.cpu_threads < 0:
            raise ValueError(f"cpu_threads は0以上である必要がある: {self.cpu_threads}")
        if not self.language:
            raise ValueError("language が空である")

    def replace(self, **changes: object) -> "AsrParams":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AsrParams":
        known = {field.name for field in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"未知のパラメータ: {sorted(unknown)}")
        return cls(**payload)


def load_params(path: str | Path) -> AsrParams:
    """``configs/baselines/asr.yaml`` からパラメータを読む。"""
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AsrParams.from_dict(payload.get("params", payload))


def mora_rate_from_transcript(
    transcript: str,
    duration_sec: float,
    *,
    counter: Callable[[str], int] | None = None,
) -> float:
    """書き起こしと区間長から毎秒モーラ数を求める。

    Args:
        transcript: 書き起こし文字列。空文字列なら0モーラとする（無音は0モーラ）。
        duration_sec: 区間長（秒）。正の値。
        counter: モーラ数の数え方。既定は ``spkrate.labels.mora.count_mora``。

    Raises:
        ValueError: ``duration_sec`` が正でない場合。
    """
    if not (duration_sec > 0.0):
        raise ValueError(f"区間長は正の値である必要がある: {duration_sec}")
    text = transcript.strip()
    if not text:
        return 0.0
    if counter is None:
        from spkrate.labels.mora import count_mora

        counter = count_mora
    return float(counter(text)) / float(duration_sec)


class AsrSpeedEstimator:
    """書き起こしから毎秒モーラ数を返す推定器。

    ``spkrate.eval.runner`` の推定器の型
    （``(samples, sample_rate) -> 毎秒モーラ数``）に合わせてある。
    """

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        counter: Callable[[str], int] | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._counter = counter
        self.last_transcript: str = ""

    def __call__(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate <= 0:
            raise ValueError(f"標本化周波数は正の値である必要がある: {sample_rate}")
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"モノラルの1次元配列が必要である: 次元数{array.ndim}")
        duration = float(array.size) / float(sample_rate)
        if duration <= 0.0:
            return 0.0
        self.last_transcript = self._transcriber(array, sample_rate)
        return mora_rate_from_transcript(
            self.last_transcript, duration, counter=self._counter
        )


class FasterWhisperTranscriber:
    """faster-whisper（CTranslate2）による書き起こし。

    モデルの読み込みは最初の呼び出しまで遅延する。復号の設定は ``AsrParams`` に従う。
    """

    def __init__(self, params: AsrParams | None = None, *, model: object | None = None) -> None:
        self.params = params or AsrParams()
        self._model = model

    @property
    def model(self) -> object:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.params.model_name,
                device=self.params.device,
                compute_type=self.params.compute_type,
                cpu_threads=self.params.cpu_threads,
            )
        return self._model

    def __call__(self, samples: np.ndarray, sample_rate: int) -> str:
        from spkrate.eval.audio import TARGET_SAMPLE_RATE

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"faster-whisper は{TARGET_SAMPLE_RATE}Hzの入力を前提とする: {sample_rate}"
            )
        segments, _info = self.model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=self.params.language,
            beam_size=self.params.beam_size,
            condition_on_previous_text=self.params.condition_on_previous_text,
            without_timestamps=self.params.without_timestamps,
            vad_filter=self.params.vad_filter,
        )
        return "".join(segment.text for segment in segments)
