"""書き起こしベースライン（docs/PLAN.md 第3段階 3-3）の単体テスト。

音声認識モデルの取得や推論には依存させない。``Transcriber``（音声 → 文字列）を
偽物に差し替え、書き起こしからモーラ数・毎秒モーラ数を出す部分だけを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from spkrate.baselines.asr import (
    AsrParams,
    AsrSpeedEstimator,
    FasterWhisperTranscriber,
    load_params,
    mora_rate_from_transcript,
)
from spkrate.eval.runner import EvalSegment, evaluate


class FakeTranscriber:
    """呼ばれるたびに決められた文字列を順に返す偽の音声認識。"""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, samples: np.ndarray, sample_rate: int) -> str:
        self.calls.append((len(samples), sample_rate))
        return self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]


def silence(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


# --------------------------------------------------------------------------------------
# mora_rate_from_transcript


def test_毎秒モーラ数はモーラ数を区間長で割った値になる():
    # 「ニホンゴ」= 4モーラ。2秒なら毎秒2モーラ。
    assert mora_rate_from_transcript("ニホンゴ", 2.0) == pytest.approx(2.0)


def test_モーラ数の数え方を差し替えられる():
    assert mora_rate_from_transcript("なんでも", 4.0, counter=lambda text: 12) == pytest.approx(3.0)


@pytest.mark.parametrize("transcript", ["", "   ", "\n"])
def test_書き起こしが空なら0モーラとする(transcript):
    # docs/spec.md「無音は0モーラ」。空文字列を読み変換に渡さない。
    assert mora_rate_from_transcript(transcript, 3.0, counter=_never) == 0.0


def _never(text: str) -> int:  # pragma: no cover - 呼ばれないことを確かめるため
    raise AssertionError("空の書き起こしでモーラ計数を呼んではいけない")


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_区間長が正でなければエラーになる(duration):
    with pytest.raises(ValueError):
        mora_rate_from_transcript("ニホンゴ", duration)


def test_漢字仮名交じりの書き起こしでも読み変換を経てモーラを数える():
    # 「日本語」→ ニホンゴ（4モーラ）。count_mora と同じ経路を通ることの確認。
    assert mora_rate_from_transcript("日本語", 1.0) == pytest.approx(4.0)


# --------------------------------------------------------------------------------------
# AsrSpeedEstimator


def test_推定器は音声長から毎秒モーラ数を求める():
    estimator = AsrSpeedEstimator(FakeTranscriber("ニホンゴ"), counter=len)
    assert estimator(silence(2.0), 16000) == pytest.approx(2.0)


def test_推定器は書き起こしを保持する():
    estimator = AsrSpeedEstimator(FakeTranscriber("こんにちは"), counter=len)
    estimator(silence(1.0), 16000)
    assert estimator.last_transcript == "こんにちは"


def test_推定器は書き起こしに波形と標本化周波数をそのまま渡す():
    transcriber = FakeTranscriber("ア")
    estimator = AsrSpeedEstimator(transcriber, counter=len)
    estimator(silence(1.5), 16000)
    assert transcriber.calls == [(24000, 16000)]


def test_長さ0の音声では書き起こしを呼ばず0を返す():
    transcriber = FakeTranscriber("ア")
    estimator = AsrSpeedEstimator(transcriber, counter=len)
    assert estimator(np.zeros(0, dtype=np.float32), 16000) == 0.0
    assert transcriber.calls == []


def test_標本化周波数が正でなければエラーになる():
    estimator = AsrSpeedEstimator(FakeTranscriber("ア"), counter=len)
    with pytest.raises(ValueError):
        estimator(silence(1.0), 0)


def test_多次元の入力はエラーになる():
    estimator = AsrSpeedEstimator(FakeTranscriber("ア"), counter=len)
    with pytest.raises(ValueError):
        estimator(np.zeros((2, 16000), dtype=np.float32), 16000)


def test_推定器はrunnerの評価に渡せる():
    """runner の推定器の型（(samples, sample_rate) -> 毎秒モーラ数）に合っていること。"""
    estimator = AsrSpeedEstimator(FakeTranscriber("ニホンゴ", "ニホンゴ"), counter=len)
    segments = [
        EvalSegment("a", "dummy-a", true_mora=4.0, duration_sec=2.0),
        EvalSegment("b", "dummy-b", true_mora=8.0, duration_sec=2.0),
    ]
    result = evaluate(
        estimator,
        segments,
        audio_loader=lambda path: (silence(2.0), 16000),
    )
    # 予測はどちらも毎秒2モーラ。正解は毎秒2と毎秒4なので、平均絶対誤差は1.0。
    assert result.metrics.mae_moras_per_sec == pytest.approx(1.0)
    assert result.metrics.num_segments == 2


# --------------------------------------------------------------------------------------
# AsrParams と設定ファイル


def test_既定のパラメータは設定ファイルと一致する(tmp_path):
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "configs" / "baselines" / "asr.yaml"
    assert load_params(config).as_dict() == {
        "model_name": "Systran/faster-whisper-small",
        "device": "cpu",
        "compute_type": "float32",
        "cpu_threads": 0,
        "language": "ja",
        "beam_size": 1,
        "condition_on_previous_text": False,
        "without_timestamps": True,
        "vad_filter": False,
    }


def test_設定ファイルを往復できる(tmp_path):
    params = AsrParams(model_name="dummy", beam_size=3)
    path = tmp_path / "asr.yaml"
    path.write_text(yaml.safe_dump({"params": params.as_dict()}), encoding="utf-8")
    assert load_params(path) == params


def test_未知のパラメータは拒否する():
    with pytest.raises(ValueError, match="未知のパラメータ"):
        AsrParams.from_dict({"model_name": "dummy", "unknown_key": 1})


@pytest.mark.parametrize(
    "changes",
    [{"beam_size": 0}, {"cpu_threads": -1}, {"model_name": ""}, {"language": ""}],
)
def test_不正なパラメータは拒否する(changes):
    with pytest.raises(ValueError):
        AsrParams().replace(**changes)


def test_replaceは元の値を残す():
    params = AsrParams().replace(beam_size=5)
    assert params.beam_size == 5
    assert params.model_name == AsrParams().model_name


# --------------------------------------------------------------------------------------
# FasterWhisperTranscriber（モデルを差し替えて検証。ダウンロードも推論もしない）


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def transcribe(self, samples, **kwargs):
        self.kwargs = kwargs
        return iter([FakeSegment("こん"), FakeSegment("にちは")]), object()


def test_書き起こしは分割された断片を連結する():
    transcriber = FasterWhisperTranscriber(AsrParams(), model=FakeWhisperModel())
    assert transcriber(silence(1.0), 16000) == "こんにちは"


def test_復号の設定がモデルへ渡る():
    model = FakeWhisperModel()
    params = AsrParams(beam_size=5, language="ja", vad_filter=True)
    FasterWhisperTranscriber(params, model=model)(silence(1.0), 16000)
    assert model.kwargs["beam_size"] == 5
    assert model.kwargs["language"] == "ja"
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["condition_on_previous_text"] is False


def test_16kHz以外の入力は拒否する():
    # docs/spec.md「入力: 16kHzモノラル音声」。再標本化は eval.audio 側の責務。
    transcriber = FasterWhisperTranscriber(AsrParams(), model=FakeWhisperModel())
    with pytest.raises(ValueError):
        transcriber(silence(1.0, 22050), 22050)


def test_モデルは渡されたものを使いダウンロードしない():
    model = FakeWhisperModel()
    assert FasterWhisperTranscriber(AsrParams(), model=model).model is model
