"""窓の切り出し方式に関する3診断 D1・D2・D3（docs/decisions/005-window-strategy.md 4.3節）。

第4段階4-3が第5段階5-4に課した診断である。5-4の判断サブエージェントは前向き計算を
行えないため未測定のまま残り、第6段階の測定サブエージェントが実施する
（`docs/experiments/003-first-model.md` 7節の申し送り1）。

| 記号 | 診断 | 出力する数値 |
| --- | --- | --- |
| D1 | 分割整合性 | ``mean(|Σ − P|) / (2.0m)``（mora/s） |
| D2 | 無音・雑音窓の出力 | 予測モーラ数の平均（mora） |
| D3 | 連結加法性 | ``mean(|差|) / 連結後の長さ``（mora/s） |

いずれも正解ラベルを必要としない。ここには**計算の骨格だけ**を置き、どのクリップを
使うか・結果をどう書くかは ``scripts/window_diagnostics.py`` が持つ。

``predict`` は「波形の列 → その区間のモーラ数の列」という呼び出し可能オブジェクトで、
``make_predictor`` が学習済みモデルから作る。試験では加法的な偽の推定器を渡して、
診断の値が理論どおりになることを確かめる（``tests/test_window_diag.py``）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GAP_SEC",
    "SAMPLE_RATE",
    "WINDOW_SEC",
    "ConcatResult",
    "PredictFn",
    "SplitResult",
    "concat_additivity",
    "concatenate_with_silence",
    "make_predictor",
    "split_consistency",
    "split_into_windows",
    "summarize",
]

#: docs/spec.md「モデルの入力単位: 固定長窓2.0秒」
WINDOW_SEC = 2.0
#: docs/spec.md「入力: 16kHzモノラル音声」
SAMPLE_RATE = 16000
#: D3 で2件の間に入れる無音の長さ（005 4.3節）
GAP_SEC = 0.2

#: 波形の列を受け取り、それぞれのモーラ数を返すもの。
PredictFn = Callable[[Sequence[np.ndarray]], np.ndarray]


# --------------------------------------------------------------------------------------
# 波形の切り出しと連結（実データに依存しない純粋な操作）


def split_into_windows(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = WINDOW_SEC,
) -> tuple[list[np.ndarray], np.ndarray]:
    """先頭から重なりなしに ``m = floor(T / window_sec)`` 個の窓を取る（D1）。

    Returns:
        ``(窓の列, 同じ 2.0m 秒の連続区間)``。連続区間は窓を順に並べたものと標本単位で
        一致する（``np.concatenate(windows)`` と等しい）。末尾の半端な部分は捨てる。

    Raises:
        ValueError: 窓を1つも取れない長さのとき。
    """
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"1次元のモノラル波形を渡すこと: ndim={waveform.ndim}")
    window_samples = int(round(float(window_sec) * int(sample_rate)))
    if window_samples <= 0:
        raise ValueError(f"窓長が短すぎる: {window_sec}秒 × {sample_rate}Hz")
    count = int(waveform.size // window_samples)
    if count < 1:
        raise ValueError(
            f"窓を1つも取れない: 長さ {waveform.size} 標本 < 窓 {window_samples} 標本"
        )
    covered = np.ascontiguousarray(waveform[: count * window_samples])
    windows = [
        np.ascontiguousarray(covered[index * window_samples : (index + 1) * window_samples])
        for index in range(count)
    ]
    return windows, covered


def concatenate_with_silence(
    first: np.ndarray,
    second: np.ndarray,
    *,
    gap_sec: float = GAP_SEC,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """2つの波形をデジタル無音を挟んで連結する（D3）。

    Returns:
        ``(連結した波形, 挟んだ無音そのもの)``。無音は ``P(無音)`` を測るのに使う。
    """
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("1次元のモノラル波形を2つ渡すこと")
    gap_samples = int(round(float(gap_sec) * int(sample_rate)))
    if gap_samples < 0:
        raise ValueError(f"無音の長さは0以上: {gap_sec}")
    gap = np.zeros(gap_samples, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([a, gap, b])), gap


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """診断の値の要約（件数・平均・中央値・第9十分位・最大・標準偏差）。float64 は使わない。"""
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        raise ValueError("要約する値が無い")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "std": float(array.std()),
    }


# --------------------------------------------------------------------------------------
# D1 分割整合性


@dataclass(frozen=True)
class SplitResult:
    """1クリップ分の D1 の内訳。"""

    key: str
    num_windows: int
    covered_sec: float
    sum_windows: float  # Σ（各窓を単独で推論した値の総和）
    whole: float  # P（同じ 2.0m 秒を一括で推論した値）

    @property
    def diff(self) -> float:
        """``Σ − P``（モーラ）。符号つき。"""
        return self.sum_windows - self.whole

    @property
    def error_mora_per_sec(self) -> float:
        """``|Σ − P| / (2.0m)``（mora/s）。005 4.3節の D1 の定義そのもの。"""
        return abs(self.diff) / self.covered_sec


def split_consistency(
    predict: PredictFn,
    waveforms: Sequence[tuple[str, np.ndarray]],
    *,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = WINDOW_SEC,
) -> list[SplitResult]:
    """D1（分割整合性）を測る。

    各波形について「重なりなしの2.0秒窓を単独で推論した値の総和 Σ」と
    「同じ 2.0m 秒の区間を一括で推論した値 P」を求める。``predict`` の呼び出しは
    全クリップ分をまとめて1回にする（バッチ化は ``predict`` 側の責任）。
    """
    requests: list[np.ndarray] = []
    spans: list[tuple[str, int, int, float]] = []
    for key, waveform in waveforms:
        windows, covered = split_into_windows(
            waveform, sample_rate=sample_rate, window_sec=window_sec
        )
        start = len(requests)
        requests.extend(windows)
        requests.append(covered)
        spans.append((key, start, len(windows), float(covered.size) / float(sample_rate)))

    predictions = np.asarray(predict(requests), dtype=np.float32)
    if predictions.shape != (len(requests),):
        raise ValueError(f"推定値の個数が合わない: {predictions.shape} != {(len(requests),)}")

    results: list[SplitResult] = []
    for key, start, count, covered_sec in spans:
        results.append(
            SplitResult(
                key=key,
                num_windows=count,
                covered_sec=covered_sec,
                sum_windows=float(predictions[start : start + count].sum()),
                whole=float(predictions[start + count]),
            )
        )
    return results


# --------------------------------------------------------------------------------------
# D3 連結加法性


@dataclass(frozen=True)
class ConcatResult:
    """1組分の D3 の内訳。"""

    key: str
    total_sec: float
    pair: float  # P(AB)
    first: float  # P(A)
    second: float  # P(B)
    gap: float  # P(無音)

    @property
    def diff(self) -> float:
        """``P(AB) − (P(A) + P(B) + P(無音))``（モーラ）。符号つき。"""
        return self.pair - (self.first + self.second + self.gap)

    @property
    def error_mora_per_sec(self) -> float:
        """``|差| / 連結後の長さ``（mora/s）。005 4.3節の D3 の定義そのもの。"""
        return abs(self.diff) / self.total_sec


def concat_additivity(
    predict: PredictFn,
    pairs: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    gap_sec: float = GAP_SEC,
    sample_rate: int = SAMPLE_RATE,
) -> list[ConcatResult]:
    """D3（連結加法性）を測る。

    各組 ``(A, B)`` について ``P(AB)`` と ``P(A) + P(B) + P(無音)`` を比べる。
    無音は組ごとに同じ長さだが、``P(無音)`` も他と同じ経路で推論する。
    """
    requests: list[np.ndarray] = []
    spans: list[tuple[str, int, float]] = []
    for key, first, second in pairs:
        joined, gap = concatenate_with_silence(
            first, second, gap_sec=gap_sec, sample_rate=sample_rate
        )
        start = len(requests)
        requests.extend([joined, np.asarray(first, dtype=np.float32),
                         np.asarray(second, dtype=np.float32), gap])
        spans.append((key, start, float(joined.size) / float(sample_rate)))

    predictions = np.asarray(predict(requests), dtype=np.float32)
    if predictions.shape != (len(requests),):
        raise ValueError(f"推定値の個数が合わない: {predictions.shape} != {(len(requests),)}")

    results: list[ConcatResult] = []
    for key, start, total_sec in spans:
        results.append(
            ConcatResult(
                key=key,
                total_sec=total_sec,
                pair=float(predictions[start]),
                first=float(predictions[start + 1]),
                second=float(predictions[start + 2]),
                gap=float(predictions[start + 3]),
            )
        )
    return results


# --------------------------------------------------------------------------------------
# 学習済みモデルから推定器を作る


def make_predictor(
    model: Any,
    normalizer: Any,
    device: Any,
    *,
    batch_size: int = 64,
    mel: Any = None,
    progress: Callable[[int, int], None] | None = None,
) -> PredictFn:
    """``波形の列 → モーラ数の列`` を返す推定器を作る。

    経路は推論時と同じ「波形 → 対数メル → 正規化 → モデル」である
    （``runs/exp001/eval_dev_full.py`` の処理時間の測り方と同じ範囲）。
    長さが揃わないので、**長さ順に並べ替えてから**バッチにし、詰め物の量を抑える。
    詰め物のフレームは ``lengths`` でマスクされるため出力に影響しない
    （``tests/test_cnn.py::test_output_does_not_depend_on_padding``）。

    float64 は使わない（CLAUDE.md「実行環境」）。
    """
    import torch

    from spkrate.features.melspec import LogMelSpectrogram

    transform = mel if mel is not None else LogMelSpectrogram()

    def predict(waveforms: Sequence[np.ndarray]) -> np.ndarray:
        features = [
            np.asarray(normalizer(transform(np.asarray(w, dtype=np.float32), SAMPLE_RATE)),
                       dtype=np.float32)
            for w in waveforms
        ]
        order = sorted(range(len(features)), key=lambda i: features[i].shape[0])
        output = np.zeros(len(features), dtype=np.float32)
        done = 0
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                lengths = [features[i].shape[0] for i in chunk]
                max_frames = max(lengths)
                n_mels = features[chunk[0]].shape[1]
                padded = np.zeros((len(chunk), max_frames, n_mels), dtype=np.float32)
                for row, index in enumerate(chunk):
                    padded[row, : features[index].shape[0]] = features[index]
                tensor = torch.from_numpy(padded).to(device)
                length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
                prediction = model(tensor, length_tensor)
                values = prediction.detach().to("cpu").numpy().astype(np.float32)
                for row, index in enumerate(chunk):
                    output[index] = values[row]
                done += len(chunk)
                if progress is not None:
                    progress(done, len(order))
        return output

    return predict
