"""評価指標の計算（docs/PLAN.md 第3段階 3-1）。

docs/spec.md「評価指標」のうち、区間ごとの正解・推定から計算できるものを扱う。

- 毎秒モーラ数の平均絶対誤差
- 話速帯別（4未満、4以上6未満、6以上8未満、8以上）の平均絶対誤差
- 予測値と正解の相関係数（ピアソンの積率相関係数）

処理時間とモデルサイズは指標ではなく実行環境の測定値なので、ここでは扱わず
``spkrate.eval.runner`` が metrics.csv に記録する。

入力の単位

``compute_metrics`` は区間ごとの**モーラ数**（正解・推定）と**区間長（秒）**を受け取る。
毎秒モーラ数はこの関数の内部で ``mora / duration_sec`` として求める。推定器が毎秒
モーラ数を直接返す場合は、呼び出し側で ``rate * duration_sec`` に戻してから渡す
（``runner.evaluate`` がそうしている）。この向きに統一しておくと、窓長の異なる区間が
混ざっても指標の定義が一意に決まる。

設計上の取り決め（戻り値の解釈に必要）

- **話速帯の区分は正解の毎秒モーラ数で行う。** 推定値で区分すると、手法が特定の帯へ
  偏って予測したときに帯ごとの件数が変わり、手法間の比較ができなくなるため。
- **該当区間が0件の帯の平均絶対誤差は ``float("nan")`` とする。** 0.0 を返すと
  「誤差が無い」と読めてしまうため、値なしを NaN で表す。各帯の件数は
  ``n_band_*`` に入るので、NaN が「0件」由来かどうかは件数で判別できる。
- **相関係数は、正解または推定のいずれかの分散が0のとき、および区間数が2未満のとき
  ``float("nan")`` とする。** 定義式の分母が0になり値が定まらないため。
  0.0（無相関）を返すと「相関が無い」と読めてしまうので区別する。

戻り値は ``SpeedRateMetrics``（frozen dataclass）で、``as_dict`` のキー名は
results/metrics.csv の列名とそのまま対応する。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

__all__ = [
    "BAND_KEYS",
    "METRIC_COLUMNS",
    "SPEED_BANDS",
    "SpeedRateMetrics",
    "band_of",
    "compute_metrics",
    "mora_per_second",
    "pearson_correlation",
]

# 話速帯（docs/spec.md）。(キー, 下限以上, 上限未満) の組。
SPEED_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under4", -math.inf, 4.0),
    ("4to6", 4.0, 6.0),
    ("6to8", 6.0, 8.0),
    ("over8", 8.0, math.inf),
)

BAND_KEYS: tuple[str, ...] = tuple(key for key, _, _ in SPEED_BANDS)


@dataclass(frozen=True)
class SpeedRateMetrics:
    """docs/spec.md の評価指標のうち、正解と推定から計算できるもの。

    Attributes:
        num_segments: 評価した区間数。
        mae_moras_per_sec: 毎秒モーラ数の平均絶対誤差（全区間）。
        mae_band_under4: 正解が毎秒4モーラ未満の区間の平均絶対誤差。0件なら NaN。
        mae_band_4to6: 正解が4以上6未満の区間の平均絶対誤差。0件なら NaN。
        mae_band_6to8: 正解が6以上8未満の区間の平均絶対誤差。0件なら NaN。
        mae_band_over8: 正解が8以上の区間の平均絶対誤差。0件なら NaN。
        n_band_under4: 各帯の区間数（NaN が0件由来かを判別するために持つ）。
        correlation: 正解と推定の毎秒モーラ数のピアソン相関係数。分散0なら NaN。
    """

    num_segments: int
    mae_moras_per_sec: float
    mae_band_under4: float
    mae_band_4to6: float
    mae_band_6to8: float
    mae_band_over8: float
    n_band_under4: int
    n_band_4to6: int
    n_band_6to8: int
    n_band_over8: int
    correlation: float

    def as_dict(self) -> dict[str, float | int]:
        """metrics.csv の列名と対応する辞書にする。キー名は安定である。"""
        return asdict(self)


# metrics.csv に書く指標列の順序。SpeedRateMetrics の定義順と一致させる。
METRIC_COLUMNS: tuple[str, ...] = tuple(SpeedRateMetrics.__dataclass_fields__)


def band_of(rate: float) -> str:
    """毎秒モーラ数から話速帯のキーを返す。

    区分は正解の毎秒モーラ数に対して使う（推定値には使わない）。
    NaN は境界比較が全て偽になるため ``ValueError`` とする。
    """
    if math.isnan(rate):
        raise ValueError("話速帯を決められない: rate が NaN である")
    for key, low, high in SPEED_BANDS:
        if low <= rate < high:
            return key
    raise ValueError(f"話速帯に該当しない値: {rate}")  # pragma: no cover


def mora_per_second(moras: Sequence[float], durations_sec: Sequence[float]) -> list[float]:
    """区間ごとのモーラ数と区間長から毎秒モーラ数を求める。

    区間長が0以下の場合は割り算が定義できないので ``ValueError`` を送出する。
    """
    if len(moras) != len(durations_sec):
        raise ValueError(
            f"長さが一致しない: moras={len(moras)}, durations_sec={len(durations_sec)}"
        )
    rates: list[float] = []
    for index, (mora, duration) in enumerate(zip(moras, durations_sec, strict=True)):
        if not duration > 0.0:
            raise ValueError(f"区間長は正の値である必要がある: index={index}, value={duration}")
        rates.append(float(mora) / float(duration))
    return rates


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """ピアソンの積率相関係数。

    どちらかの分散が0のとき、および要素数が2未満のときは ``float("nan")`` を返す
    （定義式の分母が0になり値が定まらないため）。
    """
    if len(xs) != len(ys):
        raise ValueError(f"長さが一致しない: {len(xs)} != {len(ys)}")
    n = len(xs)
    if n < 2:
        return math.nan
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x = sum(d * d for d in dx)
    var_y = sum(d * d for d in dy)
    if var_x <= 0.0 or var_y <= 0.0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(var_x * var_y)


def compute_metrics(
    true_moras: Sequence[float],
    pred_moras: Sequence[float],
    durations_sec: Sequence[float],
) -> SpeedRateMetrics:
    """区間ごとの正解・推定モーラ数と区間長から全指標を求める。

    Args:
        true_moras: 区間ごとの正解モーラ数。
        pred_moras: 区間ごとの推定モーラ数（推定器が毎秒モーラ数を返す場合は
            呼び出し側で区間長を掛けてモーラ数に戻す）。
        durations_sec: 区間長（秒）。すべて正の値であること。

    Returns:
        SpeedRateMetrics。区間が0件の場合、全ての平均絶対誤差と相関係数は NaN になる。

    Raises:
        ValueError: 3つの列の長さが揃わない場合、または区間長が0以下の場合。
    """
    if not (len(true_moras) == len(pred_moras) == len(durations_sec)):
        raise ValueError(
            "長さが一致しない: "
            f"true={len(true_moras)}, pred={len(pred_moras)}, duration={len(durations_sec)}"
        )
    true_rates = mora_per_second(true_moras, durations_sec)
    pred_rates = mora_per_second(pred_moras, durations_sec)
    errors = [abs(p - t) for t, p in zip(true_rates, pred_rates, strict=True)]

    band_errors: dict[str, list[float]] = {key: [] for key in BAND_KEYS}
    for rate, error in zip(true_rates, errors, strict=True):
        band_errors[band_of(rate)].append(error)

    def mean_or_nan(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    return SpeedRateMetrics(
        num_segments=len(errors),
        mae_moras_per_sec=mean_or_nan(errors),
        mae_band_under4=mean_or_nan(band_errors["under4"]),
        mae_band_4to6=mean_or_nan(band_errors["4to6"]),
        mae_band_6to8=mean_or_nan(band_errors["6to8"]),
        mae_band_over8=mean_or_nan(band_errors["over8"]),
        n_band_under4=len(band_errors["under4"]),
        n_band_4to6=len(band_errors["4to6"]),
        n_band_6to8=len(band_errors["6to8"]),
        n_band_over8=len(band_errors["over8"]),
        correlation=pearson_correlation(true_rates, pred_rates),
    )
