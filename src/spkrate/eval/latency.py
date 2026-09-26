"""推論時間の測定（新方式。docs/decisions/007-latency-measurement.md）。

比較する全モデルを**同一プロセス内で連続して**測り、2.0秒窓1回の推論時間の中央値と最大値を
記録する。同じ測定で測った行には同じ測定セッションの識別子（``latency_session_id``）を付け、
セッションが異なる値を並べて比較しようとしたらエラーにする（``compare_latency``）。

## 1回の推論に含めるもの

旧方式（``scripts/eval_dev_full.py`` の ``measure_latency`` の ``total_ms``。metrics.csv の
``latency_ms_per_inference``）と同じ範囲にそろえる。

    対数メル計算（CPU、numpy/torchaudio）→ 固定値の正規化 → テンソル化とデバイスへの転送
    → モデルの前向き計算 → デバイスの同期

音声の読み込み・再標本化は含めない。理由: 実運用では0.25秒ごとに2.0秒窓から対数メルを計算して
モデルに通すので、利用者が待つ時間はメル計算を含む（docs/spec.md「推論間隔: 0.25秒」）。
旧方式と範囲をそろえておけば、旧方式の平均と新方式の中央値の差が測り方（プロセス・統計量）の
違いだけになる。

## 計時

1回ごとに ``synchronize(); t0 = perf_counter(); 推論; synchronize(); t1 = perf_counter()``。
前の同期で前回までの非同期の計算がデバイスに残っていないことを保証し、後の同期で今回の計算の
完了までを含める（mps は非同期に実行されるため、同期しないと投入の時間しか測れない）。

## ウォームアップと測定順

- ウォームアップ: 各モデルを ``warmup`` 回（既定 ``DEFAULT_WARMUP`` = 20）呼んでから測る。
  測定と同じ交互の順で行う。
- 交互測定（既定で採用）: 1周ごとに全モデルを1回ずつ測り、周ごとに開始するモデルを1つずつ
  ずらす（周 r ではモデル i が (i + r) mod n 番目）。時間とともに変わる条件（発熱による
  クロック低下、他のプロセス、メモリの状態）が全モデルに均等に乗り、各モデルが各順位を同じ回数
  経験する。``interleave=False`` ではモデルごとに連続して測る（比較用）。
"""

from __future__ import annotations

import csv
import secrets
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spkrate.eval.metrics import METRIC_COLUMNS

__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_WARMUP",
    "LATENCY_SEED",
    "LATENCY_SPLIT",
    "LatencyResult",
    "LatencySession",
    "LatencySessionMismatchError",
    "WINDOW_SAMPLES",
    "check_same_session",
    "compare_latency",
    "device_synchronizer",
    "latency_csv_row",
    "make_cnn_inference",
    "make_window",
    "measure_session",
    "measurement_order",
    "new_session_id",
    "read_latency_rows",
]

SAMPLE_RATE = 16000
WINDOW_SEC = 2.0  # docs/spec.md「モデルの入力単位: 固定長窓2.0秒」
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
LATENCY_SEED = 20260921  # 旧方式（scripts/eval_dev_full.py）と同じ入力を作る種

# 2.0秒窓1回の推論を測る回数。タスクの規定（100回）。
DEFAULT_REPEATS = 100
# ウォームアップの回数。既定20の理由: 旧方式（eval_dev_full.py の measure_latency）と同じ回数で、
# mps では最初の数回にカーネル（シェーダ）の生成とメモリ確保が入り桁違いに遅いが、それは数回で
# 収まる。20回ならその数倍の余裕があり、100回の測定に比べて測定全体の時間も大きく増えない。
DEFAULT_WARMUP = 20

# metrics.csv で新方式の推論時間の行に書く split の値（精度評価の行と区別する）。
LATENCY_SPLIT = "latency_2s_window"


class LatencySessionMismatchError(ValueError):
    """異なる測定セッションの推論時間を並べて比較しようとした。"""


@dataclass(frozen=True)
class LatencyResult:
    """1モデルの測定結果（ミリ秒）。``times_ms`` は測定順の各回の値（ウォームアップを除く）。"""

    name: str
    times_ms: tuple[float, ...]
    warmup: int

    @property
    def repeats(self) -> int:
        return len(self.times_ms)

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.times_ms))

    @property
    def max_ms(self) -> float:
        return float(max(self.times_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.times_ms))

    def summary(self) -> dict[str, float | int | str]:
        return {"name": self.name, "repeats": self.repeats, "warmup": self.warmup,
                "median_ms": self.median_ms, "max_ms": self.max_ms, "mean_ms": self.mean_ms,
                "min_ms": float(min(self.times_ms))}


@dataclass(frozen=True)
class LatencySession:
    """1回の測定セッション（同一プロセス内で連続して測った全モデル）。"""

    session_id: str
    results: dict[str, LatencyResult]
    interleave: bool
    rounds: tuple[tuple[str, ...], ...] = field(default=())  # 測定の各周のモデルの順


def new_session_id(now: datetime | None = None) -> str:
    """測定セッションの識別子。ローカル時刻（秒まで）と8桁の16進乱数（例: lat-20260924T153012-3f9a1c2b）。"""
    moment = (now or datetime.now(timezone.utc).astimezone())
    return f"lat-{moment.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}"


def make_window(seed: int = LATENCY_SEED) -> np.ndarray:
    """測定に使う2.0秒窓（32000標本、float32）。旧方式と同じ作り方（標準正規×0.05）。"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(WINDOW_SAMPLES) * 0.05).astype(np.float32)


def device_synchronizer(device) -> Callable[[], None]:  # noqa: ANN001
    """デバイスの同期関数。mps は ``torch.mps.synchronize``、cuda は ``torch.cuda.synchronize``、
    cpu は何もしない。"""
    import torch

    kind = torch.device(device).type
    if kind == "mps":
        return torch.mps.synchronize
    if kind == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def make_cnn_inference(model, normalizer, device) -> Callable[[np.ndarray], object]:  # noqa: ANN001
    """2.0秒窓の波形から1回推論する関数を作る（モジュール docstring「1回の推論に含めるもの」）。

    同期は含めない（``measure_session`` が計時の前後で行う）。``model`` は ``device`` 上で
    ``eval()`` にしておくこと。
    """
    import torch

    from spkrate.features.melspec import log_mel_spectrogram

    def infer(samples: np.ndarray):
        with torch.no_grad():
            feature = normalizer(log_mel_spectrogram(samples, SAMPLE_RATE))
            x = torch.from_numpy(np.ascontiguousarray(feature[None], dtype=np.float32)).to(device)
            n = torch.tensor([feature.shape[0]], dtype=torch.long, device=device)
            return model(x, n)

    return infer


def measurement_order(names: Sequence[str], round_index: int, *, interleave: bool = True) -> tuple[str, ...]:
    """周 ``round_index`` でモデルを測る順。交互測定では開始位置を周ごとに1つずらす。"""
    if not interleave or not names:
        return tuple(names)
    shift = round_index % len(names)
    return tuple(names[shift:]) + tuple(names[:shift])


def measure_session(
    inferences: Mapping[str, Callable[[np.ndarray], object]],
    window: np.ndarray | None = None,
    *,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    interleave: bool = True,
    synchronize: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
    session_id: str | None = None,
) -> LatencySession:
    """全モデルを同一プロセス内で連続して測る。

    Args:
        inferences: モデル名から「2.0秒窓の波形を受け取り1回推論する関数」への対応
            （``make_cnn_inference``）。挿入順がモデルの順になる。
        window: 入力の窓。省略時は ``make_window()``。
        repeats: 各モデルの測定回数（既定100）。
        warmup: 各モデルのウォームアップ回数（既定20。0なら行わない）。
        interleave: 交互測定にするか（既定 True。モジュール docstring 参照）。
        synchronize: デバイスの同期関数（``device_synchronizer``）。計時の前後で呼ぶ。
        clock: 秒を返す時計（テスト用の差し替え口）。
        session_id: 省略時は ``new_session_id()``。
    """
    if repeats < 1:
        raise ValueError(f"repeats は1以上: {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup は0以上: {warmup}")
    names = list(inferences)
    if not names:
        raise ValueError("測るモデルが無い")
    samples = make_window() if window is None else window

    # ウォームアップ（測定と同じ順）。結果は捨てる。
    for index in range(warmup):
        for name in measurement_order(names, index, interleave=interleave):
            inferences[name](samples)
    synchronize()

    times: dict[str, list[float]] = {name: [] for name in names}
    rounds: list[tuple[str, ...]] = []

    def timed(name: str) -> None:
        synchronize()
        started = clock()
        inferences[name](samples)
        synchronize()
        times[name].append((clock() - started) * 1000.0)

    if interleave:
        for index in range(repeats):
            order = measurement_order(names, index, interleave=True)
            rounds.append(order)
            for name in order:
                timed(name)
    else:
        for name in names:
            for _ in range(repeats):
                timed(name)
        rounds.append(tuple(names))

    results = {name: LatencyResult(name, tuple(times[name]), warmup) for name in names}
    return LatencySession(session_id or new_session_id(), results, interleave, tuple(rounds))


def latency_csv_row(
    result: LatencyResult,
    session_id: str,
    *,
    experiment_id: str,
    method: str,
    config_path: str,
    model_size_bytes: int | None,
    commit_hash: str,
    commit_dirty: str,
    timestamp: str | None = None,
    device: str = "",
    host: str | None = None,
) -> dict[str, object]:
    """新方式の測定結果を metrics.csv の1行（``append_metrics_row`` に渡す辞書）にする。

    精度の指標列と旧方式の ``latency_ms_per_inference`` は空欄。``split`` は ``LATENCY_SPLIT``。
    ``device`` は測ったデバイスの種類、``host`` は計算機の呼び名（省略時は
    ``spkrate.device.host_label``）。
    """
    from spkrate.device import host_label

    row: dict[str, object] = {column: "" for column in METRIC_COLUMNS}
    row.update({
        "experiment_id": experiment_id,
        "timestamp": timestamp or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "commit_hash": commit_hash,
        "commit_dirty": commit_dirty,
        "config_path": config_path,
        "method": method,
        "split": LATENCY_SPLIT,
        "latency_ms_per_inference": "",
        "model_size_bytes": "" if model_size_bytes is None else model_size_bytes,
        "latency_session_id": session_id,
        "latency_ms_median": result.median_ms,
        "latency_ms_max": result.max_ms,
        "host": host_label() if host is None else host,
        "device": device,
    })
    return row


def read_latency_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """metrics.csv から新方式の推論時間の行（``latency_session_id`` が空でない行）を読む。"""
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if (row.get("latency_session_id") or "").strip()]


def check_same_session(rows: Sequence[Mapping[str, object]]) -> str:
    """並べる行がすべて同じ測定セッションのものか確かめ、その識別子を返す。

    Raises:
        LatencySessionMismatchError: 識別子が空の行（旧方式）がある、または識別子が複数ある場合。
    """
    sessions = [str(row.get("latency_session_id") or "").strip() for row in rows]
    if any(not session for session in sessions):
        raise LatencySessionMismatchError(
            "測定セッションの識別子が無い行（旧方式の latency_ms_per_inference）は比較に使えない")
    unique = sorted(set(sessions))
    if len(unique) != 1:
        raise LatencySessionMismatchError(
            f"異なる測定セッションの推論時間を並べようとした: {unique}。同じセッションで測り直すこと")
    return unique[0]


def compare_latency(
    csv_path: str | Path,
    experiment_ids: Sequence[str],
    *,
    session_id: str | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    """指定した実験の推論時間（新方式の中央値・最大値）を同じセッションの値で並べる。

    ``session_id`` を省くと、指定した全実験を含むセッションのうち最も新しいもの（行の
    timestamp の最大）を使う。

    Returns:
        ``(セッションID, {実験ID: {"median_ms": ..., "max_ms": ...}})``。

    Raises:
        LatencySessionMismatchError: 全実験を含むセッションが無い（各実験の値が別々の
            セッションにしか無い）、または指定したセッションに無い実験がある場合。
    """
    rows = read_latency_rows(csv_path)
    wanted = list(dict.fromkeys(experiment_ids))
    by_session: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["experiment_id"] in wanted:
            by_session.setdefault(row["latency_session_id"], {})[row["experiment_id"]] = row

    if session_id is None:
        complete = [sid for sid, found in by_session.items() if all(e in found for e in wanted)]
        if not complete:
            seen = {e: sorted(sid for sid, found in by_session.items() if e in found) for e in wanted}
            raise LatencySessionMismatchError(
                "指定した実験をすべて同じセッションで測った値が無い。"
                f"実験ごとのセッション: {seen}。同じセッションで測り直すこと")
        session_id = max(complete, key=lambda sid: max(r["timestamp"] for r in by_session[sid].values()))
    found = by_session.get(session_id, {})
    missing = [e for e in wanted if e not in found]
    if missing:
        raise LatencySessionMismatchError(f"セッション {session_id} に無い実験がある: {missing}")
    selected = [found[e] for e in wanted]
    check_same_session(selected)
    values = {e: {"median_ms": float(found[e]["latency_ms_median"]),
                  "max_ms": float(found[e]["latency_ms_max"])} for e in wanted}
    return session_id, values
