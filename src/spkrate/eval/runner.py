"""評価の実行と results/metrics.csv への追記（docs/PLAN.md 第3段階 3-1）。

## 推定器のインタフェース

推定器は docs/PLAN.md の定義どおり「音声を受け取り毎秒モーラ数を返す関数」である。

    def estimator(samples: numpy.ndarray, sample_rate: int) -> float: ...

- ``samples``: 16kHzモノラル float32 の1次元配列（``spkrate.eval.audio.load_audio``
  が返すもの。docs/spec.md「入力: 16kHzモノラル音声」）
- ``sample_rate``: 常に ``audio.TARGET_SAMPLE_RATE``（16000）
- 戻り値: その区間の**毎秒モーラ数**（非負の実数）

``evaluate`` は戻り値に区間長を掛けてモーラ数へ戻し、
``spkrate.eval.metrics.compute_metrics`` へ渡す。指標の定義が区間長に依存しない形で
一意に決まるようにするためである（metrics.py の docstring を参照）。

## テストセットの使用禁止

``configs/splits/test.json`` は docs/PLAN.md 第10段階まで使用禁止である。
``load_eval_client_ids`` および ``build_eval_segments`` は、テスト分割のファイルを
渡されると既定で ``RuntimeError`` を送出する（``spkrate.data.splits.load_test_split``
と同じ仕組みで、``stage10_approved=True`` を明示しない限り拒否する）。
第10段階に到達するまで、この引数を True にするコードを書かないこと。

## metrics.csv の列

CLAUDE.md の規定（実験ID、日時、コミットハッシュ、設定ファイルのパス、全指標）と
docs/PLAN.md 3-1 の追加要求（手法名、処理時間、モデルサイズ）の双方を満たす列構成を
``METRICS_CSV_COLUMNS`` に固定する。列順は変更しない。追記時にヘッダが無ければ書き、
既存ヘッダが ``METRICS_CSV_COLUMNS`` と異なる場合は ``ValueError`` とする
（列がずれた行を混ぜないため）。NaN の指標は空欄として書く。
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from spkrate.data.splits import load_split, load_test_split
from spkrate.eval.audio import load_audio
from spkrate.eval.metrics import METRIC_COLUMNS, SpeedRateMetrics, compute_metrics

__all__ = [
    "CommitInfo",
    "DEFAULT_METRICS_CSV",
    "EvalSegment",
    "EvaluationResult",
    "Estimator",
    "METRICS_CSV_COLUMNS",
    "MetricsRow",
    "append_metrics_row",
    "build_eval_segments",
    "evaluate",
    "git_commit_info",
    "load_eval_client_ids",
    "model_size_bytes",
    "run_and_record",
    "segments_from_clip_records",
]

DEFAULT_METRICS_CSV = Path("results/metrics.csv")

# metrics.csv の列順（固定）。指標列は METRIC_COLUMNS をそのまま展開する。
METRICS_CSV_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "timestamp",
    "commit_hash",
    "commit_dirty",
    "config_path",
    "method",
    "split",
    *METRIC_COLUMNS,
    "latency_ms_per_inference",
    "model_size_bytes",
)


class Estimator(Protocol):
    """音声を受け取り毎秒モーラ数を返す関数。モジュール docstring を参照。"""

    def __call__(self, samples: np.ndarray, sample_rate: int) -> float: ...


@dataclass(frozen=True)
class EvalSegment:
    """評価する1区間。

    Attributes:
        segment_id: 区間の識別子（Common Voice ではクリップID）。
        audio_path: 音声ファイルのパス。
        true_mora: 正解のモーラ数。
        duration_sec: 区間長（秒）。None の場合は読み込んだ音声の長さを使う。
    """

    segment_id: str
    audio_path: Path
    true_mora: float
    duration_sec: float | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """評価の結果。指標と実行時の測定値。

    Attributes:
        metrics: docs/spec.md の指標。
        latency_ms_per_inference: 推定器1回あたりの平均処理時間（ミリ秒）。
            音声の読み込み・再標本化の時間は含まない（手法そのものの重さを比べるため）。
        total_inference_sec: 推定器の呼び出しに要した合計秒数。
        total_audio_sec: 評価した音声の合計秒数。
        predictions: 区間IDから推定した毎秒モーラ数への対応（再利用・検証用）。
    """

    metrics: SpeedRateMetrics
    latency_ms_per_inference: float
    total_inference_sec: float
    total_audio_sec: float
    predictions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitInfo:
    """評価時点のコミット状態。"""

    commit_hash: str
    dirty: bool

    @property
    def dirty_flag(self) -> str:
        """metrics.csv に書く文字列。"""
        return "dirty" if self.dirty else "clean"


@dataclass(frozen=True)
class MetricsRow:
    """metrics.csv の1行分。``as_row`` のキー順は METRICS_CSV_COLUMNS に従う。"""

    experiment_id: str
    method: str
    config_path: str
    split: str
    metrics: SpeedRateMetrics
    latency_ms_per_inference: float
    model_size_bytes: int | None = None
    commit: CommitInfo | None = None
    timestamp: str | None = None

    def as_row(self) -> dict[str, object]:
        commit = self.commit or git_commit_info()
        row: dict[str, object] = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "commit_hash": commit.commit_hash,
            "commit_dirty": commit.dirty_flag,
            "config_path": self.config_path,
            "method": self.method,
            "split": self.split,
            "latency_ms_per_inference": self.latency_ms_per_inference,
            "model_size_bytes": "" if self.model_size_bytes is None else self.model_size_bytes,
        }
        row.update(self.metrics.as_dict())
        return {column: row[column] for column in METRICS_CSV_COLUMNS}


def git_commit_info(repo_dir: str | Path | None = None) -> CommitInfo:
    """``git rev-parse HEAD`` と ``git status --porcelain`` で現在のコミット状態を取る。

    git が使えない、またはリポジトリでない場合は ``CommitInfo("unknown", True)`` を
    返す（不明な状態を「汚れている」側に倒して、後から誤解しないようにする）。
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return CommitInfo("unknown", True)
    return CommitInfo(commit, bool(status))


def model_size_bytes(path: str | Path | None) -> int | None:
    """モデルファイルの大きさ（バイト）。パスが None または存在しなければ None。"""
    if path is None:
        return None
    candidate = Path(path)
    return candidate.stat().st_size if candidate.is_file() else None


def load_eval_client_ids(
    split_path: str | Path, *, stage10_approved: bool = False
) -> list[str]:
    """評価に使う分割ファイルを読み、client_id の一覧を返す。

    テスト分割（``split`` が "test"、またはファイル名が test.json）の場合は
    ``spkrate.data.splits.load_test_split`` を通すため、``stage10_approved=True`` を
    明示しない限り ``RuntimeError`` になる。docs/PLAN.md 禁止事項による。
    """
    path = Path(split_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") == "test" or path.name == "test.json":
        return load_test_split(path, stage10_approved=stage10_approved)
    return load_split(path)


def segments_from_clip_records(
    records: Iterable, audio_root: str | Path
) -> list[EvalSegment]:
    """``ClipRecord`` の列から評価区間を作る。

    ``ClipRecord.audio_path`` は ``audio_root`` からの相対パスとして解決する。
    """
    root = Path(audio_root)
    return [
        EvalSegment(
            segment_id=record.clip_id,
            audio_path=root / record.audio_path,
            true_mora=float(record.mora),
            duration_sec=float(record.duration_sec),
        )
        for record in records
    ]


def build_eval_segments(
    clips_jsonl: str | Path,
    split_path: str | Path,
    audio_root: str | Path,
    *,
    limit: int | None = None,
    stage10_approved: bool = False,
) -> list[EvalSegment]:
    """clips.jsonl と分割ファイルから、その分割に属する評価区間を作る。

    ``limit`` を与えると先頭から その件数だけに絞る（試走用）。
    テスト分割は ``stage10_approved=True`` を明示しない限り拒否される。
    """
    from spkrate.data.splits import load_clip_records

    client_ids = set(load_eval_client_ids(split_path, stage10_approved=stage10_approved))
    records = []
    for record in load_clip_records(clips_jsonl):
        if record.client_id in client_ids:
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return segments_from_clip_records(records, audio_root)


def evaluate(
    estimator: Estimator,
    segments: Sequence[EvalSegment],
    *,
    audio_loader: Callable[[Path], tuple[np.ndarray, int]] | None = None,
) -> EvaluationResult:
    """推定器を評価区間に適用して指標を求める。

    各区間について、音声を16kHzモノラルで読み、``estimator(samples, sample_rate)`` を
    呼び、戻り値（毎秒モーラ数）に区間長を掛けてモーラ数へ戻してから指標を計算する。

    Args:
        estimator: 音声を受け取り毎秒モーラ数を返す関数（モジュール docstring 参照）。
        segments: 評価区間。
        audio_loader: 音声読み込みの差し替え口（テスト用）。既定は
            ``spkrate.eval.audio.load_audio``。

    Raises:
        ValueError: 推定器が有限でない値を返した場合。
    """
    loader = audio_loader or (lambda path: load_audio(path))

    true_moras: list[float] = []
    pred_moras: list[float] = []
    durations: list[float] = []
    predictions: dict[str, float] = {}
    total_inference_sec = 0.0

    for segment in segments:
        samples, sample_rate = loader(segment.audio_path)
        duration = segment.duration_sec
        if duration is None:
            duration = float(len(samples)) / float(sample_rate)
        started = time.perf_counter()
        rate = float(estimator(samples, sample_rate))
        total_inference_sec += time.perf_counter() - started
        if not np.isfinite(rate):
            raise ValueError(f"推定器が有限でない値を返した: {segment.segment_id} -> {rate}")
        predictions[segment.segment_id] = rate
        true_moras.append(segment.true_mora)
        pred_moras.append(rate * duration)
        durations.append(duration)

    metrics = compute_metrics(true_moras, pred_moras, durations)
    count = len(segments)
    latency_ms = (total_inference_sec / count * 1000.0) if count else float("nan")
    return EvaluationResult(
        metrics=metrics,
        latency_ms_per_inference=latency_ms,
        total_inference_sec=total_inference_sec,
        total_audio_sec=sum(durations),
        predictions=predictions,
    )


def _format_value(value: object) -> object:
    """NaN は空欄にする。それ以外はそのまま csv へ渡す。"""
    if isinstance(value, float) and not np.isfinite(value):
        return "" if np.isnan(value) else value
    return value


def append_metrics_row(
    row: MetricsRow | dict[str, object],
    csv_path: str | Path = DEFAULT_METRICS_CSV,
) -> Path:
    """metrics.csv に1行追記する。ヘッダが無ければ書く。

    既存ファイルのヘッダが ``METRICS_CSV_COLUMNS`` と異なる場合は ``ValueError``。
    列がずれた行が混ざるのを防ぐため、黙って合わせることはしない。
    """
    values = row.as_row() if isinstance(row, MetricsRow) else dict(row)
    missing = [column for column in METRICS_CSV_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"metrics.csv の列が足りない: {missing}")
    unknown = [key for key in values if key not in METRICS_CSV_COLUMNS]
    if unknown:
        raise ValueError(f"metrics.csv に未知の列がある: {unknown}")

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != METRICS_CSV_COLUMNS:
            raise ValueError(
                "metrics.csv のヘッダが想定と異なる。"
                f"期待: {list(METRICS_CSV_COLUMNS)} / 実際: {header}"
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_CSV_COLUMNS))
        if not exists:
            writer.writeheader()
        writer.writerow({key: _format_value(value) for key, value in values.items()})
    return path


def run_and_record(
    estimator: Estimator,
    segments: Sequence[EvalSegment],
    *,
    experiment_id: str,
    method: str,
    config_path: str,
    split: str,
    model_path: str | Path | None = None,
    csv_path: str | Path = DEFAULT_METRICS_CSV,
    repo_dir: str | Path | None = None,
    audio_loader: Callable[[Path], tuple[np.ndarray, int]] | None = None,
) -> EvaluationResult:
    """評価を実行し、結果を metrics.csv に1行追記する。

    ``split`` は評価に使った分割の名前（"dev" など）。テスト分割の区間を作るには
    ``build_eval_segments`` で ``stage10_approved=True`` が必要である。
    """
    result = evaluate(estimator, segments, audio_loader=audio_loader)
    append_metrics_row(
        MetricsRow(
            experiment_id=experiment_id,
            method=method,
            config_path=config_path,
            split=split,
            metrics=result.metrics,
            latency_ms_per_inference=result.latency_ms_per_inference,
            model_size_bytes=model_size_bytes(model_path),
            commit=git_commit_info(repo_dir),
        ),
        csv_path=csv_path,
    )
    return result
