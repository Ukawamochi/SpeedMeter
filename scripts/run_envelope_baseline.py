"""信号処理ベースラインの調整と評価（docs/PLAN.md 第3段階 3-2）。

実装の本体は src/spkrate/baselines/envelope.py にある。このスクリプトは3つの動作を持つ。

- ``tune``: 検証セット（dev）から固定シードで抽出した部分集合でパラメータを探索し、
  ``results/envelope_baseline_tuning.md`` と ``configs/baselines/envelope.yaml`` を書く
- ``timeit``: 少数のクリップで1クリップあたりの所要時間を測り、全件評価の見積もりを出す
- ``eval``: 調整後のパラメータで dev を評価し、``results/metrics.csv`` に1行追記する

**テストセット（configs/splits/test.json）は使わない。** 分割の指定は既定で dev のみであり、
``spkrate.eval.runner`` 側にもテスト分割を拒否する仕組みがある。

## 探索を速くするための作り

費用の大半は mp3 の復号と16kHzへの再標本化である。そこで ``tune`` では、部分集合の
**デシベル包絡だけ**を並列に計算して npz に保存し、平滑化から先（帯域・閾値・数え方・
換算係数）の探索はその配列の上で繰り返す。包絡の窓長と移動幅は探索対象から外し、
docs/spec.md の特徴量と同じ 25ミリ秒 / 10ミリ秒 に固定する。

``eval`` では音声の復号だけを複数プロセスで先読みし、推定器そのものは主プロセスで
逐次実行する（``runner.evaluate`` が測る1推論あたりの処理時間を歪めないため）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.baselines.envelope import (  # noqa: E402
    EnvelopeParams,
    EnvelopeSpeedEstimator,
    count_syllables_from_log_envelope,
    load_params,
)
from spkrate.eval.audio import load_audio  # noqa: E402
from spkrate.eval.runner import EvalSegment, build_eval_segments, run_and_record  # noqa: E402

CLIPS_JSONL = ROOT / "data" / "processed" / "clips.jsonl"
DEV_SPLIT = ROOT / "configs" / "splits" / "dev.json"
CONFIG_PATH = ROOT / "configs" / "baselines" / "envelope.yaml"
TUNING_MD = ROOT / "results" / "envelope_baseline_tuning.md"
METRICS_CSV = ROOT / "results" / "metrics.csv"
CACHE_PATH = Path("/tmp/envelope_dev_subset_envelopes.npz")

# 調整用の部分集合を選ぶ固定シード。この値を変えない限り同じ部分集合になる。
TUNE_SEED = 20260921
TUNE_SIZE = 2000

# 包絡の作り方は探索対象から外す（docs/spec.md の特徴量と同じ刻み）。
BASE_PARAMS = EnvelopeParams(frame_length_sec=0.025, hop_sec=0.010, floor_db=-60.0)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------------------
# 部分集合の抽出


def load_dev_segments() -> list[EvalSegment]:
    return build_eval_segments(CLIPS_JSONL, DEV_SPLIT, ROOT)


def sample_segments(
    segments: list[EvalSegment], size: int, seed: int
) -> list[EvalSegment]:
    """固定シードで部分集合を選ぶ。件数が足りなければ全件を返す。"""
    if size >= len(segments):
        return list(segments)
    indices = sorted(random.Random(seed).sample(range(len(segments)), size))
    return [segments[index] for index in indices]


# --------------------------------------------------------------------------------------
# 包絡のキャッシュ（tune 用）


def _envelope_worker(task: tuple[str, str]) -> tuple[str, np.ndarray, float]:
    segment_id, path = task
    samples, sample_rate = load_audio(path)
    estimator = EnvelopeSpeedEstimator(BASE_PARAMS)
    envelope = estimator.log_envelope_of(samples, sample_rate)
    return segment_id, envelope, float(len(samples)) / float(sample_rate)


@dataclass
class EnvelopeCache:
    """部分集合のデシベル包絡と正解。"""

    segment_ids: list[str]
    envelopes: list[np.ndarray]
    audio_durations: np.ndarray
    record_durations: np.ndarray
    true_moras: np.ndarray

    @property
    def true_rates(self) -> np.ndarray:
        return self.true_moras / self.record_durations


def build_cache(segments: list[EvalSegment], workers: int) -> EnvelopeCache:
    tasks = [(segment.segment_id, str(segment.audio_path)) for segment in segments]
    started = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(_envelope_worker, tasks, chunksize=16)
    elapsed = time.perf_counter() - started
    log(f"包絡の計算: {len(results)}件, {elapsed:.1f}秒 ({elapsed / max(1, len(results)) * 1000:.1f}ミリ秒/件, {workers}並列)")
    order = {segment_id: index for index, (segment_id, _, _) in enumerate(results)}
    return EnvelopeCache(
        segment_ids=[segment.segment_id for segment in segments],
        envelopes=[results[order[segment.segment_id]][1] for segment in segments],
        audio_durations=np.array(
            [results[order[segment.segment_id]][2] for segment in segments], dtype=np.float64
        ),
        record_durations=np.array(
            [float(segment.duration_sec) for segment in segments], dtype=np.float64
        ),
        true_moras=np.array([float(segment.true_mora) for segment in segments], dtype=np.float64),
    )


def save_cache(cache: EnvelopeCache, path: Path) -> None:
    lengths = np.array([envelope.size for envelope in cache.envelopes], dtype=np.int64)
    np.savez(
        path,
        segment_ids=np.array(cache.segment_ids),
        flat=np.concatenate(cache.envelopes) if cache.envelopes else np.zeros(0, np.float32),
        lengths=lengths,
        audio_durations=cache.audio_durations,
        record_durations=cache.record_durations,
        true_moras=cache.true_moras,
    )


def load_cache(path: Path) -> EnvelopeCache:
    payload = np.load(path, allow_pickle=False)
    lengths = payload["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = payload["flat"]
    envelopes = [
        flat[offsets[index] : offsets[index + 1]].astype(np.float32)
        for index in range(len(lengths))
    ]
    return EnvelopeCache(
        segment_ids=list(payload["segment_ids"]),
        envelopes=envelopes,
        audio_durations=payload["audio_durations"],
        record_durations=payload["record_durations"],
        true_moras=payload["true_moras"],
    )


# --------------------------------------------------------------------------------------
# 探索


def syllable_counts(cache: EnvelopeCache, params: EnvelopeParams) -> np.ndarray:
    rate = params.envelope_rate
    return np.array(
        [count_syllables_from_log_envelope(envelope, rate, params) for envelope in cache.envelopes],
        dtype=np.float64,
    )


def fit_conversion_factor(counts: np.ndarray, cache: EnvelopeCache) -> float:
    """平均絶対誤差を最小にする換算係数を求める。

    予測は ``k * counts / 音声長``、正解は ``モーラ数 / 区間長`` である。誤差の平均は
    ``k`` について区分線形かつ凸なので、最小点は重み付き中央値で厳密に求まる
    （重みは ``counts / 音声長``）。音節が0個のクリップは ``k`` に依存しない定数項に
    なるので重みから外れる。
    """
    slopes = counts / cache.audio_durations
    targets = cache.true_rates
    positive = slopes > 0.0
    if not np.any(positive):
        return 1.0
    ratios = targets[positive] / slopes[positive]
    weights = slopes[positive]
    order = np.argsort(ratios)
    ratios, weights = ratios[order], weights[order]
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(ratios[min(index, ratios.size - 1)])


def mean_absolute_error(counts: np.ndarray, cache: EnvelopeCache, factor: float) -> float:
    predicted = factor * counts / cache.audio_durations
    return float(np.mean(np.abs(predicted - cache.true_rates)))


@dataclass(frozen=True)
class Trial:
    label: str
    params: EnvelopeParams
    factor: float
    mae: float
    zero_count_ratio: float


def run_trial(cache: EnvelopeCache, params: EnvelopeParams, label: str) -> Trial:
    counts = syllable_counts(cache, params)
    factor = fit_conversion_factor(counts, cache)
    params = params.replace(mora_per_syllable=round(factor, 4))
    return Trial(
        label=label,
        params=params,
        factor=round(factor, 4),
        mae=mean_absolute_error(counts, cache, params.mora_per_syllable),
        zero_count_ratio=float(np.mean(counts == 0.0)),
    )


def search_stage(
    cache: EnvelopeCache,
    base: EnvelopeParams,
    name: str,
    candidates: list[dict[str, object]],
) -> tuple[EnvelopeParams, list[Trial]]:
    """候補を順に試し、平均絶対誤差が最小のものを返す。"""
    trials: list[Trial] = []
    for changes in candidates:
        label = ", ".join(f"{key}={value}" for key, value in changes.items()) or "(既定)"
        try:
            params = base.replace(**changes)
        except ValueError as error:
            log(f"  {name}: {label} は不正なので飛ばす（{error}）")
            continue
        trial = run_trial(cache, params, label)
        trials.append(trial)
        log(f"  {name}: {label} -> MAE={trial.mae:.4f}, k={trial.factor}")
    best = min(trials, key=lambda trial: trial.mae)
    log(f"{name} の最良: {best.label} (MAE={best.mae:.4f})")
    return best.params, trials


def constant_baseline_mae(cache: EnvelopeCache) -> tuple[float, float]:
    """常に同じ値を答える場合の平均絶対誤差（比較の下敷き）。最小は中央値。"""
    value = float(np.median(cache.true_rates))
    return value, float(np.mean(np.abs(cache.true_rates - value)))


def stages_for(method: str) -> list[tuple[str, list[dict[str, object]]]]:
    """数え方ごとの探索段階。その数え方に効かないパラメータは候補に入れない。

    突出量（``peak_prominence``）はピークを数える場合にしか使われないため、
    ゼロ交差の枝では動かさない（同じ MAE の候補が並ぶだけで意味がないため）。

    帯域の候補は docs/PLAN.md 3-2 の「2Hzから10Hzの帯域で変動成分を取り出す」の
    内側に限る。これより広い帯域も試したが（``REFERENCE_BANDS``）、採用しない。
    閾値の類は範囲の端に最小点が貼り付かないよう、両側に余裕を持たせてある。
    """
    stages: list[tuple[str, list[dict[str, object]]]] = [
        (
            "帯域",
            [
                {"band_low_hz": low, "band_high_hz": high}
                for low in (2.0, 2.5, 3.0)
                for high in (8.0, 9.0, 10.0)
            ],
        ),
    ]
    if method == "peak":
        stages.append(
            (
                "突出量と平滑化",
                [
                    {"peak_prominence": prominence, "smoothing_sec": smoothing}
                    for prominence in (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
                    for smoothing in (0.0, 0.02, 0.03, 0.05)
                ],
            )
        )
    else:
        stages.append(
            ("平滑化", [{"smoothing_sec": smoothing} for smoothing in (0.0, 0.02, 0.03, 0.05)])
        )
    stages.extend(
        [
            (
                "無音の閾値",
                [
                    {"silence_floor_db": value}
                    for value in (-50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0)
                ],
            ),
            (
                "最小間隔",
                [
                    {"min_peak_distance_sec": value}
                    for value in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
                ],
            ),
        ]
    )
    return stages


# 参考として試す、docs/PLAN.md 3-2 の指定（2Hz〜10Hz）より広い帯域。採用はしない。
REFERENCE_BANDS: tuple[tuple[float, float], ...] = (
    (2.0, 12.0),
    (2.0, 14.0),
    (2.0, 16.0),
    (1.5, 10.0),
    (1.25, 12.0),
    (1.0, 16.0),
)

# 座標降下法を何巡させるか。1巡目で選んだ値が後の段階に依存する場合があるため、
# 2巡目で変化が無くなることを確かめる。
NUM_PASSES = 2


@dataclass
class Branch:
    """ある数え方について探索した結果。"""

    method: str
    final: Trial
    stages: list[tuple[str, list[Trial]]]


def tune(args: argparse.Namespace) -> None:
    segments = load_dev_segments()
    log(f"dev の全クリップ数: {len(segments)}")
    subset = sample_segments(segments, args.size, args.seed)
    log(f"調整用の部分集合: {len(subset)}件 (seed={args.seed})")

    if args.use_cache and CACHE_PATH.exists():
        cache = load_cache(CACHE_PATH)
        log(f"包絡のキャッシュを読んだ: {CACHE_PATH}")
    else:
        cache = build_cache(subset, args.workers)
        save_cache(cache, CACHE_PATH)
        log(f"包絡のキャッシュを書いた: {CACHE_PATH}")

    constant_value, constant_mae = constant_baseline_mae(cache)
    log(f"常に中央値({constant_value:.3f})と答える場合の MAE = {constant_mae:.4f}")

    # 「ピークを数える」と「ゼロ交差を数える」を別々に最後まで調整してから比べる。
    # 既定値のまま一度に比べると、片方に不利な初期値で優劣が決まってしまうため。
    branches: list[Branch] = []
    for method in ("peak", "zero_cross"):
        log(f"=== 数え方 {method} の探索 ===")
        params = BASE_PARAMS.replace(count_method=method)
        stage_trials: list[tuple[str, list[Trial]]] = []
        for pass_index in range(1, NUM_PASSES + 1):
            for name, candidates in stages_for(method):
                label = f"{method} / {pass_index}巡目 / {name}"
                log(label)
                params, trials = search_stage(cache, params, label, candidates)
                stage_trials.append((label, trials))
        final = run_trial(cache, params, f"{method} の最終")
        log(f"=== 数え方 {method} の最終: MAE={final.mae:.4f}, k={final.factor} ===")
        branches.append(Branch(method=method, final=final, stages=stage_trials))

    best = min(branches, key=lambda branch: branch.final.mae)
    log(f"採用した数え方: {best.method}")
    log(f"選んだパラメータ: {best.final.params.as_dict()}")
    log(f"部分集合での MAE = {best.final.mae:.4f}, k = {best.final.factor}")

    # 指定より広い帯域も測っておく（採用はしない。差が小さいことを記録に残すため）。
    reference: list[Trial] = []
    for low, high in REFERENCE_BANDS:
        trial = run_trial(
            cache,
            best.final.params.replace(band_low_hz=low, band_high_hz=high),
            f"band_low_hz={low}, band_high_hz={high}",
        )
        reference.append(trial)
        log(f"参考（不採用）: {trial.label} -> MAE={trial.mae:.4f}, k={trial.factor}")

    write_config(best.final.params, args)
    write_tuning_report(best, branches, cache, constant_value, constant_mae, args, reference)


def write_config(params: EnvelopeParams, args: argparse.Namespace) -> None:
    import yaml

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 信号処理ベースライン（docs/PLAN.md 第3段階 3-2）のパラメータ。\n"
        "# 実装: src/spkrate/baselines/envelope.py\n"
        "# 生成: scripts/run_envelope_baseline.py tune\n"
        f"# 調整に使った集合: dev から seed={args.seed} で抽出した {args.size} 件\n"
        "# 調整の記録: results/envelope_baseline_tuning.md\n"
        "# テストセットは調整に使っていない。\n"
    )
    body = yaml.safe_dump(
        {"params": params.as_dict()}, allow_unicode=True, sort_keys=True, default_flow_style=False
    )
    CONFIG_PATH.write_text(header + body, encoding="utf-8")
    log(f"設定を書いた: {CONFIG_PATH}")


def write_tuning_report(
    best: Branch,
    branches: list[Branch],
    cache: EnvelopeCache,
    constant_value: float,
    constant_mae: float,
    args: argparse.Namespace,
    reference: list[Trial],
) -> None:
    final = best.final
    lines: list[str] = []
    lines.append("# 信号処理ベースラインの調整（docs/PLAN.md 第3段階 3-2）")
    lines.append("")
    lines.append("## 調整の条件")
    lines.append("")
    lines.append("- 実装: `src/spkrate/baselines/envelope.py`")
    lines.append("- 実行: `scripts/run_envelope_baseline.py tune`")
    lines.append(
        f"- 対象: 検証セット（dev）から固定シード `{args.seed}` で無作為抽出した **{len(cache.segment_ids)}件**"
    )
    lines.append("- テストセット（`configs/splits/test.json`）は一切使っていない")
    lines.append("- 指標は毎秒モーラ数の平均絶対誤差（MAE）。小さいほど良い")
    lines.append(
        f"- 比較の下敷き: 常に中央値 {constant_value:.3f} と答えた場合の MAE = **{constant_mae:.4f}**"
    )
    lines.append("")
    lines.append("## 探索の方法")
    lines.append("")
    lines.append(
        "包絡の窓長25ミリ秒・移動10ミリ秒（docs/spec.md の特徴量と同じ刻み）は固定し、"
        "残りを段階ごとに順に決める座標降下法で探索した。全ての組み合わせを試すと"
        "候補数が数千になり、部分集合の復号を何度も繰り返すことになるためである。"
        "各段階では前段階までに選んだ値を固定し、その段階のパラメータだけを動かす。"
    )
    lines.append("")
    lines.append(
        "「ピークを数える」と「正方向のゼロ交差を数える」は、どちらも最後まで別々に"
        "調整してから比べた。既定値のまま一度に比べると、片方に不利な初期値のまま"
        "優劣が決まってしまうためである。"
    )
    lines.append("")
    lines.append(
        "換算係数 `mora_per_syllable` は各候補ごとに、その候補の音節数に対して"
        "平均絶対誤差を最小にする値を厳密に求めている（誤差は換算係数について"
        "区分線形かつ凸なので、重み付き中央値が最小点になる）。したがって下表の MAE は"
        "いずれも「その設定で最良の換算係数を使ったときの値」である。"
    )
    lines.append("")
    lines.append("## 数え方ごとの最終結果")
    lines.append("")
    lines.append("| 数え方 | 換算係数 | 部分集合での MAE |")
    lines.append("| --- | --- | --- |")
    for branch in sorted(branches, key=lambda branch: branch.final.mae):
        mark = " ←採用" if branch.method == best.method else ""
        lines.append(
            f"| {branch.method}{mark} | {branch.final.factor} | {branch.final.mae:.4f} |"
        )
    lines.append("")
    lines.append("## 探索した範囲と各候補の平均絶対誤差")
    lines.append("")
    for branch in branches:
        for name, trials in branch.stages:
            stage_best = min(trials, key=lambda trial: trial.mae)
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| 候補 | 換算係数 | MAE |")
            lines.append("| --- | --- | --- |")
            for trial in sorted(trials, key=lambda trial: trial.mae):
                mark = " ←採用" if trial.label == stage_best.label else ""
                lines.append(f"| {trial.label}{mark} | {trial.factor} | {trial.mae:.4f} |")
            lines.append("")
    lines.append("## 選んだ値")
    lines.append("")
    lines.append("| パラメータ | 値 |")
    lines.append("| --- | --- |")
    for key, value in final.params.as_dict().items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append(
        f"- 部分集合での MAE = **{final.mae:.4f}**（常に中央値と答える場合は {constant_mae:.4f}）"
    )
    lines.append(f"- 音節が1つも検出されなかったクリップの割合 = {final.zero_count_ratio * 100:.2f}%")
    lines.append("- 設定ファイル: `configs/baselines/envelope.yaml`")
    lines.append("")
    lines.append("## 参考: 指定より広い帯域（不採用）")
    lines.append("")
    lines.append(
        "帯域の候補は docs/PLAN.md 3-2 の「2Hzから10Hz」の内側に限って選んだ。"
        "その外側も測っておいたが、改善はわずかで、手法の定義を指定から外してまで"
        "得る価値は無いと判断して採用しなかった。"
    )
    lines.append("")
    lines.append("| 帯域 | 換算係数 | MAE | 採用値との差 |")
    lines.append("| --- | --- | --- | --- |")
    for trial in sorted(reference, key=lambda trial: trial.mae):
        lines.append(
            f"| {trial.label} | {trial.factor} | {trial.mae:.4f} | {trial.mae - final.mae:+.4f} |"
        )
    lines.append("")
    lines.append("## 最終評価の条件")
    lines.append("")
    lines.append("（`scripts/run_envelope_baseline.py eval` の実行後に追記する）")
    lines.append("")
    TUNING_MD.parent.mkdir(parents=True, exist_ok=True)
    TUNING_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"調整の記録を書いた: {TUNING_MD}")


# --------------------------------------------------------------------------------------
# 評価


def _load_audio_worker(path: str) -> tuple[np.ndarray, int]:
    return load_audio(path)


class BatchParallelLoader:
    """``runner.evaluate`` に渡す音声読み込み。復号だけを複数プロセスで行う。

    ``evaluate`` は区間の順に ``loader(path)`` を呼ぶので、同じ順に一定数ずつ
    まとめて並列復号し、その結果を1件ずつ返す。推定器の実行は主プロセスのままなので、
    測定される1推論あたりの処理時間には影響しない。
    """

    def __init__(self, segments: list[EvalSegment], workers: int, batch_size: int = 256) -> None:
        self._paths = [str(segment.audio_path) for segment in segments]
        self._batch_size = batch_size
        self._pool = mp.Pool(workers)
        self._buffer: list[tuple[np.ndarray, int]] = []
        self._next = 0
        self._served = 0
        self._started = time.perf_counter()

    def __call__(self, path) -> tuple[np.ndarray, int]:
        if not self._buffer:
            chunk = self._paths[self._next : self._next + self._batch_size]
            if not chunk:
                raise RuntimeError("読み込む音声が残っていない")
            self._buffer = list(self._pool.map(_load_audio_worker, chunk, chunksize=8))
            self._next += len(chunk)
            elapsed = time.perf_counter() - self._started
            done = self._next
            total = len(self._paths)
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 else float("nan")
            log(f"  復号 {done}/{total} 件, 経過 {elapsed / 60:.1f}分, 残り約 {remaining / 60:.1f}分")
        self._served += 1
        return self._buffer.pop(0)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()


def timeit(args: argparse.Namespace) -> None:
    """少数のクリップで所要時間を測り、dev 全件の見積もりを出す。"""
    segments = load_dev_segments()
    subset = sample_segments(segments, args.size, args.seed)
    estimator = EnvelopeSpeedEstimator(load_params(CONFIG_PATH) if CONFIG_PATH.exists() else BASE_PARAMS)

    decode_started = time.perf_counter()
    loader = BatchParallelLoader(subset, args.workers, batch_size=len(subset))
    loaded = [loader(segment.audio_path) for segment in subset]
    loader.close()
    decode_elapsed = time.perf_counter() - decode_started

    infer_started = time.perf_counter()
    for samples, sample_rate in loaded:
        estimator(samples, sample_rate)
    infer_elapsed = time.perf_counter() - infer_started

    count = len(subset)
    total_per_clip = (decode_elapsed + infer_elapsed) / count
    log(f"計測: {count}件, 並列{args.workers}")
    log(f"  復号+再標本化(並列): {decode_elapsed:.1f}秒 = {decode_elapsed / count * 1000:.1f}ミリ秒/件")
    log(f"  推定(逐次): {infer_elapsed:.2f}秒 = {infer_elapsed / count * 1000:.2f}ミリ秒/件")
    log(f"  合計 {total_per_clip * 1000:.1f}ミリ秒/件")
    log(f"  dev 全件({len(segments)}件)の見積もり: {total_per_clip * len(segments) / 60:.1f}分")


def evaluate_split(args: argparse.Namespace) -> None:
    params = load_params(CONFIG_PATH)
    log(f"パラメータ: {params.as_dict()}")
    segments = load_dev_segments()
    total_dev = len(segments)
    if args.size and args.size < total_dev:
        segments = sample_segments(segments, args.size, args.seed)
        experiment_id = f"{args.experiment_id}-subset{len(segments)}-seed{args.seed}"
        note = f"dev から seed={args.seed} で抽出した {len(segments)}/{total_dev} 件"
    else:
        experiment_id = args.experiment_id
        note = f"dev 全件 {total_dev} 件"
    log(f"評価対象: {note}")

    loader = BatchParallelLoader(segments, args.workers, batch_size=args.batch_size)
    started = time.perf_counter()
    try:
        result = run_and_record(
            EnvelopeSpeedEstimator(params),
            segments,
            experiment_id=experiment_id,
            method="envelope-baseline (音量包絡の帯域通過+ピーク計数)",
            config_path=str(CONFIG_PATH.relative_to(ROOT)),
            split="dev",
            csv_path=METRICS_CSV,
            repo_dir=ROOT,
            audio_loader=loader,
        )
    finally:
        loader.close()
    elapsed = time.perf_counter() - started

    metrics = result.metrics.as_dict()
    log(f"評価が終わった: {elapsed / 60:.1f}分")
    for key, value in metrics.items():
        log(f"  {key} = {value}")
    log(f"  latency_ms_per_inference = {result.latency_ms_per_inference:.3f}")
    log(f"metrics.csv に追記した: {METRICS_CSV}")

    summary = {
        "experiment_id": experiment_id,
        "note": note,
        "elapsed_min": round(elapsed / 60, 2),
        "latency_ms_per_inference": result.latency_ms_per_inference,
        **metrics,
    }
    Path("/tmp/envelope_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    tune_parser = sub.add_parser("tune", help="dev の部分集合でパラメータを探索する")
    tune_parser.add_argument("--size", type=int, default=TUNE_SIZE)
    tune_parser.add_argument("--seed", type=int, default=TUNE_SEED)
    tune_parser.add_argument("--workers", type=int, default=8)
    tune_parser.add_argument("--use-cache", action="store_true", help="包絡のキャッシュを再利用する")
    tune_parser.set_defaults(func=tune)

    time_parser = sub.add_parser("timeit", help="1クリップあたりの所要時間を測る")
    time_parser.add_argument("--size", type=int, default=200)
    time_parser.add_argument("--seed", type=int, default=7)
    time_parser.add_argument("--workers", type=int, default=8)
    time_parser.set_defaults(func=timeit)

    eval_parser = sub.add_parser("eval", help="dev を評価して metrics.csv に追記する")
    eval_parser.add_argument("--size", type=int, default=0, help="0 なら dev 全件")
    eval_parser.add_argument("--seed", type=int, default=TUNE_SEED)
    eval_parser.add_argument("--workers", type=int, default=8)
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--experiment-id", default="003-envelope-baseline")
    eval_parser.set_defaults(func=evaluate_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
