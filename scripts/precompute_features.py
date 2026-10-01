"""特徴量の事前計算（docs/PLAN.md 第4段階 4-1）。

学習（train）・検証（dev）の全クリップについて、docs/spec.md の対数メルスペクトログラム
（``spkrate.features.melspec``）を計算し ``data/processed/features/`` へ保存する。
保存形式と読み出し方は docs/decisions/004-feature-storage.md を参照。

## 保存の形

``data/processed/features/<split>/shard_XXXX.npy``
    float16、形は ``(そのシャードの全フレーム数, 80)``。時間方向が第0軸。
    クリップの特徴量は連続した行として並ぶ。
``data/processed/features/<split>/shard_XXXX.json``
    そのシャードの索引。``clips`` に ``{clip_id, client_id, offset, n_frames, mora,
    duration_sec, audio_duration_sec}`` が並ぶ。``failures`` に読めなかったクリップ。
``data/processed/features/<split>/manifest.json``
    分割全体の集計（クリップ数、フレーム数、容量、失敗件数、メルの設定）。

シャードは clips.jsonl の並び順を ``--shard-size`` 件ずつに切ったもので、
同じ入力なら常に同じ区切りになる。**中断・再開**は、npy と json の両方が揃った
シャードを飛ばすことで行う（書き込みは一時ファイル → rename で原子的に行う）。

## 使い方

    # 100件で所要時間と1件あたりの容量を実測する（本実行の見積もり）
    uv run python scripts/precompute_features.py estimate --size 100

    # 本実行（バックグラウンド）
    uv run python scripts/precompute_features.py run --splits train dev

    # 学習データ全体から正規化の平均と標準偏差を求め configs/normalization.yaml を書く
    uv run python scripts/precompute_features.py stats

``stats`` は **train 分割のみ**から計算する（docs/spec.md「学習データ全体から算出した
平均と標準偏差を固定値として使う」）。dev は含めない。

## テストセットについて

``configs/splits/test.json`` は docs/PLAN.md 第10段階（人間の指示 2026-10-02）で使う。
既定の ``--splits`` は train dev のまま。test は ``--splits test`` と明示した場合だけ処理し、
``data/processed/features/test/`` に出す。正規化の値は train 由来の configs/normalization.yaml
の固定値のままで、test から再計算しない（``stats`` は train のみ）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.data.splits import load_clip_records  # noqa: E402
from spkrate.eval.split_profile import load_split_ids  # noqa: E402
from spkrate.eval.audio import load_audio  # noqa: E402
from spkrate.features.melspec import MEL_DEFAULTS, LogMelSpectrogram  # noqa: E402

CLIPS_JSONL = ROOT / "data" / "processed" / "clips.jsonl"
SPLITS_DIR = ROOT / "configs" / "splits"
FEATURES_DIR = ROOT / "data" / "processed" / "features"
NORMALIZATION_YAML = ROOT / "configs" / "normalization.yaml"

ALLOWED_SPLITS = ("train", "dev")
# 第10段階（2026-10-02 の人間の指示）で test を明示指定の場合に限り扱えるようにした。既定は train・dev のまま。
# test でも正規化の値は configs/normalization.yaml の固定値（train 由来）を使い、test から再計算しない
EXPLICIT_SPLITS = ("test",)
SELECTABLE_SPLITS = ALLOWED_SPLITS + EXPLICIT_SPLITS
DEFAULT_SHARD_SIZE = 1000
STORAGE_DTYPE = np.float16


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}時間{minutes:02d}分{secs:02d}秒"


# --------------------------------------------------------------------------------------
# クリップ一覧とシャードの決め方


@dataclass(frozen=True)
class ClipTask:
    clip_id: str
    client_id: str
    audio_path: str
    mora: int
    duration_sec: float


_TASK_CACHE: dict[str, list[ClipTask]] = {}


def load_split_tasks(split: str) -> list[ClipTask]:
    """clips.jsonl を読み、その分割に属するクリップを**ファイルの並び順のまま**返す。

    clips.jsonl は147MiBあり読み込みに時間がかかるので、分割ごとに結果を覚えておく。
    """
    if split in _TASK_CACHE:
        return _TASK_CACHE[split]
    if split not in SELECTABLE_SPLITS:
        raise ValueError(f"扱えない分割: {split}（train・dev・test のみ）")
    client_ids = set(load_split_ids(SPLITS_DIR / f"{split}.json", allow_test=split in EXPLICIT_SPLITS))
    tasks: list[ClipTask] = []
    for record in load_clip_records(CLIPS_JSONL):
        if record.client_id in client_ids:
            tasks.append(
                ClipTask(
                    clip_id=record.clip_id,
                    client_id=record.client_id,
                    audio_path=record.audio_path,
                    mora=record.mora,
                    duration_sec=record.duration_sec,
                )
            )
    _TASK_CACHE[split] = tasks
    return tasks


def shard_paths(split: str, index: int) -> tuple[Path, Path]:
    directory = FEATURES_DIR / split
    return directory / f"shard_{index:04d}.npy", directory / f"shard_{index:04d}.json"


def shard_is_done(split: str, index: int) -> bool:
    """npy と json の両方が揃っていれば計算済みとみなす（再開用）。"""
    npy_path, json_path = shard_paths(split, index)
    if not (npy_path.exists() and json_path.exists()):
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("complete"))


# --------------------------------------------------------------------------------------
# 特徴量の計算（ワーカー）

_TRANSFORM: LogMelSpectrogram | None = None


def get_transform() -> LogMelSpectrogram:
    global _TRANSFORM
    if _TRANSFORM is None:
        import torch

        # プロセスを並べるので、1プロセスあたりのスレッドは1本に絞る。
        torch.set_num_threads(1)
        _TRANSFORM = LogMelSpectrogram(MEL_DEFAULTS)
    return _TRANSFORM


def compute_one(task: ClipTask) -> tuple[np.ndarray, float]:
    """1クリップの対数メルスペクトログラム（float16）と音声長（秒）を返す。"""
    samples, sample_rate = load_audio(ROOT / task.audio_path)
    feature = get_transform()(samples, sample_rate)
    return feature.astype(STORAGE_DTYPE), float(len(samples)) / float(sample_rate)


def process_shard(job: tuple[str, int, list[ClipTask]]) -> dict:
    """1シャード分を計算して npy と json を書く。戻り値は集計。"""
    split, index, tasks = job
    npy_path, json_path = shard_paths(split, index)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    arrays: list[np.ndarray] = []
    clips: list[dict] = []
    failures: list[dict] = []
    offset = 0
    for task in tasks:
        try:
            feature, audio_duration = compute_one(task)
        except Exception as error:  # 壊れた mp3 などは飛ばして続ける
            failures.append(
                {
                    "clip_id": task.clip_id,
                    "audio_path": task.audio_path,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        arrays.append(feature)
        clips.append(
            {
                "clip_id": task.clip_id,
                "client_id": task.client_id,
                "offset": offset,
                "n_frames": int(feature.shape[0]),
                "mora": task.mora,
                "duration_sec": task.duration_sec,
                "audio_duration_sec": round(audio_duration, 3),
            }
        )
        offset += int(feature.shape[0])

    if arrays:
        stacked = np.concatenate(arrays, axis=0)
    else:
        stacked = np.zeros((0, MEL_DEFAULTS.n_mels), dtype=STORAGE_DTYPE)
    # np.save は拡張子が .npy でないと .npy を足すので、一時名も .npy で終わらせる。
    tmp_npy = npy_path.with_name(npy_path.stem + ".tmp.npy")
    np.save(tmp_npy, stacked)
    os.replace(tmp_npy, npy_path)

    payload = {
        "complete": True,
        "split": split,
        "shard": index,
        "dtype": str(np.dtype(STORAGE_DTYPE)),
        "n_mels": MEL_DEFAULTS.n_mels,
        "num_frames": int(stacked.shape[0]),
        "num_clips": len(clips),
        "num_failed": len(failures),
        "array": npy_path.name,
        "clips": clips,
        "failures": failures,
    }
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_json, json_path)

    return {
        "split": split,
        "shard": index,
        "num_clips": len(clips),
        "num_failed": len(failures),
        "num_frames": int(stacked.shape[0]),
        "bytes": npy_path.stat().st_size,
        "elapsed_sec": time.perf_counter() - started,
        "failures": failures,
    }


# --------------------------------------------------------------------------------------
# 見積もり


def command_estimate(args: argparse.Namespace) -> None:
    """少数のクリップで所要時間と1件あたりの容量を実測し、全体を見積もる。"""
    tasks = load_split_tasks("train")
    step = max(1, len(tasks) // args.size)
    sample = tasks[::step][: args.size]
    log(f"train {len(tasks)}件から {len(sample)}件を等間隔で抽出して実測する")

    started = time.perf_counter()
    frames = 0
    audio_sec = 0.0
    failures = 0
    for task in sample:
        try:
            feature, duration = compute_one(task)
        except Exception as error:
            failures += 1
            log(f"失敗: {task.clip_id}: {type(error).__name__}: {error}")
            continue
        frames += int(feature.shape[0])
        audio_sec += duration
    elapsed = time.perf_counter() - started
    ok = len(sample) - failures
    if ok == 0:
        log("全件失敗した。見積もれない")
        return

    per_clip_sec = elapsed / ok
    frames_per_clip = frames / ok
    bytes_float16 = frames * MEL_DEFAULTS.n_mels * 2
    bytes_float32 = frames * MEL_DEFAULTS.n_mels * 4

    totals = {split: len(load_split_tasks(split)) for split in ALLOWED_SPLITS}
    total_clips = sum(totals.values())
    scale = total_clips / ok
    log(f"実測: {ok}件, {elapsed:.1f}秒, 1件 {per_clip_sec * 1000:.0f}ミリ秒（1プロセス）")
    log(f"実測: 平均 {frames_per_clip:.1f}フレーム/件, 音声 {audio_sec / ok:.3f}秒/件")
    log(
        f"1件あたりの容量: float16 {bytes_float16 / ok / 1024:.1f}KiB, "
        f"float32 {bytes_float32 / ok / 1024:.1f}KiB"
    )
    log(f"対象クリップ数: {totals} 合計 {total_clips}件")
    log(
        f"全体の見積もり容量: float16 {bytes_float16 * scale / 1024**3:.1f}GiB, "
        f"float32 {bytes_float32 * scale / 1024**3:.1f}GiB"
    )
    for workers in (1, args.workers):
        log(
            f"全体の見積もり時間（{workers}並列）: "
            f"{format_duration(per_clip_sec * total_clips / workers)}"
        )


# --------------------------------------------------------------------------------------
# 本実行


def command_run(args: argparse.Namespace) -> None:
    splits = list(args.splits)
    for split in splits:
        if split not in SELECTABLE_SPLITS:
            raise SystemExit(f"扱えない分割: {split}（train・dev・test のみ。test は明示指定）")

    jobs: list[tuple[str, int, list[ClipTask]]] = []
    planned: dict[str, int] = {}
    skipped_shards = 0
    for split in splits:
        tasks = load_split_tasks(split)
        planned[split] = len(tasks)
        shards = [
            tasks[start : start + args.shard_size]
            for start in range(0, len(tasks), args.shard_size)
        ]
        log(f"{split}: {len(tasks)}件 → {len(shards)}シャード（{args.shard_size}件ごと）")
        for index, shard in enumerate(shards):
            if shard_is_done(split, index):
                skipped_shards += 1
                continue
            jobs.append((split, index, shard))

    remaining_clips = sum(len(job[2]) for job in jobs)
    log(
        f"計算対象: {len(jobs)}シャード / {remaining_clips}件"
        f"（計算済みで飛ばすシャード: {skipped_shards}）"
    )
    if not jobs:
        log("計算済み。manifest を書き直して終了する")
        for split in splits:
            write_manifest(split, args.shard_size)
        return

    started = time.perf_counter()
    done_clips = 0
    done_shards = 0
    total_frames = 0
    total_bytes = 0
    total_failed = 0
    failure_log = FEATURES_DIR / "failures.jsonl"
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(process_shard, jobs):
            done_shards += 1
            done_clips += result["num_clips"] + result["num_failed"]
            total_frames += result["num_frames"]
            total_bytes += result["bytes"]
            total_failed += result["num_failed"]
            if result["failures"]:
                with open(failure_log, "a", encoding="utf-8") as handle:
                    for failure in result["failures"]:
                        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            elapsed = time.perf_counter() - started
            rate = done_clips / elapsed if elapsed > 0 else 0.0
            eta = (remaining_clips - done_clips) / rate if rate > 0 else 0.0
            log(
                f"{result['split']} shard {result['shard']:04d} 完了 | "
                f"{done_clips}/{remaining_clips}件 "
                f"({done_clips / remaining_clips * 100:.1f}%) | "
                f"{rate:.1f}件/秒 | 経過 {format_duration(elapsed)} | "
                f"残り見込み {format_duration(eta)} | "
                f"累計 {total_bytes / 1024**3:.2f}GiB | 失敗 {total_failed}件"
            )

    elapsed = time.perf_counter() - started
    log(
        f"計算終了: {done_clips}件, {total_frames}フレーム, "
        f"{total_bytes / 1024**3:.2f}GiB, 失敗 {total_failed}件, "
        f"所要 {format_duration(elapsed)}"
    )
    for split in splits:
        manifest = write_manifest(split, args.shard_size)
        log(
            f"manifest: {split} clips={manifest['num_clips']} "
            f"frames={manifest['num_frames']} "
            f"size={manifest['total_bytes'] / 1024**3:.2f}GiB "
            f"failed={manifest['num_failed']}"
        )


def write_manifest(split: str, shard_size: int) -> dict:
    """分割ごとの manifest.json を書き、その中身を返す。"""
    directory = FEATURES_DIR / split
    shard_jsons = sorted(directory.glob("shard_*.json"))
    num_clips = 0
    num_frames = 0
    num_failed = 0
    total_bytes = 0
    shards: list[dict] = []
    for path in shard_jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            continue
        array_path = directory / payload["array"]
        size = array_path.stat().st_size if array_path.exists() else 0
        num_clips += payload["num_clips"]
        num_frames += payload["num_frames"]
        num_failed += payload["num_failed"]
        total_bytes += size
        shards.append(
            {
                "shard": payload["shard"],
                "array": payload["array"],
                "index": path.name,
                "num_clips": payload["num_clips"],
                "num_frames": payload["num_frames"],
                "bytes": size,
            }
        )

    manifest = {
        "split": split,
        "source": "data/processed/clips.jsonl",
        "generated_by": "scripts/precompute_features.py",
        "implementation": "src/spkrate/features/melspec.py",
        "dtype": str(np.dtype(STORAGE_DTYPE)),
        "layout": "shard_XXXX.npy は (フレーム数, n_mels)。クリップは連続した行",
        "shard_size": shard_size,
        "n_mels": MEL_DEFAULTS.n_mels,
        "sample_rate": MEL_DEFAULTS.sample_rate,
        "hop_length": MEL_DEFAULTS.hop_length,
        "n_fft": MEL_DEFAULTS.n_fft,
        "num_shards": len(shards),
        "num_clips": num_clips,
        "num_frames": num_frames,
        "num_failed": num_failed,
        "total_bytes": total_bytes,
        "shards": shards,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------------------
# 正規化の統計（train のみ）


def command_stats(args: argparse.Namespace) -> None:
    """train 分割の保存済み特徴量から平均と標準偏差を求め configs/normalization.yaml を書く。

    dev は含めない（docs/spec.md「学習データ全体から算出した平均と標準偏差」）。
    累積は float32 の配列で行う（CLAUDE.md「float64は使わない」）。numpy の総和は
    対分割（pairwise）で足すため、シャード内 45万点・シャード間 240点程度の規模では
    float32 でも相対誤差は 1e-6 を下回る。
    """
    directory = FEATURES_DIR / "train"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("train の manifest.json が無い。先に run を実行すること")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    n_mels = manifest["n_mels"]
    shard_sums = np.zeros((len(manifest["shards"]), n_mels), dtype=np.float32)
    shard_squares = np.zeros_like(shard_sums)
    frames_per_shard = np.zeros(len(manifest["shards"]), dtype=np.int64)
    minimum = np.full(n_mels, np.inf, dtype=np.float32)
    maximum = np.full(n_mels, -np.inf, dtype=np.float32)

    started = time.perf_counter()
    for position, shard in enumerate(manifest["shards"]):
        array = np.load(directory / shard["array"], mmap_mode="r")
        block = np.asarray(array, dtype=np.float32)
        shard_sums[position] = block.sum(axis=0, dtype=np.float32)
        shard_squares[position] = np.square(block).sum(axis=0, dtype=np.float32)
        frames_per_shard[position] = block.shape[0]
        if block.shape[0]:
            minimum = np.minimum(minimum, block.min(axis=0))
            maximum = np.maximum(maximum, block.max(axis=0))
        if (position + 1) % 25 == 0 or position + 1 == len(manifest["shards"]):
            log(
                f"統計: {position + 1}/{len(manifest['shards'])}シャード "
                f"経過 {format_duration(time.perf_counter() - started)}"
            )

    total_frames = int(frames_per_shard.sum())
    if total_frames == 0:
        raise SystemExit("フレームが0件。統計を取れない")
    total_sum = shard_sums.sum(axis=0, dtype=np.float32)
    total_square = shard_squares.sum(axis=0, dtype=np.float32)

    per_mel_mean = total_sum / np.float32(total_frames)
    per_mel_var = np.maximum(total_square / np.float32(total_frames) - per_mel_mean**2, 0.0)
    per_mel_std = np.sqrt(per_mel_var)

    global_mean = float(total_sum.sum(dtype=np.float32) / (total_frames * n_mels))
    global_var = max(
        float(total_square.sum(dtype=np.float32) / (total_frames * n_mels)) - global_mean**2,
        0.0,
    )
    global_std = float(np.sqrt(global_var))

    log(f"train: {total_frames}フレーム / {manifest['num_clips']}クリップ")
    log(f"全体で1組: mean={global_mean:.6f} std={global_std:.6f}")
    log(
        f"メル次元ごと: mean {per_mel_mean.min():.4f}〜{per_mel_mean.max():.4f}, "
        f"std {per_mel_std.min():.4f}〜{per_mel_std.max():.4f}"
    )

    # 抜き取りで検算する（対分割和の誤差確認）。
    rng = np.random.default_rng(20260921)
    picks = rng.choice(len(manifest["shards"]), size=min(5, len(manifest["shards"])), replace=False)
    sample = np.concatenate(
        [np.asarray(np.load(directory / manifest["shards"][p]["array"], mmap_mode="r"), dtype=np.float32) for p in picks]
    )
    log(
        f"抜き取り検算（{len(picks)}シャード, {sample.shape[0]}フレーム）: "
        f"mean={float(sample.mean()):.6f} std={float(sample.std()):.6f}"
    )

    write_normalization_yaml(
        mean_per_mel=per_mel_mean,
        std_per_mel=per_mel_std,
        global_mean=global_mean,
        global_std=global_std,
        total_frames=total_frames,
        num_clips=manifest["num_clips"],
        minimum=minimum,
        maximum=maximum,
        mode=args.mode,
    )
    log(f"書き出し: {NORMALIZATION_YAML}")


def write_normalization_yaml(
    *,
    mean_per_mel: np.ndarray,
    std_per_mel: np.ndarray,
    global_mean: float,
    global_std: float,
    total_frames: int,
    num_clips: int,
    minimum: np.ndarray,
    maximum: np.ndarray,
    mode: str,
) -> None:
    """configs/normalization.yaml を書く（pyyaml を使わず、読みやすい形で直に書く）。"""

    def vector(values: np.ndarray, indent: str = "  ") -> str:
        return "\n".join(f"{indent}- {float(value):.6f}" for value in values)

    text = f"""# 対数メルスペクトログラムの正規化（docs/spec.md「正規化」、docs/PLAN.md 第4段階 4-1）
#
# docs/spec.md:「学習データ全体から算出した平均と標準偏差を固定値として使う。
# 値はconfigsに記録する。発話ごとの正規化は行わない」
#
# 出所: data/processed/features/train（train 分割のみ。dev は含まない）
# 生成: uv run python scripts/precompute_features.py stats
# 特徴量: src/spkrate/features/melspec.py（n_fft=400, hop_length=160, n_mels=80,
#         f_min=0, f_max=8000, power=2.0, log(mel + 1e-6), 保存は float16）
#
# 適用: normalized = (log_mel - mean) / std
# mode が per_mel なら mean/std はメル次元ごとの80要素、global なら全体で1組の値。
# 両方を保存してあるので、実験で切り替えられる。

mode: {mode}

source:
  split: train
  features_dir: data/processed/features/train
  num_clips: {num_clips}
  num_frames: {total_frames}

global:
  mean: {global_mean:.6f}
  std: {global_std:.6f}

per_mel:
  n_mels: {len(mean_per_mel)}
  mean:
{vector(mean_per_mel, "    ")}
  std:
{vector(std_per_mel, "    ")}

# 参考値（正規化には使わない）。float16 で保存した値の範囲。
range:
  min: {float(minimum.min()):.6f}
  max: {float(maximum.max()):.6f}
"""
    NORMALIZATION_YAML.parent.mkdir(parents=True, exist_ok=True)
    NORMALIZATION_YAML.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="少数で実測し全体を見積もる")
    estimate.add_argument("--size", type=int, default=100)
    estimate.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    estimate.set_defaults(func=command_estimate)

    run = subparsers.add_parser("run", help="全クリップの特徴量を計算して保存する")
    run.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    run.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    run.set_defaults(func=command_run)

    stats = subparsers.add_parser("stats", help="train から正規化の平均と標準偏差を求める")
    stats.add_argument("--mode", choices=("per_mel", "global"), default="per_mel")
    stats.set_defaults(func=command_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
