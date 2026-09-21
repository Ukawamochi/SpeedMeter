"""データ拡張と特徴量再計算のコスト実測（docs/PLAN.md 第4段階 4-2）。

docs/decisions/006-augmentation.md の「拡張を波形に適用するか特徴量に適用するか」を
決めるための実測を行う。測るのは1クリップあたりのミリ秒である。

    uv run python scripts/benchmark_augment.py --clips 20

測る項目。

- ``load``      : mp3 の復号と16kHzへの再標本化（``spkrate.eval.audio.load_audio``）
- 各拡張        : 時間伸縮・残響・雑音重畳・帯域制限・音量変化を個別に
- ``pipeline``  : ``augment_waveform`` ひとそろい（既定の適用確率）
- ``mel``       : 拡張後の波形から対数メルスペクトログラムを計算し直すコスト
- ``read_npy``  : 事前計算済み特徴量（float16 シャード）を読み出すコスト（比較用）

data/ は読み取りのみ。結果は標準出力に表とまとめを出す。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_clip_paths(num_clips: int) -> list[Path]:
    """clips.jsonl から等間隔に音声ファイルを選ぶ（先頭に偏らせない）。"""
    clips_path = REPO_ROOT / "data" / "processed" / "clips.jsonl"
    if not clips_path.is_file():
        raise SystemExit(f"clips.jsonl が無い: {clips_path}")
    with clips_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    step = max(1, len(records) // num_clips)
    selected = records[::step][:num_clips]
    return [REPO_ROOT / record["audio_path"] for record in selected]


def summarize(name: str, durations_ms: list[float]) -> str:
    return (
        f"{name:<14} 平均 {statistics.mean(durations_ms):7.2f} ms "
        f"中央 {statistics.median(durations_ms):7.2f} ms "
        f"最大 {max(durations_ms):7.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=20, help="測るクリップ数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from spkrate.data.augment import (
        AugmentConfig,
        MusanNoiseSource,
        add_noise,
        apply_reverb,
        augment_waveform,
        band_limit,
        change_volume,
        generate_rir,
        time_stretch,
    )
    from spkrate.eval.audio import load_audio
    from spkrate.features.melspec import log_mel_spectrogram

    rng = np.random.default_rng(args.seed)
    paths = load_clip_paths(args.clips)
    musan_root = REPO_ROOT / "data" / "musan"
    noise_source = MusanNoiseSource(musan_root) if (musan_root / "noise").is_dir() else None
    config = AugmentConfig()

    timings: dict[str, list[float]] = {
        key: []
        for key in (
            "load",
            "time_stretch",
            "reverb(rir生成)",
            "reverb(畳み込み)",
            "noise",
            "band_limit",
            "volume",
            "pipeline",
            "mel",
        )
    }
    durations: list[float] = []

    # 最初の1件は numba などの初期化を含むので捨てる。
    warm, _ = load_audio(paths[0])
    time_stretch(warm, 1.2)
    log_mel_spectrogram(warm)
    if noise_source is not None:
        noise_source.sample(warm.size, rng)

    for path in paths:
        start = time.perf_counter()
        samples, _ = load_audio(path)
        timings["load"].append((time.perf_counter() - start) * 1000.0)
        durations.append(samples.size / 16000.0)

        start = time.perf_counter()
        stretched = time_stretch(samples, 1.3)
        timings["time_stretch"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        rir = generate_rir(rng, rt60_range=config.rt60_range, max_order=config.reverb_max_order)
        timings["reverb(rir生成)"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        apply_reverb(samples, rir)
        timings["reverb(畳み込み)"].append((time.perf_counter() - start) * 1000.0)

        if noise_source is not None:
            noise = noise_source.sample(samples.size, rng)
            start = time.perf_counter()
            add_noise(samples, noise, 10.0)
            timings["noise"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        band_limit(samples, low_hz=200.0, high_hz=3400.0)
        timings["band_limit"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        change_volume(samples, -6.0)
        timings["volume"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        result = augment_waveform(samples, rng, config=config, noise_source=noise_source)
        timings["pipeline"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        log_mel_spectrogram(result.samples)
        timings["mel"].append((time.perf_counter() - start) * 1000.0)

        del stretched

    print(f"クリップ数 {len(paths)}、平均音声長 {statistics.mean(durations):.3f} 秒")
    print(f"MUSAN: {'あり' if noise_source is not None else '無し（雑音重畳は測っていない）'}")
    for name, values in timings.items():
        if values:
            print(summarize(name, values))

    # 比較: 事前計算済み特徴量の読み出し。
    features_dir = REPO_ROOT / "data" / "processed" / "features" / "train"
    manifest_path = features_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = manifest["shards"][: args.clips]
        read_ms: list[float] = []
        for shard in shards:
            index = json.loads((features_dir / shard["index"]).read_text(encoding="utf-8"))
            array = np.load(features_dir / shard["array"], mmap_mode="r")
            entry = index["clips"][0]
            start = time.perf_counter()
            np.asarray(
                array[entry["offset"] : entry["offset"] + entry["n_frames"]], dtype=np.float32
            )
            read_ms.append((time.perf_counter() - start) * 1000.0)
        print(summarize("read_npy", read_ms))


if __name__ == "__main__":
    main()
