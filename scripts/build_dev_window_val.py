"""方式Bの検証に使う dev_window の抽出窓と、その特徴量を作る（docs/decisions/009-method-b.md 1.7節）。

設定: configs/eval/dev_window_val.yaml（種・件数・dev_window の設定・除外の一覧・出力先）
実装: src/spkrate/eval/dev_window_val.py
出力（Git管理外）: data/processed/dev_window_val/ の windows.npz（窓の番号・正解・単一/連結の別）、
features.npy（float16、(件数, 201, 80)、正規化前の対数メル）、meta.json（設定・件数・進み具合）

- 窓の波形は ``DevWindowAudio.window_at``（clean）で再生成する。雑音下の版は作らない
- 窓は番号の昇順に、連続した塊（``--chunk`` 件）ごとにワーカーへ渡す（同じ音源の窓が続けば
  音源を1回だけ組み立てる）。塊ごとに features.npy へ書き、meta.json の ``computed`` を進める。
  途中で止めても、同じ設定で再実行すれば続きから計算する
- configs/splits/test.json は使わない。data/ 以下のうち書くのは出力先だけ

実行（リポジトリ直下から）:
    uv run python scripts/build_dev_window_val.py --limit 512        # 所要時間の見積り（先頭512件）
    uv run python scripts/build_dev_window_val.py > runs/dev_window_val/build.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spkrate.eval.dev_window_val import (  # noqa: E402
    config_as_dict,
    load_suspect_ids,
    load_val_config,
    main_indicator_mask,
    select_val_windows,
)

_STATE: dict = {}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def _init_worker(window_config_path: str) -> None:
    from spkrate.data.splits import load_split
    from spkrate.eval.dev_window import DevWindowAudio, load_dev_clips, load_dev_window, load_window_config
    from spkrate.features.melspec import LogMelSpectrogram

    config = load_window_config(window_config_path)
    ws = load_dev_window(_resolve(config.output_dir))
    _, paths = load_dev_clips(_resolve(config.clips_jsonl), load_split(_resolve(config.dev_split)))
    root = _resolve(config.audio_root)
    _STATE["ws"] = ws
    _STATE["audio"] = DevWindowAudio(config, {k: root / v for k, v in paths.items()})
    _STATE["mel"] = LogMelSpectrogram()
    _STATE["sr"] = config.sample_rate


def _compute_chunk(indices: np.ndarray) -> np.ndarray:
    ws, audio, mel, sr = _STATE["ws"], _STATE["audio"], _STATE["mel"], _STATE["sr"]
    out = []
    for index in indices:
        waveform = audio.window_at(ws, int(index))
        out.append(mel(waveform, sr).astype(np.float16))
    return np.stack(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval/dev_window_val.yaml")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None, help="先頭の件数だけ計算して止める（見積り用）")
    args = ap.parse_args(argv)

    from spkrate.eval.dev_window import load_dev_window, load_known_no_speech, load_window_config
    from spkrate.features.melspec import num_frames

    config = load_val_config(_resolve(args.config))
    window_config_path = str(_resolve(config.dev_window_config))
    wconfig = load_window_config(window_config_path)
    ws = load_dev_window(_resolve(wconfig.output_dir))
    known = load_known_no_speech(_resolve(wconfig.known_no_speech_list))
    suspect = load_suspect_ids(_resolve(config.suspect_list))
    mask = main_indicator_mask(ws, known | suspect)
    picked = select_val_windows(mask, config.num_windows, config.seed)
    frames = num_frames(wconfig.window_samples)
    n_mels = 80

    out = _resolve(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    kind_counts = Counter(int(k) for k in ws.kind[picked])
    base_meta = {
        "config": config_as_dict(config),
        "config_path": args.config,
        "dev_window_windows": int(len(ws)),
        "main_windows": int(mask.sum()),
        "known_no_speech_ids": len(known),
        "suspect_ids": len(suspect),
        "count": int(picked.size),
        "single": int(kind_counts.get(0, 0)),
        "concat": int(kind_counts.get(1, 0)),
        "zero_label": int((ws.mora[picked] == 0).sum()),
        "frames": int(frames),
        "n_mels": n_mels,
        "dtype": "float16",
        "features": "正規化前の対数メル（spkrate.features.melspec）",
    }

    # 再開: 同じ抽出で途中まで計算済みなら続きから
    computed = 0
    features_path = out / "features.npy"
    if meta_path.is_file() and features_path.is_file() and (out / "windows.npz").is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(out / "windows.npz") as arrays:
            same = np.array_equal(arrays["window_index"], picked)
        if same and old.get("config") == base_meta["config"]:
            computed = int(old.get("computed", 0))
    if computed == 0:
        np.savez(
            out / "windows.npz",
            window_index=picked,
            mora=ws.mora[picked].astype(np.float32),
            kind=ws.kind[picked].astype(np.int8),
            source_index=ws.source_index[picked].astype(np.int32),
        )
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.float16, shape=(picked.size, frames, n_mels)
        )
        del features
    features = np.load(features_path, mmap_mode="r+")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    def write_meta(done: int) -> None:
        payload = dict(base_meta, computed=int(done), complete=bool(done >= picked.size), commit=commit,
                       updated_at=datetime.now().isoformat(timespec="seconds"))
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(meta_path)

    target = picked.size if args.limit is None else min(picked.size, int(args.limit))
    log(f"母集団（主指標）{int(mask.sum()):,} / 全 {len(ws):,} 窓 → 抽出 {picked.size:,} 窓"
        f"（単一 {base_meta['single']:,}、連結 {base_meta['concat']:,}）。計算済み {computed:,}、今回 {target:,} まで")
    write_meta(computed)
    chunks = [np.arange(s, min(s + args.chunk, target)) for s in range(computed, target, args.chunk)]
    started = time.perf_counter()
    done = computed
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(window_config_path,)) as pool:
        for rows, block in zip(chunks, pool.imap(_compute_chunk, [picked[c] for c in chunks])):
            features[rows[0] : rows[-1] + 1] = block
            done = int(rows[-1] + 1)
            features.flush()
            write_meta(done)
            elapsed = time.perf_counter() - started
            rate = (done - computed) / elapsed
            log(f"{done:,}/{picked.size:,} 件 {rate:.1f} 件/秒 残り見込み {(picked.size - done) / rate / 60:.1f} 分")
    log(f"終了: {done:,}/{picked.size:,} 件（{time.perf_counter() - started:.1f} 秒）→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
