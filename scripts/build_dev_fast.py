"""速めた音声の評価セット dev_fast を作り、data/processed/dev_fast/ に保存する
（docs/directives/2026-09-28-fast-speech.md タスク1 評価2）。

処理の本体は src/spkrate/eval/fast_speech.py、設定は configs/eval/dev_fast.yaml。

1. dev_window（data/processed/dev_window/）の音源から、種を固定して ``num_sources`` 件を選ぶ
2. 各音源の clean の波形（``DevWindowAudio.source_waveform``。dev_window と同じ組み立て）を
   速さの条件ごとに WSOLA（audiotsm）で速め、条件ごとの ``audio.npy`` に順に書く
3. モーラ時刻を 1/speed 倍にして dev_window と同じ規則で窓と正解を作り、
   ``<output_dir>/x<speed>/`` に sources.jsonl・windows.npz・meta.json を書く

``--verify N`` は保存済みの dev_fast から先頭寄り・末尾寄りの N 音源を作り直し、保存した波形と
ビット一致するか（と窓の定義が一致するか）を確かめるだけで、何も書かない。遠隔機
（ubuntu-desktop）に ``scripts/sync_to_remote.sh --data`` で送った後の確認や、遠隔機で同じ手順で
作れることの確認に使う。

configs/splits/test.json は使わない。data/ 以下の既存のファイルは読むだけで変更しない。

実行（リポジトリ直下から）:
    uv run python scripts/build_dev_fast.py > runs/dev_fast_build/build.log 2>&1
    uv run python scripts/build_dev_fast.py --verify 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from spkrate.data.splits import load_split  # noqa: E402
from spkrate.eval.dev_window import (  # noqa: E402
    DevWindowAudio,
    load_alignments,
    load_dev_clips,
    load_dev_window,
    load_window_config,
)
from spkrate.eval.fast_speech import (  # noqa: E402
    RATE_BIN_KEYS,
    build_fast_windows,
    condition_name,
    rate_bin_codes,
    select_sources,
    wsola_speed_up,
)

CONFIG = "configs/eval/dev_fast.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def load_fast_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def wsola_kwargs(cfg: dict) -> dict:
    w = cfg["wsola"]
    if cfg.get("method") != "wsola" or w.get("library") != "audiotsm":
        raise ValueError("dev_fast の方式は wsola（audiotsm）だけを実装している")
    return {"frame_length": int(w["frame_length"]), "synthesis_hop": int(w["synthesis_hop"]),
            "tolerance": int(w["tolerance"]), "pad_samples": int(w["pad_samples"])}


def prepare(cfg: dict):
    """(dev_window の設定, dev_window, アライメント, clip_id → 音声のパス, 選んだ音源の番号)。"""
    wcfg = load_window_config(_resolve(cfg["dev_window_config"]))
    ws = load_dev_window(_resolve(wcfg.output_dir))
    alignments = load_alignments(_resolve(wcfg.alignments))
    _, clip_paths = load_dev_clips(_resolve(wcfg.clips_jsonl), load_split(_resolve(wcfg.dev_split)))
    paths = {k: _resolve(Path(wcfg.audio_root) / v) for k, v in clip_paths.items()}
    numbers = select_sources(ws, seed=int(cfg["seed"]), num_sources=int(cfg["num_sources"]),
                             max_speed=max(float(s) for s in cfg["speeds"]),
                             window_samples=wcfg.window_samples)
    return wcfg, ws, alignments, paths, numbers


def window_counts(fast, window_sec: float) -> dict:
    rates = fast.mora / np.float32(window_sec)
    codes = rate_bin_codes(rates) if rates.size else np.zeros(0, dtype=np.int8)
    out = {"windows": len(fast), "single_windows": int((fast.kind == 0).sum()),
           "concat_windows": int((fast.kind == 1).sum()), "zero_windows": int((fast.mora == 0).sum()),
           "true_rate_mean": float(rates.mean()) if rates.size else float("nan"),
           "by_bin": {k: int((codes == i).sum()) for i, k in enumerate(RATE_BIN_KEYS)}}
    for name, code in (("single", 0), ("concat", 1)):
        c = codes[fast.kind == code]
        out[f"{name}_by_bin"] = {k: int((c == i).sum()) for i, k in enumerate(RATE_BIN_KEYS)}
    return out


def cmd_build(cfg: dict, args: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    wcfg, ws, alignments, paths, numbers = prepare(cfg)
    kwargs = wsola_kwargs(cfg)
    speeds = [float(s) for s in cfg["speeds"]]
    out_root = _resolve(cfg["output_dir"])
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"{out_root} が空でない（既存の dev_fast を上書きしない）")
    out_root.mkdir(parents=True, exist_ok=True)
    selected = [ws.sources[int(n)] for n in numbers]
    n_concat = sum(1 for s in selected if s.kind == "concat")
    log(f"selected sources={len(selected)} (single={len(selected) - n_concat}, concat={n_concat}) "
        f"clips={sum(len(s.clip_ids) for s in selected)} "
        f"audio_hours={sum(s.num_samples for s in selected) / wcfg.sample_rate / 3600:.3f} commit={_commit()}")
    (out_root / "selection.json").write_text(json.dumps({
        "seed": int(cfg["seed"]), "num_sources": int(cfg["num_sources"]),
        "dev_window_sources": len(ws.sources), "source_numbers": [int(n) for n in numbers],
        "source_ids": [s.source_id for s in selected]}, ensure_ascii=False, indent=1), encoding="utf-8")

    plans = {}
    for speed in speeds:
        fast, lengths = build_fast_windows(ws, numbers, alignments, speed=speed, window_sec=wcfg.window_sec,
                                           hop_sec=wcfg.hop_sec, sample_rate=wcfg.sample_rate)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        cond_dir = out_root / condition_name(speed)
        cond_dir.mkdir()
        audio = np.lib.format.open_memmap(cond_dir / "audio.npy", mode="w+", dtype=np.float32,
                                          shape=(int(lengths.sum()),))
        plans[speed] = {"fast": fast, "lengths": lengths, "offsets": offsets, "dir": cond_dir, "audio": audio,
                        "max_abs": 0.0, "over1": 0}
        log(f"{condition_name(speed)}: windows={len(fast)} audio_samples={int(lengths.sum())}")

    loader = DevWindowAudio(wcfg, paths)
    t_wave = t_wsola = 0.0
    for i, source in enumerate(selected):
        t0 = time.perf_counter()
        waveform = loader.source_waveform(source)
        t_wave += time.perf_counter() - t0
        for speed, plan in plans.items():
            t0 = time.perf_counter()
            y = wsola_speed_up(waveform, speed, **kwargs)
            t_wsola += time.perf_counter() - t0
            if y.size != plan["lengths"][i] or not np.isfinite(y).all():
                raise ValueError(f"{source.source_id} x{speed:g}: 長さか値が不正")
            o = int(plan["offsets"][i])
            plan["audio"][o:o + y.size] = y
            plan["max_abs"] = max(plan["max_abs"], float(np.abs(y).max()) if y.size else 0.0)
            plan["over1"] += int((np.abs(y) > 1.0).sum())
        if (i + 1) % 200 == 0 or i + 1 == len(selected):
            log(f"sources {i + 1}/{len(selected)} wave={t_wave:.0f}s wsola={t_wsola:.0f}s")

    versions = {"audiotsm": metadata.version("audiotsm"), "numpy": np.__version__}
    for speed, plan in plans.items():
        plan["audio"].flush()
        del plan["audio"]
        fast, cond_dir = plan["fast"], plan["dir"]
        with open(cond_dir / "sources.jsonl", "w", encoding="utf-8") as handle:
            for source, length, offset in zip(fast.sources, plan["lengths"], plan["offsets"], strict=True):
                row = source.to_json()
                row.update({"speed": speed, "stretched_samples": int(length), "audio_offset": int(offset)})
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.savez_compressed(cond_dir / "windows.npz", source_index=fast.source_index,
                            start_sample=fast.start_sample, mora=fast.mora, kind=fast.kind,
                            known_no_speech=fast.known_no_speech)
        meta = {"condition": condition_name(speed), "speed": speed, "stretch": 1.0 / speed,
                "config_path": args.config, "config": cfg, "dev_window_dir": wcfg.output_dir,
                "commit": _commit(), "created": datetime.now().isoformat(timespec="seconds"),
                "method": "WSOLA (audiotsm.wsola)", "versions": versions, "wsola": kwargs,
                "sources": len(fast.sources),
                "sources_without_windows": int(len(fast.sources) - np.unique(fast.source_index).size),
                "audio_samples": int(plan["lengths"].sum()),
                "audio_hours": float(plan["lengths"].sum() / wcfg.sample_rate / 3600),
                "audio_max_abs": plan["max_abs"], "audio_samples_abs_over_1": plan["over1"],
                "counts": window_counts(fast, wcfg.window_sec),
                "timing_sec": {"source_waveform": t_wave, "wsola_all_conditions": t_wsola}}
        (cond_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"{condition_name(speed)}: saved {cond_dir} {json.dumps(meta['counts'], ensure_ascii=False)}")
    log(f"BUILD_DONE elapsed={time.perf_counter() - t_start:.0f}s versions={json.dumps(versions)}")
    return 0


def cmd_verify(cfg: dict, args: argparse.Namespace) -> int:
    wcfg, ws, alignments, paths, numbers = prepare(cfg)
    kwargs = wsola_kwargs(cfg)
    out_root = _resolve(cfg["output_dir"])
    saved_ids = json.loads((out_root / "selection.json").read_text(encoding="utf-8"))["source_ids"]
    ids = [ws.sources[int(n)].source_id for n in numbers]
    if ids != saved_ids:
        raise SystemExit("選んだ音源が保存時と違う")
    log(f"selection identical: {len(ids)} sources")
    half = max(1, args.verify // 2)
    picks = sorted(set(range(min(half, len(ids)))) | set(range(max(0, len(ids) - half), len(ids))))
    loader = DevWindowAudio(wcfg, paths)
    ok = True
    for speed in (float(s) for s in cfg["speeds"]):
        cond_dir = out_root / condition_name(speed)
        saved = load_dev_window(cond_dir)
        fast, lengths = build_fast_windows(ws, numbers, alignments, speed=speed, window_sec=wcfg.window_sec,
                                           hop_sec=wcfg.hop_sec, sample_rate=wcfg.sample_rate)
        same_windows = all(np.array_equal(getattr(saved, k), getattr(fast, k))
                           for k in ("source_index", "start_sample", "mora", "kind", "known_no_speech"))
        with open(cond_dir / "sources.jsonl", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        audio = np.load(cond_dir / "audio.npy", mmap_mode="r")
        identical = 0
        max_diff = 0.0
        for i in picks:
            y = wsola_speed_up(loader.source_waveform(saved.sources[i]), speed, **kwargs)
            o, n = rows[i]["audio_offset"], rows[i]["stretched_samples"]
            stored = np.asarray(audio[o:o + n])
            if stored.size == y.size and np.array_equal(stored, y):
                identical += 1
            else:
                max_diff = max(max_diff, float(np.abs(stored - y).max()) if stored.size == y.size else float("inf"))
        log(f"{condition_name(speed)}: windows_identical={same_windows} audio_identical={identical}/{len(picks)} "
            f"max_abs_diff={max_diff:g}")
        ok &= same_windows and identical == len(picks)
    log("VERIFY_OK" if ok else "VERIFY_MISMATCH")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--verify", type=int, default=0, help="保存済みの dev_fast を N 音源で作り直して比べる")
    args = parser.parse_args(argv)
    cfg = load_fast_config(_resolve(args.config))
    return cmd_verify(cfg, args) if args.verify else cmd_build(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
