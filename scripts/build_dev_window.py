"""窓単位の評価セット dev_window の窓の定義を生成し、data/processed/dev_window/ に保存する。

処理の本体は src/spkrate/eval/dev_window.py、設定は configs/eval/dev_window.yaml。
音声は読まない（長さはアライメントの duration_sec）。``--check-waveforms N`` を付けると、
保存後に読み戻した定義から先頭寄りの代表 N 窓（単一・連続の両方）の波形を clean と
雑音下（設定の最初の SNR）で再生成し、長さと2回の生成のビット一致を確かめる。
既定の設定（configs/eval/dev_window.yaml）では configs/splits/test.json は使わない。
``--config configs/eval/test_window.yaml`` は test の分割（第10段階、2026-10-02 の人間の指示）で
data/processed/test_window/ に作る。data/ 以下の元データは読むだけで変更しない。

実行（リポジトリ直下から）:
    uv run python scripts/build_dev_window.py --check-waveforms 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spkrate.eval.dev_window import (  # noqa: E402
    KIND_CONCAT,
    KIND_SINGLE,
    DevWindowAudio,
    build_dev_window,
    load_alignments,
    load_dev_clips,
    load_dev_window,
    load_known_no_speech,
    load_window_config,
    save_dev_window,
)
from spkrate.eval.noisy import make_noise_source  # noqa: E402
from spkrate.eval.split_profile import load_split_ids  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def check_waveforms(config, window_set, clip_paths, count: int) -> list[dict]:
    """代表窓の波形を再生成して長さ・決定性を確かめる。"""
    noise_source = make_noise_source(config.noisy, repo_root=ROOT)
    paths = {k: _resolve(Path(config.audio_root) / v) for k, v in clip_paths.items()}
    snr = config.noisy.snr_db[0]
    results = []
    for kind_code, kind in ((0, KIND_SINGLE), (1, KIND_CONCAT)):
        indices = np.flatnonzero(window_set.kind == kind_code)[: max(1, count // 2)]
        for index in indices:
            outputs = []
            for _ in range(2):
                audio = DevWindowAudio(config, paths, noise_source=noise_source)
                clean = audio.window_at(window_set, int(index))
                noisy = audio.window_at(window_set, int(index), snr)
                outputs.append((clean, noisy))
            (c1, n1), (c2, n2) = outputs
            assert c1.size == n1.size == config.window_samples
            results.append({
                "index": int(index),
                "kind": kind,
                "source_id": window_set.sources[int(window_set.source_index[index])].source_id,
                "start_sample": int(window_set.start_sample[index]),
                "mora": float(window_set.mora[index]),
                "samples": int(c1.size),
                "clean_identical": bool(np.array_equal(c1, c2)),
                "noisy_identical": bool(np.array_equal(n1, n2)),
                "clean_rms": float(np.sqrt(np.mean(c1**2))),
                "noisy_rms": float(np.sqrt(np.mean(n1**2))),
            })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/dev_window.yaml")
    parser.add_argument("--check-waveforms", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_window_config(_resolve(args.config))
    clip_to_client, clip_paths = load_dev_clips(
        _resolve(config.clips_jsonl),
        load_split_ids(_resolve(config.dev_split), allow_test=config.allow_test_split),
    )
    alignments = load_alignments(_resolve(config.alignments))
    known = load_known_no_speech(
        None if config.known_no_speech_list is None else _resolve(config.known_no_speech_list)
    )
    window_set = build_dev_window(
        config, alignments=alignments, clip_to_client=clip_to_client, known_no_speech=known
    )
    window_set.meta.update(
        {
            "config_path": args.config,
            "commit": _commit(),
            "created": datetime.now().isoformat(timespec="seconds"),
        }
    )
    out = save_dev_window(window_set, _resolve(config.output_dir))
    print(json.dumps(window_set.meta["counts"], ensure_ascii=False, indent=1))
    print("point_moras_in_eligible_clips:", window_set.meta["point_moras_in_eligible_clips"])
    print("saved:", out)

    if args.check_waveforms:
        reloaded = load_dev_window(out)
        assert np.array_equal(reloaded.mora, window_set.mora)
        checks = check_waveforms(config, reloaded, clip_paths, args.check_waveforms)
        for row in checks:
            print(json.dumps(row, ensure_ascii=False))
        window_set.meta["waveform_checks"] = checks
        with open(out / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(window_set.meta, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
