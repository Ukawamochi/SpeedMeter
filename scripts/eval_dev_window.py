"""dev_window（窓単位の評価セット）で exp004 と包絡ベースラインを評価する（指示書 2026-09-25 タスク5-2）。

サブコマンド:

- ``predict``: 窓ごとの予測（窓内のモーラ数）を ``runs/window_eval/`` に保存する。
  条件は clean と SNR 5・10・15dB（``configs/eval/dev_window.yaml`` の noisy 節）。
  音源（単一クリップまたは連結音声）の波形を ``DevWindowAudio.source_waveform`` で1回だけ
  組み立て、その中の窓（``DevWindowAudio.window`` と同じ切り出し）を両手法に通す。
  音源を ``--chunk-sources`` 件ずつの区切りで処理し、区切りごとに ``parts/`` へ保存する
  （途中で止まっても保存済みの区切りを飛ばして再開できる）。すべて終わると
  ``pred_<条件>.npz``（配列 exp004・envelope、窓の添字順）にまとめる
  ``--sample-sources N`` を付けると、無作為に選んだ N 音源だけで処理時間を測り、全体の
  所要時間を見積もって終わる（予測は保存しない）
- ``summarize``: 保存した予測から指標を計算し ``runs/window_eval/summary.json`` と
  ``runs/window_eval/tables.md`` に書く。``--append-metrics`` で results/metrics.csv に追記する

``--split test``（第10段階、2026-10-02 の人間の指示）: 対象を test_window（``configs/eval/test_window.yaml``、
``data/processed/test_window/``）にし、既定の出力先を ``runs/test_window_eval/``、除外の一覧を
``data/processed/no_speech/suspect_test.tsv``（test には known_no_speech は無い）、metrics.csv の split 列を
``test_window``・``test_window_noisy_snr5`` など（先頭の dev を test にした値）にする。既定の dev の動作は変わらない。
``--metrics-csv`` で追記先を作業用の写しに向けられる（summarize）。

推論の経路（exp004）: 窓の波形 → 対数メル（``LogMelSpectrogram``、学習の waveform 経路と同じ）
→ ``configs/normalization.yaml``（per_mel、学習時と同じ）→ ``runs/exp004/checkpoint_best.pt``
（008 と同じ）を mps で前向き計算（``spkrate.eval.window_diag.make_predictor``）。
出力は窓内のモーラ数。包絡ベースラインは ``configs/baselines/envelope.yaml`` のパラメータで
窓の毎秒モーラ数を返すので、窓長（2.0秒）を掛けてモーラ数にする。

除外（主指標）: known_no_speech（``results/error_cases/to_listen.tsv``）または no_speech_suspect
（``results/no_speech_suspect_dev.tsv``。規則は ``configs/eval/no_speech.yaml``）のクリップを由来に
持つ窓（単一クリップの窓と、そのクリップを含む連結の組の全窓）を除く。除かない値も併記する。

他のモデル（exp005 など）: ``--model-key``・``--checkpoint``（predict と summarize）、
``--model-config``・``--experiment-id``・``--method-name``（summarize）で対象のモデルを切り替え、
``--no-envelope``（両方）で包絡ベースラインを省く。既定値は exp004 と包絡の評価と同じ。
推論の経路は exp004 と同じ（モデルの出力は窓内のモーラ数）。例:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_window.py predict \\
        --out-dir runs/exp005/window_eval --model-key exp005 \\
        --checkpoint runs/exp005/checkpoint_best.pt --no-envelope
    uv run python scripts/eval_dev_window.py summarize --out-dir runs/exp005/window_eval \\
        --model-key exp005 --checkpoint runs/exp005/checkpoint_best.pt --no-envelope \\
        --model-config configs/exp005.yaml --experiment-id 010-exp005-method-b \\
        --method-name "..." --append-metrics

ONNX（第8段階 8-1）: ``predict --onnx <model.onnx>`` で、モデルの前向き計算を ONNX Runtime
（CPU の実行プロバイダ）に替える（``spkrate.export.to_onnx.make_onnx_predictor``。正規化はグラフ内）。
``--checkpoint`` は元のチェックポイント（best_epoch の記録用）を渡す。``summarize`` の
``--conditions clean`` で clean だけを集計し（雑音下の行は書かない）、``--model-file`` で
metrics.csv の model_size_bytes に ONNX ファイルの大きさを書く。例:
    uv run python scripts/eval_dev_window.py predict --conditions clean \\
        --out-dir runs/onnx_exp005/window_eval_int8 --model-key onnx_int8 \\
        --checkpoint runs/exp005/checkpoint_best.pt --onnx runs/onnx_exp005/model_int8.onnx --no-envelope
    uv run python scripts/eval_dev_window.py summarize --conditions clean \\
        --out-dir runs/onnx_exp005/window_eval_int8 --model-key onnx_int8 \\
        --checkpoint runs/exp005/checkpoint_best.pt --model-file runs/onnx_exp005/model_int8.onnx \\
        --no-envelope --model-config configs/exp005.yaml --experiment-id ... --method-name ... --append-metrics

configs/splits/test.json は使わない。data/ 以下は読むだけ。

実行（リポジトリ直下から）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_window.py predict --sample-sources 300
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_window.py predict > runs/window_eval/predict.log 2>&1
    uv run python scripts/eval_dev_window.py summarize --append-metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spkrate.eval.split_profile import SPLITS  # noqa: E402

CONFIG = "configs/eval/dev_window.yaml"
NO_SPEECH_CONFIG = "configs/eval/no_speech.yaml"
SUSPECT_TSV = "results/no_speech_suspect_dev.tsv"
MODEL_CONFIG = "configs/exp004.yaml"
CHECKPOINT = "runs/exp004/checkpoint_best.pt"
ENVELOPE_CONFIG = "configs/baselines/envelope.yaml"
NORMALIZATION = "configs/normalization.yaml"
OUT_DIR = "runs/window_eval"
METHODS = ("exp004", "envelope")
EXPERIMENT_IDS = {"exp004": "009-window-eval-exp004", "envelope": "009-window-eval-envelope"}
METHOD_NAMES = {
    "exp004": "cnn (話速推定CNN、方式A: クリップ全体入力、拡張あり、無音サンプルあり) 2.0秒窓ごと",
    "envelope": "envelope-baseline (音量包絡の帯域通過+ピーク計数) 2.0秒窓ごと",
}
METHOD_CONFIGS = {"exp004": MODEL_CONFIG, "envelope": ENVELOPE_CONFIG}
SHIFT_SEC = 0.05


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def method_keys(args: argparse.Namespace) -> tuple[str, ...]:
    """予測・集計の対象（モデルの名前と、省かなければ包絡）。既定は ``METHODS`` と同じ。"""
    return (args.model_key,) if args.no_envelope else (args.model_key, "envelope")


def condition_names(config) -> list[str]:
    return ["clean"] + [f"snr{float(s):g}" for s in config.noisy.snr_db]


def condition_snr(name: str) -> float | None:
    return None if name == "clean" else float(name[3:])


# ------------------------------------------------------------------------------ predict


def cmd_predict(args: argparse.Namespace) -> int:
    import torch

    from spkrate.baselines.envelope import EnvelopeSpeedEstimator, load_params
    from spkrate.eval.dev_window import DevWindowAudio, load_dev_clips, load_dev_window, load_window_config
    from spkrate.eval.noisy import make_noise_source
    from spkrate.eval.split_profile import get_profile, load_split_ids
    from spkrate.eval.window_diag import make_predictor
    from spkrate.eval.window_eval import source_window_ranges
    from spkrate.train.data import load_normalization
    from spkrate.train.train import MpsFallbackWatcher, load_checkpoint

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("eval_dev_window")

    profile = get_profile(args.split)
    config = load_window_config(_resolve(profile.window_config))
    ws = load_dev_window(_resolve(config.output_dir))
    ranges = source_window_ranges(ws)
    _, clip_paths = load_dev_clips(
        _resolve(config.clips_jsonl), load_split_ids(_resolve(config.dev_split), allow_test=config.allow_test_split)
    )
    paths = {k: _resolve(Path(config.audio_root) / v) for k, v in clip_paths.items()}
    noise_source = make_noise_source(config.noisy, repo_root=ROOT)
    audio = DevWindowAudio(config, paths, noise_source=noise_source)
    W = config.window_samples

    from spkrate.device import setup_device, synchronize

    model, payload = load_checkpoint(str(_resolve(args.checkpoint)), map_location="cpu")
    if args.onnx:
        import onnxruntime

        from spkrate.device import host_label
        from spkrate.export.to_onnx import make_onnx_predictor, make_session

        # ONNX Runtime は CPU の実行プロバイダだけを使う。torch のデバイスは使わない。
        device = torch.device("cpu")
        session = make_session(_resolve(args.onnx), intra_op_threads=args.onnx_threads)
        device_record = {"host_label": host_label(), "runtime": "onnxruntime",
                         "onnxruntime": onnxruntime.__version__, "providers": session.get_providers(),
                         "intra_op_threads": args.onnx_threads, "onnx": args.onnx}
        logger.info("onnxruntime %s providers=%s onnx=%s", onnxruntime.__version__,
                    session.get_providers(), args.onnx)
        predict = make_onnx_predictor(session, batch_size=args.batch_size)
    else:
        device, device_record = setup_device(args.device, logger)
        model = model.to(device).eval()
        normalizer = load_normalization(_resolve(NORMALIZATION))
        predict = make_predictor(model, normalizer, device, batch_size=args.batch_size)
    envelope = None if args.no_envelope else EnvelopeSpeedEstimator(load_params(_resolve(ENVELOPE_CONFIG)))
    methods = method_keys(args)
    log(f"device={device} checkpoint={args.checkpoint} best_epoch={payload['epoch']} "
        f"methods={','.join(methods)} windows={len(ws)} sources={len(ws.sources)} commit={_commit()}")

    conditions = condition_names(config) if args.conditions is None else args.conditions.split(",")
    with_windows = np.flatnonzero(ranges[:, 1] > ranges[:, 0])

    def run_sources(numbers, snr):
        """音源の番号の列を処理し、(窓の添字, exp004, envelope, 時間の内訳) を返す。"""
        t_wave = t_env = 0.0
        index_parts, waves, env = [], [], []
        for number in numbers:
            lo, hi = ranges[number]
            source = ws.sources[number]
            t0 = time.perf_counter()
            waveform = audio.source_waveform(source, snr)
            t_wave += time.perf_counter() - t0
            t0 = time.perf_counter()
            for i in range(lo, hi):
                start = int(ws.start_sample[i])
                window = np.ascontiguousarray(waveform[start:start + W])
                if window.size != W:
                    raise ValueError(f"窓が音源の外にはみ出す: {source.source_id} {start}")
                waves.append(window)
                if envelope is not None:
                    env.append(float(envelope(window, config.sample_rate)) * config.window_sec)
            t_env += time.perf_counter() - t0
            index_parts.append(np.arange(lo, hi, dtype=np.int64))
        t0 = time.perf_counter()
        model_mora = predict(waves) if waves else np.zeros(0, dtype=np.float32)
        synchronize(device)
        t_model = time.perf_counter() - t0
        index = np.concatenate(index_parts) if index_parts else np.zeros(0, dtype=np.int64)
        env_mora = np.asarray(env, dtype=np.float32)
        if not (np.isfinite(model_mora).all() and np.isfinite(env_mora).all()):
            raise ValueError("有限でない予測がある")
        # 窓切り出し・包絡の時間は包絡側に含める（t_env）。t_model は対数メル・正規化・前向き計算
        return index, np.asarray(model_mora, dtype=np.float32), env_mora, {
            "wave_sec": t_wave, "envelope_sec": t_env, "model_sec": t_model}

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with MpsFallbackWatcher(logger) as watcher:
        if args.sample_sources:
            rng = np.random.default_rng(args.seed)
            numbers = np.sort(rng.choice(with_windows, size=min(args.sample_sources, with_windows.size),
                                         replace=False))
            n_windows = int(sum(ranges[n, 1] - ranges[n, 0] for n in numbers))
            clip_count = int(sum(len(ws.sources[n].clip_ids) for n in numbers))
            estimate = {"sample_sources": int(numbers.size), "sample_windows": n_windows,
                        "sample_clips": clip_count, "conditions": {}}
            total_windows = len(ws)
            total_clips = int(sum(len(ws.sources[n].clip_ids) for n in with_windows))
            total_audio = float(sum(ws.sources[n].num_samples for n in with_windows))
            sample_audio = float(sum(ws.sources[n].num_samples for n in numbers))
            predict([np.zeros(W, dtype=np.float32)] * 8)  # ウォームアップ
            for name in conditions:
                audio._cache_key = None
                _, _, _, t = run_sources(numbers, condition_snr(name))
                scaled = {
                    "wave_sec": t["wave_sec"] * total_audio / sample_audio,
                    "envelope_sec": t["envelope_sec"] * total_windows / n_windows,
                    "model_sec": t["model_sec"] * total_windows / n_windows,
                }
                scaled["total_sec"] = sum(scaled.values())
                estimate["conditions"][name] = {"sample": t, "estimated_full": scaled,
                                                "per_window_ms": {k: v / n_windows * 1000 for k, v in t.items()}}
                log(f"estimate {name}: sample {json.dumps(t)} -> full {json.dumps(scaled)}")
            estimate["total_windows"] = total_windows
            estimate["total_clips_in_sources"] = total_clips
            estimate["estimated_total_hours"] = sum(
                c["estimated_full"]["total_sec"] for c in estimate["conditions"].values()) / 3600
            estimate["mps_cpu_fallback_events"] = watcher.events
            (out_dir / "estimate.json").write_text(json.dumps(estimate, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
            log(f"ESTIMATE_DONE total_hours={estimate['estimated_total_hours']:.2f} "
                f"fallback_events={len(watcher.events)}")
            return 0

        parts_dir = out_dir / "parts"
        parts_dir.mkdir(exist_ok=True)
        chunks = [with_windows[i:i + args.chunk_sources]
                  for i in range(0, with_windows.size, args.chunk_sources)]
        timing: dict[str, dict[str, float]] = {}
        for name in conditions:
            final = out_dir / f"pred_{name}.npz"
            if final.exists():
                log(f"{name}: {final.name} があるので飛ばす")
                continue
            snr = condition_snr(name)
            totals = {"wave_sec": 0.0, "envelope_sec": 0.0, "model_sec": 0.0}
            started = time.perf_counter()
            for number, chunk in enumerate(chunks):
                part = parts_dir / f"{name}_{number:04d}.npz"
                if part.exists():
                    continue
                index, model_mora, env_mora, t = run_sources(chunk, snr)
                arrays = {args.model_key: model_mora}
                if envelope is not None:
                    arrays["envelope"] = env_mora
                np.savez(part, index=index, **arrays,
                         **{k: np.float32(v) for k, v in t.items()})
                for k in totals:
                    totals[k] += t[k]
                log(f"{name} chunk {number + 1}/{len(chunks)} windows={index.size} "
                    f"elapsed={time.perf_counter() - started:.0f}s {json.dumps({k: round(v, 1) for k, v in t.items()})}")
            merged = {m: np.full(len(ws), np.nan, dtype=np.float32) for m in methods}
            part_totals = {"wave_sec": 0.0, "envelope_sec": 0.0, "model_sec": 0.0}
            for number in range(len(chunks)):
                with np.load(parts_dir / f"{name}_{number:04d}.npz") as data:
                    for m in methods:
                        merged[m][data["index"]] = data[m]
                    for k in part_totals:
                        part_totals[k] += float(data[k])
            missing = {m: int(np.isnan(v).sum()) for m, v in merged.items()}
            if any(missing.values()):
                raise ValueError(f"{name}: 予測の無い窓がある {missing}")
            np.savez(final, **merged)
            timing[name] = part_totals
            log(f"{name}: saved {final.name} timing={json.dumps(part_totals)}")
        meta_path = out_dir / "predict_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.setdefault("timing", {}).update(timing)
        meta.update({"commit": _commit(), "checkpoint": args.checkpoint, "methods": list(methods),
                     "best_epoch": int(payload["epoch"]),
                     "device": str(device), "host": device_record, "batch_size": args.batch_size,
                     "chunk_sources": args.chunk_sources, "windows": len(ws),
                     "mps_cpu_fallback_events": meta.get("mps_cpu_fallback_events", []) + watcher.events})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"mps_cpu_fallback_events={len(watcher.events)}")
    log("PREDICT_DONE")
    return 0


# ------------------------------------------------------------------------------ summarize


def load_suspect_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return {row["clip_id"] for row in csv.DictReader(lines, delimiter="\t")}


def cmd_summarize(args: argparse.Namespace) -> int:
    from spkrate.eval.dev_window import load_alignments, load_dev_window, load_known_no_speech, load_window_config
    from spkrate.eval.metrics import BAND_KEYS, compute_metrics
    from spkrate.eval.no_speech import windows_from_clips
    from spkrate.eval.split_profile import get_profile
    from spkrate.eval.runner import MetricsRow, append_metrics_row, model_size_bytes
    from spkrate.eval.window_eval import (
        rate_summary,
        shifted_window_labels,
        short_clip_window_mask,
        zero_window_summary,
    )

    profile = get_profile(args.split)
    config = load_window_config(_resolve(profile.window_config))
    ws = load_dev_window(_resolve(config.output_dir))
    out_dir = _resolve(args.out_dir)
    conditions = condition_names(config) if args.conditions is None else args.conditions.split(",")
    if conditions[0] != "clean" or not set(conditions) <= set(condition_names(config)):
        raise SystemExit(f"--conditions は clean から始まる {condition_names(config)} の部分列: {conditions}")
    noisy_names = conditions[1:]
    with_noisy_all = noisy_names == condition_names(config)[1:]
    methods = method_keys(args)
    experiment_ids = {**EXPERIMENT_IDS, args.model_key: args.experiment_id or EXPERIMENT_IDS.get(args.model_key)}
    method_names = {**METHOD_NAMES, args.model_key: args.method_name or METHOD_NAMES.get(args.model_key)}
    method_configs = {**METHOD_CONFIGS, args.model_key: args.model_config}
    if args.append_metrics and not (experiment_ids[args.model_key] and method_names[args.model_key]):
        raise SystemExit(f"--model-key {args.model_key} には --experiment-id と --method-name が要る")
    preds = {}
    for name in conditions:
        with np.load(out_dir / f"pred_{name}.npz") as data:
            preds[name] = {m: data[m].astype(np.float32) for m in methods}

    known_ids = load_known_no_speech(
        None if config.known_no_speech_list is None else _resolve(config.known_no_speech_list)
    )
    suspect_ids = load_suspect_ids(_resolve(profile.suspect_tsv))
    flagged = windows_from_clips(ws, known_ids | suspect_ids)
    if not np.array_equal(ws.known_no_speech, ws.known_no_speech & flagged):
        raise ValueError("known_no_speech の窓が除外に含まれていない")
    keep = ~flagged
    single = ws.kind == 0
    concat = ws.kind == 1
    zero = ws.mora == 0
    short = short_clip_window_mask(ws, config.window_samples)
    true = ws.mora.astype(np.float32)
    W_SEC = config.window_sec

    subsets = {"all": np.ones(len(ws), dtype=bool), "single": single, "concat": concat}
    summary: dict = {
        "commit": _commit(),
        "windows": len(ws),
        "excluded_windows": int(flagged.sum()),
        "excluded_single": int((flagged & single).sum()),
        "excluded_concat": int((flagged & concat).sum()),
        "excluded_zero": int((flagged & zero).sum()),
        "known_ids": len(known_ids),
        "suspect_ids": len(suspect_ids),
        "short_concat_windows": {"all": int(short.sum()), "main": int((short & keep).sum())},
        "results": {},
    }
    for name in conditions:
        for method in methods:
            pred = preds[name][method]
            entry: dict = {}
            for scope, base in (("main", keep), ("unfiltered", np.ones(len(ws), dtype=bool))):
                block: dict = {}
                for sub, mask in subsets.items():
                    m = base & mask
                    block[sub] = rate_summary(true[m], pred[m], W_SEC)
                    block[sub]["zero"] = zero_window_summary(pred[m & zero], W_SEC)
                block["short_concat"] = rate_summary(true[base & short], pred[base & short], W_SEC)
                entry[scope] = block
            summary["results"][f"{name}/{method}"] = entry

    # 雑音下3条件のまとめ（3条件がそろうときだけ）
    for method in (methods if with_noisy_all else ()):
        entry = {}
        for scope, base in (("main", keep), ("unfiltered", np.ones(len(ws), dtype=bool))):
            t = np.concatenate([true[base]] * 3)
            p = np.concatenate([preds[n][method][base] for n in conditions[1:]])
            z = np.concatenate([(zero & base)[base]] * 3)
            s = rate_summary(t, p, W_SEC)
            s["zero"] = zero_window_summary(p[z], W_SEC)
            entry[scope] = {"all": s}
        summary["results"][f"noisy_all/{method}"] = entry

    # 窓の正解の不確かさ（±50ms）
    alignments = load_alignments(_resolve(config.alignments))
    zero_check, zero_clamped = shifted_window_labels(ws, alignments, 0.0, window_sec=W_SEC,
                                                     sample_rate=config.sample_rate)
    uncertainty: dict = {"shift0_identical": bool(np.array_equal(zero_check, ws.mora)),
                         "shift0_max_abs_diff": float(np.max(np.abs(zero_check - ws.mora))),
                         "shift0_clamped": zero_clamped}
    diffs = {}
    for shift in (SHIFT_SEC, -SHIFT_SEC):
        labels, clamped = shifted_window_labels(ws, alignments, shift, window_sec=W_SEC,
                                                sample_rate=config.sample_rate)
        diffs[f"{shift:+g}"] = np.abs(labels - ws.mora) / np.float32(W_SEC)
        uncertainty[f"clamped_{shift:+g}"] = clamped
    from spkrate.eval.window_eval import band_codes
    codes = band_codes(true / np.float32(W_SEC))
    for scope, base in (("main", keep), ("unfiltered", np.ones(len(ws), dtype=bool))):
        block = {}
        groups = {"all": base, "single": base & single, "concat": base & concat}
        for i, key in enumerate(BAND_KEYS):
            groups[f"band_{key}"] = base & (codes == i)
            groups[f"single_band_{key}"] = base & single & (codes == i)
            groups[f"concat_band_{key}"] = base & concat & (codes == i)
        for g, mask in groups.items():
            row = {"n": int(mask.sum())}
            for label, d in diffs.items():
                row[label] = float(d[mask].mean()) if mask.any() else float("nan")
                row[f"{label}_frac_changed"] = float((d[mask] > 0).mean()) if mask.any() else float("nan")
            both = (diffs[f"{SHIFT_SEC:+g}"] + diffs[f"{-SHIFT_SEC:+g}"]) / np.float32(2.0)
            row["mean_of_both"] = float(both[mask].mean()) if mask.any() else float("nan")
            block[g] = row
        uncertainty[scope] = block
    summary["label_uncertainty"] = uncertainty

    meta_path = out_dir / "predict_meta.json"
    if meta_path.exists():
        summary["predict_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    log(f"summary written: {out_dir / 'summary.json'}")

    if args.append_metrics:
        size = model_size_bytes(_resolve(args.model_file or args.checkpoint))
        # host・device 列: exp004 は予測を作った計算機とデバイス（predict_meta.json）、包絡は cpu。
        meta_path = out_dir / "predict_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        predict_host = (meta.get("host") or {}).get("host_label")
        predict_device = str(meta["device"]).split(":")[0] if meta.get("device") else None
        durations = [W_SEC] * int(keep.sum())
        split_names = {"clean": profile.window_split,
                       **{n: profile.window_noisy_split(n) for n in conditions[1:]}}
        for method in methods:
            rows = []
            parts_t, parts_p = [], []
            for name in conditions:
                p = preds[name][method][keep]
                metrics = compute_metrics(true[keep].tolist(), p.tolist(), durations)
                rows.append((split_names[name], metrics))
                if name != "clean":
                    parts_t.append(true[keep])
                    parts_p.append(p)
            if with_noisy_all:
                pooled = compute_metrics(np.concatenate(parts_t).tolist(), np.concatenate(parts_p).tolist(),
                                         [W_SEC] * int(keep.sum()) * len(parts_t))
                rows.append((profile.window_noisy_pooled_split, pooled))
            config_path = f"{method_configs[method]};{profile.window_config};{NO_SPEECH_CONFIG}"
            is_model = method == args.model_key
            for split, metrics in rows:
                append_metrics_row(MetricsRow(
                    experiment_id=experiment_ids[method], method=method_names[method],
                    config_path=config_path, split=split, metrics=metrics,
                    latency_ms_per_inference=float("nan"),
                    model_size_bytes=size if is_model else None,
                    host=predict_host,
                    device=predict_device if is_model else "cpu",
                ), csv_path=_resolve(args.metrics_csv))
                log(f"metrics.csv: {experiment_ids[method]} {split} mae={metrics.mae_moras_per_sec:.4f}")
    log("SUMMARIZE_DONE")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--out-dir", default=None, help="既定は runs/window_eval（--split test は runs/test_window_eval）")
    p.add_argument("--conditions", default=None, help="clean,snr5,snr10,snr15 の一部（既定はすべて）")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-sources", type=int, default=500)
    p.add_argument("--sample-sources", type=int, default=0, help="見積もり用に無作為に選ぶ音源の数")
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--device", default="mps",
                   help="mps・cuda・cpu（既定 mps）。使えない場合は開始前に止める")
    p.add_argument("--onnx", default=None,
                   help="ONNX のモデル（第8段階）。与えると ONNX Runtime の CPU 実行で推論する")
    p.add_argument("--onnx-threads", type=int, default=None,
                   help="ONNX Runtime の intra_op_num_threads（省略時は ONNX Runtime の既定）")
    s = sub.add_parser("summarize")
    s.add_argument("--out-dir", default=None, help="既定は runs/window_eval（--split test は runs/test_window_eval）")
    s.add_argument("--append-metrics", action="store_true")
    s.add_argument("--metrics-csv", default="results/metrics.csv",
                   help="追記先の csv（既定は results/metrics.csv。作業用の写しに向けられる）")
    s.add_argument("--model-config", default=MODEL_CONFIG, help="metrics.csv の config_path に書くモデルの設定")
    s.add_argument("--experiment-id", default=None, help="metrics.csv の実験ID（既定は exp004 の値）")
    s.add_argument("--method-name", default=None, help="metrics.csv の method（既定は exp004 の値）")
    s.add_argument("--conditions", default=None,
                   help="集計する条件（clean から始まる部分列。既定はすべて。3条件そろわなければ雑音下まとめを省く）")
    s.add_argument("--model-file", default=None,
                   help="metrics.csv の model_size_bytes に大きさを書くファイル（既定は --checkpoint）")
    for q in (p, s):
        q.add_argument("--split", choices=SPLITS, default="dev",
                       help="対象の分割（既定 dev。test は第10段階、2026-10-02 の人間の指示。configs/eval/test_window.yaml）")
        q.add_argument("--model-key", default="exp004", help="予測の配列名（npz のキー）")
        q.add_argument("--checkpoint", default=CHECKPOINT)
        q.add_argument("--no-envelope", action="store_true", help="包絡ベースラインを省く")
    p.set_defaults(func=cmd_predict)
    s.set_defaults(func=cmd_summarize)
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = OUT_DIR if args.split == "dev" else "runs/test_window_eval"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
