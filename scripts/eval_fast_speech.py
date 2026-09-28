"""高速度域の評価（docs/directives/2026-09-28-fast-speech.md タスク1）の推論・集計・記録。

評価1（dev_window・clean・主指標の、正解の話速の区間別）と評価2（速めた音声 dev_fast の区間別）。
区間・集計の定義は ``src/spkrate/eval/fast_speech.py``、dev_fast の定義は ``configs/eval/dev_fast.yaml``
（作成は ``scripts/build_dev_fast.py``）。

サブコマンド:

- ``predict``: dev_fast の各条件（x1.5・x2）の窓をモデルに通し、``<out-dir>/pred_<条件>.npz``
  （配列名は ``--model-key``、窓の添字順）と ``predict_meta.json`` を書く。波形は
  ``data/processed/dev_fast/<条件>/audio.npy`` から切り出す（WSOLA をやり直さない）。
  推論の経路は ``scripts/eval_dev_window.py predict`` と同じ（``window_diag.make_predictor``、
  ``configs/normalization.yaml``）。``--device`` は mps・cuda・cpu
- ``summarize``: 評価1は ``--window-eval-dir``（``scripts/eval_dev_window.py predict`` が保存した
  ``pred_clean.npz``。推論をやり直さない）、評価2は ``--out-dir`` の ``pred_<条件>.npz`` から、
  主指標（dev_window と同じ除外）と全窓の区間別の集計を ``<out-dir>/fast_speech_summary.json`` と
  ``fast_speech_tables.md`` に書く。どちらか一方だけでもよい（``--skip-window`` / ``--skip-fast``）。
  ``--append-metrics`` で results/metrics.csv に主指標の行を追記する

metrics.csv の行（既存の列だけを使う。値は主指標、窓長2.0秒の ``compute_metrics``）:

- 評価1: split ``dev_window_bin_<区間>``（``6to7``〜``13to14``、``ge14``。窓が0の区間は書かない）。
  config_path は ``<モデルの設定>;configs/eval/dev_window.yaml;configs/eval/no_speech.yaml``
- 評価2: split ``dev_fast_<条件>``（全窓）と ``dev_fast_<条件>_bin_<区間>``（例 ``dev_fast_x1.5``・
  ``dev_fast_x2_bin_12to13``）。config_path は ``<モデルの設定>;configs/eval/dev_fast.yaml;configs/eval/no_speech.yaml``
- 実験ID は ``--experiment-id``（推奨: ``013-fast-speech-<モデル>``、例 ``013-fast-speech-exp005``）。
  num_segments は窓数、latency は空欄（NaN）、host・device は予測を作った計算機（predict_meta.json）
- 偏り・正解と出力の平均・出力の10%点と90%点は列が無いので json と md にだけ出す

例（exp005、Mac の mps）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_fast_speech.py predict \\
        --checkpoint runs/exp005/checkpoint_best.pt --model-key exp005 --device mps \\
        --out-dir runs/exp005/fast_speech > runs/exp005/fast_speech_predict.log 2>&1
    uv run python scripts/eval_fast_speech.py summarize --model-key exp005 \\
        --checkpoint runs/exp005/checkpoint_best.pt --model-config configs/exp005.yaml \\
        --window-eval-dir runs/exp005/window_eval --out-dir runs/exp005/fast_speech \\
        --experiment-id 013-fast-speech-exp005 --method-name "..." --append-metrics

configs/splits/test.json は使わない。data/ 以下は読むだけ。
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
import yaml  # noqa: E402

WINDOW_CONFIG = "configs/eval/dev_window.yaml"
FAST_CONFIG = "configs/eval/dev_fast.yaml"
NO_SPEECH_CONFIG = "configs/eval/no_speech.yaml"
SUSPECT_TSV = "results/no_speech_suspect_dev.tsv"
NORMALIZATION = "configs/normalization.yaml"
WINDOW_SEC = 2.0


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def fast_conditions(args: argparse.Namespace) -> tuple[dict, list[str]]:
    from spkrate.eval.fast_speech import condition_name

    with open(_resolve(args.fast_config), encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    names = [condition_name(float(s)) for s in cfg["speeds"]]
    if args.conditions:
        wanted = args.conditions.split(",")
        if not set(wanted) <= set(names):
            raise SystemExit(f"--conditions は {names} の部分集合: {wanted}")
        names = [n for n in names if n in wanted]
    return cfg, names


def load_fast_set(cfg: dict, name: str):
    """(窓の定義, 音源ごとの (audio_offset, stretched_samples), 波形の memmap)。"""
    from spkrate.eval.dev_window import load_dev_window

    cond_dir = _resolve(cfg["output_dir"]) / name
    ws = load_dev_window(cond_dir)
    with open(cond_dir / "sources.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    spans = np.array([[r["audio_offset"], r["stretched_samples"]] for r in rows], dtype=np.int64)
    audio = np.load(cond_dir / "audio.npy", mmap_mode="r")
    if spans.shape[0] != len(ws.sources) or int(spans[-1].sum()) != audio.shape[0]:
        raise ValueError(f"{cond_dir}: sources.jsonl と audio.npy が合わない")
    return ws, spans, audio


# ------------------------------------------------------------------------------ predict


def cmd_predict(args: argparse.Namespace) -> int:
    from spkrate.device import setup_device, synchronize
    from spkrate.eval.window_diag import make_predictor
    from spkrate.eval.window_eval import source_window_ranges
    from spkrate.train.data import load_normalization
    from spkrate.train.train import MpsFallbackWatcher, load_checkpoint

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("eval_fast_speech")
    cfg, names = fast_conditions(args)
    W = int(round(WINDOW_SEC * 16000))

    model, payload = load_checkpoint(str(_resolve(args.checkpoint)), map_location="cpu")
    device, device_record = setup_device(args.device, logger)
    model = model.to(device).eval()
    normalizer = load_normalization(_resolve(NORMALIZATION))
    predict = make_predictor(model, normalizer, device, batch_size=args.batch_size)
    log(f"device={device} checkpoint={args.checkpoint} best_epoch={payload['epoch']} "
        f"model_key={args.model_key} conditions={','.join(names)} commit={_commit()}")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, dict] = {}
    with MpsFallbackWatcher(logger) as watcher:
        for name in names:
            final = out_dir / f"pred_{name}.npz"
            if final.exists():
                log(f"{name}: {final.name} があるので飛ばす")
                continue
            ws, spans, audio = load_fast_set(cfg, name)
            ranges = source_window_ranges(ws)
            with_windows = np.flatnonzero(ranges[:, 1] > ranges[:, 0])
            if args.max_sources:
                # 動作確認用: 先頭の音源だけを通す（残りの窓は NaN のまま。npz には保存しない）
                with_windows = with_windows[:args.max_sources]
            pred = np.full(len(ws), np.nan, dtype=np.float32)
            t_model = 0.0
            started = time.perf_counter()
            chunks = [with_windows[i:i + args.chunk_sources]
                      for i in range(0, with_windows.size, args.chunk_sources)]
            for number, chunk in enumerate(chunks):
                waves, index = [], []
                for n in chunk:
                    offset, length = (int(v) for v in spans[n])
                    waveform = np.array(audio[offset:offset + length], dtype=np.float32)  # memmap から書き込める配列へ写す
                    lo, hi = ranges[n]
                    for i in range(lo, hi):
                        start = int(ws.start_sample[i])
                        window = np.ascontiguousarray(waveform[start:start + W])
                        if window.size != W:
                            raise ValueError(f"窓が音源の外にはみ出す: {ws.sources[n].source_id} {start}")
                        waves.append(window)
                        index.append(i)
                t0 = time.perf_counter()
                out = predict(waves)
                synchronize(device)
                t_model += time.perf_counter() - t0
                pred[np.asarray(index, dtype=np.int64)] = np.asarray(out, dtype=np.float32)
                log(f"{name} chunk {number + 1}/{len(chunks)} windows={len(index)} "
                    f"elapsed={time.perf_counter() - started:.0f}s")
            if args.max_sources:
                done = np.isfinite(pred)
                log(f"{name}: smoke windows={int(done.sum())} pred_mean_rate={float(pred[done].mean()) / WINDOW_SEC:.3f} "
                    f"true_mean_rate={float(ws.mora[done].mean()) / WINDOW_SEC:.3f}（保存しない）")
                continue
            if not np.isfinite(pred).all():
                raise ValueError(f"{name}: 予測の無い窓か有限でない予測がある")
            np.savez(final, **{args.model_key: pred})
            timing[name] = {"model_sec": t_model, "total_sec": time.perf_counter() - started, "windows": len(ws)}
            log(f"{name}: saved {final.name} windows={len(ws)} timing={json.dumps(timing[name])}")
        meta_path = out_dir / "predict_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.setdefault("timing", {}).update(timing)
        meta.update({"commit": _commit(), "checkpoint": args.checkpoint, "model_key": args.model_key,
                     "best_epoch": int(payload["epoch"]), "device": str(device), "host": device_record,
                     "batch_size": args.batch_size, "fast_config": args.fast_config,
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


def evaluate(ws, pred: np.ndarray, flagged_ids: set[str]) -> tuple[dict, np.ndarray]:
    """(主指標と全窓の集計, 主指標の窓の真偽配列)。"""
    from spkrate.eval.fast_speech import binned_summary
    from spkrate.eval.no_speech import windows_from_clips
    from spkrate.eval.window_eval import rate_summary

    flagged = windows_from_clips(ws, flagged_ids)
    if not np.array_equal(ws.known_no_speech, ws.known_no_speech & flagged):
        raise ValueError("known_no_speech の窓が除外に含まれていない")
    keep = ~flagged
    true = ws.mora.astype(np.float32)
    out: dict = {"windows": len(ws), "excluded_windows": int(flagged.sum())}
    for scope, mask in (("main", keep), ("unfiltered", np.ones(len(ws), dtype=bool))):
        out[scope] = {"overall": rate_summary(true[mask], pred[mask], WINDOW_SEC),
                      "bins": binned_summary(true[mask], pred[mask], WINDOW_SEC)}
    return out, keep


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if value != value else f"{value:.{digits}f}".replace("-", "−")


def bins_table(title: str, bins: dict) -> list[str]:
    lines = [f"### {title}", "",
             "| 区間（正解の mora/s） | 窓数 | 正解の平均 | 出力の平均 | 出力の10%点 | 出力の90%点 | 偏り | MAE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, b in bins.items():
        lines.append(f"| {key} | {b['n']:,} | {_fmt(b['true_mean'])} | {_fmt(b['pred_mean'])} | "
                     f"{_fmt(b['pred_p10'])} | {_fmt(b['pred_p90'])} | {_fmt(b['bias'])} | {_fmt(b['mae'])} |")
    return lines + [""]


def cmd_summarize(args: argparse.Namespace) -> int:
    from spkrate.eval.dev_window import load_dev_window, load_known_no_speech, load_window_config
    from spkrate.eval.fast_speech import binned_metrics_rows
    from spkrate.eval.runner import MetricsRow, append_metrics_row, model_size_bytes

    if args.append_metrics and not (args.experiment_id and args.method_name and args.model_config):
        raise SystemExit("--append-metrics には --experiment-id・--method-name・--model-config が要る")
    wcfg = load_window_config(_resolve(WINDOW_CONFIG))
    flagged_ids = load_known_no_speech(_resolve(wcfg.known_no_speech_list)) | load_suspect_ids(_resolve(SUSPECT_TSV))
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"commit": _commit(), "model_key": args.model_key, "checkpoint": args.checkpoint,
                     "flagged_clip_ids": len(flagged_ids), "window_eval": None, "dev_fast": {}}
    md = [f"# 高速度域の評価（{args.model_key}）", "",
          "値は毎秒モーラ数。区間は窓の正解の毎秒モーラ数。偏り = 出力 − 正解 の平均。"
          "主指標は dev_window と同じ除外（known_no_speech・no_speech_suspect）の後。", ""]
    rows: list[tuple[str, str, object, dict]] = []  # (config_path, split, metrics, predict_meta)

    def predict_meta(directory: Path) -> dict:
        path = directory / "predict_meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    if not args.skip_window:
        wdir = _resolve(args.window_eval_dir)
        ws = load_dev_window(_resolve(wcfg.output_dir))
        with np.load(wdir / "pred_clean.npz") as data:
            pred = data[args.model_key].astype(np.float32)
        entry, keep = evaluate(ws, pred, flagged_ids)
        entry["source"] = str(args.window_eval_dir)
        summary["window_eval"] = entry
        o = entry["main"]["overall"]
        log(f"dev_window clean main: n={o['n']} mae={o['mae']:.4f} bias={o['bias']:.4f} "
            "（scripts/eval_dev_window.py summarize の clean・主指標と一致するはず）")
        md += ["## 評価1: dev_window（clean）の正解の話速の区間別", "",
               f"予測: `{args.window_eval_dir}/pred_clean.npz`。主指標 {o['n']:,}窓、MAE {_fmt(o['mae'], 4)}、"
               f"偏り {_fmt(o['bias'], 4)}。", ""]
        md += bins_table("主指標", entry["main"]["bins"])
        md += bins_table("全窓", entry["unfiltered"]["bins"])
        config_path = f"{args.model_config};{WINDOW_CONFIG};{NO_SPEECH_CONFIG}"
        meta = predict_meta(wdir)
        for split, metrics in binned_metrics_rows("dev_window", ws.mora[keep], pred[keep], include_overall=False):
            rows.append((config_path, split, metrics, meta))

    if not args.skip_fast:
        cfg, names = fast_conditions(args)
        meta = predict_meta(out_dir)
        md += ["## 評価2: dev_fast（速めた音声）の正解の話速の区間別", "",
               f"定義: `{args.fast_config}`（WSOLA、audiotsm）。予測: `{args.out_dir}/pred_<条件>.npz`。", ""]
        for name in names:
            from spkrate.eval.dev_window import load_dev_window as _load

            ws = _load(_resolve(cfg["output_dir"]) / name)
            with np.load(out_dir / f"pred_{name}.npz") as data:
                pred = data[args.model_key].astype(np.float32)
            entry, keep = evaluate(ws, pred, flagged_ids)
            summary["dev_fast"][name] = entry
            o = entry["main"]["overall"]
            log(f"dev_fast {name} main: n={o['n']} mae={o['mae']:.4f} bias={o['bias']:.4f}")
            md += [f"### {name}: 主指標 {o['n']:,}窓（除外 {entry['excluded_windows']:,}）、MAE {_fmt(o['mae'], 4)}、"
                   f"偏り {_fmt(o['bias'], 4)}、相関係数 {_fmt(o['correlation'], 4)}、"
                   f"正解の平均 {_fmt(o['true_mean'])}、出力の平均 {_fmt(o['pred_mean'])}", ""]
            md += bins_table(f"{name} 主指標", entry["main"]["bins"])
            md += bins_table(f"{name} 全窓", entry["unfiltered"]["bins"])
            config_path = f"{args.model_config};{args.fast_config};{NO_SPEECH_CONFIG}"
            for split, metrics in binned_metrics_rows(f"dev_fast_{name}", ws.mora[keep], pred[keep],
                                                      include_overall=True):
                rows.append((config_path, split, metrics, meta))

    (out_dir / "fast_speech_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    (out_dir / "fast_speech_tables.md").write_text("\n".join(md), encoding="utf-8")
    log(f"summary written: {out_dir / 'fast_speech_summary.json'} / fast_speech_tables.md")

    if args.append_metrics:
        size = model_size_bytes(_resolve(args.checkpoint)) if args.checkpoint else None
        for config_path, split, metrics, meta in rows:
            host = (meta.get("host") or {}).get("host_label")
            device = str(meta["device"]).split(":")[0] if meta.get("device") else None
            append_metrics_row(MetricsRow(
                experiment_id=args.experiment_id, method=args.method_name, config_path=config_path,
                split=split, metrics=metrics, latency_ms_per_inference=float("nan"),
                model_size_bytes=size, host=host, device=device,
            ), csv_path=_resolve(args.metrics_csv))
            log(f"metrics.csv: {args.experiment_id} {split} n={metrics.num_segments} "
                f"mae={metrics.mae_moras_per_sec:.4f}")
    log("SUMMARIZE_DONE")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="mps", help="mps・cuda・cpu（既定 mps）。使えない場合は開始前に止める")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-sources", type=int, default=200)
    p.add_argument("--max-sources", type=int, default=0,
                   help="動作確認用。各条件の先頭 N 音源だけを通して平均を表示し、予測は保存しない")
    s = sub.add_parser("summarize")
    s.add_argument("--checkpoint", default=None, help="metrics.csv の model_size_bytes に大きさを書くファイル")
    s.add_argument("--window-eval-dir", default=None,
                   help="評価1の予測（scripts/eval_dev_window.py predict の --out-dir。pred_clean.npz を読む）")
    s.add_argument("--skip-window", action="store_true", help="評価1を省く")
    s.add_argument("--skip-fast", action="store_true", help="評価2を省く")
    s.add_argument("--append-metrics", action="store_true")
    s.add_argument("--metrics-csv", default="results/metrics.csv")
    s.add_argument("--model-config", default=None, help="metrics.csv の config_path に書くモデルの設定")
    s.add_argument("--experiment-id", default=None, help="metrics.csv の実験ID（例 013-fast-speech-exp005）")
    s.add_argument("--method-name", default=None, help="metrics.csv の method")
    for q in (p, s):
        q.add_argument("--model-key", required=True, help="予測の配列名（npz のキー。例 exp005）")
        q.add_argument("--out-dir", required=True, help="dev_fast の予測と集計の置き場所（例 runs/exp005/fast_speech）")
        q.add_argument("--fast-config", default=FAST_CONFIG)
        q.add_argument("--conditions", default=None, help="x1.5,x2 の一部（既定はすべて）")
    p.set_defaults(func=cmd_predict)
    s.set_defaults(func=cmd_summarize)
    args = parser.parse_args(argv)
    if args.command == "summarize" and not args.skip_window and not args.window_eval_dir:
        parser.error("評価1には --window-eval-dir が要る（省くなら --skip-window）")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
