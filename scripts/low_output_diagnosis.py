"""低出力事例の切り分け（docs/PLAN.md 第6段階の追加診断）。

誤差上位100件（``results/error_cases/index.tsv``）のうち推定毎秒モーラ数が1.0未満の件を
対象群、dev 全件のうち毎秒モーラ数の絶対誤差が dev 全体の中央値以下の件から同数を
固定シードで無作為抽出した件を比較群とし、各クリップの

1. 波形の実効値・ピーク振幅・無音標本の割合
2. 標本化周波数（元ファイル・読み込み後）・チャンネル数・クリップ長
3. 対数メルスペクトログラムの平均・標準偏差・最大・最小
4. 正規化後の特徴量の平均・標準偏差・絶対値の最大
5. フレームごとの softplus 出力の要約統計

を算出し、群の差を Mann-Whitney U 検定（両側）と Holm 法で調べる。

## 推論の与え方（dev の評価と同じ）

``runs/exp001/eval_dev_full.py`` と同じく、``data/processed/features/dev`` の事前計算済み
float16 特徴量をクリップ全体のまま float32 に戻し、``configs/normalization.yaml``
（per_mel）で正規化して ``SpeechRateCNN.frame_counts`` に1件ずつ（詰め物なし）与える。
詰め物は ``lengths`` でマスクされ出力に影響しないため、バッチで与えた評価時と値は一致する
（一致の程度は結果に記録する）。

音声を聴く判断はしない。data/ 以下は読むだけで書き込まない。
``configs/splits/test.json`` は使わない。

実行:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/low_output_diagnosis.py \\
        > runs/diag006/low_output_diagnosis.log 2>&1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.data.splits import load_clip_records, load_split  # noqa: E402
from spkrate.eval.audio import load_audio  # noqa: E402
from spkrate.eval.low_output_diag import (  # noqa: E402
    FRAME_THRESHOLDS,
    SILENCE_AMPLITUDE_THRESHOLD,
    feature_stats,
    frame_output_stats,
    group_summary,
    holm_adjust,
    normalized_feature_stats,
    select_comparison,
    waveform_stats,
)
from spkrate.features.melspec import log_mel_spectrogram  # noqa: E402
from spkrate.train.data import FeatureClipDataset, load_normalization  # noqa: E402
from spkrate.train.train import load_checkpoint  # noqa: E402

INDEX_TSV = "results/error_cases/index.tsv"
PREDICTIONS = "runs/exp001/predictions_dev_full.json"
CHECKPOINT = "runs/exp001/checkpoint_best.pt"
NORMALIZATION = "configs/normalization.yaml"
CLIPS_JSONL = "data/processed/clips.jsonl"
DEV_SPLIT = "configs/splits/dev.json"
FEATURES_DEV = "data/processed/features/dev"
OUTPUT_TSV = "results/low_output_diagnosis.tsv"
OUTPUT_MD = "results/low_output_diagnosis.md"
OUTPUT_JSON = "runs/diag006/low_output_summary.json"

LOW_OUTPUT_MPS = 1.0
SEED = 20260921
ALPHA = 0.05
EXPECTED_TARGETS = 63

#: 検定する項目（列名, 表示名, 区分）
ITEMS: list[tuple[str, str, str]] = [
    ("rms", "波形の実効値", "1 波形"),
    ("peak", "ピーク振幅", "1 波形"),
    ("silence_frac", f"無音標本の割合（絶対値 < {SILENCE_AMPLITUDE_THRESHOLD:g}）", "1 波形"),
    ("orig_sample_rate", "元ファイルの標本化周波数（Hz）", "2 形式"),
    ("loaded_sample_rate", "読み込み後の標本化周波数（Hz）", "2 形式"),
    ("orig_channels", "元ファイルのチャンネル数", "2 形式"),
    ("loaded_channels", "読み込み後のチャンネル数", "2 形式"),
    ("orig_duration_sec", "元ファイルの長さ（秒）", "2 形式"),
    ("loaded_duration_sec", "読み込み後の長さ（秒）", "2 形式"),
    ("logmel_mean", "対数メルの平均", "3 対数メル"),
    ("logmel_std", "対数メルの標準偏差", "3 対数メル"),
    ("logmel_max", "対数メルの最大", "3 対数メル"),
    ("logmel_min", "対数メルの最小", "3 対数メル"),
    ("norm_mean", "正規化後の平均", "4 正規化後"),
    ("norm_std", "正規化後の標準偏差", "4 正規化後"),
    ("norm_absmax", "正規化後の絶対値の最大", "4 正規化後"),
    ("frame_mean", "フレーム出力の平均", "5 フレーム出力"),
    ("frame_median", "フレーム出力の中央値", "5 フレーム出力"),
    ("frame_p90", "フレーム出力の90%点", "5 フレーム出力"),
    ("frame_p99", "フレーム出力の99%点", "5 フレーム出力"),
    ("frame_max", "フレーム出力の最大", "5 フレーム出力"),
    ("frame_min", "フレーム出力の最小", "5 フレーム出力"),
    ("frame_frac_gt_0p01", "フレーム出力 >0.01 の割合", "5 フレーム出力"),
    ("frame_frac_gt_0p05", "フレーム出力 >0.05 の割合", "5 フレーム出力"),
    ("frame_top10pct_share", "上位10%フレームが総和に占める割合", "5 フレーム出力"),
    ("frame_sum", "フレーム出力の総和（推定モーラ数）", "5 フレーム出力"),
]

EXTRA_COLUMNS = ["group", "clip_id", "mps_true", "mps_pred_saved", "abs_error_mps",
                 "n_frames", "label_duration_sec", "mps_pred_recomputed",
                 "stored_vs_recomputed_logmel_maxabs"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_index(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------- 群の決定
    index_rows = read_index(INDEX_TSV)
    targets = [r["clip_id"] for r in index_rows if float(r["mora_per_second_pred"]) < LOW_OUTPUT_MPS]
    log(f"index.tsv {len(index_rows)} 件のうち推定 < {LOW_OUTPUT_MPS} mora/s: {len(targets)} 件")

    predictions: dict[str, float] = json.loads(Path(PREDICTIONS).read_text(encoding="utf-8"))
    dev_clients = set(load_split(DEV_SPLIT))
    records = {r.clip_id: r for r in load_clip_records(CLIPS_JSONL)
               if r.client_id in dev_clients and r.clip_id in predictions}
    if len(records) != len(predictions):
        raise ValueError(f"予測 {len(predictions)} 件と clips.jsonl の突き合わせ {len(records)} 件")
    abs_errors = {cid: abs(float(predictions[cid]) - float(r.mora_per_second))
                  for cid, r in records.items()}
    comparison, median_err, pool_size = select_comparison(
        abs_errors, len(targets), seed=args.seed, exclude=targets)
    log(f"比較群: 中央値 {median_err:.4f} mora/s 以下 {pool_size} 件から {len(comparison)} 件")

    # ---------------------------------------------------------------- 準備
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_checkpoint(CHECKPOINT, map_location="cpu")
    model = model.to(device).eval()
    normalizer = load_normalization(NORMALIZATION)
    dataset = FeatureClipDataset(FEATURES_DEV, client_ids=sorted(dev_clients), normalizer=None)
    position = {entry["clip_id"]: i for i, entry in enumerate(dataset.entries)}
    log(f"device={device} best_epoch={payload['epoch']} 正規化={normalizer.mode} "
        f"dev特徴量 {len(dataset)} 件")

    fallback_messages: list[str] = []
    rows: list[dict[str, object]] = []
    groups = [("target", cid) for cid in targets] + [("comparison", cid) for cid in comparison]
    for count, (group, cid) in enumerate(groups, start=1):
        record = records[cid]
        audio_path = ROOT / record.audio_path
        info = sf.info(str(audio_path))
        samples, loaded_sr = load_audio(audio_path)
        row: dict[str, object] = {
            "group": group,
            "clip_id": cid,
            "mps_true": float(record.mora_per_second),
            "mps_pred_saved": float(predictions[cid]),
            "abs_error_mps": abs_errors[cid],
            "orig_sample_rate": float(info.samplerate),
            "loaded_sample_rate": float(loaded_sr),
            "orig_channels": float(info.channels),
            "loaded_channels": 1.0 if samples.ndim == 1 else float(samples.shape[1]),
            "orig_duration_sec": float(info.frames) / float(info.samplerate),
            "loaded_duration_sec": float(samples.size) / float(loaded_sr),
            "label_duration_sec": float(record.duration_sec),
        }
        row.update(waveform_stats(samples))
        recomputed = log_mel_spectrogram(samples, loaded_sr)
        row.update(feature_stats(recomputed))
        stored = dataset[position[cid]].features  # float16 保存値を float32 に戻したもの
        if stored.shape == recomputed.shape:
            row["stored_vs_recomputed_logmel_maxabs"] = float(np.abs(stored - recomputed).max())
        else:
            row["stored_vs_recomputed_logmel_maxabs"] = float("nan")
            log(f"形状不一致 {cid}: stored {stored.shape} recomputed {recomputed.shape}")
        normalized = normalizer(stored)
        row.update(normalized_feature_stats(normalized))

        tensor = torch.from_numpy(np.ascontiguousarray(normalized[None], dtype=np.float32)).to(device)
        lengths = torch.tensor([normalized.shape[0]], dtype=torch.long, device=device)
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            frames = model.frame_counts(tensor, lengths)[0, : normalized.shape[0]]
            frames_np = frames.detach().to("cpu").numpy().astype(np.float32)
        for w in caught:
            text = str(w.message)
            if "MPS" in text or "fall back" in text.lower() or "fallback" in text.lower():
                fallback_messages.append(f"{w.filename}:{w.lineno}: {text}")
        row.update(frame_output_stats(frames_np))
        row["mps_pred_recomputed"] = float(row["frame_sum"]) / float(record.duration_sec)
        rows.append(row)
        if count % 20 == 0:
            log(f"{count}/{len(groups)} 件")

    # ---------------------------------------------------------------- 検定
    from scipy.stats import mannwhitneyu

    tests: list[dict[str, object]] = []
    for key, label, section in ITEMS:
        a = [float(r[key]) for r in rows if r["group"] == "target"]
        b = [float(r[key]) for r in rows if r["group"] == "comparison"]
        entry: dict[str, object] = {"key": key, "label": label, "section": section,
                                    "target": group_summary(a), "comparison": group_summary(b)}
        if len(set(a) | set(b)) <= 1:
            entry.update(U=math.nan, p=math.nan, note="両群とも全値が同一のため検定せず")
        else:
            result = mannwhitneyu(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32),
                                  alternative="two-sided")
            entry.update(U=float(result.statistic), p=float(result.pvalue), note="")
        tests.append(entry)
    adjusted = holm_adjust([float(t["p"]) for t in tests])
    for entry, p_holm in zip(tests, adjusted, strict=True):
        entry["p_holm"] = p_holm
        entry["significant"] = bool(not math.isnan(p_holm) and p_holm < ALPHA)

    # ---------------------------------------------------------------- 出力
    columns = EXTRA_COLUMNS + [k for k, _, _ in ITEMS if k not in EXTRA_COLUMNS]
    Path(OUTPUT_TSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as fh:
        fh.write("# 低出力事例の診断（クリップごとの値）。定義は results/low_output_diagnosis.md\n")
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] if isinstance(r[c], str) else f"{float(r[c]):.6g}" for c in columns])

    recompute_diff = [abs(float(r["mps_pred_recomputed"]) - float(r["mps_pred_saved"])) for r in rows]
    label_diff = [abs(float(r["label_duration_sec"]) - float(r["loaded_duration_sec"])) for r in rows]
    feature_diff = [float(r["stored_vs_recomputed_logmel_maxabs"]) for r in rows]
    summary = {
        "n_index_rows": len(index_rows),
        "n_target": len(targets),
        "expected_target": EXPECTED_TARGETS,
        "n_comparison": len(comparison),
        "seed": args.seed,
        "median_abs_error_mps_dev": median_err,
        "comparison_pool_size": pool_size,
        "n_dev": len(records),
        "device": str(device),
        "best_epoch": int(payload["epoch"]),
        "normalization_mode": normalizer.mode,
        "max_abs_diff_mps_pred": float(max(recompute_diff)),
        "max_abs_diff_label_vs_loaded_duration": float(max(label_diff)),
        "max_stored_vs_recomputed_logmel": float(np.nanmax(np.asarray(feature_diff, dtype=np.float32))),
        "mps_fallback_messages": sorted(set(fallback_messages)),
        "frame_distribution": frame_distribution(rows),
        "tests": tests,
    }
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(OUTPUT_MD).write_text(render_markdown(summary), encoding="utf-8")
    sig = [t["key"] for t in tests if t["significant"]]
    log(f"補正後に差がある項目: {sig}")
    log(f"MPS フォールバック警告: {len(set(fallback_messages))} 種")
    log("LOW_OUTPUT_DONE")
    return 0


def frame_distribution(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    """群ごとに、フレーム出力の形で分けたクリップ数を数える。"""
    out: dict[str, dict[str, int]] = {}
    for group in ("target", "comparison"):
        members = [r for r in rows if r["group"] == group]
        out[group] = {
            "n": len(members),
            "max_le_0p01": sum(float(r["frame_max"]) <= 0.01 for r in members),
            "max_le_0p05": sum(float(r["frame_max"]) <= 0.05 for r in members),
            "frac_gt_0p01_eq_0": sum(float(r["frame_frac_gt_0p01"]) == 0.0 for r in members),
            "frac_gt_0p01_lt_0p1": sum(float(r["frame_frac_gt_0p01"]) < 0.1 for r in members),
            "frac_gt_0p01_ge_0p5": sum(float(r["frame_frac_gt_0p01"]) >= 0.5 for r in members),
            "frac_gt_0p05_eq_0": sum(float(r["frame_frac_gt_0p05"]) == 0.0 for r in members),
        }
    return out


def fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.4g}"
    if magnitude >= 1e-3:
        return f"{value:.4f}"
    return f"{value:.3e}"


def fmt_p(value: float) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.3e}" if value < 1e-3 else f"{value:.4f}"


def render_markdown(s: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 低出力事例の切り分け（第6段階 追加診断）")
    add("")
    add("- 生成: `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/low_output_diagnosis.py`"
        "（計算は `src/spkrate/eval/low_output_diag.py`）")
    add(f"- モデル: `{CHECKPOINT}`（最良エポック {s['best_epoch']}）。デバイス `{s['device']}`")
    add("- 対象は dev のみ。`configs/splits/test.json` は使っていない。音声は聴いていない")
    add("- **本文書は数値と事実のみを記す。原因の推測・改善案は書かない**")
    add("- クリップごとの値: `results/low_output_diagnosis.tsv`")
    add("")
    add("## 群の定義")
    add("")
    add(f"- **対象群**: `{INDEX_TSV}` の {s['n_index_rows']} 件のうち `mora_per_second_pred < "
        f"{LOW_OUTPUT_MPS}` の件。**{s['n_target']} 件**"
        + ("（想定の63件と一致）" if s["n_target"] == EXPECTED_TARGETS
           else f"（**想定の{EXPECTED_TARGETS}件と異なる**）"))
    add(f"- **比較群**: dev 全 {s['n_dev']:,} 件の予測（`{PREDICTIONS}`）について毎秒モーラ数の"
        f"絶対誤差を求め、その中央値 **{s['median_abs_error_mps_dev']:.4f} mora/s 以下**の "
        f"{s['comparison_pool_size']:,} 件（対象群を除く）から **{s['n_comparison']} 件**を無作為抽出")
    add(f"  - 抽出: 候補を clip_id 順に並べ `np.random.default_rng({s['seed']}).choice(候補数, "
        f"size={s['n_comparison']}, replace=False)`。シード **{s['seed']}**")
    add("")
    add("## 各値の算出方法")
    add("")
    add("| 区分 | 算出方法 |")
    add("| --- | --- |")
    add("| 1 波形 | `spkrate.eval.audio.load_audio`（soundfile で読み、モノラル化、librosa で16kHzへ再標本化。"
        "学習・評価の特徴量計算と同じ読み込み）後の float32 波形。実効値 = sqrt(mean(x²))、ピーク = 振幅の絶対値の最大、"
        f"無音標本の割合 = 振幅の絶対値が **{SILENCE_AMPLITUDE_THRESHOLD:g}** 未満（-60dBFS）の標本の割合 |")
    add("| 2 形式 | 元ファイル: `soundfile.info` の samplerate・channels・frames/samplerate。"
        "読み込み後: `load_audio` の戻り値（16kHzモノラル）の標本化周波数・チャンネル数・長さ |")
    add("| 3 対数メル | 上の波形から `spkrate.features.melspec.log_mel_spectrogram`（docs/spec.md の設定。"
        "`scripts/precompute_features.py` と同じ関数）で計算し直した値の、全フレーム×80次元での統計 |")
    add("| 4 正規化後 | dev 評価の入力そのもの: `data/processed/features/dev` の float16 保存値を float32 に戻し、"
        "`configs/normalization.yaml`（mode=per_mel、80次元ごとの平均・標準偏差）で `(x-mean)/std` した値の統計 |")
    add("| 5 フレーム出力 | 4 の特徴量をクリップ全体のまま1件ずつ（詰め物なし、lengths=全フレーム数）"
        "`SpeechRateCNN.frame_counts` に与えた softplus 出力（モーラ/フレーム、1フレーム10ミリ秒）。"
        "dev 評価（`runs/exp001/eval_dev_full.py`）もクリップ全体を lengths 付きで与えており、"
        "詰め物は出力に影響しない。閾値は **0.01**（=1 mora/s 相当）と **0.05**（=5 mora/s 相当）|")
    add("")
    add("入力の再現性の確認:")
    add("")
    add(f"- 保存済み特徴量（float16）と計算し直した対数メルの差の絶対値の最大: "
        f"{s['max_stored_vs_recomputed_logmel']:.4g}（全{s['n_target'] + s['n_comparison']}件中の最大）")
    add(f"- `{CLIPS_JSONL}` の duration_sec（毎秒モーラ数の分母）と読み込み後の長さの差の絶対値の最大: "
        f"{s['max_abs_diff_label_vs_loaded_duration']:.3e} 秒")
    add("- 元ファイルの長さは `soundfile.info` の frames（mp3 のヘッダ・復号器が報告する値）から求めたもので、"
        "読み込み後（実際に復号した標本数から求めた長さ）とは一致しない場合がある")
    add(f"- フレーム出力の総和 ÷ クリップ長 と `{PREDICTIONS}` の保存値の差の絶対値の最大: "
        f"{s['max_abs_diff_mps_pred']:.3e} mora/s")
    fb = s["mps_fallback_messages"]
    add(f"- MPS の CPU フォールバック警告: " + ("**なし**" if not fb else f"**{len(fb)} 種**"))
    for message in fb:
        add(f"  - `{message}`")
    add("")
    add("## 群ごとの要約と検定")
    add("")
    add(f"Mann-Whitney U 検定（両側）。Holm 法で補正（族は検定した {sum(1 for t in s['tests'] if not math.isnan(t['p']))} 項目）。"
        f"有意水準 {ALPHA}。**太字の項目**が補正後 p < {ALPHA}。値は 中央値 [第1四分位, 第3四分位]。")
    add("")
    add("| 区分 | 項目 | 対象群 | 比較群 | U | p | p（Holm） | 補正後の差 |")
    add("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for t in s["tests"]:
        a, b = t["target"], t["comparison"]
        label = f"**{t['label']}**" if t["significant"] else t["label"]
        verdict = "**あり**" if t["significant"] else ("—" if math.isnan(t["p"]) else "なし")
        u = "—" if math.isnan(t["U"]) else f"{t['U']:.0f}"
        note = f"（{t['note']}）" if t["note"] else ""
        add(f"| {t['section']} | {label}{note} | {fmt(a['median'])} [{fmt(a['q1'])}, {fmt(a['q3'])}] | "
            f"{fmt(b['median'])} [{fmt(b['q1'])}, {fmt(b['q3'])}] | {u} | {fmt_p(t['p'])} | "
            f"{fmt_p(t['p_holm'])} | {verdict} |")
    add("")
    add("最小・最大:")
    add("")
    add("| 項目 | 対象群 最小 | 対象群 最大 | 比較群 最小 | 比較群 最大 |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for t in s["tests"]:
        a, b = t["target"], t["comparison"]
        add(f"| {t['label']} | {fmt(a['min'])} | {fmt(a['max'])} | {fmt(b['min'])} | {fmt(b['max'])} |")
    add("")
    sig = [t["label"] for t in s["tests"] if t["significant"]]
    add(f"**補正後に差がある項目（{len(sig)} 項目）**: " + ("、".join(sig) if sig else "なし"))
    add("")
    add("## フレーム出力の形（全フレームが低いのか、一部のみか）")
    add("")
    d = s["frame_distribution"]
    add("クリップ数。閾値は モーラ/フレーム。")
    add("")
    add("| 条件 | 対象群 | 比較群 |")
    add("| --- | ---: | ---: |")
    rows_def = [
        ("件数", "n"),
        ("フレーム出力の最大 ≤ 0.01（全フレームが 0.01 以下）", "max_le_0p01"),
        ("フレーム出力の最大 ≤ 0.05（全フレームが 0.05 以下）", "max_le_0p05"),
        ("0.01 を超えるフレームが 0 個", "frac_gt_0p01_eq_0"),
        ("0.01 を超えるフレームの割合 < 10%", "frac_gt_0p01_lt_0p1"),
        ("0.01 を超えるフレームの割合 ≥ 50%", "frac_gt_0p01_ge_0p5"),
        ("0.05 を超えるフレームが 0 個", "frac_gt_0p05_eq_0"),
    ]
    for label, key in rows_def:
        add(f"| {label} | {d['target'][key]} | {d['comparison'][key]} |")
    add("")
    add("参考: 2.0秒のデジタル無音窓（201フレーム）の出力は 0.0824 モーラ（`results/window_diagnostics.md` D2）、"
        "1フレームあたり約 4.1e-4。")
    add("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
