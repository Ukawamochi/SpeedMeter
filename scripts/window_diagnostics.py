"""D1・D2・D3 の3診断を実施する（docs/decisions/005-window-strategy.md 4.3節）。

第4段階4-3が第5段階5-4に課した診断で、5-4では前向き計算ができず未測定のまま残った
（`docs/experiments/003-first-model.md` 7節の申し送り1）。第6段階の測定サブエージェントが
実施する。計算の骨格は `src/spkrate/eval/window_diag.py` にある。

| 記号 | 対象 | 出力 |
| --- | --- | --- |
| D1 分割整合性 | dev の4.0秒以上（13,322件）から無作為1,000件 | ``mean(|Σ − P|) / (2.0m)``（mora/s） |
| D2 無音・雑音窓 | デジタル無音の2.0秒窓500本、MUSAN noise の2.0秒窓500本 | 予測モーラ数の平均（mora） |
| D3 連結加法性 | dev から2件ずつ500組（間に0.2秒の無音） | ``mean(|差|) / 連結後の長さ``（mora/s） |

方式Bへ移る条件の閾値（B1: D1>0.25 mora/s、B2: D2>1.0 モーラ、B5: D3>0.25 mora/s かつ
D1>0.25）に照らして**超過の有無だけ**を記録する。採否の判断は第7段階7-1の仕事なので書かない。

`configs/splits/test.json` は使わない。`data/` は読むだけで変更しない。

実行（背景で実行しログをファイルへ）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/window_diagnostics.py \\
        > /tmp/window_diagnostics.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from spkrate.data.augment import MusanNoiseSource
from spkrate.eval.audio import load_audio
from spkrate.eval.window_diag import (
    GAP_SEC,
    SAMPLE_RATE,
    WINDOW_SEC,
    concat_additivity,
    make_predictor,
    split_consistency,
    summarize,
)
from spkrate.train.data import load_normalization, select_clip_records
from spkrate.train.train import MpsFallbackWatcher, load_checkpoint

CHECKPOINT = "runs/exp001/checkpoint_best.pt"
EXPERIMENT_ID = "005-first-model"
CLIPS_JSONL = "data/processed/clips.jsonl"
DEV_SPLIT = "configs/splits/dev.json"
MUSAN_ROOT = "data/musan"
NORMALIZATION = "configs/normalization.yaml"
OUTPUT_MD = "results/window_diagnostics.md"
OUTPUT_JSON = "results/window_diagnostics.json"

D1_MIN_DURATION_SEC = 4.0
D1_CLIPS = 1000
D2_WINDOWS = 500
D3_PAIRS = 500
SEED = 20260921

#: 方式Bへ移る条件の閾値（005 5節）。判断はしない。超過の有無だけを記録する。
THRESHOLD_B1 = 0.25  # D1（mora/s）
THRESHOLD_B2 = 1.0  # D2（mora）
THRESHOLD_B5 = 0.25  # D3（mora/s）。D1 との併発時のみ


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_waveforms(records) -> list[tuple[str, np.ndarray]]:
    """クリップを16kHzモノラルで読む（``audio_path`` はリポジトリ直下からの相対。読み取りのみ）。"""
    loaded: list[tuple[str, np.ndarray]] = []
    for index, record in enumerate(records):
        samples, sample_rate = load_audio(record.audio_path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"標本化周波数が16kHzでない: {sample_rate}")
        loaded.append((record.clip_id, np.asarray(samples, dtype=np.float32)))
        if (index + 1) % 250 == 0:
            log(f"  読み込み {index + 1}/{len(records)}")
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d1-clips", type=int, default=D1_CLIPS)
    parser.add_argument("--d2-windows", type=int, default=D2_WINDOWS)
    parser.add_argument("--d3-pairs", type=int, default=D3_PAIRS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--output-json", default=OUTPUT_JSON)
    parser.add_argument("--output-md", default=OUTPUT_MD)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("window_diagnostics")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device={device} mps_available={torch.backends.mps.is_available()}")
    if device.type != "mps":
        logger.warning("mps が使えないため %s で実行する（CLAUDE.md の規定は mps）", device)

    model, payload = load_checkpoint(args.checkpoint, map_location="cpu")
    model = model.to(device).eval()
    log(f"checkpoint={args.checkpoint} best_epoch={payload['epoch']} "
        f"receptive_field={model.receptive_field_frames}フレーム")

    normalizer = load_normalization(NORMALIZATION)
    predict = make_predictor(model, normalizer, device, batch_size=args.batch_size)

    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "experiment_id": args.experiment_id,
        "best_epoch": int(payload["epoch"]),
        "device": str(device),
        "seed": int(args.seed),
        "window_sec": WINDOW_SEC,
        "gap_sec": GAP_SEC,
        "receptive_field_frames": int(model.receptive_field_frames),
    }

    with MpsFallbackWatcher(logger) as watcher:
        # ------------------------------------------------------------------ D1
        log("D1: dev の4.0秒以上のクリップを列挙する")
        pool = select_clip_records(CLIPS_JSONL, DEV_SPLIT,
                                   min_duration_sec=D1_MIN_DURATION_SEC)
        log(f"D1: 母集団 {len(pool)} 件（dev の {D1_MIN_DURATION_SEC} 秒以上）")
        chosen = [pool[i] for i in rng.choice(len(pool), size=min(args.d1_clips, len(pool)),
                                              replace=False)]
        d1_waveforms = load_waveforms(chosen)
        started = time.perf_counter()
        d1_results = split_consistency(predict, d1_waveforms)
        log(f"D1: 推論 {time.perf_counter() - started:.1f} 秒")
        d1_errors = [r.error_mora_per_sec for r in d1_results]
        d1_signed = [r.diff / r.covered_sec for r in d1_results]
        summary["D1"] = {
            "population": len(pool),
            "sampled": len(d1_results),
            "min_duration_sec": D1_MIN_DURATION_SEC,
            "abs_error_mora_per_sec": summarize(d1_errors),
            "signed_error_mora_per_sec_mean": float(np.mean(
                np.asarray(d1_signed, dtype=np.float32))),
            "mean_num_windows": float(np.mean(
                np.asarray([r.num_windows for r in d1_results], dtype=np.float32))),
            "mean_covered_sec": float(np.mean(
                np.asarray([r.covered_sec for r in d1_results], dtype=np.float32))),
            "mean_sum_windows_mora": float(np.mean(
                np.asarray([r.sum_windows for r in d1_results], dtype=np.float32))),
            "mean_whole_mora": float(np.mean(
                np.asarray([r.whole for r in d1_results], dtype=np.float32))),
            "threshold_B1": THRESHOLD_B1,
            "exceeds_B1": bool(float(np.mean(np.asarray(d1_errors, dtype=np.float32)))
                               > THRESHOLD_B1),
        }
        log(f"D1: mean={summary['D1']['abs_error_mora_per_sec']['mean']:.4f} mora/s")

        # ------------------------------------------------------------------ D2
        log("D2: 無音窓と MUSAN noise の窓を推論する")
        window_samples = int(round(WINDOW_SEC * SAMPLE_RATE))
        silence = [np.zeros(window_samples, dtype=np.float32) for _ in range(args.d2_windows)]
        silence_pred = predict(silence)
        noise_source = MusanNoiseSource(MUSAN_ROOT)
        log(f"D2: MUSAN noise のファイル数 {len(noise_source.paths)}")
        noise_rng = np.random.default_rng(args.seed + 1)
        noise = [np.asarray(noise_source.sample(window_samples, noise_rng), dtype=np.float32)
                 for _ in range(args.d2_windows)]
        noise_rms = [float(np.sqrt(np.mean(np.square(w, dtype=np.float32)))) for w in noise]
        noise_pred = predict(noise)
        # 参考（005の定義には無い追加測定）。005 は MUSAN noise の音量を指定していないので、
        # 元の振幅のままの値に加え、dev の音声と実効値を揃えた場合の値も測っておく。
        speech_rms = [
            float(np.sqrt(np.mean(np.square(w, dtype=np.float32))))
            for _, w in d1_waveforms
        ]
        target_rms = float(np.median(np.asarray(speech_rms, dtype=np.float32)))
        scaled = [
            np.asarray(w * np.float32(target_rms / max(r, 1e-8)), dtype=np.float32)
            for w, r in zip(noise, noise_rms, strict=True)
        ]
        scaled_pred = predict(scaled)
        d2_value = max(float(np.mean(silence_pred)), float(np.mean(noise_pred)))
        summary["D2"] = {
            "silence_mora": summarize(silence_pred.tolist()),
            "musan_noise_mora": summarize(noise_pred.tolist()),
            "musan_noise_rms": summarize(noise_rms),
            "dev_speech_rms_median": target_rms,
            "musan_noise_rms_matched_mora": summarize(scaled_pred.tolist()),
            "value_for_threshold_mora": d2_value,
            "threshold_B2": THRESHOLD_B2,
            "exceeds_B2_silence": bool(float(np.mean(silence_pred)) > THRESHOLD_B2),
            "exceeds_B2_musan_noise": bool(float(np.mean(noise_pred)) > THRESHOLD_B2),
            "exceeds_B2": bool(d2_value > THRESHOLD_B2),
        }
        log(f"D2: silence_mean={float(np.mean(silence_pred)):.4f} "
            f"noise_mean={float(np.mean(noise_pred)):.4f} "
            f"noise_rms_matched_mean={float(np.mean(scaled_pred)):.4f} mora")

        # ------------------------------------------------------------------ D3
        log("D3: dev から2件ずつ連結する")
        dev_all = select_clip_records(CLIPS_JSONL, DEV_SPLIT)
        log(f"D3: 母集団 {len(dev_all)} 件（dev 全件）")
        need = min(args.d3_pairs * 2, len(dev_all) - len(dev_all) % 2)
        picked = [dev_all[i] for i in rng.choice(len(dev_all), size=need, replace=False)]
        d3_waveforms = load_waveforms(picked)
        pairs = [
            (f"{d3_waveforms[2 * i][0]}+{d3_waveforms[2 * i + 1][0]}",
             d3_waveforms[2 * i][1], d3_waveforms[2 * i + 1][1])
            for i in range(len(d3_waveforms) // 2)
        ]
        started = time.perf_counter()
        d3_results = concat_additivity(predict, pairs)
        log(f"D3: 推論 {time.perf_counter() - started:.1f} 秒")
        d3_errors = [r.error_mora_per_sec for r in d3_results]
        summary["D3"] = {
            "pairs": len(d3_results),
            "gap_sec": GAP_SEC,
            "abs_error_mora_per_sec": summarize(d3_errors),
            "signed_diff_mora_mean": float(np.mean(
                np.asarray([r.diff for r in d3_results], dtype=np.float32))),
            "abs_diff_mora": summarize([abs(r.diff) for r in d3_results]),
            "mean_total_sec": float(np.mean(
                np.asarray([r.total_sec for r in d3_results], dtype=np.float32))),
            "mean_gap_mora": float(np.mean(
                np.asarray([r.gap for r in d3_results], dtype=np.float32))),
            "threshold_B5": THRESHOLD_B5,
            "exceeds_B5_own_threshold": bool(
                float(np.mean(np.asarray(d3_errors, dtype=np.float32))) > THRESHOLD_B5),
        }
        log(f"D3: mean={summary['D3']['abs_error_mora_per_sec']['mean']:.4f} mora/s")

    summary["mps_cpu_fallback_events"] = watcher.events
    summary["output_json"] = args.output_json
    log(f"MPSのCPUフォールバック: {len(watcher.events)} 件")

    Path(args.output_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.output_md).write_text(render_markdown(summary), encoding="utf-8")
    log(f"書き出し: {args.output_json} / {args.output_md}")
    log("DIAGNOSTICS_DONE")
    return 0


def render_markdown(s: dict) -> str:
    """results/window_diagnostics.md を組み立てる（事実のみ。判断は書かない）。"""
    d1, d2, d3 = s["D1"], s["D2"], s["D3"]
    b1 = "超過" if d1["exceeds_B1"] else "超過しない"
    b2 = "超過" if d2["exceeds_B2"] else "超過しない"
    b2_silence = "超過" if d2["exceeds_B2_silence"] else "超過しない"
    b2_noise = "超過" if d2["exceeds_B2_musan_noise"] else "超過しない"
    b5_own = "超過" if d3["exceeds_B5_own_threshold"] else "超過しない"
    b5 = "該当" if (d3["exceeds_B5_own_threshold"] and d1["exceeds_B1"]) else "該当しない"
    events = s["mps_cpu_fallback_events"]
    fallback = (
        "無し"
        if not events
        else "\n".join(f"  - 演算子 `{e['operator']}` 発生箇所 `{e['location']}`" for e in events)
    )
    return f"""\
# 窓の切り出し方式の診断 D1・D2・D3（docs/decisions/005-window-strategy.md 4.3節）

- 実施: 第6段階の測定サブエージェント（第4段階4-3が第5段階5-4に課した診断。5-4では前向き計算ができず未測定だった）
- 生成: `uv run python scripts/window_diagnostics.py`（計算の骨格は `src/spkrate/eval/window_diag.py`）
- モデル: `{s['checkpoint']}`（実験ID `{s['experiment_id']}`、最良エポック {s['best_epoch']}、受容野 {s['receptive_field_frames']} フレーム）
- デバイス: `{s['device']}`。窓長 {s['window_sec']} 秒、D3の無音 {s['gap_sec']} 秒、乱数の種 {s['seed']}
- 対象は dev のみ。`configs/splits/test.json` は使っていない
- **本文書は測定値と閾値超過の有無だけを記録する。結果の解釈・採否の判断は書かない**（第7段階7-1の仕事）

---

## D1 分割整合性

先頭から重なりなしに2.0秒窓を m = floor(T/2.0) 個とり、各窓を単独で推論した値の総和 Σ と、
同じ 2.0m 秒の区間を一括で推論した値 P を比べる。診断値は `|Σ − P| / (2.0m)`（mora/s）。

| 項目 | 値 |
| --- | ---: |
| 母集団（dev の4.0秒以上） | {d1['population']:,} 件 |
| 標本 | {d1['sampled']:,} 件（無作為） |
| 1件あたりの窓数（平均） | {d1['mean_num_windows']:.3f} |
| 1件あたりの対象区間長（平均） | {d1['mean_covered_sec']:.3f} 秒 |
| Σ の平均 | {d1['mean_sum_windows_mora']:.4f} モーラ |
| P の平均 | {d1['mean_whole_mora']:.4f} モーラ |
| **D1 = mean(\\|Σ − P\\|/(2.0m))** | **{d1['abs_error_mora_per_sec']['mean']:.4f} mora/s** |
| 中央値 | {d1['abs_error_mora_per_sec']['median']:.4f} mora/s |
| 第9十分位 | {d1['abs_error_mora_per_sec']['p90']:.4f} mora/s |
| 最大 | {d1['abs_error_mora_per_sec']['max']:.4f} mora/s |
| 標準偏差 | {d1['abs_error_mora_per_sec']['std']:.4f} mora/s |
| 符号つき (Σ − P)/(2.0m) の平均 | {d1['signed_error_mora_per_sec_mean']:+.4f} mora/s |

- 閾値 **B1: D1 > {d1['threshold_B1']} mora/s** に対して **{b1}**（測定値 {d1['abs_error_mora_per_sec']['mean']:.4f} vs 閾値 {d1['threshold_B1']}）

## D2 無音・雑音窓の出力

デジタル無音の2.0秒窓と MUSAN noise のみの2.0秒窓を推論する。仕様は「無音は0モーラ」。

| 入力 | 本数 | 平均（モーラ） | 中央値 | 最大 | 標準偏差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| デジタル無音 | {d2['silence_mora']['count']} | **{d2['silence_mora']['mean']:.4f}** | {d2['silence_mora']['median']:.4f} | {d2['silence_mora']['max']:.4f} | {d2['silence_mora']['std']:.4f} |
| MUSAN noise | {d2['musan_noise_mora']['count']} | **{d2['musan_noise_mora']['mean']:.4f}** | {d2['musan_noise_mora']['median']:.4f} | {d2['musan_noise_mora']['max']:.4f} | {d2['musan_noise_mora']['std']:.4f} |

MUSAN noise の窓は元ファイルの振幅のままで、音量を合わせる操作はしていない（005 は D2 の音量を指定していない）。
実効値は平均 {d2['musan_noise_rms']['mean']:.5f}、中央値 {d2['musan_noise_rms']['median']:.5f}、最大 {d2['musan_noise_rms']['max']:.5f}。

**参考（005の定義には無い追加測定）**: 同じ500本を dev 音声の実効値の中央値 {d2['dev_speech_rms_median']:.5f} に
揃え直して推論すると、平均 {d2['musan_noise_rms_matched_mora']['mean']:.4f} モーラ（中央値 {d2['musan_noise_rms_matched_mora']['median']:.4f}、最大 {d2['musan_noise_rms_matched_mora']['max']:.4f}）。

閾値 **B2: D2 > {d2['threshold_B2']} モーラ** に対して:

- デジタル無音 {d2['silence_mora']['mean']:.4f} モーラ → **{b2_silence}**
- MUSAN noise {d2['musan_noise_mora']['mean']:.4f} モーラ → **{b2_noise}**
- 2種のうち大きい方 {d2['value_for_threshold_mora']:.4f} モーラ → **{b2}**

## D3 連結加法性

dev から2件ずつ組にして 0.2秒の無音を挟んで連結し、`P(AB)` と `P(A)+P(B)+P(無音)` を比べる。
診断値は `|差| / 連結後の長さ`（mora/s）。

| 項目 | 値 |
| --- | ---: |
| 組数 | {d3['pairs']} 組（母集団は dev 全件） |
| 連結後の長さ（平均） | {d3['mean_total_sec']:.3f} 秒 |
| P(無音 {d3['gap_sec']}秒) の平均 | {d3['mean_gap_mora']:.4f} モーラ |
| 差 `P(AB) − (P(A)+P(B)+P(無音))` の平均（符号つき） | {d3['signed_diff_mora_mean']:+.4f} モーラ |
| \\|差\\| の平均 | {d3['abs_diff_mora']['mean']:.4f} モーラ |
| **D3 = mean(\\|差\\|/連結後の長さ)** | **{d3['abs_error_mora_per_sec']['mean']:.4f} mora/s** |
| 中央値 | {d3['abs_error_mora_per_sec']['median']:.4f} mora/s |
| 第9十分位 | {d3['abs_error_mora_per_sec']['p90']:.4f} mora/s |
| 最大 | {d3['abs_error_mora_per_sec']['max']:.4f} mora/s |

- 閾値 **B5: D3 > {d3['threshold_B5']} mora/s**（D1 との併発時のみ）に対して、D3 単独では **{b5_own}**（測定値 {d3['abs_error_mora_per_sec']['mean']:.4f}）。D1 との併発を含めた B5 の成立は **{b5}**

---

## 閾値に対する超過の有無（まとめ）

| 条件 | 閾値 | 測定値 | 超過 |
| --- | ---: | ---: | --- |
| B1 | D1 > {d1['threshold_B1']} mora/s | D1={d1['abs_error_mora_per_sec']['mean']:.4f} | **{b1}** |
| B2（デジタル無音） | D2 > {d2['threshold_B2']} モーラ | {d2['silence_mora']['mean']:.4f} | **{b2_silence}** |
| B2（MUSAN noise） | D2 > {d2['threshold_B2']} モーラ | {d2['musan_noise_mora']['mean']:.4f} | **{b2_noise}** |
| B5 | D3 > {d3['threshold_B5']} mora/s かつ B1 成立 | D3={d3['abs_error_mora_per_sec']['mean']:.4f} / B1={'成立' if d1['exceeds_B1'] else '不成立'} | **{b5}** |

採否の判断は行わない（第7段階7-1）。

## 実行環境

- デバイス `{s['device']}`。float64 は使っていない。`torch.compile` は使っていない
- MPS未対応演算によるCPUフォールバック: {fallback}
- 生の値は `{s['output_json']}` にある
"""


if __name__ == "__main__":
    raise SystemExit(main())
