#!/usr/bin/env bash
# 第10段階（テストセットでの評価、2026-10-02 の人間の指示）の一連の実行。遠隔機（ubuntu-desktop）の
# ~/SpeedMeter で実行する。候補モデルが決まってから、統括Agentが呼ぶ。
#
# 使い方:
#   scripts/run_test_eval.sh <実験ID> <チェックポイント> <設定yaml> [<method の文字列>]
#   例: scripts/run_test_eval.sh exp005 runs/exp005/checkpoint_best.pt configs/exp005.yaml
#
# 出力: runs/test_eval/<実験ID>/（status.txt、各段のログ、window_eval/、clip/、fast_speech/、
#   window_diagnostics_test.json・.md、metrics_rows_test.csv）。
#   metrics.csv の正本は Mac の results/metrics.csv。ここでは写し（<出力>/metrics_work.csv）に追記し、
#   追記された行だけを metrics_rows_test.csv に取り出す（Mac で統括Agentが metrics.csv に足す）。
#   実験IDの列は <実験ID>-test-window-eval（窓単位）・<実験ID>-test-full（クリップ単位）・
#   <実験ID>-test-fast-speech（高速度域）。split 列は test・test_noisy_*・test_window*・test_fast_* 。
# 準備の成果物（既にあれば飛ばす。data/processed/test_prep/<段>.done が印）:
#   features/test/、alignments/test.jsonl、test_window/、test_fast/、no_speech/test_metrics.jsonl・suspect_test.tsv。
#   準備の途中で止まった場合は、特徴量・アライメント・無発話指標は同じコマンドの再実行で続きから進む
#   （既存の clip_id・シャードを飛ばす）。test_window・test_fast は出力先が空でないと作り直しを拒むので、
#   途中で止まったときは人間が出力先を確かめてから消す（このスクリプトは data/ を消さない）。
# 状態: <出力>/status.txt に段ごとに記録する。どれかの段が失敗したらその場で 0 以外で終わる。
#   status.txt が既にあれば上書きせず終わる（再実行は別の実験ID か、status.txt を人間が退避してから）。
# git を操作しない。MPS は使わない（cuda 固定）。
set -uo pipefail
unset VIRTUAL_ENV
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ $# -lt 3 ]; then
  echo "使い方: $0 <実験ID> <チェックポイント> <設定yaml> [<methodの文字列>]" >&2
  exit 2
fi
ID=$1
CK=$2
CFG=$3
cd ~/SpeedMeter || exit 1
UV=~/.local/bin/uv
DEVICE=cuda
M=${4:-"cnn ($ID) テストセット評価（第10段階）"}
OUT=runs/test_eval/$ID
PREP=data/processed/test_prep
mkdir -p "$OUT" "$PREP"
if [ -e "$OUT/status.txt" ]; then echo "$OUT/status.txt が既にある。上書きしない" >&2; exit 3; fi
if [ ! -f "$CK" ]; then echo "チェックポイントが無い: $CK" >&2; exit 4; fi
if [ ! -f "$CFG" ]; then echo "設定が無い: $CFG" >&2; exit 4; fi
if [ ! -f configs/splits/test.json ]; then echo "configs/splits/test.json が無い" >&2; exit 4; fi
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "START $(date -Iseconds) commit=$COMMIT host=$(hostname -s) device=$DEVICE id=$ID checkpoint=$CK config=$CFG" > "$OUT/status.txt"

# 1段を実行して status.txt に記録する。失敗したらそこで終わる。
step() {
  local name=$1 log=$2; shift 2
  local S=$(date +%s)
  "$@" > "$OUT/$log" 2>&1
  local rc=$?
  echo "$name rc=$rc sec=$(( $(date +%s) - S )) $(date -Iseconds)" >> "$OUT/status.txt"
  if [ $rc -ne 0 ]; then echo "FAIL $name rc=$rc" >> "$OUT/status.txt"; exit $rc; fi
}
# 準備の段。印があれば飛ばす。成功したら印を付ける。
prep() {
  local name=$1 log=$2; shift 2
  if [ -e "$PREP/$name.done" ]; then
    echo "SKIP $name（$PREP/$name.done がある）" >> "$OUT/status.txt"
    return 0
  fi
  step "$name" "$log" "$@"
  date -Iseconds > "$PREP/$name.done"
}

# ------------------------------------------------------------------ 準備（1〜4）
prep prep_features prep_features.log $UV run --frozen python scripts/precompute_features.py run --splits test
prep prep_align prep_align.log $UV run --frozen python scripts/align_dev.py --split test --device $DEVICE
prep prep_align_inspect prep_align_inspect.log $UV run --frozen python scripts/inspect_alignment_dev.py \
  --alignments data/processed/alignments/test.jsonl --out $PREP/alignment_inspect_test.json
prep prep_window prep_window.log $UV run --frozen python scripts/build_dev_window.py \
  --config configs/eval/test_window.yaml --check-waveforms 4
prep prep_fast prep_fast.log $UV run --frozen python scripts/build_dev_fast.py --config configs/eval/test_fast.yaml
prep prep_fast_verify prep_fast_verify.log $UV run --frozen python scripts/build_dev_fast.py \
  --config configs/eval/test_fast.yaml --verify 20
prep prep_no_speech_detect prep_no_speech_detect.log $UV run --frozen python scripts/detect_no_speech.py \
  --split test --device $DEVICE --log-every 1000
prep prep_no_speech_apply prep_no_speech_apply.log $UV run --frozen python scripts/analyze_no_speech.py apply --split test

# ------------------------------------------------------------------ 評価（5）
WORK=$OUT/metrics_work.csv
cp results/metrics.csv "$WORK"
step window_predict window_predict.log $UV run --frozen python scripts/eval_dev_window.py predict --split test \
  --out-dir "$OUT/window_eval" --model-key "$ID" --checkpoint "$CK" --no-envelope --device $DEVICE
step window_summarize window_summarize.log $UV run --frozen python scripts/eval_dev_window.py summarize --split test \
  --out-dir "$OUT/window_eval" --model-key "$ID" --checkpoint "$CK" --no-envelope --model-config "$CFG" \
  --experiment-id "$ID-test-window-eval" --method-name "$M 2.0秒窓ごと" --append-metrics --metrics-csv "$WORK"
step window_diagnostics window_diagnostics.log $UV run --frozen python scripts/window_diagnostics.py --split test \
  --checkpoint "$CK" --experiment-id "$ID-test" --musan-noise-split configs/splits/musan_noise.json \
  --d1-noisy-config configs/eval/test_noisy.yaml \
  --output-json "$OUT/window_diagnostics_test.json" --output-md "$OUT/window_diagnostics_test.md" --device $DEVICE
step eval_dev_full eval_dev_full.log $UV run --frozen python scripts/eval_dev_full.py --split test \
  --run-dir "$(dirname "$CK")" --checkpoint "$CK" --experiment-id "$ID-test-full" --config "$CFG" --method "$M" \
  --output-dir "$OUT/clip" --metrics-csv "$WORK" --device $DEVICE
step fast_predict fast_predict.log $UV run --frozen python scripts/eval_fast_speech.py predict --split test \
  --checkpoint "$CK" --model-key "$ID" --device $DEVICE --out-dir "$OUT/fast_speech"
step fast_summarize fast_summarize.log $UV run --frozen python scripts/eval_fast_speech.py summarize --split test \
  --model-key "$ID" --checkpoint "$CK" --model-config "$CFG" --window-eval-dir "$OUT/window_eval" \
  --out-dir "$OUT/fast_speech" --experiment-id "$ID-test-fast-speech" \
  --method-name "$M 高速度域の評価（区間別・test_fast）、2.0秒窓ごと" --append-metrics --metrics-csv "$WORK"

# 追記された行だけを取り出す（Mac の統括Agentが results/metrics.csv に足す）
diff results/metrics.csv "$WORK" | grep '^>' | sed 's/^> //' > "$OUT/metrics_rows_test.csv"
echo "metrics rows: $(wc -l < "$OUT/metrics_rows_test.csv")" >> "$OUT/status.txt"
echo "DONE $(date -Iseconds)" >> "$OUT/status.txt"
