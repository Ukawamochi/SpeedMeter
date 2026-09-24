# exp002（拡張あり）の dev 評価と exp001 との数値の並記

- 対象: `runs/exp002/checkpoint_best.pt`（実験ID `007-augmentation`、設定 `configs/exp002.yaml`、最良エポック 6）
- 比較対象: `runs/exp001/checkpoint_best.pt`（実験ID `005-first-model`、設定 `configs/exp001.yaml`、最良エポック 6）
- 対象は dev のみ（`configs/splits/dev.json`、29,518件）。`configs/splits/test.json` は使っていない。音声は聴いていない
- デバイス `mps`。float64・`torch.compile` は使っていない。MPS の CPU フォールバック: 評価・診断とも **0 件**
- **本文書は数値のみを並べる。解釈・採否の判断・推奨は書かない**

## 1. dev 全件の指標

- 生成: `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_dev_full.py --run-dir runs/exp002 --experiment-id 007-augmentation --config configs/exp002.yaml --method "cnn (話速推定CNN、方式A: クリップ全体入力、拡張あり)"`
  （`runs/exp001/eval_dev_full.py` と同じ手順を引数化したもの。コミット 43c2c59）
- 出力: `runs/exp002/eval_dev_full.json`、`runs/exp002/predictions_dev_full.json`、`results/metrics.csv` の `007-augmentation` 行（コミット ed6f07a、clean）
- 手順の再現確認: 同じスクリプトで exp001 を評価し直すと、指標は metrics.csv の `005-first-model` 行と全桁一致し、29,518件の予測値も `runs/exp001/predictions_dev_full.json` と完全一致した

| 指標 | exp001 | exp002 |
| --- | ---: | ---: |
| 毎秒モーラ数 MAE | 0.5013 | 0.5992 |
| 帯別 MAE 4未満（9,686件） | 0.4033 | 0.5180 |
| 帯別 MAE 4以上6未満（11,135件） | 0.4099 | 0.4917 |
| 帯別 MAE 6以上8未満（6,682件） | 0.5300 | 0.6200 |
| 帯別 MAE 8以上（2,015件） | 1.3824 | 1.5145 |
| 相関係数 | 0.8905 | 0.8663 |
| 1推論あたりの処理時間（2.0秒窓、対数メル→正規化→前向き、ms） metrics.csv 記録値 | 1.7455（2026-09-21 測定） | 4.1898（2026-09-24 測定） |
| 同上、2026-09-24 に同じスクリプトで連続して測り直した値（ms） | 4.1766 | 4.1898 |
| 同上のうち前向き計算のみ（2026-09-24、ms） | 3.2494 | 3.2583 |
| モデルのファイルサイズ（checkpoint_best.pt、バイト） | 2,005,357 | 2,005,357 |

処理時間は測定日によって値が異なる（exp001 の 2026-09-21 の値 1.7455 ms と 2026-09-24 の値 4.1766 ms）。
同じ日に連続して測った2つのモデルの値は上の2行目・3行目。

## 2. 低出力事例（推定毎秒モーラ数 < 1.0）の件数

推定毎秒モーラ数は `results/low_output_diagnosis.md` と同じく、dev 全件評価の予測（`predictions_dev_full.json` の値 = 推定モーラ数 ÷ クリップ長）。
`results/error_cases/index.tsv` の `mora_per_second_pred` と `runs/exp001/predictions_dev_full.json` の差の絶対値の最大は 4.97e-05（表示桁の丸め）。

| 項目 | exp001 | exp002 |
| --- | ---: | ---: |
| (a) dev 全 29,518 件のうち 1.0 未満 | 215 | 148 |
| (a') 両方で 1.0 未満 | 130 | 130 |
| (b) exp001 の誤差上位100件のうち exp001 で 1.0 未満だった 63 件のうち 1.0 未満 | 63 | 50 |
| (b) 同 63 件の推定毎秒モーラ数の中央値 | 0.1583 | 0.5963 |
| (b) 同 63 件の exp002 での第1四分位・第3四分位 | — | 0.4944 / 0.8474 |
| (b) 同 63 件の exp002 での最小・最大 | — | 0.1821 / 3.2786 |

## 3. 窓の診断 D1・D2・D3

- 生成: `scripts/window_diagnostics.py`（exp001 と同じスクリプト、乱数の種 20260921、同じ入力）。
  exp002 は `--checkpoint runs/exp002/checkpoint_best.pt --experiment-id 007-augmentation --output-json runs/exp002/window_diagnostics.json --output-md runs/exp002/window_diagnostics.md`
- 生の値: `results/window_diagnostics_exp002.json`（`runs/exp002/window_diagnostics.json` の写し）。exp001 は `results/window_diagnostics.json`
- 再現確認: 引数追加後のスクリプト（コミット ed6f07a）で exp001 を測り直すと D1・D2・D3 の JSON の値は `results/window_diagnostics.json` と完全一致した

| 項目 | exp001 | exp002 |
| --- | ---: | ---: |
| **D1** 分割整合性 mean(\|Σ−P\|/(2.0m))（mora/s、1,000件） | **0.4360** | **0.7667** |
| D1 中央値 / 第9十分位 / 最大 | 0.4019 / 0.7830 / 1.6505 | 0.7700 / 1.2136 / 1.9136 |
| D1 符号つき (Σ−P)/(2.0m) の平均 | −0.4333 | −0.7378 |
| D1 Σ の平均 / P の平均（モーラ） | 33.3204 / 35.9307 | 30.6540 / 35.1091 |
| **D2** デジタル無音 2.0秒窓の平均（モーラ、500本） | **0.0824** | **1.4410** |
| **D2** MUSAN noise 2.0秒窓の平均（モーラ、500本） | **4.7024** | **1.7378** |
| D2 MUSAN noise 中央値 / 最大 | 4.1652 / 21.4650 | 1.3084 / 13.1240 |
| D2 参考: MUSAN noise を dev 音声の実効値中央値に揃えた場合の平均 | 4.6909 | 1.5706 |
| **D3** 連結加法性 mean(\|差\|/連結後の長さ)（mora/s、500組） | **0.0568** | **0.2438** |
| D3 中央値 / 第9十分位 / 最大 | 0.0297 / 0.1447 / 0.4425 | 0.2220 / 0.4002 / 0.7322 |
| D3 差の平均（符号つき、モーラ） | +0.4437 | −1.9815 |
| D3 P(無音0.2秒) の平均（モーラ） | 0.0001 | 0.9186 |

閾値（docs/decisions/005-window-strategy.md 5節）に対する超過の有無:

| 条件 | exp001 | exp002 |
| --- | --- | --- |
| B1: D1 > 0.25 mora/s | 超過 | 超過 |
| B2（デジタル無音）: > 1.0 モーラ | 超過しない | 超過 |
| B2（MUSAN noise）: > 1.0 モーラ | 超過 | 超過 |
| B5: D3 > 0.25 mora/s かつ B1 | 該当しない（D3 単独で超過しない） | 該当しない（D3 単独で超過しない） |

## 4. 1エポックあたりの所要時間

`runs/*/log.txt` のエポック終了行（「エポックN 学習損失=…」）から求めた。

- 学習時間: エポック終了行の末尾の秒数（`train_seconds`、学習ループのみ）
- dev 評価時間: 「前のエポック終了行（エポック1は「データ:」行）からそのエポック終了行までの経過」−学習時間。
  dev 全 29,518 件の評価のほか、前エポックのチェックポイント保存（エポック1はデータローダ等の準備）を含む

| エポック | exp001 学習（秒） | exp001 dev評価（秒） | exp002 学習（秒） | exp002 dev評価（秒） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1318.2 | 50.5 | 2399.0 | 54.1 |
| 2 | 1318.3 | 49.5 | 2433.0 | 47.0 |
| 3 | 1334.1 | 50.3 | 1850.2 | 45.2 |
| 4 | 1367.8 | 50.3 | 1753.8 | 46.2 |
| 5 | 1367.1 | 50.3 | 2284.0 | 57.1 |
| 6 | 1370.4 | 51.4 | 2188.2 | 46.0 |
| 平均 | 1346.0 | 50.4 | 2151.4 | 49.3 |
| 1エポック合計の平均 | 1396.4 | | 2200.6 | |

学習データの経路は exp001 が事前計算特徴量（features）、exp002 が波形から都度計算（waveform、拡張あり）。dev 評価は両方とも features 経路・拡張なし。
