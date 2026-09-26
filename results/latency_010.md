# 推論時間（新方式、同一セッション）: exp004・exp005・exp006

指示書 docs/directives/2026-09-26.md タスク4の「推論時間（同一セッションでexp004と交互に測る）」の測定結果。
判断は含めない。

- 規則: `docs/decisions/007-latency-measurement.md`（2.0秒窓1回、ウォームアップ20回、各100回、交互測定、中央値と最大値）
- 計算機: mac、デバイス: mps（docs/directives/2026-09-26-rtx3060.md 0節4。推論時間は Mac だけで測る）
- 対象: exp004 `runs/exp004/checkpoint_best.pt`、exp005 `runs/exp005/checkpoint_best.pt`（方式B）、exp006 `runs/exp006_resume/checkpoint_best.pt`（方式A、ubuntu-desktop の cuda で学習）。3つとも同じ CNN 構造
- 3モデルを1回の `scripts/measure_latency.py` で測った（コードの変更なし）。Mac では他の学習・推論は動いていなかった。exp001〜exp003 は含めていない
- 交互の順: 周ごとに開始するモデルを1つずつずらす（1周目 exp004→exp005→exp006、2周目 exp005→exp006→exp004、3周目 exp006→exp004→exp005）。100回は3の倍数でないため、各順位の回数は完全には等しくない
- セッション: `lat-20260927T071126-c8e62bc7`。ログ `runs/latency_010.log`、詳細 `runs/latency_010.json`
- CPU フォールバック: 0件
- 測定時のコミット: `a48308e`（作業ツリーは clean）

| 実験 | 中央値 (ms) | 最大値 (ms) | 平均 (ms) | 最小 (ms) | model_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 009-exp004-silence-aug | 1.7436 | 2.2445 | 1.7656 | 1.6741 | 2007725 |
| 010-exp005-method-b | 1.7493 | 2.2724 | 1.7706 | 1.6762 | 2008749 |
| 010-exp006-method-a-control | 1.7391 | 2.2177 | 1.7596 | 1.6699 | 2008685 |

参考: 前回のセッション `lat-20260925T141147-613088cb`（results/exp003_exp004_eval.md 6節）の exp004 は中央値 1.7218 ms・最大値 2.2726 ms。
別セッションの値なので、比較には使わない（007 の5節）。

## results/metrics.csv に追記した行

- `009-exp004-silence-aug`・`010-exp005-method-b`・`010-exp006-method-a-control` の `latency_2s_window`（3行、host `mac`、device `mps`、コミット `a48308e`、`commit_dirty` は `clean`）
