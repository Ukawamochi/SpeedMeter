# exp005 の ONNX 書き出しと整合確認（指示書 2026-09-27 系統B、docs/plan.md 8-1〜8-3）

- 対象: exp005（`runs/exp005/checkpoint_best.pt`、最良エポック 26、方式B。設定 `configs/exp005.yaml`）
- コード: ブランチ `feat/onnx-export`（書き出し `src/spkrate/export/to_onnx.py`、評価 `scripts/eval_dev_window.py --onnx`、推論時間 `scripts/measure_latency.py --onnx`、参照値 `scripts/make_melspec_fixtures.py`）。窓ごとの予測のコミットは `9953e32`（書き出しも同じ）、集計・metrics.csv への追記・推論時間の測定のコミットは `4ded00c`（metrics.csv の commit_dirty は clean）
- 計算機: mac（Apple M5、物理コア10）。ONNX Runtime 1.30.0 の **CPUExecutionProvider** だけを使った（CoreML などは使っていない）。torch 2.14.0、onnx 1.23.0
- 本書は測定値だけを記録する。解釈と採否は書かない。configs/splits/test.json は使っていない。CPU フォールバックは0件（評価2回と推論時間の測定のすべて）

## 1. 書き出したファイル（8-1。runs/ の下、リポジトリには入れない）

| ファイル | 大きさ (バイト) | sha256 の先頭 |
| --- | ---: | --- |
| `runs/onnx_exp005/model_fp32.onnx` | 2,005,372 | d87eb2d8f59f |
| `runs/onnx_exp005/model_int8.onnx` | 529,968 | 0e7fa3366b6d |
| 参考: `runs/exp005/checkpoint_best.pt` | 2,008,749 | — |

- グラフ: 入力 `log_mel`（batch, frames, 80）float32 = **正規化前**の対数メル → グラフ内で `configs/normalization.yaml`（per_mel）の固定値で正規化 → CNN（全フレーム有効）→ 出力 `mora`（batch,）= 入力区間のモーラ数。バッチとフレーム数は可変。対数メルの計算はグラフに入れていない（ブラウザ側で実装し、2節の参照値で照合する）
- 書き出し器: `torch.onnx.export(dynamo=False)`（従来の TorchScript の書き出し器。torch 2.14 の既定の dynamo 版は onnxscript が要るため、依存を増やさなかった。非推奨の警告が出る）。opset 17。`onnx.checker` を通る
- 量子化: `onnxruntime.quantization.quantize_dynamic`（重み `QInt8`、テンソル単位、`reduce_range=False`。ONNX Runtime が勧める前処理 `quant_pre_process` はしていない）。畳み込み8層と出力層がすべて `ConvInteger`（活性は実行時に `DynamicQuantizeLinear` で uint8）になった
- 再生成の手順（2回実行して同じ sha256 になることを確かめた）:
  `uv run python -m spkrate.export.to_onnx --checkpoint runs/exp005/checkpoint_best.pt --out-dir runs/onnx_exp005`
  （`runs/onnx_exp005/export_meta.json` に版・sha256・コマンドを残す。ログは `runs/onnx_exp005/export.log`）
- 依存: onnx・onnxruntime はすでに pyproject.toml にあったので、依存と uv.lock は変えていない

## 2. PyTorch と ONNX Runtime の出力の一致（8-1、tests/test_onnx_export.py）

出力はモーラ数（2.0秒窓なら毎秒モーラ数の2倍の単位）。

| 比較 | 許容誤差（テスト） | 実測 |
| --- | --- | --- |
| fp32 と PyTorch | 要素ごとに `atol 1e-4 モーラ + rtol 1e-5`（無作為の初期値のモデルと exp005。可変長を含む） | dev_window clean 全842,900窓（PyTorch は mps の既存の予測 `runs/exp005/window_eval/pred_clean.npz`）: 平均 9.3e-7、最大 7.6e-6 モーラ |
| int8 と PyTorch（無作為の初期値のモデル） | 出力に対する比 `rtol 1e-3` | 比 約1.9e-5（出力が約139でほぼ一定のため小さく出る。配線の確認だけ） |
| int8 と PyTorch（exp005） | 合成の入力16窓で、窓ごとの差の平均 ≤ 0.15、最大 ≤ 1.5 モーラ | 合成の入力: 平均 約0.07、最大 約0.95。dev_window clean 全842,900窓: 平均 0.0915、99%点 0.268、99.9%点 0.319、最大 0.528 モーラ。符号つきの平均は +0.089 モーラ（int8 が大きい。毎秒 +0.045） |

- exp005 のテストは `runs/exp005/checkpoint_best.pt` が無い環境では飛ばす
- int8 の許容誤差は要素ごとの一致ではなく、量子化を誤ったとき（差が数モーラ以上）を検出する目的の値である。精度への影響は3節の dev_window で見る

## 3. dev_window の clean・主指標（8-1。量子化の前後）

`scripts/eval_dev_window.py predict --onnx ... --conditions clean` → `summarize --conditions clean --append-metrics`。窓・除外は docs/spec.md「窓単位の評価」節のとおり（主指標 827,467窓）。毎秒モーラ数。

| モデル | MAE | 偏り | 相関係数 | 4未満 | 4以上6未満 | 6以上8未満 | 8以上 | 8以上の偏り | 正解0の窓の出力平均（モーラ） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch（mps、results/exp005_eval.md） | 0.4755 | −0.0342 | 0.9665 | 0.2893 | 0.4572 | 0.4692 | 0.6810 | −0.3655 | 0.1738 |
| ONNX fp32 | 0.4755 | −0.0342 | 0.9665 | 0.2893 | 0.4572 | 0.4692 | 0.6810 | −0.3655 | 0.1738 |
| ONNX int8 | 0.4783 | +0.0111 | 0.9665 | 0.2924 | 0.4666 | 0.4812 | 0.6705 | −0.2855 | 0.1757 |

- 全窓（除外しない、842,900窓）の MAE: fp32 0.4923、int8 0.4951
- 窓単位の必達目標（全体0.70、4未満0.70、4以上6未満0.60、6以上8未満0.60、8以上0.90）に対して、fp32・int8 とも全項目で目標以下
- results/metrics.csv の行: `onnx-exp005-fp32`・`onnx-exp005-int8`（split `dev_window`、主指標。host mac、device cpu、model_size_bytes は ONNX ファイルの大きさ、コミット 4ded00c）
- 予測と集計: `runs/onnx_exp005/window_eval_{fp32,int8}/`（pred_clean.npz、summary.json、predict_meta.json）。推論の所要時間は fp32 約30分、int8 約54分（対数メル計算を含む、既定のスレッド数）

## 4. 推論時間とファイルサイズ（8-3）

docs/decisions/007-latency-measurement.md の規則（2.0秒窓1回 = 対数メル計算 → 正規化（グラフ内）→ 前向き計算、入力は種 20260921 の標準正規×0.05、ウォームアップ20回、各100回、交互測定）。ONNX Runtime の CPU 実行は呼び出しの中で終わるので、デバイスの同期はしていない。測定の前に、Mac で他の学習・推論が動いていないことを確かめた（ps で上位の CPU 使用率は数%以下）。

| 実験ID | スレッド数（intra_op） | 中央値 (ms) | 平均 (ms) | 最大 (ms) | 最小 (ms) | ファイル (バイト) | セッション |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| onnx-exp005-fp32 | 既定（物理コア数） | 1.916 | 2.078 | 7.932 | 1.793 | 2,005,372 | lat-20260927T154516-d43b1af7 |
| onnx-exp005-int8 | 既定（物理コア数） | 3.765 | 3.811 | 5.454 | 3.677 | 529,968 | lat-20260927T154516-d43b1af7 |
| onnx-exp005-fp32-1thread | 1 | 2.829 | 2.868 | 5.336 | 2.807 | 2,005,372 | lat-20260927T154526-9d99aca1 |
| onnx-exp005-int8-1thread | 1 | 3.238 | 3.245 | 5.532 | 2.897 | 529,968 | lat-20260927T154526-9d99aca1 |

- 詳細（各回の値）: `runs/onnx_exp005/latency.json`・`latency_1thread.json`、ログは同じ名前の .log。metrics.csv に4行（split `latency_2s_window`、device cpu）
- fp32 の最大 7.932 ms は100回中1回だけの値（平均と中央値の差の主因）
- 内訳の参考（上とは別の測定。100回の中央値）: 対数メル計算だけ 0.19 ms。ONNX Runtime の実行だけ fp32 1.37 ms（1スレッド 2.68 ms）、int8 3.08 ms（1スレッド 3.04 ms）。この計算機の CPU では int8（`ConvInteger`）の実行が fp32 より遅く、スレッド数を増やしても速くならなかった
- PyTorch（mps）と同じセッションで測らなかった理由: 試しに PyTorch（mps）と ONNX 4種を1つのセッションで交互に測ったところ（`lat-20260927T154450-2e4cba47`、`runs/onnx_exp005/latency_mixed_torch_onnx.{log,json}`、metrics.csv には書いていない）、PyTorch の中央値が 5.27 ms になった。PyTorch だけのセッションでは 1.76 ms（results/latency_010.md では 1.749 ms）。ONNX Runtime のスレッドが実行の後もしばらく CPU を回り続ける（既定の spinning）ことで、直後に測る他のモデルが遅くなったと考えられるが、原因は確かめていない。このため ONNX の2種だけを1つのセッションで測った。PyTorch（mps）の値との比較は、別セッションの値になるので 007 の5節により比較に使わない

## 5. 特徴量の参照値（8-2）

- `tests/fixtures/melspec/sine_1000hz.json`（1000Hz・振幅0.5・0.5秒、51フレーム）、`multitone_with_silence.json`（先頭0.1秒が無音、続けて 250・1500・5000Hz の和、51フレーム）、`mel_filterbank.json`（201ビン × 80 の0でない要素）。作成は `uv run python scripts/make_melspec_fixtures.py`
- 波形は int16 に丸めた標本で保存し、入力は `int16 / 32768`（float32）。対数メルは float32 の値（`[frame][mel]`）。作成時の特徴量の設定（`MEL_DEFAULTS`）を各ファイルに記録した
- 照合テスト `tests/test_melspec_fixtures.py` の許容誤差（対数メルの値の絶対誤差）: 全要素 2e-3、値が −5 を超える要素 1e-4。根拠: 同じ波形を numpy（float64）で独立に計算した値（tests/test_melspec.py の参照実装）との差の最大が、エネルギーのほとんど無い帯（値が −13 前後）で 1.0e-3、値が −5 を超える帯で 1.8e-5 だった。ブラウザ側も同じ基準で照合する想定
- 無音のフレーム（先頭9フレーム）は log(1e-6) = −13.815511 に一致することもテストした
- docs/spec.md の「特徴量」に、実装の計算パラメータ（窓関数、切り出し、メルフィルタ、対数）を転記した（plan.md 8-2 の指示。値は実装から転記したもので、変更ではない。ブランチ feat/onnx-export の c99a063）
- 実際のクリップ1件: dev の `common_voice_ja_19485242` の対数メルを `runs/melspec_fixture_clip/clip_common_voice_ja_19485242.json`（約1.0MB）に書いただけで、リポジトリには入れていない。入れてよいかは docs/questions.md（2026-09-27 実際のクリップ1件の特徴量の値）で判断を仰いでいる

## 6. テスト

- Mac の worktree（feat/onnx-export、4ded00c）で pytest 全件: 746 passed、3 skipped
- 追加: tests/test_onnx_export.py（8件）、tests/test_melspec_fixtures.py（8件）
