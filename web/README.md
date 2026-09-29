# 確認用ページ（web/）

ブラウザでモデルの動作を確かめるための最小限のページ（ユーザーの指示による。仕様の実装ではない）。

## 起動

リポジトリ直下で:

```sh
uv run python scripts/setup_web_model.py --source runs/onnx_exp005 runs/onnx_exp009   # 比べるモデルを並べる（--symlink でリンク）
uv run python -m http.server -d web 8000
```

2026-09-29 時点で比べているのは、学習を終えた条件の 8 モデル（exp007 は exp005 と seed だけが違うので外した）:

```sh
uv run python scripts/setup_web_model.py --source runs/onnx_exp005 runs/onnx_exp008 runs/onnx_exp009 runs/onnx_exp010 \
  runs/onnx_exp011 runs/onnx_exp012 runs/onnx_exp014 runs/onnx_exp015
```

書き出していない実験は先に `uv run python -m spkrate.export.to_onnx --checkpoint runs/<実験>/checkpoint_best.pt --out-dir runs/onnx_<実験>`。

`--source` の各 `runs/onnx_<名前>/model_fp32.onnx` を `web/models/<名前>.onnx` に置き、一覧 `web/models/models.json`
（`--source` の順、8個まで）を書く。省略時は exp005 だけ。check.html は `exp005.onnx` を読むので exp005 を含めておく。

http://localhost:8000 を開く（マイクは localhost なら使える）。web/models/ は .gitignore の対象。
onnxruntime-web 1.30.0（wasm、1スレッド）は cdn.jsdelivr.net から読むのでネットワークが要る。

## index.html（リアルタイム表示）

- **録音開始 / 録音停止**。録音中は推論を動かし続ける。docs/spec.md の推論の定義どおり、直近 2.0 秒の音声で 0.25 秒ごとに推論し、
  出力（モーラ数）÷ 2.0 秒を毎秒モーラ数とする。
- 音声: getUserMedia（エコー除去・雑音抑圧・自動利得は切る）をブラウザの標本化周波数のまま AudioWorklet で取り（チャンネル平均で
  モノラル）、直近 4 秒を環形バッファに持つ。推論の時点ごとに直近 2.0 秒（+ 変換の余白 0.01 秒）を 16kHz へ変換する
  （hann 窓付き sinc 補間、lowpass_filter_width=6、rolloff=0.99。torchaudio の既定と同じ核。学習・評価の Python 側は librosa/soxr なので同一ではない）。
- 推論の時点は録音の標本数で数える（壁時計ではない）。推論中に次の時点が来たらその時点は間引き、遅れを溜めない。
  推論回数・間引いた回数・直近の処理時間を小さく表示する。
- 複数のモデルの比較: 一覧の全モデルに、時点ごとに同じ対数メルを順に通す。グラフの上のチェックボックスで、グラフに描くモデルを
  切り替える（推論はチェックの有無によらず全モデルで行うので、途中でチェックを付けても線は欠けない）。色は一覧の順に固定
  （表示を切り替えても塗り替えない）。現在値は全モデル分の行を出し、チェックを外したモデルは薄くする。縦軸の上限は表示中の
  モデルの値で決める。推論の時間は全モデルの合計なので、モデルを増やして 0.25 秒に間に合わなくなると全モデルそろって間引く
  （表示の「直近の処理」で確かめる。ヘッドレスの Chrome で 8 モデルの合計 約 90〜110 ms、間引き 0）。
- 表示: モデルごとの現在の毎秒モーラ数（大きく）と、直近 30 秒の折れ線グラフ（canvas、外部ライブラリなし。縦軸は 0〜10、超えたら広げる）。
  薄い破線 4・6・8 は評価で使う話速帯の境界であり、**早口の閾値は未定義**（docs/questions.md で人間の判断待ち）。
  値は平滑化せずそのまま描く（無音の表示の定義は仕様に無い）。

## check.html（計算の照合）

`reference/onnx_exp005_fp32.json` の合成波形（3.0 秒）をブラウザの経路（dsp.js → onnxruntime-web）で処理し、Python
（対数メル → ONNX Runtime CPU）の出力との差を表示する。窓ごと（2.0 秒窓・0.25 秒ずらし）と、確認用のクリップ全体1回（仕様外）。
目安は max|差| ≤ 1e-3 モーラ（テストでの実測は 1.9e-6）。ブラウザの計算が Python と一致するかを確かめる唯一の手段。

## テスト

`node --test web/tests/*.test.mjs`（pytest の `tests/test_web_demo.py` からも呼ぶ。node が無ければ skip）。
- melspec.test.mjs: 対数メルを tests/fixtures/melspec/ と tests/test_melspec_fixtures.py と同じ許容誤差で照合。再標本化・窓・パース
- realtime.test.mjs: 環形バッファ、窓の切り出しと変換、推論のスケジュール（間引き）、グラフの座標
- `tests/test_web_demo.py` は JS で作った入力を Python の ONNX Runtime に通して参照値と比べる（モデルが無ければ skip）。
  参照値の作り直しは `uv run python scripts/make_web_reference.py`。
