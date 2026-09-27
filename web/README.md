# 確認用ページ（web/）

ブラウザでモデルの動作を確かめるための最小限のページ（ユーザーの指示による。仕様の実装ではない）。

## 起動

リポジトリ直下で:

```sh
uv run python scripts/setup_web_model.py        # runs/onnx_exp005/model_fp32.onnx を web/models/ にコピー（--symlink でリンク）
uv run python -m http.server -d web 8000
```

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
- 表示: 現在の毎秒モーラ数（大きく）と、直近 30 秒の折れ線グラフ（canvas、外部ライブラリなし。縦軸は 0〜10、超えたら広げる）。
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
