# 確認用ページ（web/）

ブラウザでモデルの動作を確かめるための最小限のページ（ユーザーの指示による。仕様の実装ではない）。
録音 → 16kHz へ再標本化 → 対数メル（JS）→ onnxruntime-web 1.30.0（CDN、wasm）で `model_fp32.onnx` を実行し、
出力テンソル `mora` のナマの値と、それをパースした値を表示する。

## 起動

リポジトリ直下で:

```sh
uv run python scripts/setup_web_model.py        # runs/onnx_exp005/model_fp32.onnx を web/models/ にコピー（--symlink でリンク）
uv run python -m http.server -d web 8000
```

http://localhost:8000 を開く（マイクは localhost なら使える）。web/models/ は .gitignore の対象。
onnxruntime-web は cdn.jsdelivr.net から読むのでネットワークが要る。

## 画面

- **録音開始 / 録音停止**: getUserMedia（エコー除去・雑音抑圧・自動利得は切る）で録る。ブラウザの標本化周波数のまま
  AudioWorklet で取り（チャンネル平均でモノラル）、停止後に 16kHz へ再標本化する（hann 窓付き sinc 補間、
  lowpass_filter_width=6、rolloff=0.99。torchaudio の既定と同じ核。学習・評価の Python 側は librosa/soxr なので同一ではない）。
- **入力**: 元の標本化周波数・標本数、再標本化の方法、16kHz の標本数と秒数、マイクの設定。16kHz に変換後の音声を再生できる。
- **窓ごと**（docs/spec.md の窓）: 2.0 秒窓を 0.25 秒ずつずらし、窓全体が収まるものだけ（2.0 秒未満なら窓なし）。
  各窓の対数メル (201, 80) を並べて (窓数, 201, 80) で1回実行。ナマの値は出力テンソルの型・形・値の配列。
  パースした値は窓ごとのモーラ数と毎秒モーラ数（÷2.0 秒）。
- **確認用: クリップ全体を1回入力**（仕様外）: (1, 1 + 標本数 // 160, 80) で1回実行。総モーラ数と、毎秒モーラ数（÷録音の秒数）。
- **参照入力で実行**: `reference/onnx_exp005_fp32.json` の合成波形（3.0 秒）を同じ経路に通し、Python（対数メル → ONNX Runtime CPU）
  の出力との差を表示する。目安は max|差| ≤ 1e-3 モーラ（テストでの実測は 1.9e-6）。

## テスト

`node --test web/tests/melspec.test.mjs`（pytest の `tests/test_web_demo.py` からも呼ぶ）。対数メルは tests/fixtures/melspec/ と
tests/test_melspec_fixtures.py と同じ許容誤差で照合する。`tests/test_web_demo.py` は JS で作った入力を Python の ONNX Runtime に
通して参照値と比べる（モデルが無ければ skip）。参照値の作り直しは `uv run python scripts/make_web_reference.py`。
