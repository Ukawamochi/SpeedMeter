# 確認用ページ（web/）

学習したモデルをブラウザで動かし、マイクの音声から話速（毎秒モーラ数）をリアルタイムに表示するページ。
複数のモデルに同じ音声を通して、折れ線グラフで比べられる。確認用のページであり、docs/spec.md の仕様の実装ではない。

- `index.html`: リアルタイム表示（録音 → 推論 → 現在値と直近 30 秒のグラフ）
- `check.html`: 計算の照合（ブラウザの計算が Python と一致するかを、固定の合成波形で確かめる）

ビルドの手順は無い。素の HTML と ES モジュールで、ローカルの HTTP サーバーから配信する。

## 使い方

リポジトリ直下で実行する（`uv sync` 済みであること）。

```sh
uv run python scripts/setup_web_model.py        # web/models.txt のモデルを web/models/ に置く
uv run python -m http.server -d web 8000
```

ブラウザで http://localhost:8000 を開き、「録音開始」を押す。マイクは localhost なら使える。
onnxruntime-web 1.30.0 は cdn.jsdelivr.net から読むので、ネットワークにつながっている必要がある。

- モデルごとの現在値が上に並び、グラフに線で描かれる。グラフの上のチェックボックスで、描くモデルを切り替える
- 画面の下に、推論の回数・間引いた回数・直近の処理時間（全モデルの合計）が出る

## 表示するモデルを追加する・外す

表示するモデルは **`web/models.txt`** で決める（Git で管理する一覧）。1行に1つ、ONNX の書き出し先を書く。

1. 学習を終えた実験を ONNX に書き出す（`runs/onnx_<実験>` が既にあれば不要）

   ```sh
   uv run python -m spkrate.export.to_onnx --checkpoint runs/<実験>/checkpoint_best.pt --out-dir runs/onnx_<実験>
   ```

   `runs/onnx_<実験>/` に `model_fp32.onnx`・`model_int8.onnx`・`export_meta.json` ができる。ページが使うのは `model_fp32.onnx`。
2. `web/models.txt` に `runs/onnx_<実験>` の行を足す（外すときは行を消すか `#` でコメントにする）。行の後ろに `# 説明` を書ける
3. `uv run python scripts/setup_web_model.py` を実行する
4. ページを再読み込みする（サーバーは止めなくてよい）

守ること:

- **8 個まで**。線の色が 8 色（`web/realtime.js` の `SERIES_COLORS`）なので、スクリプトが 9 個目を拒む。
  色は一覧の上からの順に割り当てる。行を入れ替えると色も入れ替わる
- **`runs/onnx_exp005` を残す**。check.html の参照値が exp005 のモデルで作られている（外すとスクリプトが警告する）
- ページに出る名前は、書き出し先のディレクトリ名から `onnx_` を除いたもの（`runs/onnx_exp015` → `exp015`）。重複は拒む
- 推論は、チェックの有無によらず全モデルで行う（途中でチェックを付けても線が欠けないように）。処理時間は
  モデルの数に比例する。0.25 秒に間に合わない時点は全モデルそろって間引くので、増やした後は画面下の
  「間引き」と「直近の処理」を見る（Mac のヘッドレス Chrome で 8 モデルの合計 約 90〜110 ms、間引き 0）

`setup_web_model.py` がすること: 一覧の各 `model_fp32.onnx` を `web/models/<名前>.onnx` にコピーし（`--symlink` で
シンボリックリンク）、`export_meta.json` の SHA-256 と照合し、ページが読む一覧 `web/models/models.json` を書く。
一覧に無くなった `web/models/*.onnx` は消す。`--source runs/onnx_a runs/onnx_b` で一覧ファイルを使わずに一時的に並べることもできる。
`web/models/` は .gitignore の対象で、モデルのファイルはリポジトリに入れない（別の計算機では、`runs/onnx_*` を
用意してから上の手順 3 を実行する）。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `index.html` / `app.js` | リアルタイム表示のページ。録音、推論の実行、現在値・チェックボックス・グラフの描画（canvas、外部ライブラリなし） |
| `realtime.js` | DOM に依存しない部品: 環形バッファ、窓の切り出しと 16kHz への変換、推論のスケジュール（間引き）、グラフの座標、系列の色 |
| `dsp.js` | 信号処理: 再標本化、窓の切り出し、対数メル（Python の `spkrate.features.melspec` と同じ計算） |
| `model.js` | onnxruntime-web の読み込み（版を固定）、`models/models.json` の読み込み、セッションの作成 |
| `recorder-worklet.js` | AudioWorklet。マイクの音声をチャンネル平均でモノラルにして app.js に送る |
| `check.html` / `check.js` | 計算の照合のページ |
| `models.txt` | 表示するモデルの一覧（Git で管理） |
| `models/` | `setup_web_model.py` が置くモデルと `models.json`（Git 管理外） |
| `reference/onnx_exp005_fp32.json` | check.html とテストが使う参照値（合成波形と Python の出力） |
| `tests/` | node:test のテスト |

関係するファイル（リポジトリの他の場所）: `scripts/setup_web_model.py`（モデルの配置）、`scripts/make_web_reference.py`
（参照値の作成）、`src/spkrate/export/to_onnx.py`（ONNX の書き出し）、`tests/test_web_demo.py`・`tests/test_setup_web_model.py`。

## 処理の流れ（index.html）

1. getUserMedia でマイクを開く（エコー除去・雑音抑圧・自動利得は切る）。ブラウザの標本化周波数のまま AudioWorklet で受け取り、
   直近 4 秒を環形バッファに持つ
2. 録音の標本数で 0.25 秒ごとの時点を数える（壁時計ではない）。各時点で直近 2.0 秒（+ 変換の余白 0.01 秒）を 16kHz へ変換する
   （hann 窓付き sinc 補間、lowpass_filter_width=6、rolloff=0.99。学習・評価の Python 側は librosa/soxr なので同一ではない）
3. 対数メル（201 フレーム × 80）を1回だけ計算し、一覧の全モデルに順に通す。出力（モーラ数）÷ 2.0 秒を毎秒モーラ数とする
   （docs/spec.md の推論の定義）
4. 推論中に次の時点が来たらその時点は間引き、遅れを溜めない
5. 現在値とグラフを更新する。値は平滑化しない（無音の表示の定義は仕様に無い）。グラフの薄い破線 4・6・8 は評価で使う
   話速帯の境界で、早口の閾値ではない（閾値は未定義。docs/questions.md）。縦軸は 0〜10 で、表示中のモデルの値が超えたら広げる

## 見た目や動きを変えるとき

| 変えたいもの | 場所 |
|---|---|
| 線の色・色の数 | `realtime.js` の `SERIES_COLORS`（数を変えたら `scripts/setup_web_model.py` の `MAX_MODELS` も合わせる） |
| グラフの横軸の長さ（秒） | `app.js` の `SPAN_SEC` |
| 破線の位置 | `realtime.js` の `BAND_BOUNDARIES` |
| 縦軸の既定の上限 | `realtime.js` の `yAxisMax` |
| 配置・文字の大きさ | `index.html` の `<style>` |
| onnxruntime-web の版 | `model.js`（2か所の URL） |

窓の長さ・推論の間隔・対数メルの設定（`dsp.js` の `WINDOW_SEC`・`HOP_SEC`・`MEL_CONFIG`）は学習と同じ値でなければならない
（docs/spec.md）。変えるとモデルの出力の意味が変わるので、仕様の変更なしに触らない。

## check.html（計算の照合）

`reference/onnx_exp005_fp32.json` の合成波形（3.0 秒）をブラウザの経路（dsp.js → onnxruntime-web）で処理し、Python
（対数メル → ONNX Runtime CPU）の出力との差を表示する。窓ごと（2.0 秒窓・0.25 秒ずらし）と、確認用のクリップ全体1回（仕様外）。
目安は max|差| ≤ 1e-3 モーラ（テストでの実測は 1.9e-6）。dsp.js を変えたら必ずここで OK になることを確かめる。

## テスト

```sh
node --test web/tests/*.test.mjs
uv run python -m pytest -q tests/test_web_demo.py tests/test_setup_web_model.py
```

- `melspec.test.mjs`: 対数メルを tests/fixtures/melspec/ の参照値と照合（許容誤差は tests/test_melspec_fixtures.py と同じ）。再標本化・窓・パース
- `realtime.test.mjs`: 環形バッファ、窓の切り出しと変換、推論のスケジュール（間引き）、グラフの座標、系列の色、表示中の系列の値
- `tests/test_web_demo.py`: node のテストを呼び（node が無ければ skip）、JS で作った入力を Python の ONNX Runtime に通して参照値と比べる
  （`runs/onnx_exp005/model_fp32.onnx` が無ければ skip）。参照値の作り直しは `uv run python scripts/make_web_reference.py`
- `tests/test_setup_web_model.py`: 一覧の読み方、配置、`web/models.txt` が有効であること（8 個以下、重複なし、exp005 を含む）

マイクを使った動作はテストでは確かめられないので、変更後はブラウザで録音して確かめる。
