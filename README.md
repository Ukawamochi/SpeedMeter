# speakmeter

日本語音声から話速（毎秒モーラ数）を推定する小型CNNを開発するプロジェクト。
一定区間に含まれるモーラ数を推定し、最終的にONNXへ書き出してブラウザ上で動作させることを目指す。

## 現在の実装

実装は `src/spkrate/` 以下にある（main の状態）。

- `labels/`: 正解の作成
  - `mora.py`: テキストのカタカナ読み変換とモーラ数の計数（pyopenjtalk）
  - `filter.py`: 学習データから除外する文の判定
  - `alignment.py`: モーラ単位の強制アライメント
  - `train_selection.py`: train のアライメント付与・無発話の指標・学習に使うクリップの選別
- `data/`: データの一覧と分割
  - `common_voice.py`: Common Voice 日本語の validated.tsv からクリップ一覧を作る
  - `splits.py`: 話者（client_id）単位の学習・検証・テストの分割
  - `musan_split.py`: MUSAN noise のファイル単位の学習用・評価用の分割
  - `augment.py`: データ拡張（時間伸縮・残響・雑音重畳・帯域制限・音量変化・周波数マスク）
- `features/melspec.py`: 対数メルスペクトログラム
- `models/cnn.py`: 話速推定CNN
- `train/`: 学習
  - `train.py`: 学習ループ（`python -m spkrate.train.train`）
  - `data.py`: 学習・検証のデータ供給（方式A）
  - `method_b.py`: 方式B（窓単位の正解）の学習データ
  - `silence.py`: 無音・雑音のみのサンプル（正解モーラ数0）
- `eval/`: 評価
  - `runner.py`: 評価の実行と `results/metrics.csv` への追記
  - `metrics.py`: 評価指標、`audio.py`: 評価時の音声読み込み
  - `dev_window.py`・`dev_window_val.py`・`window_eval.py`: 窓単位の評価セット dev_window とその集計
  - `noisy.py`: 雑音下評価セット dev_noisy、`no_speech.py`: 無発話クリップの自動検出
  - `latency.py`: 推論時間の測定
  - `window_diag.py`・`low_output_diag.py`: 窓の切り出し方式・低出力事例の診断
- `baselines/`: 比較用のベースライン（`envelope.py` 信号処理、`asr.py` 書き起こし）
- `device.py`: 計算デバイス（mps・cuda・cpu）の選択、精度の設定（cuda の TF32 無効化）、実行した計算機とデバイスの記録
- `export/`: ONNX 書き出し（未実装）

`scripts/` には、データの準備（`build_clips.py`、`build_splits.py`、`precompute_features.py` など）、アライメント（`align_dev.py`、`align_train.py`）、評価（`eval_dev_full.py`、`eval_dev_window.py`、`measure_latency.py`）、ベースライン（`run_envelope_baseline.py`、`run_asr_baseline.py`）、診断、環境確認（`check_env.py`）、遠隔機との転送（`sync_to_remote.sh`、`fetch_from_remote.sh`）の入口がある。設定は `configs/`（実験は `configs/exp*.yaml`、評価セットは `configs/eval/`、分割は `configs/splits/`）に置く。

学習は次の形で実行する。`--device` で設定ファイルのデバイスを上書きできる。

```bash
uv run python -m spkrate.train.train --config configs/<設定>.yaml [--device mps|cuda|cpu]
```

ONNX 書き出しとブラウザでの動作は今後の実装対象。
仕様は [docs/spec.md](docs/spec.md)、確認事項と回答は [docs/questions.md](docs/questions.md) に記録している。

## 開発環境

Pythonのバージョンは `.python-version`、依存関係は `pyproject.toml` と `uv.lock` で管理する。
torch は macOS では通常の版、Linux では PyTorch 公式の cu130 版が入る（`pyproject.toml` の `tool.uv.index`）。

```bash
uv sync
uv run python scripts/check_env.py
uv run python -m pytest -q
```

`check_env.py` はPython・PyTorchのバージョン、MPS・CUDAの利用可否と演算、pyopenjtalkの読み変換を確認する。

1行1文のテキストについて読みとモーラ数を確認するには、次を実行する。

```bash
uv run python scripts/inspect_mora.py path/to/text.txt
```

### 計算機の構成

計算機は2台で、呼び名で区別する。詳細は [docs/decisions/010-compute-environment.md](docs/decisions/010-compute-environment.md) を参照する。

| 呼び名 | 役割 | デバイス |
| --- | --- | --- |
| mac | 作業の本拠地。git、docs、results、metrics.csv の正本を置く。推論時間の測定はここだけで行う | mps |
| ubuntu-desktop | 学習と推論の実行だけを行う。Mac から `ssh ubuntu-desktop` で接続する。コミットしない | cuda（RTX 3060 12GB） |

どちらでも float64・torch.compile・混合精度は使わず、cuda では TF32 を無効にする（`src/spkrate/device.py`）。
実験ごとにどちらの計算機で実行したかは [docs/compute.md](docs/compute.md) で管理する。
`results/metrics.csv` の `host` 列には生のホスト名ではなく呼び名（`mac`・`ubuntu-desktop`）を書く。

### ubuntu-desktop の環境構築

リポジトリは遠隔機の `~/SpeedMeter` に置く。

1. 必要なもの: git、rsync、tmux、NVIDIA ドライバ（`nvidia-smi` で GPU が見えること。CUDA Toolkit は不要で、torch の wheel が実行時ライブラリを含む）
2. pyopenjtalk は sdist だけで uv sync のときにビルドされるため、C/C++ コンパイラと cmake を入れる

   ```bash
   sudo apt-get install -y build-essential cmake
   ```

3. uv をユーザー権限で導入する（`~/.local/bin/uv` に入る）

   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

4. Mac から `scripts/sync_to_remote.sh --data` でリポジトリとデータを送る（下記）
5. 遠隔機で依存を入れて確認する。Python は `.python-version` の版を uv が取得する（システムの python3 は使わない）

   ```bash
   cd ~/SpeedMeter
   ~/.local/bin/uv sync --frozen
   ~/.local/bin/uv run --frozen python scripts/check_env.py
   ~/.local/bin/uv run --frozen python -m pytest -q
   ```

非対話の ssh（`ssh ubuntu-desktop '<コマンド>'`）では `~/.bashrc` が読まれず uv が PATH に入らないため、`~/.local/bin/uv` と絶対パスで呼ぶ。
学習は ssh が切れても続くよう tmux の中で起動する。

```bash
ssh ubuntu-desktop "cd ~/SpeedMeter && tmux new-session -d -s <実験ID> '~/.local/bin/uv run --frozen python -m spkrate.train.train --config configs/<設定>.yaml > runs/<実験ID>.stdout.log 2> runs/<実験ID>.stderr.log'"
```

### sync_to_remote.sh と fetch_from_remote.sh

Mac から遠隔機へリポジトリを送るには `scripts/sync_to_remote.sh` を使う。

```bash
scripts/sync_to_remote.sh                  # .git と作業ツリーだけ
scripts/sync_to_remote.sh --data           # データも送る
scripts/sync_to_remote.sh --src ../<worktree>   # worktree の内容を送る
```

- `.git`（worktree から実行した場合も元のリポジトリの .git）と作業ツリーを送る。`data/`・`runs/`・`.venv/`・`listening/`・`results/error_cases/` と `results/alignment_check/` の音声・キャッシュは送らず、遠隔機の側でも消さない
- 遠隔機で `git checkout --force --detach <送り元の HEAD>` を行い、送り元が clean なら遠隔機の `git status --porcelain` が空であることを確かめる。遠隔機で行う git の操作はこれだけで、コミットはしない
- `--data` のときだけ `data/common_voice_ja`・`data/musan`・`data/processed` を送り（`--delete` なし）、ファイル数と合計バイト数を Mac と遠隔機で比べる
- 主な引数: `--host`（既定 ubuntu-desktop）、`--src`（送り元の作業ツリー）、`--commit`（checkout するコミット）、`--remote-dir`（既定 `SpeedMeter`、ホームからの相対パス）、`--data-src`、`--retries`（既定4）。ssh の接続失敗と rsync の失敗は再試行する

遠隔機の `~/SpeedMeter/runs/` の結果を Mac の `runs/` に戻すには `scripts/fetch_from_remote.sh` を使う。

```bash
scripts/fetch_from_remote.sh <実行名> [<実行名またはファイル名> ...]
```

- 実行名（`runs/` の下のディレクトリ）またはファイル名を複数指定できる。戻し先は元のリポジトリの `runs/`（`--dest` で変更可）
- Mac の `runs/` に同じ名前があれば何も送らずに停止する（既存の結果を上書きしない）。戻した後にファイル数とバイト数を比べる
- 主な引数: `--host`、`--remote-dir`、`--dest`、`--retries`

遠隔機で評価した場合も、結果は Mac の `results/` と `metrics.csv` に書く。

## データセット

データセットは手動で取得し、`data/` 以下に配置する。
配置先と確認済みの取得記録は [data/DATASETS.md](data/DATASETS.md) を参照する。
利用前に必要なファイルがそろっているか確認する。
データ本体・アーカイブはGit管理外とし、配置案内の `data/DATASETS.md` のみ管理する。

## ディレクトリと文書

- `src/spkrate/`: 実装（上記「現在の実装」）
- `scripts/`: データの準備・評価・診断・環境確認・遠隔機との転送の入口
- `tests/`: 単体テスト
- `configs/`: 学習・評価の設定と、`configs/splits/` の固定した分割
- `runs/`: 学習ログの出力先（Git管理外）
- `results/`: 評価結果の出力先
- [docs/spec.md](docs/spec.md): 入出力・モーラ・特徴量・モデル・評価の仕様
- [docs/plan.md](docs/plan.md): 今後の段階を含む開発計画
- [docs/compute.md](docs/compute.md): 実験ごとの計算機と状態
- [docs/progress.md](docs/progress.md): セッションごとの実施記録
- [docs/questions.md](docs/questions.md): 仕様の確認事項と回答
- [CLAUDE.md](CLAUDE.md): リポジトリで作業する際の規則
