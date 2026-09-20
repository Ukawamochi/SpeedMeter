# speakmeter

日本語の音声から話速（毎秒モーラ数）をリアルタイムに推定する小型CNNを開発するプロジェクト。
書き起こしは行わず、一定区間に含まれるモーラ数のみを推定する。最終的にONNXへ書き出し、ブラウザ上で動作させることを目指す。

## 環境の再現手順

Python 3.12を前提とし、[uv](https://docs.astral.sh/uv/)でパッケージ管理を行う。

```bash
uv sync
uv run python scripts/check_env.py
```

`scripts/check_env.py`は以下を出力する。

- Pythonのバージョン
- torchのバージョン
- `torch.backends.mps.is_available()`の結果
- MPS上でのConv2d順伝播の成否
- pyopenjtalkによる読み変換の動作確認（入力「今日は快晴です」）

## ディレクトリ構成

```
pyproject.toml       依存関係とプロジェクト設定
docs/                 仕様・計画・進捗・疑問点の文書
configs/              モデル・特徴量・分割などの設定ファイル
configs/splits/       話者単位の学習・検証・テスト分割の固定ファイル
src/spkrate/labels/   テキストからのモーラ数ラベル生成
src/spkrate/data/     データセットの読み込み・分割処理
src/spkrate/features/ 特徴量（対数メルスペクトログラム）の計算・保存
src/spkrate/models/   モデル定義
src/spkrate/train/    学習スクリプト
src/spkrate/eval/     評価指標・評価スクリプト
src/spkrate/baselines/ 信号処理・書き起こしベースライン
src/spkrate/export/   ONNX書き出し・量子化
scripts/              環境確認などの補助スクリプト
tests/                単体テスト
data/                 コーパス・音声データ（git管理外）
runs/                 学習ログ（git管理外）
results/              評価指標の記録（results/metrics.csvはgit管理下）
```

## ドキュメントの役割

- `docs/spec.md`: 仕様の正。入力・出力・特徴量・モデル構造・評価指標などを固定する
- `docs/plan.md`: 開発段階ごとの担当・成果物・完了条件
- `docs/progress.md`: セッションごとの実施内容・数値・未解決点の記録
- `docs/questions.md`: 仕様への疑問、環境の問題、人間の判断が必要な事項の記録
- `CLAUDE.md`: このリポジトリで作業する際の規則
