# speakmeter

日本語音声から話速（毎秒モーラ数）を推定する小型CNNを開発するプロジェクト。
一定区間に含まれるモーラ数を推定し、最終的にONNXへ書き出してブラウザ上で動作させることを目指す。

## 現在の実装

- テキストのカタカナ読み変換とモーラ数カウント: `src/spkrate/labels/mora.py`
- モーラ処理の単体テスト: `tests/test_mora.py`
- 環境確認: `scripts/check_env.py`
- テキストファイルの読み・モーラ数の確認: `scripts/inspect_mora.py`

データ読み込み、特徴量計算、モデル、学習、評価、ONNX書き出しは今後の実装対象。
モーラ処理の追加修正に関する回答と未解決点は [docs/questions.md](docs/questions.md) に記録している。

## 開発環境

Pythonのバージョンは `.python-version`、依存関係は `pyproject.toml` と `uv.lock` で管理する。

```bash
uv sync
uv run python scripts/check_env.py
uv run python -m pytest -q
```

`check_env.py` はPython・PyTorchのバージョン、MPSの利用可否と演算、pyopenjtalkの読み変換を確認する。

1行1文のテキストについて読みとモーラ数を確認するには、次を実行する。

```bash
uv run python scripts/inspect_mora.py path/to/text.txt
```

## データセット

データセットは手動で取得し、`data/` 以下に配置する。
配置先と確認済みの取得記録は [data/DATASETS.md](data/DATASETS.md) を参照する。
利用前に必要なファイルがそろっているか確認する。
データ本体・アーカイブはGit管理外とし、配置案内の `data/DATASETS.md` のみ管理する。

## ディレクトリと文書

- `src/spkrate/`: 実装。`labels/` 以外の機能別ディレクトリは実装用のひな形
- `scripts/`: 環境確認・モーラ処理確認用の補助スクリプト
- `tests/`: 単体テスト
- `configs/`: 今後作成する設定と話者単位の分割。`configs/splits/` の分割ファイルは未作成
- `runs/`: 学習ログの出力先（Git管理外）
- `results/`: 評価結果の出力先
- [docs/spec.md](docs/spec.md): 入出力・モーラ・特徴量・モデル・評価の仕様
- [docs/plan.md](docs/plan.md): 今後の段階を含む開発計画
- [docs/progress.md](docs/progress.md): セッションごとの実施記録
- [docs/questions.md](docs/questions.md): 仕様の確認事項と回答
- [CLAUDE.md](CLAUDE.md): リポジトリで作業する際の規則
