# 録音手順書（実環境評価）

20文を、3つのマイク条件 × 3つの話速 = 9回に分けて録音する。リポジトリの直下で実行する。

## 準備（1回だけ）

```
uv run python scripts/select_eval_sentences.py   # sentences_labeled.tsv が無いときだけ
uv run python scripts/record_eval.py --list-devices
```

一覧の `*` が今の既定の入力。別のマイクを使うときは `--device 番号` を付ける。

## 起動（9条件）

| マイク | 通常 | 速め | かなり速め |
|---|---|---|---|
| 内蔵マイク | `--mic builtin --rate normal` | `--mic builtin --rate fast` | `--mic builtin --rate veryfast` |
| イヤホンマイク | `--mic earphone --rate normal` | `--mic earphone --rate fast` | `--mic earphone --rate veryfast` |
| 対面で1m以上離れて置いたマイク | `--mic far --rate normal` | `--mic far --rate fast` | `--mic far --rate veryfast` |

例:

```
uv run python scripts/record_eval.py --mic builtin --rate normal
uv run python scripts/record_eval.py --mic earphone --rate fast --device 3
```

## 操作

1. 画面の文を確かめて **Enter** → 録音開始
2. 読み終えたら **Enter** → 録音終了・保存
3. **Enter** で次の文へ。言い直し・読み間違いがあったら **r** + Enter で同じ文を録り直す（上書きされる）
4. **q** + Enter で終了。途中から再開するときは `--start 番号` を付ける

保存先は `data/eval_real/`、ファイル名は `マイク_話速_番号_文ID.wav`（例: `builtin_normal_01_BASIC5000_0036.wav`）。
