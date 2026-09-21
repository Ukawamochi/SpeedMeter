# 誤り事例（docs/PLAN.md 第6段階 6-1）

生成: `uv run python scripts/extract_errors.py`

## 対象

- 検証セット dev の全29,518件。予測は `runs/exp001/predictions_dev_full.json`
  （第5段階5-3の初回学習 exp001 の最良チェックポイント、方式A、拡張なし）
- `configs/splits/test.json` は使っていない

## 誤差の定義

順位付けに使った誤差は **毎秒モーラ数の絶対誤差** である。

    誤差 = |推定毎秒モーラ数 − 正解毎秒モーラ数|

`docs/spec.md` の評価指標の第1項が「毎秒モーラ数の平均絶対誤差」であり、実際に表示する量も
毎秒モーラ数なので、この指標を押し上げている事例をそのまま取り出せる定義を選んだ。
参考として **モーラ数の絶対誤差** `|推定モーラ数 − 正解モーラ数|` も `abs_error_mora` 列に
出してある（こちらはクリップが長いほど大きくなりやすいため、順位付けには使っていない）。

推定モーラ数は `推定毎秒モーラ数 × クリップ長` で戻した値である（モデルの出力はモーラ数だが、
保存されている予測値が毎秒モーラ数のため）。

## index.tsv の列

| 列 | 内容 |
| --- | --- |
| `rank` | 誤差の大きい順の順位（1が最大） |
| `clip_id` | クリップの識別子 |
| `client_id` | 話者の識別子 |
| `audio_file` | 同ディレクトリにコピーした音声のファイル名 |
| `duration_sec` | クリップ長（秒） |
| `mora_true` | 正解モーラ数 |
| `mora_pred` | 推定モーラ数 |
| `mora_per_second_true` | 正解の毎秒モーラ数 |
| `mora_per_second_pred` | 推定の毎秒モーラ数 |
| `abs_error_mora_per_second` | **順位付けに使った誤差**（毎秒モーラ数の絶対誤差） |
| `abs_error_mora` | モーラ数の絶対誤差（参考） |
| `signed_error_mora_per_second` | 符号つきの誤差（推定 − 正解） |
| `band_true` | 正解の話速帯（under4 / 4to6 / 6to8 / over8） |
| `sentence` | 原文 |
| `kana` | カタカナ読み |

## 音声ファイルをコミットしない理由

`docs/PLAN.md`「データセットと役割」のとおり **Common Voice は再ホスト・再共有が禁止**されている。
本リポジトリは Public で運用しているため、`results/error_cases/` に置いた音声をコミットすると
Common Voice の音声を再配布することになる。

対処として `.gitignore` に次を追加し、**音声だけを追跡対象から外した**。

    /results/error_cases/*.mp3

`index.tsv` と本 README.md はテキストなのでコミットする。音声は上のコマンドを実行すれば
`data/common_voice_ja/clips/` から手元に再生成できる（元ファイルは読むだけで変更しない）。
第6段階6-3の聴取を行う人は、このディレクトリのコピーを聴くこと。
