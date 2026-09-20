# Common Voice 日本語データ 内容確認

- 対象ディレクトリ: `data/common_voice_ja`
- 版: cv-corpus-27.0-2026-09-11（Common Voice 日本語）
- 生成スクリプト: `scripts/inspect_common_voice.py`
- 無作為抽出の固定シード: `20260920`（`random.Random(20260920)`）
- data/ 以下は読み取りのみ行い、変更・削除はしていない

## 1. 各tsvファイルの行数

行数はいずれもヘッダ行を除いたデータ行数である。
「改行数-1」は改行を数えた値、「csvレコード数」はタブ区切り・引用符なし（QUOTE_NONE）で解析したレコード数からヘッダ1行を引いた値。両者が一致すれば1レコード1行である。

| ファイル | データ行数（改行数-1） | csvレコード数 | サイズ |
| --- | ---: | ---: | ---: |
| clip_durations.tsv | 585,330 | 585,330 | 18.99 MiB |
| dev.tsv | 9,020 | 9,020 | 2.73 MiB |
| invalidated.tsv | 55,390 | 55,390 | 18.50 MiB |
| other.tsv | 229,625 | 229,625 | 70.38 MiB |
| reported.tsv | 839 | 839 | 133.76 KiB |
| test.tsv | 9,020 | 9,020 | 2.69 MiB |
| train.tsv | 19,696 | 19,696 | 6.12 MiB |
| unvalidated_sentences.tsv | 6,946 | 6,946 | 1.01 MiB |
| validated.tsv | 300,315 | 300,315 | 92.20 MiB |
| validated_sentences.tsv | 44,606 | 44,606 | 6.88 MiB |

すべてのファイルで両者が一致した（1レコード1行）。

## 2. validated.tsv のカラム一覧と先頭5行

カラム数: 13

1. `client_id`
2. `path`
3. `sentence_id`
4. `sentence`
5. `sentence_domain`
6. `up_votes`
7. `down_votes`
8. `age`
9. `gender`
10. `accents`
11. `variant`
12. `locale`
13. `segment`

先頭5行（データ行）。client_id と sentence_id は長いため先頭12文字のみ示す。

| client_id | path | sentence_id | sentence | sentence_domain | up_votes | down_votes | age | gender | accents | variant | locale | segment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00bafc5b7e22… | common_voice_ja_41788246.mp3 | d21965e1a257… | 私はその人の記憶を呼び起すごとに、すぐ「先生」といいたくなる。 |  | 2 | 0 |  |  |  |  | ja |  |
| 013d4d9b1402… | common_voice_ja_39038752.mp3 | 25c7bc49a044… | 忘れてしまった記憶は、どこに行くのだろうか |  | 8 | 4 | teens | male_masculine |  |  | ja |  |
| 0142e35941e7… | common_voice_ja_41143600.mp3 | cec33c42740f… | 元気の出る曲をかけて | general | 2 | 0 | fourties | male_masculine |  |  | ja |  |
| 02f461e83afc… | common_voice_ja_44940582.mp3 | 6de8f0a3bea8… | 国の滅亡を嘆くことのたとえ。 |  | 2 | 0 |  |  |  |  | ja |  |
| 032d7636ea3a… | common_voice_ja_41759859.mp3 | 3061d18c4cf3… | 青春時代、清掃員が清掃して清々しい |  | 27 | 8 |  |  |  |  | ja |  |

## 3. 音声ファイル（clips/）

- ファイル総数: 585,330
- 合計サイズ: 14.69 GiB（15,772,228,726 バイト）
- 拡張子が .mp3 でないファイル: 0
- 参考: validated.tsv の行数 300,315 に対し、clips/ には 585,330 件（other/invalidated 等も含む全クリップ）が存在する

## 4. validated.tsv の話者数

- client_id のユニーク数: 6,932
- validated.tsv のデータ行数: 300,315
- 1話者あたりの平均クリップ数: 43.3

## 5. 無作為10件の音声のサンプリングレートとチャンネル数

validated.tsv の path 列から固定シード 20260920 で10件を抽出し、soundfile（libsndfile）で読み取った。

| クリップ | サンプリングレート(Hz) | チャンネル数 | 形式 | 長さ(秒) |
| --- | ---: | ---: | --- | ---: |
| common_voice_ja_36311807.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 4.255 |
| common_voice_ja_38999615.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 7.440 |
| common_voice_ja_29765183.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 6.056 |
| common_voice_ja_41885959.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 2.635 |
| common_voice_ja_41936837.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 3.156 |
| common_voice_ja_45086904.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 2.455 |
| common_voice_ja_42022527.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 3.012 |
| common_voice_ja_36049697.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 5.516 |
| common_voice_ja_41764942.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 3.840 |
| common_voice_ja_39905722.mp3 | 32,000 | 1 | MP3/MPEG_LAYER_III | 3.716 |

10件すべてで サンプリングレート [32000]、チャンネル数 [1] であった。

## 6. 無作為1000件のクリップ長

**使用した方法: clip_durations.tsv の値**（duration[ms] をミリ秒として読み、秒に換算した）。validated.tsv の path 列から固定シード 20260920 で1000件を抽出した（上記10件の抽出のあと、同じ乱数列から抽出）。

- 対象件数: 1,000（clip_durations.tsv に無かった件数: 0）
- 最小: 1.116 秒
- 最大: 13.320 秒
- 平均: 4.631 秒
- 中央値: 3.960 秒
- 合計: 1.29 時間

### 実音声との突き合わせ

clip_durations.tsv の値が信用できるか確認するため、5節の10件について soundfile で読んだ実長（frames / samplerate）と比較した。

| クリップ | clip_durations.tsv(秒) | 実音声(秒) | 差(秒) |
| --- | ---: | ---: | ---: |
| common_voice_ja_36311807.mp3 | 4.248 | 4.255 | 0.007 |
| common_voice_ja_38999615.mp3 | 7.416 | 7.440 | 0.024 |
| common_voice_ja_29765183.mp3 | 6.048 | 6.056 | 0.008 |
| common_voice_ja_41885959.mp3 | 2.628 | 2.635 | 0.007 |
| common_voice_ja_41936837.mp3 | 3.132 | 3.156 | 0.024 |
| common_voice_ja_45086904.mp3 | 2.448 | 2.455 | 0.007 |
| common_voice_ja_42022527.mp3 | 2.988 | 3.012 | 0.024 |
| common_voice_ja_36049697.mp3 | 5.508 | 5.516 | 0.008 |
| common_voice_ja_41764942.mp3 | 3.816 | 3.840 | 0.024 |
| common_voice_ja_39905722.mp3 | 3.708 | 3.716 | 0.007 |

差の最大は 0.024 秒であった。

## 7. .gitignore の確認

`.gitignore` の `/data/*`（例外は `/data/DATASETS.md` のみ）により、`data/.archives/` は無視される / `data/common_voice_ja/` は無視される（`git check-ignore` で確認）。

