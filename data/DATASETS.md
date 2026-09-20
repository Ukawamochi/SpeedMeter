# データセットの配置

データセットはユーザーが手動で取得・配置する。使用前に対象ファイルの存在と構成を確認する。
取得中のファイルやアーカイブはそのまま保持する。

| データセット | 配置先 | 使用する内容 |
|---|---|---|
| JSUT | data/jsut/ | basic5000/transcript_utf8.txt（テキスト） |
| JVS | data/jvs/ | 音声・書き起こし |
| MUSAN | data/musan/ | noise/ 以下の雑音 |
| Common Voice 日本語 | data/common_voice_ja/ | 日本語の音声・書き起こし |

## 確認済みの取得記録

以下は過去の取得完了時点の記録。以後の手動配置の進捗は反映していない。
JVS・Common Voiceの配置状況は手動取得後に確認する。
ファイル数とサイズは展開済みデータ本体の値で、data/.archives/ のアーカイブを含まない。

| データセット | 配置場所 | 取得日 | 取得時の状態 | ファイル数 | 合計サイズ (bytes) | 内容・取得元 |
|---|---|---|---|---:|---:|---|
| jsut | data/jsut/ | 2026-09-20T20:32:14+09:00 | 取得済み | 22 | 1,027,376 | テキストのみ。https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip |
| musan | data/musan/ | 2026-09-20T21:10:15+09:00 | 取得済み | 935 | 717,334,021 | noise サブセット。https://www.openslr.org/resources/17/musan.tar.gz; 公式MD5検証あり; mirror: https://openslr.elda.org/resources/17/musan.tar.gz |
