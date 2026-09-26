# データセットの配置

データセットはユーザーが手動で取得・配置する。使用前に対象ファイルの存在と構成を確認する。
取得中のファイルやアーカイブはそのまま保持する。

| データセット | 配置先 | 使用する内容 |
|---|---|---|
| JSUT | data/jsut/ | basic5000/transcript_utf8.txt（テキスト） |
| JVS | data/jvs/ | 音声・書き起こし |
| MUSAN | data/musan/ | noise/ 以下の雑音 |
| Common Voice 日本語 | data/common_voice_ja/ | 日本語の音声・書き起こし |

## ubuntu-desktop 側の配置

ubuntu-desktop（学習と推論を実行する遠隔機）でも、リポジトリ `~/SpeedMeter` の下に Mac と同じ相対パスで置く（例: `~/SpeedMeter/data/common_voice_ja/`）。
遠隔機へは Mac から `scripts/sync_to_remote.sh --data` で送る。送るのは data/common_voice_ja・data/musan・data/processed で、`--delete` は使わない。送った後、スクリプトが通常ファイルの数と合計バイト数を Mac と遠隔機で比べて表示する。
正本は Mac の data/ であり、遠隔機のデータを直接取得・変更しない。2026-09-27 の送り出しでは3件とも一致した（docs/decisions/010-compute-environment.md 4.2節）。

## 確認済みの取得記録

以下は過去の確認完了時点の記録。以後の手動配置の進捗は反映していない。
JVSの配置状況は手動取得後に確認する。
ファイル数とサイズは展開済みデータ本体の値で、data/.archives/ のアーカイブを含まない。

| データセット | 配置場所 | 確認日 | 状態 | ファイル数 | 合計サイズ (bytes) | 内容・取得元 |
|---|---|---|---|---:|---:|---|
| jsut | data/jsut/ | 2026-09-20T20:32:14+09:00 | 取得済み | 22 | 1,027,376 | テキストのみ。https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip |
| musan | data/musan/ | 2026-09-20T21:10:15+09:00 | 取得済み | 935 | 717,334,021 | noise サブセット。https://www.openslr.org/resources/17/musan.tar.gz; 公式MD5検証あり; mirror: https://openslr.elda.org/resources/17/musan.tar.gz |
| common_voice_ja | data/common_voice_ja/ | 2026-09-20T22:47:00+09:00 | 取得済み | 585,341 | 約17,180,000,000（約16GiB） | Common Voice 日本語（cv-corpus-27.0-2026-09-11）。clips/ に音声585,330件、train/dev/test等のtsvを含む |
