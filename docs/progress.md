# 進捗記録

各セッションの終了時に、実施内容・得られた数値・未解決点を5行以内で追記する運用。

## セッション記録

### 2026-09-20 第1段階 環境構築と文書作成
- uv（Python 3.12）でプロジェクトspeakmeterを初期化し、指定の依存を全てインストールした。torch 2.14.0/torchaudio 2.11.0/pyopenjtalk 0.4.1のビルドは3.12で問題なく成功した
- check_env.pyで確認: Python 3.12.13, torch 2.14.0, torch.backends.mps.is_available()=True, mps上のConv2d順伝播成功、pyopenjtalk g2p("今日は快晴です")→「キョーワカイセーデス」成功
- ディレクトリ構成・docs（spec/plan/progress/questions）・README・CLAUDE.mdを作成し、git初期化と初回コミットを完了した
- リモートはgit@github.com:Ukawamochi/SpeedMeter.gitを設定済み（push未実施）
- 未解決点なし。次は第2段階（ラベル生成の実装と検証）

### 2026-09-20 第2段階 ラベル生成の実装と検証
- src/spkrate/labels/mora.pyにread_kana（pyopenjtalkでカタカナ読み変換）とcount_mora（spec.mdの規則でモーラ数算出）を実装した
- tests/test_mora.pyに清音・拗音・促音・撥音・長音・漢字かな混じり・数字・英字・記号句読点・フィラーを網羅する40件のテストを追加し全件通過を確認した
- scripts/inspect_mora.pyを追加し、テキストファイルから原文・カタカナ・モーラ数のタブ区切り出力が動作することを確認した
- 環境側の問題として、editable installの.pthファイルがmacOSでhidden属性を持ちspkrateのimportが不安定になる事象を確認し、pyproject.tomlにpytestのpythonpath設定を追加して回避した
- 未解決点：小書き文字ヵ・ヶ・ヮの扱い等4件をdocs/questions.mdに記録。次は第3段階（評価セットの作成）

### 2026-09-20 データセット取得
- 4本の取得スクリプトを実装・実行。サイズ検証、進捗表示、既存ファイル再利用、MUSANの区間再開に対応し、data/DATASETS.mdへ結果を記録した。
- JSUTテキスト22ファイル（1,027,376 bytes、basic5000は5,000行）、MUSAN noise等935ファイル（717,334,021 bytes、音声930本）を配置。MUSANは公式ミラーから取得し公式MD5一致を確認した。
- JVSは直接取得でHTML応答、Common VoiceはHF_TOKEN未設定のため未取得。理由・必要操作・Common Voiceの配布移転をdocs/questions.mdに記録。取得処理のテスト10件通過。モデル実装・学習は未実施。

### 2026-09-20 モーラ処理の回答反映・確認待ち
- CLAUDE.md、spec.md、questions.mdと既存実装を確認。変更前のpytestは50件通過。
- 「ヮは直前と結合」という仕様とテスト指定「シヮ = 2」の不整合をquestions.mdへ記録。回答待ちのため実装・仕様・テストの変更を保留して停止。

### 2026-09-20 手動データ取得への運用整理
- 取得スクリプト4本・共通処理・取得専用テストを削除し、CLAUDE.mdとREADMEを現行の手動配置・実装状況に合わせて更新した。
- data/DATASETS.mdの再実行案内と古い失敗状態を整理し、配置先と過去の取得完了記録を保持。開発計画の古い「このタスク」表記を削除した。
- データ本体・アーカイブ・未解決のモーラ仕様は保持。pytest 40件通過、削除済み処理への参照と文書リンクを確認した。

### 2026-09-20 モーラ仕様の確定事項を反映
- 人間の確定回答（ヮは常に結合、シヮ=1、ヵ・ヶ単独1、先頭小書き文字は1＋警告、NFKC正規化、非カタカナ残存文は除外）を実装・テスト・spec.mdに反映した。
- src/spkrate/labels/mora.pyをto_kana・count_mora_from_kana・count_mora・has_unconvertedの4関数に再構成し、read_kanaを廃止してscripts/inspect_mora.pyを追従させた。
- docs/spec.mdのモーラの数え方を更新し前処理の節を追加。questions.mdへ疑問1〜4とシヮ確認の反映完了を追記した。
- tests/test_mora.pyを新APIに書き換えヮ結合・ヵヶ・先頭小書き文字の警告・NFKC・has_unconvertedのテストを追加し、pytest 64件全通過。
- 未解決点なし。非カタカナ残存文の除外処理そのものは後続タスク。

### 2026-09-20 第1段階 読み変換の品質測定
- 1-1 測定: JSUT BASIC5000 全5,000文でpyopenjtalkのモーラ数と人手注釈仮名（kana_level0を主・kana_level2を参考）を比較。完全一致92.34%、相対誤差0.35%（参考0.41%）、ID欠落0件。results/mora_label_eval.md と mora_mismatch.tsv（383件）を生成。
- 1-2 分類: 不一致上位200件を7分類。難読語141・固有名詞32・数字9・送り仮名7・実発話準拠9・その他2、記号空白の前処理差は0件。残り183件は全て差の絶対値1。results/mora_mismatch_analysis.md。
- 1-3 判断: 相対誤差0.35%が基準3%を大きく下回るため、ラベル生成処理をこのまま採用と結論。spec既定のhas_unconverted除外は必ず適用する前提（誤差の14.5%を除去）。docs/experiments/001-mora-label-quality.md。
- 1-4 は「除外規則を追加する場合のみ」の条件に該当せずスキップ。未解決は、3桁以上の数字列を含む文の除外提案（仕様変更のため承認待ち）と、音声と原文のずれに由来するラベル誤差が未測定であること。

### 2026-09-20 第1段階 1-4 除外規則の実装（第1段階完了）
- 人間の承認により「3桁以上の数字列を含む文は学習データから除外する」を仕様に追加し、docs/spec.mdの前処理節へ明記した。
- src/spkrate/labels/filter.py に should_exclude_from_training（別名 should_exclude）を実装。理由は "unconverted"（読み変換の失敗痕跡）と "long_digit_run"（NFKC後に[0-9]{3,}）の2種。
- 除外は学習データ構築経路からのみ呼ぶ規約をモジュール・関数docstringとspec.mdに明記し、評価セットには適用しないことを実装上明確にした。
- tests/test_filter.py 22件を追加し pytest 86件全通過。CLAUDE.mdの箇条書き改行漏れも修正した。
- 第1段階は全て完了。次は第2段階（Common Voiceの内容把握と整形）。data/common_voice_ja はcv-corpus-27.0-2026-09-11として展開済みで、版の改訂は目的に影響しないとユーザーが確認済み。

### 2026-09-20 第2段階 2-2 クリップ一覧の作成
- src/spkrate/data/common_voice.py と scripts/build_clips.py を実装。validated.tsv 300,315行から295,179件が残り（残存率98.29%）、data/processed/clips.jsonl（pyarrow未導入のためparquetから自動切替）と results/clip_filtering.md を生成した。
- 除外内訳は down_votes 0、空文 0、unconverted 2,735、long_digit_run 0（unconvertedが先に一致するため）、音声長欠落 0、毎秒モーラ数の外れ値 2,401。
- 毎秒モーラ数は最小1.001・Q1 3.813・中央値4.971・平均5.031・Q3 6.153・最大12.000、合計時間379.67時間（1クリップ平均4.630秒）。
- 実データに1万字規模の文があり pyopenjtalk が異常終了したため、to_kana を2048バイト単位の分割変換に変更した。pytest 104件全通過。
- 未解決は、parquetで保存したい場合の pyarrow 追加と、既存 scripts/classify_mora_mismatch.py の未使用import（ruff F401）。次は 2-3 の分割固定。

### 2026-09-21 第2段階 Common Voiceの内容把握と整形（完了）
- 2-1 内容確認: cv-corpus-27.0-2026-09-11は展開済み。validated 300,315行、clips 585,330件/14.69GiB、client_id 6,932人、全て32kHzモノラルmp3（16kHzへのリサンプリングが必要）。クリップ長は平均4.631秒・中央値3.960秒。results/common_voice_overview.md。
- 2-2 クリップ一覧: 300,315件から unconverted 2,735件・外れ値2,401件を除外し残存295,179件（98.29%）、合計379.67時間。毎秒モーラ数は中央値4.971・平均5.031。long_digit_runは0件で、これは実装の不具合ではなくCommon Voiceのユニーク37,736文に3桁以上の数字列が存在しないため（別途確認済み）。data/processed/clips.jsonl（pyarrow未導入のためparquetではない）、results/clip_filtering.md。
- 付随対処: pyopenjtalkがUTF-8約8192バイト超の入力でSIGABRTする問題に遭遇し、to_kanaに2048バイト単位の分割変換を追加した（仕様変更ではなく堅牢化）。.gitignoreの `data/` がsrc/spkrate/data/まで無視していたため直下のみを指す設定に修正した。
- 2-3 分割の固定: client_id単位・シード20260921の貪欲法でクリップ数が8:1:1になるよう割当。train 3,467人/236,143件/304.80時間、dev 1,711人/29,518件/37.10時間、test 1,711人/29,518件/37.76時間。話者重複0を確認。test.jsonは第10段階まで使用禁止で、load_test_splitはstage10_approved=Trueなしでは例外を送出する。
- pytest 121件全通過。未解決点なし。次は第3段階（ベースラインの実装と測定）。
