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
