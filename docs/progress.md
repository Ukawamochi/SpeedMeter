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
