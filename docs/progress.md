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

### 2026-09-21 第3段階3-1 評価基盤（完了）
- src/spkrate/eval/metrics.py（毎秒モーラ数のMAE・話速帯別MAE・ピアソン相関）、audio.py（16kHzモノラルへの読み込み）、runner.py（推定器の適用とmetrics.csvへの追記）を実装。
- 取り決め: 話速帯は正解の毎秒モーラ数で区分し、該当0件の帯のMAEと分散0・件数1以下の相関はNaN（CSVでは空欄）。各帯の件数もn_band_*列に残す。
- results/metrics.csv の列を experiment_id,timestamp,commit_hash,commit_dirty,config_path,method,split,num_segments,mae_*,n_band_*,correlation,latency_ms_per_inference,model_size_bytes に固定（ヘッダのみ変更、データ行は未追加）。テスト分割はstage10_approved=Trueなしでrunnerが拒否する。
- pytest 169件全通過（新規48件）。実データ評価は3-2以降。所見: clipsは32kHzが大半だが48kHzの個体も存在するため、リサンプリングは実ファイルの標本化周波数から行う実装にした。

### 2026-09-21 第3段階3-2 信号処理ベースライン（完了）
- src/spkrate/baselines/envelope.py を実装。RMS包絡→デシベル化→平滑化→2〜10Hz帯域通過（Butterworth零位相）→無音ゲート付きのピーク計数→換算係数でモーラ数化。デシベル領域で処理するため録音レベルに依存しない。
- 調整はdevから固定シード20260921で抽出した2000件で実施（包絡をキャッシュして座標降下法2巡）。採用値は帯域3〜10Hz・平滑化0.05秒・突出量0.5dB・最小間隔0.06秒・無音閾値-20dB・換算係数1.5278（部分集合MAE 0.8675）。ピーク計数はゼロ交差計数より良い（0.9284）。configs/baselines/envelope.yaml、results/envelope_baseline_tuning.md。
- dev全件29,518クリップ（37.1時間、0.36分）で評価しmetrics.csvに実験ID 003-envelope-baseline を追記。MAE 0.8961、帯別 under4=0.8399 / 4to6=0.7284 / 6to8=0.8737 / over8=2.1677、相関0.7670、1推論0.416ミリ秒。
- 帯域候補はPLAN 3-2の指定（2〜10Hz）の内側に限った。16Hzまで広げても改善は0.007以下で不採用（記録は調整md）。依存にscipyを明示追加。
- pytest 214件全通過（新規45件、合成パルス列で計数が一致することを確認）。未解決点は毎秒8モーラ以上の帯の誤差が他の2倍以上あること。次は3-3。

### 2026-09-21 第3段階3-4 目標値の設定（完了）
- 同一の dev 部分集合5,000件で比較: 包絡法 MAE 0.9059 / 相関0.7606 / 0.485ms、書き起こし MAE 0.2836 / 相関0.8497 / 851.7ms。精度は3.2分の1、速度は1,757倍の差。
- 包絡法の弱点は端の2帯に偏る。8以上（6.7%）でMAE 2.1992（全体誤差の16.2%を占める、換算係数固定による頭打ち）、4未満は相対誤差28.1%で最悪。最頻の4-6帯では定数予測（0.502）より悪く分解能がない。
- 目標値（dev・クリップ単位、6項目すべて必達）: 4未満 0.45 / 4-6 0.40 / 6-8 0.55 / 8以上 1.35 / 全体 0.50 / 相関 0.85以上。中央2帯は帯平均話速の相対8%、端2帯は相対15%で置き、どの帯も包絡法→書き起こしの差を6割以上詰める水準。
- 処理時間の上限（8-3で判定）: 2.0秒窓1回の推論が平均10.0ms・最大20.0ms（int8後）、モデル5MB以下。仕様の推論間隔0.25秒とブラウザ常時動作から占有率20%・wasm4倍で導出。書き起こしより38〜85倍軽くする必要がある。
- 未解決: ラベル雑音の実量（MAE 0.35を下回ったら目標を下げない歯止めを設定）、窓単位の目標未設定、wasm4倍が未実測、8以上の帯は333件で標準誤差が大きい。docs/experiments/002-baseline.md。

### 2026-09-21 第4段階4-1 特徴量の事前計算（完了）
- src/spkrate/features/melspec.py: 仕様のパラメータで対数メルスペクトログラムを計算。仕様に無い窓・詰め方・メル尺度（hann periodic / reflect / htk / norm=None）は torchaudio の既定に揃え、8-2の突き合わせ用に docstring へ全て明記。
- scripts/precompute_features.py で train 236,143件（109,860,234フレーム・16.37GiB）と dev 29,518件（13,373,969フレーム・1.99GiB）を float16 のシャード npy に保存。失敗0件、8並列で計1分43秒。docs/decisions/004-feature-storage.md。
- configs/normalization.yaml: train のみから算出。既定はメル次元ごと80組（mean -11.18〜-3.57、std 3.17〜5.65）、全体で1組は mean -7.2877 / std 4.6502。
- pytest 258件全通過（新規17件、numpy参照実装との一致でパラメータ適用を確認）。未解決は4-2の拡張を波形側で行うか拡張済み特徴量を別に作るか。次は4-2。

### 2026-09-21 第4段階4-2 データ拡張（完了）
- src/spkrate/data/augment.py: 時間伸縮0.7〜1.5倍（位相ボコーダ、n_fft=1024。440Hzが±5Hzに保たれることを確認）・雑音重畳SNR0〜20dB（MUSAN noise 930件）・残響（RT60 0.1〜0.7秒、部屋3〜10×3〜10×2.3〜4.0m、直接音と長さを保つ）・音量−12〜+6dB・帯域制限（下0〜300Hz／上3400〜7800Hz）。既定の適用確率は順に0.5/0.5/0.3/0.5/0.25。
- 拡張は波形に適用しメルは毎回計算し直す。実測で再計算0.32ms／事前計算特徴量の読み出し0.28ms（1クリップ、平均3.894秒）と同等のため作り置きの利点が無い。事前計算特徴量は検証・評価と拡張なしの対照実験に使う。docs/decisions/006-augmentation.md、scripts/benchmark_augment.py。
- MUSAN は speech を渡すと ValueError で拒否する（別話者の声がモーラ数のラベルと矛盾するため）。時間方向のマスクは実装せず、API が無いことをテストで固定。周波数方向のマスクのみ特徴量に適用。
- pytest 314件全通過（新規56件）。新規依存は無し（librosa・pyroomacoustics・scipy は既存）。
- 未解決は適用確率が暫定で根拠となる測定が無いこと（第5段階以降にdevで見直す）。次は4-3。

### 2026-09-21 第4段階4-3 窓の切り出し方式（完了）
- **方式A**（クリップ全体を入力、全体のモーラ数を正解）を採用。方式Bは第7段階7-2の探索条件として保留。PLAN.mdの誘導を追認した。docs/decisions/005-window-strategy.md。
- 根拠: 仕様のモデルがフレームごとのsoftplus出力の総和なので区間長の不一致が原理的に生じない。平均クリップ長4.630秒は2秒窓の2.32倍。2.0秒未満のクリップは方式Aでは使えるが方式Bでは切り出せず、時間伸縮0.7倍と併せると推定約28%が窓を作れない。方式Bの「データ量増」は重複した切り出し位置の増加で合計304.80時間は不変。
- 吸収できないのは窓端の切断（取りこぼし0.05〜0.1秒なら0.25〜0.50 mora/s）。5-4で分割整合性D1・無音窓D2・連結加法性D3を記録し、D1>0.25 mora/s、D2>1.0モーラ、または7-2で全体MAE最良値が0.55超のままなら方式Bへ移る。
- 第5段階への制約: 受容野は201フレーム（2.0秒）以下・推奨51〜101、詰め物フレームは総和の前にマスクし詰め物量に出力が依存しないことをテストで固定、正解はモーラ数（毎秒モーラ数ではない）。
- 未解決: クリップ長のヒストグラムが未測定で2.0秒未満の件数が確定できない（方式Aの採否には影響しないが方式Bに移るなら必須）。窓単位の目標値は依然なくD1〜D3で代替。次は第5段階5-1。

### 2026-09-21 第5段階5-1 モデル定義（完了）
- src/spkrate/models/cnn.py と configs/model/cnn_base.yaml。既定は周波数段3層（32/64/64、刻み2で80→10）＋時間段5層（128、膨張率1/2/4/8/16）、受容野69フレーム（0.69秒）、498,881パラメータ（float32で1.90MiB、int8概算0.48MiB。目標5MB以下を満たす）。
- 出力はクリップのモーラ数（毎秒モーラ数ではない）。入力は (バッチ, フレーム, 80) の正規化済み対数メル、lengths で可変長を受ける。層数・チャネル数・膨張率・カーネル・活性化・正規化は configs から指定し、受容野201フレーム超の設定は CnnConfig が拒否する。
- 詰め物は入口と各層の出力で0に戻し、softplus の後にマスクしてから総和する。単独で流した場合と詰め物0/1/100/500フレーム付きで流した場合の一致を rtol=1e-4, atol=1e-4 で固定した。バッチ・時間方向を混ぜる正規化は使えないため norm は none / layer のみ。
- pytest 351件全通過（新規37件）。mps で前向き・逆伝播とも通り、PYTORCH_ENABLE_MPS_FALLBACK=0 でもCPUフォールバックは発生しなかった。学習は未実施。
- 未解決は norm・dropout の既定値（none / 0.0）が測定に基づかないこと、時間段の入力640チャネルが第8段階の推論速度に効く可能性があること。次は5-2。

### 2026-09-21 第5段階5-2 学習ループ（完了）
- src/spkrate/train/train.py と src/spkrate/train/data.py、configs/exp000_smoke.yaml。損失は train.loss で mse / poisson を切り替え（ポアソンは log_input=False, eps=1e-8）、デバイスは mps、出力は runs/<実験ID>/ に log.txt・metrics.jsonl・config_snapshot.yaml（コミットハッシュと設定全文）・checkpoint_best.pt / checkpoint_last.pt。
- データ供給は2経路。拡張なしは事前計算特徴量を memmap で読み、拡張ありは波形から都度計算する（拡張あり×事前計算特徴量の組み合わせは ValueError で拒否）。長さでまとめるバッチ分け（無作為→塊ごとに長さで整列→バッチ順を無作為化）を既定で使う。検証は常に dev の事前計算特徴量で拡張なし。
- 完走確認: configs/exp000_smoke.yaml（学習512件・検証256件・3エポック・mse）を mps でバックグラウンド実行し約75秒で完走。metrics.jsonl 3行、最良エポック2（毎秒モーラ数MAE 1.833）。PYTORCH_ENABLE_MPS_FALLBACK=1 でもCPUフォールバックは検出されず。指標は完走確認用で比較には使わない（results/metrics.csv には残していない）。
- pytest 375件全通過（新規24件）。config_snapshot.yaml の書き出しで yaml.safe_dump が str の派生型（torch.__version__）を拒否する不具合を直した。
- 未解決は学習率・エポック数などの最適化設定が未検討で3エポック目に検証損失が跳ねたこと（5-3の本番設定で見直す）。次は5-3。

### 2026-09-21 第5段階5-3 初回学習（完了）
- configs/exp001.yaml: 損失mse・拡張なし（比較は第7段階）・学習率1e-4・6エポック・バッチ64・train全236,143件・シード20260921。学習率は本番前の短い試行（3e-3〜3e-5、runs/exp001_lrtrial/）で1e-4のみ検証損失と相関が単調改善したため採用。
- mpsでバックグラウンド実行。1エポック約20分＋dev全件評価約3分で計約2時間半。CPUフォールバックは検出されず。6エポックすべて検証指標が改善し最良は最終エポック6。
- 最良チェックポイントをdev全件29,518件で評価: MAE 0.5013、帯別 under4=0.4033 / 4to6=0.4099 / 6to8=0.5300 / over8=1.3824、相関0.8905。2.0秒窓1回の推論1.746ミリ秒（メル計算を含む）、モデル2,005,357バイト。metrics.csv に 005-first-model を追記。
- 統括Agent側の手違いで同じ評価を2回走らせ metrics.csv に重複行ができたため、判定文書が採用した側を残して削除した（精度指標は両行で同一）。
- ブランチ exp/005-first-model を main へ取り込み、pytest 375件全通過。

### 2026-09-21 第5段階5-4 初回結果の判定（完了・第5段階完了）
- 目標6項目中3到達3未到達。到達は4未満0.4033≤0.45・6-8 0.5300≤0.55・相関0.8905≥0.85。未到達は4-6 0.4099>0.40・8以上1.3824>1.35・全体0.5013>0.50（超過はいずれも僅差で、全体は+0.0013）。
- 過学習なし。6エポックすべて検証損失・検証MAEが単調改善し最良＝最終エポックで、打ち切りは所要時間から決めたepochs:6に達したため。つまり構造の限界ではなく収束前の停止。直近の改善幅0.0196/epochは未達3項目の不足幅と同程度以上。
- 帯別の弱点は2つ。8以上帯の系統的過小評価（誤差1.3824のうち1.2851が偏り、回帰の傾き0.825）と、より重い短入力での悪化（1.8-2.2秒クリップでMAE 0.6881、8以上帯の2.0秒未満では4.0048）。後者は配備単位の2.0秒窓に直結する。
- 方式Bへ移る条件はB1・B2・B5が判定不能（D1〜D3診断は前向き計算が必要で判断サブエージェントには測れない）、B3・B4は非該当。D1〜D3は第6段階の測定サブエージェントが6-1と同枠で実施する。
- 同一部分集合5,000件の再集計ではモデルMAE 0.4946・相関0.8996で、包絡0.9059と書き起こし0.2836の差を全帯で62.5〜71.0%詰めた。docs/experiments/003-first-model.md。

### 2026-09-21 第6段階6-1 誤り事例の抽出とD1〜D3診断（完了）
- scripts/extract_errors.py。dev全29,518件（runs/exp001/predictions_dev_full.json）から毎秒モーラ数の絶対誤差の大きい上位100件を results/error_cases/ へ。誤差は最大11.5986・最小6.1019 mora/s、正解の帯は over8 が75件・6to8 が25件。定義は index.tsv ヘッダと README.md に明記。
- 音声100件はコピーするが .gitignore に /results/error_cases/*.mp3 を追加してコミットしない（Common Voice は再ホスト・再共有が禁止、Public リポジトリ）。index.tsv と README.md はコミットした。
- 第4段階4-3が5-4に課したD1〜D3を実施（scripts/window_diagnostics.py、src/spkrate/eval/window_diag.py、mps、CPUフォールバック無し）。D1=0.4360 mora/s（閾値B1 0.25を超過）、D2はデジタル無音0.0824モーラ（B2 1.0を超過しない）・MUSAN noise 4.7024モーラ（超過）、D3=0.0568 mora/s（B5の0.25を超過せず、B5は不成立）。results/window_diagnostics.md。
- 採否の判断は書いていない（第7段階7-1）。docs/experiments/003-first-model.md 5.2節の更新は未実施で申し送る。pytest 390件全通過（新規15件）。

### 2026-09-21 第6段階6-2 テキスト面の分析（完了）
- index.tsv 上位100件を音声を聴かずに排他分類（優先順位 A ラベル誤り＞D 特殊表記＞B 極端な長さ＞C 分布の端＞E 判断不能）。A=6・B=30・C=25・D=2・E=37。非排他では C が52件・D が6件で、重なりは results/error_analysis_text.md の2.2節に併記した。
- 閾値は明文化。B は dev のクリップ長 第5/第95百分位 1.98/9.396秒（2.0秒未満・9.4秒以上）、C は clip_filtering.md のヒストグラムから補間した第1/第99百分位 1.46/9.39（1.5未満・9.4超）、D は NFKC後の正規表現4条件。
- A の6件は英文・ローマ字のアルファベット読み展開（taishouyatteru で+21モーラ、英文で+40モーラ超）、五六匹→ゴジューロッピキの位取り読み、昔し→ムカシシ。学習用の除外は dev に適用されないため残存。
- 上位100件は全件が過小推定で over8 が75件（dev では6.83%）、9.4超が52件（dev では1.66%）、under4・4to6 は0件。母集団を代表せず、C・B の件数は抽出方法で構造的に押し上がり、A・D の頻度は推定できない。
- E の37件は6-3の聴取対象。clip_id 一覧を results/error_analysis_text.md 5節と results/error_cases/to_listen.tsv に出した。

### 2026-09-21 第6段階 誤り分析（6-1・6-2完了、6-3で停止）
- 6-1: scripts/extract_errors.py でdev全件から誤差上位100件を results/error_cases/ に抽出（音声100件・index.tsv・README.md）。誤差は毎秒モーラ数の絶対値で定義。誤差は11.5986〜6.1019 mora/s、100件すべて推定が過小で正解帯はover8 75件・6to8 25件。Common Voiceは再ホスト禁止のため音声は .gitignore の /results/error_cases/*.mp3 で除外しコミットしない。
- 6-1（併せて実施）: 第4段階4-3が課したD1〜D3診断を src/spkrate/eval/window_diag.py と scripts/window_diagnostics.py で実施。D1分割整合性0.4360 mora/s（閾値B1 0.25を超過）、D2無音窓はデジタル無音0.0824モーラ（B2 1.0未満）だがMUSAN noiseでは4.7024モーラ（超過）、D3連結加法性0.0568 mora/s（B5非該当）。results/window_diagnostics.md。
- 6-2: 上位100件を排他分類（優先順位 A>D>B>C>E）。Aラベル誤り6件、B極端な長さ30件（全件2.0秒未満）、C分布の端25件（全件上側）、D特殊表記2件、Eテキストから判断不能37件。判定基準はdevの実測分位点から定義。results/error_analysis_text.md。
- 6-2の所見: 誤差上位抽出は正解値の大きい事例を集めるためBとCは構造的に押し上げられ、下位2帯（under4・4to6）は0件で母集団を代表しない。上位100件の推定毎秒モーラ数の中央値は0.567で63件が1.0未満。
- 6-3は人間の作業（聴取）のため停止。対象37件の一覧は results/error_cases/to_listen.tsv。

### 2026-09-24 第6段階 追加診断 低出力事例の切り分け（完了）
- 対象63件（誤差上位100件のうち推定1.0 mora/s未満）と比較63件（dev絶対誤差中央値0.3123以下からシード20260921で抽出）を比較。Holm補正後に19項目で差。中央値は実効値0.0023/0.0596、ピーク0.0215/0.49、無音標本割合0.73/0.38、クリップ長2.27/4.18秒。標本化周波数・チャンネル数には差なし。results/low_output_diagnosis.md・.tsv。
- フレーム出力は対象群の20件で全フレーム0.01以下、44件で全フレーム0.05以下（比較群は0件）。MPSのCPUフォールバックは無し。pytest 398件全通過。
- exp001は augment.enabled=false で、音量変化・雑音重畳の適用率はいずれも0。コード既定値は音量変化 p=0.5・−12〜+6 dB、雑音 p=0.5・SNR 0〜20 dB。D2は雑音のみの入力でSNR未定義のため、学習時のSNR範囲とは一致しない。results/augmentation_check.md。
- 未解決: 拡張ありの経路では音声パスの既定値が存在しない場所になる（train.py 157・data.py 396）。修正は未実施で、詳細はaugmentation_check.md 3.2節。第7段階には進んでいない。

### 2026-09-24 拡張設定の欠落を学習開始前に検出（完了）
- run_training の冒頭で check_augment_setup を呼ぶ。musan_root が欠けている場合、features 経路、確率0、無変化の範囲、依存の欠落など21条件で AugmentSetupError を投げて停止する。log.txt の先頭にコミットと拡張の一覧（無効なら「拡張=なし」）を書く。
- 一覧は results/augment_validation.md。判断を保留した4件は docs/questions.md。pytest は433件すべて通過。

### 2026-09-24 exp002（拡張あり）のdev評価と診断（完了）
- scripts/eval_dev_full.py（exp001 と同じ手順を引数化）で dev 全件を評価し、metrics.csv に 007-augmentation 行を追記した。exp001 を同じスクリプトで評価し直すと指標・予測は完全一致した。
- 低出力件数・D1〜D3（window_diagnostics.py に出力先の引数を追加し、exp001 の値の再現を確認）・エポック所要時間を results/exp002_eval.md に並記した。解釈は書いていない。
- 処理時間は測定日で値が異なる（exp001: 09-21 は 1.75ms、09-24 は 4.18ms）。CPU フォールバックは 0 件。pytest は433件すべて通過。

### 2026-09-24 拡張の効果の比較（タスク1〜4完了、承認待ちで停止）
- 音声パスの既定値を修正し（1e7ed01）、拡張設定の欠落を学習開始前に停止させるようにした（7d743c1）。exp002（configs/exp002.yaml、拡張のみ有効、6エポック）を学習・dev評価した。
- dev全件のMAEは0.5013→0.5992、帯別4つはすべて悪化、相関は0.8905→0.8663。推定1.0未満は215→148件。D1は0.436→0.767、D2は無音0.08→1.44・雑音4.70→1.74モーラ。1エポックは約23→約37分。
- exp002は未収束（最終エポックで改善幅が最大）。劣化の主因が学習の遅れか拡張とdevの不一致かは区別できない。docs/experiments/007-augmentation-effect.md。方式Bの移行判定は保留、6-3の聴取も保留中。第7段階には進んでいない。

### 2026-09-24 診断指標D1・D2・D3の定義をspec.mdに明記
- 人間の指示でdocs/spec.mdに「診断指標」節を新設し、D1・D2・D3の入力・計算・意味と閾値B1・B2・B5の根拠を005（4.3節・5節・7節）から転記した。定義とコードは変えていない。
- 005に無く実装から読み取った項目（乱数の種、D1/D3の平均の取り方、D2の雑音の切り出しと2種の扱い、D3の母集団と組の作り方）はその旨を明記した。B2の1.0とB5の0.25の数値の根拠は既存docsに記載なしと明記した。

### 2026-09-24 雑音下評価セット dev_noisy と clean・noisy 両方の評価
- configs/eval/dev_noisy.yaml と src/spkrate/eval/noisy.py を追加した。dev全件×SNR 5/10/15dBに固定の残響（6.0×4.5×2.7m、RT60目標0.5秒、max_order 30で実測0.51秒）→MUSAN noiseを掛ける。雑音はシード20260924とclip_idで決まりSNRに依存しない。波形から都度生成（dev全件×3条件で約6分の見積り）。
- runner.py と scripts/eval_dev_full.py で metrics.csv の split 列を dev / dev_noisy_snr5・10・15 / dev_noisy_all（3条件まとめ）に分けて追記する。noisy行のconfig_pathは「実験設定;dev_noisy設定」。
- exp001 で dev 先頭200件の clean・noisy 評価が通ることをテスト用の csv で確認した（results/metrics.csv には追記していない）。dev全件の noisy 評価は未実施。

### 2026-09-25 exp003（学習中）に対する8以上帯MAE揺れの調査（第7段階には未進行）
- 指定のエポック16チェックポイントは上書き保存方式のため学習継続中に消失、ユーザー承認により実行時点のcheckpoint_best.pt（epoch=25）で代替した。results/dev_distribution.md：8以上帯MAEのブートストラップ95%CI幅0.1661はエポック間揺れ幅0.166とほぼ同一（約1.0006倍）。
- results/error_direction.md：過小評価が話速帯とともに系統的に強まる（4未満44.7%→8以上81.2%、平均誤差+0.0619→-1.1174）。results/loss_definition.md：train/val損失の定義は同一（mse・同一集約）で、エポック16の比率約2.338倍は定義差だけでは説明できない。
- results/lr_schedule.md：Adam・学習率0.0001固定（スケジューラなし、resume時も設定値で上書き）、weight_decay=0.0・grad_clip=5.0・dropout=0.0。results/checkpoint_comparison.md：全体MAE最小=epoch25、8以上帯MAE最小=epoch18で異なる。
- チェックポイントは最良1つのみ保存されていたため、train.save_every_epoch（既定True）を追加しエポックごとの個別保存を有効化（再学習はしていない）。pytest 552件全通過。
- 未解決点：残差（損失比率約2.338倍のうち定義差以外の要因）は本調査の範囲外。第7段階には進んでいない。

### 2026-09-25 8以上帯の偏りの追加調査（第7段階には未進行）
- exp003は早期終了で完了（通算30エポック、最良25）。results/prediction_distribution_comparison.md：devの正解と予測（epoch25）を比較、標準偏差比（予測/正解）0.9588、回帰の傾き0.8605・R²0.8054・切片0.5724。8以上の各ビンで予測件数が正解件数を一貫して下回る。
- results/epoch_variance_over8.md：全24エポック（7〜30）のmae_band_over8の不偏標準偏差0.092862、前回CIから逆算したSE 0.042372、比2.19。全24行での実際の範囲は1.128〜1.462（幅0.3335）で、前回記載の揺れ幅1.128〜1.294（幅0.166）とは一致せず、原因未特定のまま事実として注記。
- results/loss_mae_trajectory.md：val_loss悪化かつ全体MAE改善の区間は23組中1件（epoch10→11）のみ、帯別ではunder4/4to6/over8が減少・6to8のみ増加（+0.005138）。val_lossとmae_moras_per_secの相関r=0.9534。
- 上記3件とも既存ログ（metrics.jsonl・predictions_best.json）のみで算出し、再学習・追加推論は行っていない。第7段階には進んでいない。
- 別プロセスとしてexp004（拡張あり、exp002から再開）の学習が本セッション外でバックグラウンド実行中（このセッションでは開始・変更していない）。
- docs/experiments/008-convergence-comparison.md：exp003とexp004を比較し、基準構成をexp004（無音サンプル＋拡張）とする結論（承認待ち）。dev_noisyまとめMAE 0.5956 vs 2.5053、noisyの1.0未満 562 vs 17,605件、clean MAE差0.0037。
- 無音出力はexp003が良い（D2デジタル無音 0.0043 vs 0.5227モーラ）。exp004は上限30で終了（未収束の可能性）。
- MUSAN noiseの930ファイルはdev_noisy・D2・exp004拡張・両者の無音サンプルで共通（分割なし）をコードで確認。noisy指標はexp004に有利に偏りうる。
- 方式B移行判定は保留、第7段階には進んでいない。学習・推論は行っていない（既存予測の再集計のみ）。

### 2026-09-25 統括: 保留4件の反映〜収束比較（タスク1〜6完了、承認待ちで停止）
- 保留4件の回答を実装した（拡張ごとの enabled、spec の範囲外は停止、適用率をログに記録）。D1〜D3 の定義と無音サンプルを spec.md に追記した。dev_noisy（SNR 5/10/15dB＋固定IR）と、推論時間の同一セッション測定（latency_session_id）を追加した。早期終了・再開・計時を実装した。pytest 552件通過。
- exp003（拡張なし）: 早期終了、通算30、最良25、clean 0.4346 / noisy 2.5053、無音 D2 0.0043。exp004（拡張あり）: 上限30で終了、最良27、clean 0.4383 / noisy 0.5956、無音 D2 0.5227。どちらもデータ読み込み律速ではない。
- 008 の結論は exp004 を基準構成とするもので、承認待ち。MUSAN を学習と評価で共用している点と、spec への dev_noisy の追記を questions.md に記録した。学習中に別のセッションが exp/008 へ train.py の変更（エポックごとのチェックポイント保存）を入れ、exp004 はそのコミット f38cc9f で学習した。方式Bは保留し、第7段階には進んでいない。

### 2026-09-25 実装: 指示書タスク1（spec.md に dev_noisy と推論時間の測定規則を追記）
- spec.md に「評価指標」節を新設し、configs/eval/dev_noisy.yaml（SNR 5/10/15dB、固定残響、種 20260924、3条件のまとめ dev_noisy_all）と 007 の測定規則を値・手順を変えずに転記した。冒頭の評価指標の項から参照を付けた。
- 実装（noisy.py・augment.py・runner.py・latency.py）と記述の食い違いは無かった。pytest 552件通過。

### 2026-09-25 実装: 指示書タスク2（MUSAN noise の学習用・評価用の分割）
- data/musan/noise の930ファイルを区分ごとに8対2で分け configs/splits/musan_noise.json に固定した（種 20260925。学習用 free-sound 674・sound-bible 70、評価用 169・17）。生成は scripts/build_musan_noise_split.py。
- 雑音重畳・無音サンプルは学習用、dev_noisy・D2 は評価用だけを読む。分割の指定が無ければ開始前に停止する。dev_noisy.yaml に musan_noise_split を追加し、spec.md に反映した。pytest 566件通過。
- configs/exp00*.yaml は書き換えていないため、過去の設定のまま学習・再開すると停止する。exp004 の dev_noisy は再評価していない。

### 2026-09-25 判断: 指示書タスク3（強制アライメント手段の選定）
- 採用: ひらがなCTC（vumichien/wav2vec2-large-xlsr-japanese-hiragana、Apache-2.0、rev 017225b）＋ torchaudio.functional.forced_align。次点 MMS_FA（CC-BY-NC、ローマ字変換で dev 63件が変換不能）。Julius・MFA は uv で導入できず除外。transformers 5.17.0 を依存に追加。
- dev 20件（話速帯ごと5件）で両候補とも20/20成功、モーラ数はラベルと全件一致。1件約0.107秒（読み込み込み）、dev 29,518件で約53分。候補間の境界差は中央値20ms。
- 不一致・失敗は除外して理由別に記録する。出力はピーク状（9割が1フレーム）で、窓の計数はモーラ中点で行うことを推奨。詳細は docs/decisions/008-forced-alignment.md。

### 2026-09-26 実装: 指示書タスク4-1（強制アライメントの実装）
- src/spkrate/labels/alignment.py（モーラ分割・トークン化・forced_align・区間化・008の理由コードでの除外・JSONL書き出しと再開）と入口 scripts/align_dev.py を追加。tests/test_alignment.py 35件（実モデル1件は重み未取得時 skip）。
- dev 20件（種20260925）で20/20成功、1件0.082秒（読み込み込み、mps）。CPUフォールバック警告なし。uv run pytest 601件通過。dev 全件の実行（4-2）は未実施。

### 2026-09-26 測定: 指示書タスク4-2（dev へのアライメント付与と自動検査）
- dev 全件29,518件をアライメント（51分、0.104秒/件）。成功29,518・失敗0（全理由コード0）、モーラ数の不一致0。CPUフォールバックなし。出力 data/processed/alignments/dev.jsonl。
- 区間長は89.9%が1フレーム。モーラ間隔は中央値0.100秒、極端の閾値（対数の中央値±3σ_MAD）0.037/0.273秒。先頭・末尾の無音は中央値0.71/0.84秒。
- 連結試験100組: 単独結果基準の|ずれ|中央値8ms・90%点24ms。ただし5組で端のモーラが1秒以上離れて相手側に置かれた。詳細 results/alignment_dev.md。
- 聴取確認用30件（帯別7/8/8/7、種20260926）を results/alignment_check/ に出力（wav は除外）。人間の聴取（record.md 記入）待ちで停止。

### 2026-09-26 統括: 聴取記録のコミットと指示書タスク5-1（dev_window の構築）
- 人間が記入した record.md（30件とも良好、一括判定）と error_analysis_audio.md（37件とも発話が聞き取れない）、指示書の更新、listening/ の説明をコミットした。
- 5-1: configs/eval/dev_window.yaml と src/spkrate/eval/dev_window.py を追加。端のモーラが孤立した50件を除外し、窓は842,900（単一312,956・連続529,944、正解0の窓52,813）、連続発話は7,161組。pytest 620件通過。詳細は results/dev_window_build.md。
- 37件の無発話クリップはアライメントが ok になっているため、known_no_speech の印を付けた（窓1,607、除外はしていない）。2.0秒未満の1,626件は単一クリップの窓を持たず、questions.md に確認事項として記録した。
- 人間の追加指示（無発話の自動検出と、5-2で印付きの窓を除いた値を主指標にする）を指示書に追記した。

### 2026-09-26 測定: 追加指示1（dev の無発話クリップの自動検出）
- dev 全件でひらがなCTCの貪欲デコード（文なし）の文字誤り率と音量指標を計算した（2,920秒、CPUフォールバックなし）。規則は frame_db_p90 ≤ −25.5dBFS かつ frame_db_range ≤ 33.3dB かつ cer ≥ 0.4（configs/eval/no_speech.yaml）。
- 37件を全件検出し、該当は計419件（帯別 153/119/77/70）。dev_window で印の付く窓は15,433（1.83%）。cer の閾値は37件中の1件（#21、cer 0.405）だけで決まっている。
- 聴取用20件を listening/3_無発話疑い_20件/ に出力し、記録表 results/no_speech_check.md を用意した（人間の確認は5-2と並行）。pytest 641件通過。
- 人間の回答: 2.0秒未満のクリップは今のままとし、5-2に「短いクリップを含む連続窓の帯別集計」と「±50msずらしによる正解の不確かさ」を加える（questions.md）。

### 2026-09-26 測定: 指示書タスク5-2（dev_window での窓単位の評価）
- exp004 と包絡ベースラインを dev_window の clean と SNR 5/10/15dB で評価した（1.02時間、CPUフォールバックなし）。主指標は無発話の印の付いた15,433窓を除いた値。前任のサブエージェントは人間の誤操作で止まり、推論はそのまま続けて、集計は後任が行った。
- 窓単位の MAE は、exp004 が clean 0.8314、雑音下まとめ 1.3476。包絡は 1.2324 と 2.2280。exp004 の8以上の帯は clean 1.3237（偏り −1.2552）。正解0の窓での exp004 の出力の平均は0.84モーラ。除外で MAE は約0.01下がる。
- ±50ms ずらしによる正解の不確かさは0.19（毎秒モーラ数）。2.0秒未満のクリップを含む連続窓の8以上の帯は2.70（127窓）。詳細は results/window_eval.md、metrics.csv に10行を追記した。pytest 649件通過。

### 2026-09-26 判断: 指示書タスク5-3（窓単位の評価の判断、承認待ちで停止）
- docs/experiments/009-window-eval.md: B1 に該当する（exp004 の D1 = 0.4990 > 0.25）ため、方式Bへ移行し、7-2 の最優先の条件とする結論。B2〜B5 は非該当。窓単位の偏りは −0.576（クリップ単位は −0.087）。
- 窓単位の目標値の案（clean）: 4未満 0.70、4以上6未満 0.60、6以上8未満 0.60、8以上 0.90（別案 1.40）、全体 0.70。exp004 が満たすのは4未満だけ。無発話の窓を除く前の値で判定しても結論は変わらない。
- 承認、spec.md への窓単位評価の追加、目標値の選択、no_speech_check.md の聴取を questions.md に記録した。第6.5段階は完了し、人間の承認待ちで停止した。第7段階には進んでいない。

### 2026-09-26 実装: 指示書 2026-09-26 の0節とタスク1（spec.md への反映）
- 指示書をコミットし、plan.md の第7段階に参照を追記した。0節の回答を questions.md に転記し、無発話疑い20件の聴取の振り分け（判定2が19件、判定1が#11の1件）を results/no_speech_check.md に書き写した。
- spec.md に「窓単位の評価」節（dev_window の定義、主指標の除外規則、窓単位の目標値）と、前処理の節に train からの無発話疑いの除外を加えた（28a73a2、088b169）。pytest 649件通過。
- 方式Bの学習時の窓の正解の定義はタスク3の後に反映する。無音長の範囲の表記の差と補助の条件の対象を questions.md に記録した（停止はしない）。

### 2026-09-26 測定・実装: 指示書 2026-09-26 タスク2・タスク3・タスク1後半
- タスク2: train 236,143件にアライメントと無発話の指標を1回の推論で付けた（21,896秒、失敗0、CPUフォールバックなし）。除外は1,229件（先頭の孤立97・末尾の孤立54・無発話疑い1,078、排他）で、学習に使うのは234,914件（99.48%）。無発話疑いは0.46%（dev 1.42%）。results/train_alignment.md。
- タスク3: docs/decisions/009-method-b.md（矛盾なし、細部の決定）を書き、feat/method-b に方式Bを実装した（pytest 692件、途中の一覧でスモーク完走）。exp006 用に方式Aでも学習用の一覧で絞れるようにした（feat/method-a-selection、pytest 705件）。
- タスク1後半: 方式Bの窓と正解の定義を spec.md に加えた（b9c3bd2）。正式な一覧でのスモークとマージはこの後に行う。

### 2026-09-26 測定: 指示書 2026-09-26-rtx3060 タスク1
- ubuntu-desktop（xps）: Ubuntu 26.04.1、i7-13700（16コア24スレッド）、RAM は OS から約12.8GiB（指示書の16GBより少ない）、`/`（ext4）の空き432GB。RTX 3060 12GB、ドライバ 595.91.07、CUDA 13.2。
- python3 3.14.4、rsync 3.4.1 は有。uv・tmux・git は無（tmux・git の導入は sudo のパスワードが要る）。転送は300MiBで約62〜67MiB/s。
- 停止条件（nvidia-smi・ディスク・rsync）は非該当。git が無いとコミットハッシュが unknown になり test_git_commit_info_on_this_repo が失敗する点を記録した。docs/decisions/010-compute-environment.md（c60792c）。

### 2026-09-26 統括: 指示書 2026-09-26-rtx3060 タスク1の後（git・tmux の導入待ちで一時停止、導入後に再開）
- ubuntu-desktop に git と tmux が無く、導入に sudo が要る。人間が `sudo apt install git tmux` を行うと決まった。導入が済むまで停止する（questions.md に記録）。exp005 は Mac で学習を続けている。
- 追記: 人間が ubuntu-desktop に git（2.53.0）と tmux（3.6）を導入したことを ssh で確認し、010-compute-environment.md に反映した。
- 改訂版タスク1（0aed609）: sudo -n 成功、Secure Boot 有効（ドライバは動作済みで導入不要）。uv 0.12.19 をユーザー権限で、pyopenjtalk のビルド用に build-essential と cmake 4.2.3 を sudo apt で導入し、操作の一覧を 010 の5節に記録した。

### 2026-09-26 実装: 指示書 2026-09-26-rtx3060 タスク2（cuda への対応）
- worktree（feat/cuda-device）で実装し main にマージした（f7d147e）。spkrate.device で mps・cuda・cpu を指定でき、使えないデバイスは開始前に停止する。cuda では TF32 を無効にし、pin_memory を有効にする。
- 依存: Linux は cu130 の torch 2.14.0+cu130・torchaudio 2.11.0+cu130。macOS で解決される80件の版は変わらない（uv export で比較）。
- log.txt と config_snapshot.yaml に、ホスト名・GPU名・torch/CUDA の版を記録する。metrics.csv の末尾に host（呼び名 mac/ubuntu-desktop、Public のため生のホスト名は書かない）と device の列を足した（既存29行は空欄）。
- pytest（Mac）730 passed / 3 skipped（cuda が要る試験）。exp005 の作業ツリーは変えていない。exp005 の評価の追記は main を取り込んでから行う。

### 2026-09-27 測定: 指示書 2026-09-26-rtx3060 タスク3（ubuntu-desktop の環境構築と試験）
- scripts/sync_to_remote.sh・fetch_from_remote.sh を作り main にマージした（86a8929）。.git・作業ツリー・data（common_voice_ja・musan・processed）を送り、ファイル数とバイト数は3件とも一致した。
- 遠隔機で uv sync --frozen 成功（torch 2.14.0+cu130、RTX 3060 を認識、TF32 無効）。pytest 730 passed / 3 skipped（mps 2件・CTC 重み未取得1件。cuda の試験は通った）。
- 完走確認2件は cuda で完走した。1エポックは Mac の約24秒に対し約2秒。データ待ちの比率は 0.53〜0.67（num_workers 4 で 0.36〜0.58）。メモリの used は最大約5.2GiB。結果は 010 の4節、出力は runs/*_cuda* に戻した。

### 2026-09-27 実装: 指示書 2026-09-26-rtx3060 タスク4-1（文書の更新）
- CLAUDE.md の実行環境を2台（mac=mps、ubuntu-desktop=cuda）、精度の条件（float64・torch.compile・混合精度なし、cuda は TF32 無効）、正本と git の操作は Mac だけ（遠隔機は sync_to_remote.sh の checkout だけ例外）、推論時間は Mac、並列の方針に書き改めた。
- README.md の現在の実装を main の src/ に合わせて書き直し、開発環境に2台の構成・ubuntu-desktop の環境構築・sync/fetch スクリプトの使い方を書いた（呼び名だけを使い、ホスト名・IP は書かない）。
- docs/compute.md を新設（exp005 mac 実行中 2026-09-26 14:34 開始、cuda の完走確認2件）。data/DATASETS.md に遠隔機の配置と --data での送り方を追記した。exp/010-method-b に文書だけをコミットした。

### 2026-09-27 測定: 指示書 2026-09-26-rtx3060 タスク4-2（exp006 の起動）
- configs/exp006.yaml（exp/011-method-a-control の f70044e。exp005 との違いは方式A・全件の dev 検証・bucketing・device=cuda）を作り、ubuntu-desktop の tmux で 2026-09-27 00:16 に起動した。
- エポック1: 検証 MAE 0.9510、685秒（データ待ちの比率 0.351）。CUDA allocator の OOM 警告が3回出たが、例外にはならず学習は続いている。RAM の available は最小約6GiB。

### 2026-09-27 統括: exp006 の中断（ubuntu-desktop の移設）
- 人間の指示で一時中断した。ubuntu-desktop を隣の部屋に移すため、exp006 をエポック3の途中（00:40）で Ctrl-C で止めた。checkpoint_last.pt はエポック2の終わり（検証 MAE 0.7863）で、移設の後に resume_from で再開する。exp005（Mac）は続けている。

### 2026-09-27 統括: exp006 の再開
- ubuntu-desktop の移設後、configs/exp006_resume.yaml（35b8773。experiment_id と resume_from だけが違う）で、2026-09-27 00:50 にエポック3から再開した。出力は runs/exp006_resume/。バッチの並び・拡張の系列は続けた場合と同じで、違うのはワーカーの種と最良値の測り直し（010-method-b.md の留保に書く）。

### 2026-09-27 測定: 指示書 2026-09-26 タスク4（exp005 の評価、mac・mps）
- dev_window の clean・主指標の MAE は 0.4755（exp004 0.8314）、雑音下まとめ 0.7001（1.3476）。窓単位の必達目標と補助の条件をすべて満たした。8以上の帯の偏りは −0.37（exp004 −1.26）。
- クリップ単位は dev 0.4170、dev_noisy まとめ 0.6067（クリップ全体を入力する既存の手順を方式Bにも当てはめた）。D1 0.101、D2 は無音 0.031・雑音 0.377、D3 0.020。CPU フォールバック0件。results/exp005_eval.md（53f72ba）。

### 2026-09-27 測定: 指示書 2026-09-26 タスク4（exp006 の評価は ubuntu-desktop・cuda、推論時間は mac）
- exp006（exp006_resume のエポック27）の dev_window の clean・主指標の MAE は 0.7468、雑音下まとめ 1.3869。クリップ単位は dev 0.4343、D1 0.4749（B1 に該当）。評価したコードは 9241a8c。results/exp006_eval.md（2c781bc）。
- 推論時間（同一セッション lat-20260927T071126-c8e62bc7）の中央値は exp004 1.744、exp005 1.749、exp006 1.739 ms。results/latency_010.md（840f7ba）。

### 2026-09-27 判断: 指示書 2026-09-26 タスク5（承認待ちで停止）
- docs/experiments/010-method-b.md: 以後の基準構成を exp005（方式B）とする結論。窓単位の必達目標をすべて満たすのは exp005 だけ（clean 0.4755、雑音下まとめ 0.7001。exp006 は 0.7468・1.3869、exp004 は 0.8314・1.3476）。
- 切り分け: 窓単位の clean の改善 −0.3559 のうち方式Bの差が −0.2713、データの差が −0.0846。8以上の偏りは −1.2552 → −0.9775 → −0.3655、正解0の窓の出力は 0.8409 → 0.6074 → 0.1738 モーラ。D1 は 0.1009 で B1 未満。
- 留保: exp005 は mps・exp006 は cuda、exp006 の中断と再開、exp004 の雑音の集合の違い、クリップ単位の dev_noisy は exp005 が +0.0054 悪い。承認と確認の事項を questions.md に記録した。7-1 には進んでいない。

### 2026-09-27 実装: 指示書 2026-09-27 0節3（spec.md への D1 の雑音下の定義の追加）
- spec.md「診断指標」の D1 の下に「D1 雑音下（補助）」を加えた（18cf405）。D1 と同じ1,000件のクリップ全体に dev_noisy の加工を掛け SNR 3条件で同じ手順で測る。閾値なし・B1〜B5 の判定に使わない。実装（9241a8c）との食い違いはなかった。

### 2026-09-27 実装: 指示書 2026-09-27 系統C（9-1）
- feat/record-eval（worktree SpeedMeter-wt-c）に、20文の選択 scripts/select_eval_sentences.py（JSUT basic5000、jsut-label kana_level0 で数えて20〜40モーラ、has_unconverted なし、pyopenjtalk と一致、種 20260927。候補3,081文から選び20〜39モーラ・平均27.9）と録音の道具 scripts/record_eval.py を作った。
- 道具は Enter で開始・終了、r で録り直し（上書き）、q で終了、16kHz・モノラルで data/eval_real/{mic}_{rate}_{番号}_{文ID}.wav に保存。依存に sounddevice を追加（uv.lock は追加のみ）。手順書 data/eval_scripts/README.md。pytest 765 passed / 3 skipped（cuda）。
- JSUT の文そのものは再配布の条件の確認まで追跡しない（questions.md に記録）。マージ後に main で select_eval_sentences.py を実行して文の一覧を作る必要がある。

### 2026-09-27 統括: 系統C の停止（人間の録音 9-2 を待つ）
- feat/record-eval を main にマージした（5d64e02）。main で select_eval_sentences.py を実行し、worktree と同じ20文（data/eval_scripts/、追跡外）を作った。手順書は data/eval_scripts/README.md。
- 系統C は人間の録音を待って停止する。r キーでの録り直しで notes.txt を置き換えたので、9-3 の「notes.txt に記録のあるファイルは除外する」は録り直し後のファイルだけを使う、と読み替える。系統A・B は続けている。

### 2026-09-27 測定: 指示書 2026-09-27 A-1（推論の計算機の差、exp007 の起動）
- exp005 の checkpoint_best.pt（md5 一致）を ubuntu-desktop の cuda で dev_window clean について推論した（f949ef5、570秒）。主指標の MAE は mps 0.475469、cuda 0.475469（4桁で同じ）。窓ごとの予測の差は |差| の平均 1.3e-6、最大 3.8e-5 モーラ毎秒。metrics.csv に 012-compute-diff-exp005-cuda を追記（e939a27）。出力は runs/exp005_cuda_check/。
- configs/exp007.yaml（exp/012-seed-variance の f949ef5。exp005 と seed 20260929・device cuda だけが違う）で、ubuntu-desktop の tmux で 14:27 に学習を起動した。

### 2026-09-27 実装: 指示書 2026-09-27 系統B（8-1〜8-3）
- exp005 を ONNX（fp32 2,005,372 B）と int8 動的量子化（529,968 B）に書き出した（runs/onnx_exp005/、再生成は `python -m spkrate.export.to_onnx`）。PyTorch との一致: fp32 は atol 1e-4+rtol 1e-5 で一致（dev_window 最大 7.6e-6 モーラ）、int8 は窓ごとの差の平均 0.092・最大 0.53 モーラ。
- dev_window clean・主指標の MAE: fp32 0.4755（PyTorch と同じ）、int8 0.4783。推論時間（ONNX Runtime CPU、中央値/平均/最大 ms）: fp32 1.92/2.08/7.93、int8 3.77/3.81/5.45（int8 は Mac の CPU では遅い）。
- 対数メルの参照値（合成サイン波2件とメルフィルタバンク）と照合テスト、spec.md への計算パラメータの転記を feat/onnx-export に置いた（pytest 746 passed）。実際のクリップの値はリポジトリ外に置き、入れてよいかを questions.md に記録した。結果は results/onnx_export.md。

### 2026-09-27 実装: ブラウザでの確認用ページ（ユーザーの指示）
- feat/web-demo に web/（録音 → 16kHz 再標本化 → JS の対数メル → onnxruntime-web 1.30.0 で model_fp32.onnx。spec の 2.0秒窓・0.25秒ずらしの窓ごとの出力と、確認用にクリップ全体1回の出力を、ナマの値とパースした値で表示）を置いた。
- JS の対数メルは参照値と tests/test_melspec_fixtures.py と同じ許容誤差で一致（最大 1.0e-3、値 >−5 の帯 9e-6）。JS の入力を ONNX Runtime に通した出力は Python の経路と最大 1.9e-6 モーラ差（pytest 786 passed）。実マイクでの動作はブラウザで人間が確かめる。
- 追加の指示で web/ をリアルタイム表示（録音中に直近2.0秒で0.25秒ごとに推論、間に合わない時点は間引く。現在値と直近30秒の折れ線、補助線は帯の境界4・6・8）に作り替え、参照入力の照合は web/check.html に移した（node:test 23件、pytest 786 passed）。
- 早口の閾値と無音・平滑化の見せ方は spec に無いため決めず、questions.md に記録した。

### 2026-09-27 測定: 指示書 2026-09-27 A-1（exp007 の完了と評価、揺れの記録）
- exp007 は 22:42 に上限30エポックで終了（最良エポック27、dev_window_val MAE 0.4779。exp005 は 26・0.4737）。dev_window 主指標 MAE は clean 0.4786（exp005 0.4755）、雑音下まとめ 0.6995（0.7001）。クリップ単位 dev 0.4255（0.4170）、D1 0.1031（0.1009）。metrics.csv に 012-* を追記（7d29558）。
- results/run_variance.md に推論の計算機の差（無視できる）と2回の学習の差（種と学習の計算機の差が混ざる）を表にした。全体の MAE の差は 0.001〜0.003、帯・偏り・正解0の出力・D2 はその数倍〜数十倍動く。configs/exp007.yaml は exp/012-seed-variance（f949ef5）にあり、main へのマージは統括に任せる。

### 2026-09-28 判断: 指示書 2026-09-27 A-2（探索の計画）
- docs/experiments/011-search-plan.md に6条件を計画した。精度: exp008 ポアソン損失、exp010 時間伸縮を対数一様、exp011 膨張率2倍。軽さ: exp009 周波数段のチャネル半分（mac）、exp012 時間段のチャネル半分、exp013 周波数段4分の1（exp009 が精度を保った場合だけ）。
- 採否は exp005・exp007 の範囲の外へ指標ごとの揺れの目安 s 以上出たときだけ差とみなし、目的の指標群（8以上の偏り・正解0の窓の裾）でそろうことを求める。軽さは clean MAE +0.010 等の許容幅と ONNX 1スレッド 10% 短縮で判定する。
- 所要は ubuntu で1条件約9時間15分（exp007 実測 8時間15分＋評価）、5条件で約46時間。コードの変更は exp010 の伸縮率の分布と計算量の記録の2つ。メル次元数は仕様の変更が要るため除き、questions.md に記録した。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp008 の起動（損失 mse → poisson）
- exp/013-poisson-loss に configs/exp000_poisson_smoke_cuda.yaml（1ec6475）と configs/exp008.yaml（3914f67）を置いた。ubuntu の完走確認（先頭8192件・1エポック・267バッチ）で損失は有限のまま下がった（full=False のため負の値になる）。
- exp008 を ubuntu-desktop（cuda）の tmux exp008 で 2026-09-28 00:14 に起動した（見込み約8時間15分）。docs/compute.md に記録。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp009（起動）
- configs/exp009.yaml と configs/model/cnn_freq16.yaml（exp005 から周波数段のチャネル数だけ [32, 64, 64] → [16, 32, 32]）を exp/014-freq-ch-half に置いた（9e9de29）。パラメータ数 334,305（exp005 498,881）、積和 102.4 百万（239.5）、受容野 69 フレームで 011 の見込みと一致。
- Mac の mps で 00:14 に学習を起動した（worktree SpeedMeter-wt-c から。出力は runs_dir を絶対パスにして元の runs/exp009）。

### 2026-09-28 実装: 指示書 2026-09-27 A-3（exp010 の伸縮率の分布と MACs）
- feat/stretch-dist-macs（e2be66e〜95f256b）: AugmentConfig.time_stretch_distribution（uniform / log_uniform、既定 uniform。乱数の消費は同じ1回、方式A・B両方）、WindowTrainDataset.window_label と scripts/epoch_window_label_distribution.py、cnn.estimate_macs（model_summary に macs_201_frames）。pytest 801 passed・5 skipped。
- exp005 の設定のエポック1の窓（455,286件から20,000件を抽出）の16モーラ以上の割合: uniform 20.8%、log_uniform 22.3%。MACs は 011 1.2節の見込みと一致（exp005 239.5・exp009 102.4・exp012 185.1・exp013 61.7 百万）。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp008 の完了と評価（損失 mse → poisson）
- exp008 は 08:29 に上限30エポックで終了（最良エポック30、dev_window_val MAE 0.4851。exp005 0.4737、exp007 0.4779）。26〜30 は単調でなく 011 の「未収束」には当たらない。dev_window 主指標 MAE は clean 0.4858、雑音下まとめ 0.7196、8以上の偏りは clean −0.4358・雑音下 −0.7416。
- 011 2.2節の値: G2・G3・G4（8以上）が基準を満たさず、P8 は悪化（clean は境目）、P0 は差なし、D1 は小さくなった。results/exp008_eval.md と metrics.csv（013-*）に記録した。採否は A-4 に任せる。設定は exp/013-poisson-loss（3914f67）。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp010 の起動（伸縮率の分布 一様 → 対数一様）
- exp/015-stretch-log-uniform に configs/exp010.yaml（54490e8）を置いた。exp005 との差は augment.params.time_stretch_distribution: log_uniform・device cuda・ID と notes だけ（キーは AugmentSettings の params の下に置く必要があった）。
- ubuntu-desktop（cuda）の tmux exp010 で 09:19 に起動した（見込み約8時間15分）。log.txt に「分布=log_uniform」を確認。docs/compute.md に記録。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp009（完了と評価）
- exp009 は 11:42 に上限30エポックで終了（最良エポック30、dev_window_val MAE 0.4900。exp005 は 26・0.4737）。dev_window 主指標 MAE は clean 0.4930、雑音下まとめ 0.7245、クリップ単位 dev 0.4447、D1 0.0925。ONNX fp32 1,347,066 B（exp005 2,005,372 B）。metrics.csv に 014-* を追記（d44e399）。
- results/exp009_eval.md に exp005・exp007 と並べ、011 2.3節の L1〜L7 の値を表にした。L2（clean 0.4930 > 0.4886）・L3（0.7245 > 0.7151）・L4 の4以上6未満（0.4900 > 0.4834）が上限を超え、他は以内。採否は A-4。推論時間は未測定。

### 2026-09-28 統括: A-3 の進め方の変更（exp013 を行わない、exp012 を mac で学習）
- exp009 は 011 2.3節の「精度を保つ」の L2・L3・L4 を満たさなかったため、011 の規則どおり exp013（周波数段 [8,16,16]）は行わない。
- Mac が空いたので、011 で ubuntu に割り当てた exp012（時間段 64）を mac（mps）で学習する（指示書 A-3 の「Mac で1条件を並行して学習してよい」）。ubuntu は exp010 の後に exp011 を行う。exp012 は exp005 と同じ計算機になり、差は条件と（exp007 との比較では）seed だけになる。
- exp009 の設定の runs_dir にあったユーザー名を含む絶対パスと、results/lr_schedule.md の絶対パスを、相対パスに直した（2428968、b356a51）。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp012 の起動（時間段のチャネル 128 → 64）
- configs/exp012.yaml と configs/model/cnn_tch64.yaml（exp005 から temporal_channels だけ [128×5] → [64×5]）を exp/017-temporal-ch-half に置いた（76a4b52）。パラメータ数 228,161（exp005 498,881）、積和 185.1 百万（239.5）、受容野 69 フレームで 011 の見込みと一致。
- Mac の mps で 12:32 に学習を起動した（worktree SpeedMeter-wt-c から nohup。出力は worktree の runs/exp012、runs_dir は runs のまま）。docs/compute.md に記録。

### 2026-09-28 実装: 確認用ページで複数のモデルを比較（ユーザーの指示）
- feat/web-demo の web/ を、models/models.json に並べた全モデル（setup_web_model.py --source で複数指定、8個まで）に同じ対数メルを通す形にした。グラフの上のチェックボックスで表示するモデルを切り替える（色は一覧の順に固定、値は平滑化なし）。
- ヘッドレス Chrome（疑似マイク）で exp005・exp009 の2本の表示と切り替え、check.html の照合 OK を確認（2モデル合計 約80〜140 ms、間引き0）。node:test 25件、tests/test_web_demo.py 5件通過。実マイクでの確認は人間が行う。

### 2026-09-28 測定: 指示書 2026-09-27 A-3 exp010 の完了と評価（伸縮率の分布 一様 → 対数一様）
- exp010 は 17:36 に上限30エポックで終了（最良エポック27、dev_window_val MAE 0.4744）。dev_window 主指標 MAE は clean 0.4767、雑音下まとめ 0.6922、8以上の偏りは clean −0.3789・雑音下 −0.7162。
- 011 2.2節の値: G1〜G6 はすべて満たす（G4 8以上・G6 4〜6 は上限に近い）。P8 は悪化（雑音下 1.64 s。clean は差なし）、P0 は差なし。results/exp010_eval.md と metrics.csv（015-*）に記録した。採否は A-4 に任せる。設定は exp/015-stretch-log-uniform（54490e8）。
