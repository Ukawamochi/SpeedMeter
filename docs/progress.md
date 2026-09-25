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
