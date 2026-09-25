# 強制アライメント手段の選定（docs/directives/2026-09-25.md タスク3）

- 判定日: 2026-09-25
- 読んだ資料: `CLAUDE.md`、`docs/spec.md`（モーラの数え方）、`docs/decisions/005-window-strategy.md`、`src/spkrate/labels/mora.py`、`src/spkrate/eval/audio.py`、`src/spkrate/data/splits.py`
- 試行: `scripts/trial_forced_alignment.py`（dev から20件）。出力は `data/processed/alignment_trial/`（Git管理外）
- `configs/splits/test.json` は使っていない。`docs/spec.md` は変更していない。音声を聴く判断はしていない

---

## 0. 結論

| 項目 | 決定 |
| --- | --- |
| 採用する手段 | **候補A: ひらがなCTCモデル `vumichien/wav2vec2-large-xlsr-japanese-hiragana` の出力に `torchaudio.functional.forced_align` を掛ける** |
| 追加した依存 | `transformers>=5.17.0`（uv.lock では 5.17.0）。Python 以外の実行ファイルは不要 |
| 推論デバイス | 音響モデルは mps。`forced_align` は CPU で実行する（対応デバイスは CPU・CUDA のみ。放出確率を CPU に移してから呼ぶ） |
| モーラへの変換 | かな1文字＝語彙の1トークン。`count_mora_from_kana` と同じ結合規則でモーラにまとめ、モーラの先頭トークンの開始から末尾トークンの終了までをそのモーラの区間とする |
| dev 全件の見積もり | 1件あたり約0.107秒（読み込み0.010秒＋アライメント0.097秒）× 29,518件 ≒ **約53分**（6時間未満なので、タスク4-2は dev 全件を対象にできる） |
| 次点 | 候補B: `torchaudio.pipelines.MMS_FA`（依存追加なしで動くが、ライセンスが非商用であることと、ローマ字経由の変換に規則の穴が残ることから次点） |

---

## 1. 比較した候補

| 観点 | A: ひらがなCTC＋forced_align | B: MMS_FA（torchaudio） | C: Julius 音素セグメンテーションキット | D: Montreal Forced Aligner | E: faster-whisper の単語時刻 |
| --- | --- | --- | --- | --- | --- |
| uv で導入 | 可（transformers を追加） | 可（既存の torchaudio のみ） | 不可（Julius・Perl の実行ファイルが必要） | 不可（conda 前提、Kaldi の実行ファイル） | 可（既存） |
| macOS arm64 | 動作確認済み（mps） | 動作確認済み（mps） | 自前ビルドが必要 | conda-forge で可 | 可 |
| 出力単位 | ひらがな1文字 | ローマ字1文字（a-z と `'`） | 音素 | 音素 | 単語（サブワード） |
| モーラへの変換 | かなと1対1。dev 全件で変換不能0件 | かな→ローマ字の表が自前。dev 全件で63件（0.21%）が表で変換できない | 音素→モーラの対応づけが必要 | 同左、日本語辞書・モデルの用意が必要 | 認識結果に依存し、正解の読みに強制できない |
| 重みのライセンス | Apache-2.0 | CC-BY-NC 4.0 | — | — | — |
| 判定 | **採用** | 次点 | 条件（uv・Python のみ）を満たさない | 条件を満たさない | 強制アライメントではない |

C・D・E は試行していない。C と D は「uv で導入でき pyproject.toml に記述できる」を満たさず、E は正解の読みに時刻を付ける手段ではない（認識誤りがそのまま境界の誤りになる）。

### 1.1 API の非推奨・廃止の確認（pyproject.toml・uv.lock の版で実測）

- 版: torch 2.14.0、torchaudio 2.11.0、transformers 5.17.0、Python 3.12.13、arm64
- `torchaudio.functional.forced_align` は存在し、ソース（`torchaudio/functional/_alignment.py`）に非推奨の記述・警告は無い。torchaudio パッケージ内の「deprecat」を含む記述は `return_complex`・`onesided`・再標本化方式名の3件のみで、アライメント・pipelines とは無関係
- `torchaudio.pipelines.MMS_FA`（`Wav2Vec2FABundle`）も存在し、非推奨の記述は無い
- `-W default` で試行を実行し、torch・torchaudio・transformers からの非推奨警告は出なかった（出た警告は HF Hub の未認証アクセスの通知のみ）
- 音声の読み込みは既存の `spkrate.eval.audio.load_audio`（soundfile＋librosa）を使い、`torchaudio.load` は使わない
- transformers は `Wav2Vec2ForCTC.from_pretrained` のみを使う
- mps で CPU フォールバックの警告は出なかった

### 1.2 学習済みモデルの重み

| 候補 | 取得元 | 版 | ライセンス | 備考 |
| --- | --- | --- | --- | --- |
| A | Hugging Face `vumichien/wav2vec2-large-xlsr-japanese-hiragana`（`model.safetensors`、`vocab.json`） | コミット `017225bb128a6b1c6de9d58391f891908d69bc7b` | Apache-2.0（モデルカードの `license`） | facebook/wav2vec2-large-xlsr-53 を Common Voice 日本語と JSUT で微調整したもの。語彙はひらがな・「ー」・小書き文字など86（blank は `[PAD]`=85）。キャッシュは `~/.cache/huggingface/hub/` |
| B | `https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt`（torchaudio の `MMS_FA.get_model()` が取得） | torchaudio 2.11.0 に固定の URL | CC-BY-NC 4.0（torchaudio の docstring が fairseq の MMS のライセンスを参照） | キャッシュは `~/.cache/torch/hub/checkpoints/model.pt`（1.26GB） |

タスク4では A のコミットを `from_pretrained(..., revision="017225bb128a6b1c6de9d58391f891908d69bc7b")` で固定する。

---

## 2. 試行（dev 20件）

- 抽出: dev（`configs/splits/dev.json` の話者に属する `data/processed/clips.jsonl` の29,518件）から、話速帯（4未満・4-6・6-8・8以上）ごとに5件、種 20260925
- 手順: 16kHz モノラルに読み込み → 各候補で放出確率 → `forced_align` → モーラ区間。MPS の初回カーネル生成はウォームアップで計時から除いた。計時はアライメント部分（放出確率＋forced_align＋区間化）で、音声の読み込みは別に測った

| 項目 | A: ひらがなCTC | B: MMS_FA |
| --- | ---: | ---: |
| 成功件数 | 20 / 20 | 20 / 20 |
| アライメント後のモーラ数＝ラベル（`mora`） | 20 / 20 | 20 / 20 |
| 1件あたり時間（アライメント） | 0.097 秒 | 0.070 秒 |
| 実時間比 | 0.021 | 0.015 |
| 音声の読み込み（共通） | 0.010 秒 | 0.010 秒 |
| dev 全件の見積もり（読み込み込み） | 約53分 | 約39分 |
| モーラ区間長 中央値 / 5%点 / 95%点 | 0.020 / 0.020 / 0.040 秒 | 0.060 / 0.020 / 0.100 秒 |
| 区間長が1フレーム（20ms）以下のモーラの割合 | 90.2% | 25.1% |
| 先頭の無音（最初のモーラ開始）中央値 | 0.64 秒 | 0.61 秒 |
| 末尾の無音（最後のモーラ終了から末尾）中央値 | 0.71 秒 | 0.66 秒 |

- 20件ともラベルの `mora`・`count_mora(sentence)`・かなのモーラ分割数の3つが一致した
- **2候補間のモーラ境界（開始・終了）の差**（1,202境界）: 中央値 20ms、90%点 60ms、平均 33ms。独立に学習された2つのモデルの境界が概ね1〜3フレームで一致しており、少なくともどちらかが大きく外れている兆候は無い（ただし聴取による確認ではない）
- dev 全件の読みについて、トークンへの変換可否を音声なしで調べた。A は変換不能0件。B は本試行の表で63件が変換不能（語末の「ッ」31件、「ッッ」15件、母音・ナ行の前の「ッ」、「ン」の後の「ー」など）

### 2.1 採用の理由

1. **モーラへの変換が規則だけで決まる**。A の出力単位はかな1文字で、spec のモーラ規則（小書き文字は直前と結合、ン・ッ・ーは各1モーラ）をそのまま当てられる。B はローマ字化の規則を自前で持つ必要があり、「ッ」「ー」を何のローマ字に当てるかに恣意的な判断が入り、その判断が境界位置を左右する
2. **ライセンス**。A の重みは Apache-2.0、B は CC-BY-NC 4.0
3. 処理時間はどちらも dev 全件で1時間未満で、差は採否に影響しない
4. B に対する A の不利点（transformers の追加、ピーク状の出力）は、4節の扱いで吸収できる

---

## 3. モーラ数の不一致・失敗の扱い

アライメントのモーラ列は `data/processed/clips.jsonl` の `kana` を `count_mora_from_kana` と同じ結合規則で分けて作る。したがって構成上モーラ数は `count_mora` と一致するが、次の検査を行い、**不一致・失敗のクリップは正解の窓を作らずに除外し、件数と理由を記録する**。ラベルやかなを手で直したり、アライメントに合わせてモーラ数を変えたりはしない。

| 検査 | 失敗時の理由コード |
| --- | --- |
| かなのモーラ分割数 ＝ `mora` 列 ＝ `count_mora(sentence)` | `mora_mismatch_label` |
| すべてのかなが語彙のトークンに変換できる（語彙に無い「ゎ」「ゕ」「ゖ」「ゐ」「ゑ」は「わ」「か」「け」「い」「え」に置き換える。それでも無ければ失敗） | `unknown_token` |
| `forced_align` が成功する（フレーム数 ≥ トークン数＋連続する同一トークンの数。満たさない短いクリップは失敗） | `align_failed` |
| アライメント結果から得たトークン区間の数 ＝ トークン数、かつモーラ区間の数 ＝ `mora` | `mora_mismatch_alignment` |

除外件数が dev の1%を超えた場合は、タスク4-2の記録（results/alignment_dev.md）で理由別に報告する。

---

## 4. 出力の定義（タスク4への申し送り）

- フレーム長は 20ms（wav2vec2 の出力フレーム。秒への換算は「波形長 ÷ フレーム数」を使う）
- モーラ区間: 開始＝そのモーラの先頭トークンが最初に出たフレームの開始、終了＝末尾トークンが最後に出たフレームの終了
- CTC の出力はピーク状で、A では9割のモーラの区間が1フレームである。**区間の長さはモーラの持続時間を表さない**。モーラの位置（時刻）としては使えるが、モーラ長の分布（タスク4-2）はこの性質を前提に解釈する必要がある
- 発話区間: 最初のモーラの開始〜最後のモーラの終了
- 窓の正解モーラ数の数え方（推奨。確定はタスク5-1）: モーラ区間の中点が窓 [t, t+2.0) に入るモーラを数える。区間の端点で数える方式はピーク状の出力に対して不安定なため避ける

---

## 5. 未解決点

- **学習データの重複**: A は Common Voice 日本語（版は不明）で微調整されており、本リポジトリの dev の音声を学習に含んでいる可能性がある。アライメントの品質が dev で楽観的になりうるが、dev の正解作りの用途では害は無い。ただし dev 以外（別コーパス）に同じ手段を使う場合は品質が下がる可能性がある
- **境界の精度は聴取で未確認**。候補間の一致（中央値20ms）は必要条件にすぎない。タスク4-2の人間の確認（30件）で判断する
- 語末の「ッ」や「ッッ」のように実際には発音されにくいモーラにも、アライメントは必ずどこかのフレームを割り当てる。その位置の妥当性は検査できていない
- transformers 5.17.0 は依存が大きい（typer などが入る）。重みの読み込み以外には使わない
