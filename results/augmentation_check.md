# 学習時のデータ拡張の適用状況（第6段階 追加診断）

- 対象: exp001（`runs/exp001/checkpoint_best.pt` を作った学習）。設定は `configs/exp001.yaml` と
  `runs/exp001/config_snapshot.yaml`
- 方法: コード・設定・学習ログの読み取りのみ。学習・推論は行っていない
- **本文書は事実のみを記す。原因の推測・改善案は書かない**
- 行番号は main の `8fa3b97` 時点

## 1. 結論（事実）

| 項目 | exp001 で実際に使われた値 | コード上の既定値（拡張を有効にした場合） |
| --- | --- | --- |
| 拡張全体 | **無効**（`augment.enabled: false`） | — |
| 音量変化の適用率 | **0**（経路上呼ばれない） | 0.5 |
| 音量変化の範囲 | **適用なし** | 利得 −12〜+6 dB（一様）。振幅が 0.99 を超える場合は利得を抑える |
| 雑音重畳の適用率 | **0**（経路上呼ばれない。雑音源も未設定） | 0.5（ただし `augment.musan_root` 未設定なら 0） |
| 雑音重畳の SNR 範囲 | **適用なし** | 0〜20 dB（一様）。SNR はクリップ全区間の実効値で定義 |

exp001 の学習では音量変化・雑音重畳を含むすべての拡張が一度も適用されていない。
学習ログ `runs/exp001/log.txt` 3行目にも
`データ: 学習=236143件（経路=features、拡張=なし） 検証=29518件（経路=features、拡張なし）` と記録されている。

`runs/` 以下の他の学習（`exp000_smoke`、`exp001_lrtrial/*` の5件）の `config_snapshot.yaml` もすべて
`augment.enabled: false` で、ログも「拡張=なし」である。拡張を有効にした学習の記録は存在しない。

## 2. 設定

| ファイル | 行 | 内容 |
| --- | ---: | --- |
| `configs/exp001.yaml` | 42 | `data.source: features`（事前計算済み特徴量を読む経路） |
| `configs/exp001.yaml` | 52–53 | `augment:` / `enabled: false`。`params` と `musan_root` の記載なし |
| `runs/exp001/config_snapshot.yaml` | 44–47 | `augment: {enabled: false, params: {}, musan_root: null}` |

## 3. コード経路

### 3.1 exp001 の経路で拡張が掛からないこと

| ファイル | 行 | 事実 |
| --- | ---: | --- |
| `src/spkrate/train/train.py` | 188–192 | `AugmentSettings.build()` は `enabled` が偽なら `None` を返す |
| `src/spkrate/train/train.py` | 285–289 | `train_source` は `data.source` が指定されていればそれを使う。exp001 は `features` |
| `src/spkrate/train/train.py` | 632–640 | `source == "features"` のとき学習データは `FeatureClipDataset`。拡張の引数を受け取らない |
| `src/spkrate/train/data.py` | 197–289 | `FeatureClipDataset.__getitem__`（275–289）は float16 保存値を読み正規化するだけで、`augment_waveform` / `augment_feature` を呼ばない |
| `src/spkrate/train/train.py` | 620–624 | 拡張ありで `features` 経路を選ぶと `ValueError`。exp001 は拡張なしなので該当しない |
| `src/spkrate/train/train.py` | 680–683 | `_build_noise_source` は `enabled` が偽、または `musan_root` が空なら `None` |

`augment_waveform` の呼び出し元は `src/spkrate/train/data.py` 398–402（`WaveformClipDataset.__getitem__`、
`self.augment is not None` のときのみ）だけで、この Dataset は `source == "waveform"` のとき
（`train.py` 642–660）にしか作られない。

### 3.2 拡張を有効にした場合に値が無効化・変化する経路（exp001 には未適用）

依頼にある「設定値があってもコード経路で無効になる」場合の有無を確認した結果。

| 種類 | ファイル | 行 | 事実 |
| --- | --- | ---: | --- |
| キー名不一致 | `src/spkrate/data/augment.py` | 586–600 | `AugmentConfig.from_mapping` は未知のキーを `ValueError` で拒否する。キー名の誤りは黙って無視されない |
| キー名不一致 | `src/spkrate/train/train.py` | 127–132 | `augment` 節（`enabled` / `params` / `musan_root`）も未知のキーを拒否する |
| 確率0 | `src/spkrate/data/augment.py` | 566, 576 | 既定の `noise_prob` = 0.5、`volume_prob` = 0.5（0 ではない）。`disabled()`（602–612）を使った場合のみ 0 |
| 雑音源なし | `src/spkrate/data/augment.py` | 682 | `noise_source is not None and rng.random() < noise_prob`。雑音源が無ければ `noise_prob` の値に関わらず雑音重畳は行われない |
| 雑音源なし | `src/spkrate/train/train.py` | 182, 682 | `musan_root` の既定は `None`。設定しないと雑音源は `None`（上の行により雑音重畳は常に不適用）。警告・例外は出ない |
| 無音への雑音 | `src/spkrate/data/augment.py` | 243–244 | `add_noise` は信号の電力が 1e-12 以下なら雑音を足さずに返す |
| 音量の上限 | `src/spkrate/data/augment.py` | 443–446 | `change_volume` は `peak × gain > 0.99` のとき利得を `0.99 / peak` に抑える。正の利得は指定値より小さくなりうる |
| 音声パス | `src/spkrate/train/data.py` / `train.py` | 396 / 157 | `WaveformClipDataset` は `audio_root / record.audio_path` で音声を開く。`audio_root` の既定は `data/common_voice_ja`、`clips.jsonl` の `audio_path` は `data/common_voice_ja/clips/…` なので、既定値のままでは `data/common_voice_ja/data/common_voice_ja/clips/…` となり、このパスは存在しない（`clips.jsonl` 1件目で確認）。拡張ありの経路は既定値のままでは音声の読み込みで失敗する |

適用順（`augment.py` 640–711）: 時間伸縮 → 残響 → 雑音重畳（682–687）→ 帯域制限 → 音量変化（703–707）。
各拡張の適用は独立に判定される。

## 4. D2 の SNR と学習時の SNR

| 項目 | D2（`results/window_diagnostics.md`、MUSAN noise で平均 4.7024 モーラ） | 学習時の雑音重畳 |
| --- | --- | --- |
| 入力 | MUSAN noise **のみ**の2.0秒窓500本（音声を含まない）。`scripts/window_diagnostics.py` 162–171 | 音声に MUSAN noise を重ねたもの（`augment.py` 682–687、228–249） |
| SNR | **定義されない**（信号成分が無い。音声に対する SNR は −∞ dB に相当） | 設定上 0〜20 dB。**exp001 では雑音重畳そのものが未適用** |
| 雑音の音量 | 元ファイルの振幅のまま（音量合わせなし）。実効値 平均 0.1249、中央値 0.0853、最大 0.8409 | 音声の実効値と SNR から倍率を決める |

**判定: 一致しない。** D2 の入力は SNR 0〜20 dB の範囲外（信号なし）であり、さらに exp001 の学習では
雑音重畳が一度も適用されていない。

参考（D2 の追加測定）: 同じ雑音を dev 音声の実効値の中央値 0.05749 に揃えた場合の平均は 4.6909 モーラ。
これも音声を含まない入力であり、SNR は定義されない。
