# 事前計算した特徴量の保存形式（docs/PLAN.md 第4段階 4-1）

## 決めたこと

| 項目 | 値 |
| --- | --- |
| 特徴量 | docs/spec.md の対数メルスペクトログラム（`src/spkrate/features/melspec.py`）。n_fft=400、hop_length=160、n_mels=80、f_min=0、f_max=8000、power=2.0、`log(mel + 1e-6)` |
| dtype | **float16**（計算は float32、保存の直前に float16 へ変換） |
| レイアウト | **シャード方式**。1シャード＝clips.jsonl の並び順で1000クリップ分を連結した1つの `.npy`。形は `(そのシャードの全フレーム数, 80)`、時間方向が第0軸 |
| 索引 | シャードごとの `shard_XXXX.json`（クリップごとの `offset` と `n_frames`）と、分割ごとの `manifest.json`（シャード一覧と集計） |
| 置き場所 | `data/processed/features/train/`、`data/processed/features/dev/`。`.gitignore` の対象でコミットされない |
| 生成 | `uv run python scripts/precompute_features.py run --splits train dev` |
| 正規化 | `configs/normalization.yaml`（train 分割のみから算出。メル次元ごとの80組を既定とし、全体で1組の値も併記） |

test 分割は docs/PLAN.md 第10段階まで使用禁止のため、**計算していない**。
スクリプトは `--splits test` を拒否する。

## 実測値

`scripts/precompute_features.py estimate --size 100`（train から等間隔に100件）と本実行の結果。

| 項目 | 実測 |
| --- | --- |
| 1クリップの平均フレーム数 | 465.0フレーム（平均音声長 4.644秒） |
| 1クリップあたりの容量 | float16 **72.7KiB** / float32 145.3KiB |
| 1クリップの計算時間 | 約6ミリ秒（1プロセス。mp3の復号と16kHzへの再標本化を含む） |
| train | 236,143クリップ / 109,860,234フレーム / **16.37GiB** / 失敗0件 / 1分32秒（8並列） |
| dev | 29,518クリップ / 13,373,969フレーム / **1.99GiB** / 失敗0件 / 11秒（8並列） |
| 合計 | 265,661クリップ / 123,234,203フレーム / **18.36GiB** |
| 空き容量 | 実行前 160GiB → 実行後 約152GiB |

無作為な読み出しは20シャード×50クリップで **0.12ミリ秒/件**（memmap のスライス＋float32化）。

## 選択の理由

### dtype に float16 を選んだ理由

- 容量が半分になる（36.8GiB → 18.4GiB）。学習中はページキャッシュに載る割合がそのぶん増え、
  実効的な読み出しが速くなる。空き容量（160GiB）には float32 でも収まるが、第4段階4-2の
  データ拡張や第5段階以降の再計算で同じ規模の生成物が増える余地を残したい
- 値域が float16 に収まる。実測の最小は `log(1e-6) = -13.8125`（無音）、最大は 9.52 であり、
  float16 の表現範囲（±65504）から遠い。`inf`/`NaN` は発生しない

**量子化の影響**。float16 の仮数は10ビットで、値の大きさが 8〜16 の範囲では刻みが
`2^-7 = 0.0078` である。対数メルの値はおおむね -13.8〜9.5 に分布するので、量子化誤差は
最大でも 0.004 程度（刻みの半分）になる。学習で使う正規化後の尺度に直すと、
メル次元ごとの標準偏差が 3.17〜5.65 なので **標準偏差の 0.1% 未満**であり、
話速の手掛かり（数十〜数百ミリ秒の時間変化）に対して意味のある量ではない。
mp3 の符号化誤差や再標本化の誤差のほうが桁違いに大きい。
なお学習時は float16 のまま計算せず、読み出し直後に float32 へ変換する
（CLAUDE.md「float64は使わない」。MPS で float16 のまま計算する話とは別である）。

### 1クリップ1ファイルではなくシャードにした理由

- 1クリップ1ファイルだと26万個の小さいファイルができ、`open`/`close` の回数が
  学習の1エポックあたり26万回になる。シャードなら memmap を237個（train）開くだけで済み、
  以後はスライスだけで取り出せる（実測 0.12ミリ秒/件）
- 全体を1本の巨大な memmap にする案もあるが、各クリップのフレーム数は**実際に音声を
  読むまで確定しない**（clips.jsonl の duration_sec は mp3 のメタ情報に由来し、復号後の
  標本数と一致する保証がない）。書き込み位置を事前に決められないため、並列実行と
  中断・再開が難しくなる。シャードなら1シャードが1つの独立した仕事になり、
  **完成したシャードだけを飛ばして再開**できる
- 1シャード1000クリップで約70MiB。ワーカーは一度メモリに載せてから連結して書くが、
  8並列でも1GiB程度で収まる

### 索引をシャードごとの json にした理由

`manifest.json` だけに全クリップを並べると26万要素の json になり、読み出し側が
必ず全体を読む必要が出る。シャードごとに分ければ、そのシャードを触るときだけ
1000件分を読めばよい。`manifest.json` はシャード一覧と集計だけを持つ。

## 読み出しの例

```python
import json
from pathlib import Path

import numpy as np

features_dir = Path("data/processed/features/train")
manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))

# 1. シャードを memmap で開き、索引を読む
shard = manifest["shards"][0]
array = np.load(features_dir / shard["array"], mmap_mode="r")   # (フレーム数, 80) float16
index = json.loads((features_dir / shard["index"]).read_text(encoding="utf-8"))

# 2. クリップの特徴量は連続した行なのでスライスで取り出す
entry = index["clips"][0]
feature = np.asarray(
    array[entry["offset"] : entry["offset"] + entry["n_frames"]], dtype=np.float32
)                                                                # (n_frames, 80) float32
mora = entry["mora"]                                             # 正解のモーラ数

# 3. 正規化（configs/normalization.yaml。発話ごとの正規化はしない）
import yaml

norm = yaml.safe_load(Path("configs/normalization.yaml").read_text(encoding="utf-8"))
mean = np.asarray(norm["per_mel"]["mean"], dtype=np.float32)     # mode: per_mel
std = np.asarray(norm["per_mel"]["std"], dtype=np.float32)
normalized = (feature - mean) / std
```

`shard_XXXX.json` の1件は次の形である。

```json
{"clip_id": "common_voice_ja_39013598", "client_id": "...", "offset": 1234,
 "n_frames": 271, "mora": 21, "duration_sec": 2.7, "audio_duration_sec": 2.71}
```

`duration_sec` は clips.jsonl の値、`audio_duration_sec` は実際に復号した波形の長さである。

## 正規化の値をメル次元ごとにした理由

docs/spec.md は「学習データ全体から算出した平均と標準偏差を固定値として使う。
発話ごとの正規化は行わない」とだけ定めており、メル次元ごとに80組にするか全体で1組にするかは
一意に決まらない。**既定はメル次元ごと（`mode: per_mel`）**とした。

- 対数メルは低域と高域で平均が大きく違う（実測でメル次元ごとの平均は -11.18〜-3.57）。
  全体で1組にすると、最初の畳み込み層に入る時点で次元ごとの偏りが残る
- 80組でもブラウザ側（第8段階8-2）に持たせる値は80×2個で、実装上の負担は無い

全体で1組の値（`global: mean=-7.287698, std=4.650217`）も同じ yaml に書いてあり、
実験で切り替えられる。どちらも **train 分割のみ**から算出しており dev は含まない。

## 再現と再開

- シャードの区切りは clips.jsonl の並び順と `--shard-size`（既定1000）だけで決まるので、
  同じ入力なら常に同じ区切りになる
- 書き込みは一時ファイル → `os.replace` の順で行うため、途中で落ちても壊れたシャードは残らない
- 再実行すると、`shard_XXXX.npy` と `complete: true` の `shard_XXXX.json` が揃ったシャードを飛ばす
- 読めなかったクリップは索引の `failures` と `data/processed/features/failures.jsonl` に
  clip_id と理由を記録して飛ばす（今回の実行では0件だったのでファイルは作られていない）

## 保留

- 第4段階4-2のデータ拡張（時間伸縮・雑音重畳・残響など）は波形に対して行うため、
  ここで保存した特徴量をそのまま使うことはできない。拡張を使う学習では波形から
  その場で計算するか、拡張済みの特徴量を別に作る必要がある。どちらにするかは4-2以降で決める
- 第4段階4-3で方式B（2秒窓の切り出し）を採る場合も、保存済みの特徴量は
  `offset` からの行スライスで切り出せるので、この形式のまま使える
