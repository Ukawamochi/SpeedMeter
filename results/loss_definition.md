# train_loss と val_loss の定義（exp003, 事実のみ）

調査時点のコミット: `320580a`（`exp003（無音サンプル有効・拡張なし・exp001から再開・早期終了）の設定を追加する`）。
対象: `runs/exp003/metrics.jsonl`（読み取りのみ、学習は実行中のため変更していない）、`configs/exp003.yaml`（`train.loss: mse`）、
`src/spkrate/train/train.py`、`src/spkrate/train/data.py`。過学習かどうかの判断はここには書かない。

## 1. train_loss（学習損失）の計算方法

- 損失関数は `configs/exp003.yaml` の `train.loss: mse` により `build_loss("mse")` が返す `_mse_loss`（`src/spkrate/train/train.py:506-519`）。
  ```python
  # src/spkrate/train/train.py:472-474
  def _mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
      """二乗誤差（クリップごとのモーラ数に対する平均）。"""
      return torch.nn.functional.mse_loss(prediction, target)
  ```
  `torch.nn.functional.mse_loss` は `reduction` を省略しており、PyTorchの既定値は `"mean"`（全要素の単純平均）。
- 対象量: モジュール docstring（`src/spkrate/train/train.py:3-5`）に「1件の入力はクリップ全体、1件の正解はそのクリップの**モーラ数**である。毎秒モーラ数への換算は評価のときだけ行い」とある。実際に `prediction`（モデル出力）と比較される `target` は `batch.moras`（`src/spkrate/train/train.py:1055`）で、`data.py:154` の定義は
  ```python
  moras: torch.Tensor  # (バッチ,) float32  正解モーラ数
  ```
  すなわち **クリップ単位の総モーラ数**（毎秒モーラ数ではない）。窓（2.0秒）単位ではなく、可変長のクリップ全体が入力・正解の単位（`docs/spec.md`「診断指標」節の「窓長2.0秒…」は推論経路D1〜D3の話であり、学習は方式A゠クリップ全体）。
- バッチ内の集約: `torch.nn.functional.mse_loss(prediction, target)` は形状 `(バッチ,)` の1次元テンソル同士の要素ごと二乗誤差をバッチ内で**平均**（`reduction="mean"`）。対象要素数はそのバッチのクリップ数。
- エポック内の集計（`src/spkrate/train/train.py:1054-1065, 1083`）:
  ```python
  prediction = model(batch.features, batch.lengths)
  loss = loss_fn(prediction, batch.moras)
  ...
  count = len(batch)
  total_loss += float(loss.detach().cpu()) * count
  total_clips += count
  ...
  return {
      "train_loss": total_loss / total_clips,
      ...
  }
  ```
  バッチ損失（バッチ内平均）に `count`（そのバッチのクリップ数）を掛けて `total_loss` に加算し、全バッチ終了後に累積クリップ数 `total_clips` で割る。これは「バッチ損失の単純平均」ではなく、**エポック中に処理した全クリップにわたる二乗誤差の総和 ÷ 全クリップ数**（クリップ単位で等重みの平均）に一致する。
- `total_clips`（=`train_clips`）には学習データ（`configs/splits/train.json`、236,143件）に加えて `silence_samples`（無音・雑音のみのサンプル、`enabled: true`）が含まれる。metrics.jsonl のエポック16の行で `train_clips=243227`、`train_silence_clips=7084` であり、243227 − 7084 = 236143 と学習split件数に一致する。無音サンプルの正解モーラ数は0（`docs/spec.md`「学習データ」節）。

## 2. val_loss（検証損失）の計算方法

`evaluate_dev` 関数（`src/spkrate/train/train.py:1100-1153`）:
```python
for batch in loader:
    batch = batch.to(device)
    ...
    prediction = model(batch.features, batch.lengths)
    ...
    loss = loss_fn(prediction, batch.moras)
    total_loss += float(loss.detach().cpu()) * len(batch)
    total_clips += len(batch)
    ...
extras = {
    "val_loss": total_loss / total_clips if total_clips else float("nan"),
    ...
}
```
- 損失関数は学習と同じ `loss_fn`（`build_loss(config.train.loss)` で1回だけ生成され、`_train_one_epoch` と `evaluate_dev` の両方に渡される。呼び出し箇所: `src/spkrate/train/train.py:1490, 1523, 1542`）。すなわち `_mse_loss`（`reduction="mean"`）で同一の関数オブジェクト。
- 対象量も学習と同じ `batch.moras`（クリップ単位の総モーラ数）。`model(batch.features, batch.lengths)` の出力（クリップ単位の推定モーラ数）と比較。毎秒モーラ数への換算はここでは行わず、`compute_metrics` 呼び出し（`src/spkrate/train/train.py:1145`）でのみ行う。
- バッチ内集約はバッチ損失（`mse_loss` の `reduction="mean"`＝バッチ内平均）に `len(batch)`（そのバッチのクリップ数）を掛けて加算し、最後に `total_clips`（=`val_clips`）で割る。学習と同じ「クリップ単位で等重みの平均」。
- `configs/exp003.yaml` により検証セットは `configs/splits/dev.json`（`dev_split`）、`max_dev_clips: null` で全29,518件を毎エポック評価。dev には `silence_samples` を追加しない（`docs/spec.md`「学習データ」節「検証（dev）・テストには追加しない」）。metrics.jsonl のエポック16で `val_clips` に対応する `total_clips=29518`（`data.dev_split` の件数と一致）。

## 3. 両者が同一の定義かどうか

**損失の計算式自体は同一**である。
- 同じ損失関数オブジェクト `_mse_loss`（`torch.nn.functional.mse_loss`、`reduction="mean"`）を学習・検証の両方に使う（`loss_fn` は1回だけ生成され使い回される）。
- 比較対象は両方とも「クリップ単位の総モーラ数」（`batch.moras`）であり、窓単位でも毎秒モーラ数でもない。毎秒への換算は損失計算には入らない。
- バッチ内の集約（平均）とエポック全体での集約（クリップ数で重み付けした平均＝実質は処理した全クリップにわたる単純なMSE）の方法も、学習ループとevaluate_devで同一のコードパターン（`loss * count` を加算し、最後に `total_clips` で割る）。

## 4. 定義上の相違点

損失の計算式そのものに相違はない。ただし、集計対象となる**データの構成**には以下の相違があり、これは損失の定義（数式・集約方法）ではなくデータの違いとして事実のみ記載する。
- 学習側の `total_clips` にはクリップ単位ラベルのモーラ数が0の `silence_samples`（無音・MUSAN雑音・極小音量雑音）が含まれる。エポック16で `train_clips=243227` のうち `train_silence_clips=7084`（約2.91%）。検証側の `val_clips=29518` にはこの種のサンプルは含まれない（`docs/spec.md`「検証（dev）・テストには追加しない」、`configs/exp003.yaml` の `data.dev_split` は `silence_samples` を通らない経路）。
- `configs/exp003.yaml` は `augment.enabled: false` であり、学習・検証とも拡張は掛からない（この点は両者で相違なし）。
- `_train_one_epoch` はモデルを `model.train()`、`evaluate_dev` は `@torch.no_grad()` かつモデルは学習ループ内で設定された状態のまま呼ばれる（`train.py:1116` で `model.eval()` を呼んでいる）。モデル（`src/spkrate/models/cnn.py`）には `nn.Dropout` 層が含まれる（`cnn.py:396, 421`）ため、学習時は `train()` によってDropoutが有効、検証時は `eval()` によってDropoutが無効という、損失の数式とは別の相違点が実装上存在する。

## 5. metrics.jsonl の値と、定義差でどの程度説明できるか

エポック16の行（`runs/exp003/metrics.jsonl`）:
```
train_loss = 4.933896236440812   (train_clips = 243227, うち train_silence_clips = 7084)
val_loss   = 11.535704153018138  (val_clips   = 29518)
```
比率: val_loss / train_loss ≈ 11.5357 / 4.9339 ≈ **2.338倍**。

3節の通り、学習損失と検証損失は同一の損失関数（`_mse_loss`、`reduction="mean"`）を同一の対象量（クリップ単位の総モーラ数、単位はモーラの二乗）に対して同一の集約方法（クリップ数で重み付けした平均）で計算しており、コード上に**集計単位・対象量のスケール・正規化のいずれについても定義上の相違は見つからなかった**。定義差からくる乗数は1倍（＝差なし）であるため、上記の約2.338倍という比率のうち、損失の定義差で数値的に説明できる部分は0倍分（0%）であり、約2.338倍の差は定義差だけでは説明できない。

4節に記載した「学習側にのみ含まれる正解モーラ数0のsilence_samples（約2.91%）」「Dropoutの有効・無効の違い」は損失の定義（数式・集約方法）ではなくデータ構成・モデル状態の相違であり、これらが数値差にどの程度寄与するかはこの調査の範囲（実装コードの記述の確認）では計算していない。すなわち、両者の差が損失の定義差だけでは説明しきれない部分が残ることは事実として確認できるが、その残差の原因については本タスクの範囲外であり、ここでは判断しない。
