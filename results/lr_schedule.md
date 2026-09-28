# exp003 学習率・正則化の設定確認

調査時点: exp003 は学習継続中。`runs/exp003/metrics.jsonl` に記録済みのエポック7〜29（最新到達エポック）の範囲で確認した。runs/exp003 配下・configs/exp003.yaml は変更していない（読み取りのみ）。

## 1. 最適化器（Optimizer）

- exp003の設定値（`configs/exp003.yaml` 55行目）: `train.learning_rate: 0.0001`
- 実装コード上の対応クラス: `src/spkrate/train/train.py` 1491〜1495行

  ```python
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=config.train.learning_rate,
      weight_decay=config.train.weight_decay,
  )
  ```

  `torch.optim.Adam` を使用しており、`lr` に `config.train.learning_rate`（= 0.0001）、`weight_decay` に `config.train.weight_decay`（= 0.0）を渡している。

- `TrainSettings`（train.py 336〜348行）のフィールド既定値: `learning_rate: float = 1e-3`、`weight_decay: float = 0.0`、`grad_clip: float = 5.0`。docstring（325〜326行）に「learning_rate / weight_decay: Adam の設定」「grad_clip: 勾配のノルムの上限。0以下で無効」とある。exp003は `learning_rate` をデフォルト1e-3から0.0001へ明示的に上書きしている（`weight_decay`・`grad_clip` はデフォルト値と同じ値を明示的に書いている）。

## 2. 学習率スケジュールの有無

- コード全体（`src/spkrate/train/train.py`、`configs/exp003.yaml`、`configs/exp001.yaml`、`configs/model/cnn_base.yaml`、`docs/spec.md`）を `scheduler|lr_scheduler|StepLR|CosineAnnealing|ReduceLROnPlateau|OneCycle|ExponentialLR` で検索した結果、該当なし。`torch.optim.lr_scheduler` の呼び出し箇所はコード中に存在しない。
- `runs/exp003/log.txt` 起動時ログ（再開処理の一部）に明記されている一文（学習開始直後）:

  ```
  2026-09-24 18:57:34,861 INFO 再開: 学習率スケジューラは使っていないため、読み込む状態は無い
  ```

- 結論: 学習率のスケジュール（減衰）は設定されていない。学習率は `train.learning_rate` の固定値をエポックを通じて一定で使う実装になっている。

## 3. 実際の学習率の推移（exp003, metrics.jsonl）

`runs/exp003/metrics.jsonl` の `learning_rate` フィールド（記録済み全エポック、epoch 7〜29。epoch 30は本調査時点でまだ学習ループの途中でmetrics.jsonlに未記載）:

| epoch | learning_rate |
| --- | --- |
| 7〜29（全23エポック） | 0.0001 |

全エポックで `learning_rate = 0.0001` の一定値。値の変化はない。

なお `metrics.jsonl` の `learning_rate` 列は、実装上は毎エポック `config.train.learning_rate`（設定ファイルの値）をそのまま書き出している（train.py 1553行 `"learning_rate": config.train.learning_rate,`）。オプティマイザ内部のパラメータ群の実際の `lr` を毎エポック読み出しているわけではないが、後述の再開時の上書き処理により、実際にオプティマイザに設定される `lr` も同じ0.0001に揃えられている。

### resume_from（exp001からの継続）が学習率に与える影響

- `configs/exp003.yaml` 30行目: `resume_from: runs/exp001/checkpoint_last.pt`（エポック6・optimizer_stateあり）
- 再開処理の該当コード、`src/spkrate/train/train.py` `_load_optimizer_state` 関数（1303〜1322行）:

  ```python
  def _load_optimizer_state(
      optimizer: torch.optim.Optimizer,
      state: dict[str, Any],
      settings: TrainSettings,
      logger: logging.Logger,
  ) -> None:
      """最適化器の状態を読み込み、学習率と重み減衰は設定の値で上書きする。"""
      optimizer.load_state_dict(state)
      for group in optimizer.param_groups:
          for key, value in (
              ("lr", settings.learning_rate),
              ("weight_decay", settings.weight_decay),
          ):
              if not math.isclose(float(group.get(key, value)), float(value)):
                  logger.warning(...)
              group[key] = value
  ```

  再開時は `optimizer.load_state_dict(state)` でexp001終了時点のAdamの内部状態（モーメント推定量 `exp_avg` / `exp_avg_sq` 等を含む）を読み込むが、その直後に `lr` と `weight_decay` は読み込んだ状態の値ではなく、**exp003の設定ファイルの値（0.0001／0.0）で無条件に上書きする**。exp001の `learning_rate` も0.0001（同一値）なので（`configs/exp001.yaml` 該当行）、今回のケースでは上書きの前後で数値上の変化はない。
- `runs/exp003/log.txt` 起動ログにも対応する記述がある:

  ```
  2026-09-24 18:57:34,861 INFO 再開: 再開元=runs/exp001/checkpoint_last.pt 再開元のエポック=6 → エポック7から通算30まで学習する。理由: runs/exp001/checkpoint_last.pt に optimizer_state があるため、モデルと最適化器の状態を読み込んで再開する
  ```

- まとめ: resume_from によりAdamの1次・2次モーメント推定量（勾配の履歴に基づく内部統計）はexp001から引き継がれるが、学習率（`lr`）そのものはexp003の設定値で明示的に固定・上書きされる仕組みになっており、スケジューラも存在しないため、エポック7以降も学習率は0.0001のまま変化しない。

## 4. 正則化に関する設定一覧

| 項目 | 値 | 出典 |
| --- | --- | --- |
| weight_decay（重み減衰、Adam） | 0.0 | `configs/exp003.yaml` 56行目、`runs/exp003/config_snapshot.yaml` `config.train.weight_decay: 0.0` |
| grad_clip（勾配ノルムの上限） | 5.0 | `configs/exp003.yaml` 57行目、`runs/exp003/config_snapshot.yaml` `config.train.grad_clip: 5.0`。実装: `src/spkrate/train/train.py` 1058〜1059行 `if settings.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)` |
| dropout（モデル各層後） | 0.0 | `configs/model/cnn_base.yaml` `model.dropout: 0.0`、`runs/exp003/config_snapshot.yaml` `model.dropout: 0.0`。実装: `src/spkrate/models/cnn.py` 125行 `dropout: float = 0.0`（`CnnConfig`のデフォルト）、395〜396行・420〜421行で `if cfg.dropout > 0.0: layers.append(nn.Dropout(cfg.dropout))`（値が0なので周波数方向ブロック・時間方向ブロックいずれにも `nn.Dropout` 層は追加されていない） |
| norm（正規化層） | none | `configs/model/cnn_base.yaml` `model.norm: none`。実装: `src/spkrate/models/cnn.py` `_norm` 関数（322行〜）。`none` の場合は正規化層（`ChannelLayerNorm`）を挿入しない |
| TrainSettingsのデフォルト値（参考、exp003では明示指定で上書き済みの項目を含む） | `learning_rate=1e-3`、`weight_decay=0.0`、`grad_clip=5.0` | `src/spkrate/train/train.py` 343〜345行 |

正則化に相当する他の設定（データ拡張・早期終了など）は本タスクの対象外（データ拡張はexp003で `augment.enabled: false`）。

## 参照ファイル

- `configs/exp003.yaml`
- `configs/model/cnn_base.yaml`
- `src/spkrate/train/train.py`
- `src/spkrate/models/cnn.py`
- `runs/exp003/log.txt`
- `runs/exp003/metrics.jsonl`
- `runs/exp003/config_snapshot.yaml`
