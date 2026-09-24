"""話速推定CNNの学習ループ（docs/PLAN.md 第5段階 5-2）。

docs/decisions/005-window-strategy.md の方式Aで学習する。1件の入力はクリップ全体、
1件の正解はそのクリップの**モーラ数**である。毎秒モーラ数への換算は評価のときだけ行い、
``spkrate.eval.metrics.compute_metrics`` に「モーラ数と区間長」を渡す。

## 実行

    uv run python -m spkrate.train.train --config configs/exp000_smoke.yaml

出力は ``runs/<実験ID>/`` に集約する。

| ファイル | 中身 |
| --- | --- |
| ``log.txt`` | 実行ログ（バックグラウンド実行を前提に、標準出力と同じ内容をファイルへ書く） |
| ``metrics.jsonl`` | エポックごとに1行。検証セットの全指標と学習損失 |
| ``config_snapshot.yaml`` | gitのコミットハッシュ、設定ファイルの内容、解決後の設定、モデルの要約、乱数シード |
| ``checkpoint_best.pt`` / ``checkpoint_last.pt`` | 最良と最新のチェックポイント |
| ``checkpoints/epoch_{epoch:03d}.pt`` | エポックごとの個別チェックポイント（``train.save_every_epoch``、既定有効） |

## 損失の切り替え（docs/spec.md「損失: 二乗誤差とポアソン損失を比較する」）

``train.loss`` に ``mse`` か ``poisson`` を書く。``build_loss`` が対応する関数を返す。

**ポアソン損失の実装と数値安定性**。モデルの出力はフレームごとの softplus の総和なので
常に正の実数であり、そのままポアソン分布の平均 λ とみなせる。したがって
``torch.nn.PoissonNLLLoss(log_input=False, full=False, eps=POISSON_EPS)`` を使い、

    損失 = λ - k·log(λ + eps)      （k は正解のモーラ数）

を最小化する。``log_input=True``（出力を log λ と解釈する経路）は使わない。出力は
λ そのものであり、log を取って指数へ戻すと、詰め物のフレームが厳密に0であるという
本実装の性質（005の4.2節）と噛み合わないうえ、λ が大きいときに exp が溢れるためである。
``eps`` は λ が0に潰れた場合の ``log(0) = -inf`` を防ぐ。softplus の出力は数学的には
正だが、入力が -100 程度まで下がると float32 では0に丸まるため、この保護が要る。
``full=False`` は Stirling 近似の項（k·log k - k + log(2πk)/2）を落とす指定である。
この項はパラメータに依存しない定数なので勾配に影響しない。**エポック間・設定間で
損失の値を比べるときは、この定数ぶんだけ二乗誤差と尺度が違う**点に注意する
（比較に使う指標は ``metrics.jsonl`` の毎秒モーラ数MAEであって損失の値ではない）。

## デバイスとMPS

CLAUDE.md の規定により学習デバイスは ``mps``、float64 は使わない、``torch.compile`` は
使わない。MPS未対応の演算によるCPUフォールバックが起きた場合は、PyTorch が出す警告を
``MpsFallbackWatcher`` が捕まえ、演算子の名前と発生箇所（Pythonの呼び出し位置）を
``log.txt`` に記録する。環境変数 ``PYTORCH_ENABLE_MPS_FALLBACK`` が未設定のときは
フォールバックせず例外になるので、その場合も例外の内容がログに残る。

## 早期終了と上限エポック数

``train.max_epochs`` は学習の上限エポック数（再開した場合も**通算**のエポック数）である。
書かなければ従来どおり ``train.epochs`` を上限とする（``epochs`` と ``max_epochs`` を
両方書くと誤りとして止める）。

``train.early_stopping_patience`` に整数を書くと早期終了が有効になる（既定 ``None`` で無効。
exp001・exp002 の挙動は変わらない）。監視する指標は ``train.best_metric``（既定は dev の
``mae_moras_per_sec``）で、``best_mode`` の向き（MAEなら小さいほど良い）に比べる。

**改善の判定**: ``min`` の向きなら「値 < これまでの最良 − ``early_stopping_min_delta``」の
ときだけ改善とみなす（``max`` なら「値 > 最良 + min_delta」）。``min_delta`` の既定は
``0.0`` で、このときは厳密に小さくなれば改善である（同値は改善ではない）。改善しなかった
エポックが ``patience`` 回連続した時点で、そのエポックの記録と保存を済ませてから終了する。
``checkpoint_best.pt`` の更新は従来どおり厳密な改善で行う（``min_delta=0`` なら両者は一致）。

終了理由は ``early_stopping``（早期終了）か ``max_epochs``（上限到達）で、到達エポック・
最良エポックとともに ``log.txt`` の最後と ``run_summary.json`` に書く。
``metrics.jsonl`` の各行にも ``epochs_without_improvement`` を残す。

## エポックごとの個別チェックポイント保存

``train.save_every_epoch``（既定 ``True``）が有効なら、各エポックの検証と
``checkpoint_last.pt`` / ``checkpoint_best.pt`` の更新に加えて、
``run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"`` にそのエポックのチェックポイントを
個別に保存する。保存する中身は ``checkpoint_best.pt`` と同じ形式（``save_checkpoint`` の
既定どおり、``optimizer_state`` は含まない）。目的は、学習後に「全体MAEが最小のエポック」
と「特定の話速帯のMAEが最小のエポック」が異なる場合に、どちらのエポックのチェックポイントも
後から選び直せるようにすること（``checkpoint_best.pt`` は上書き保存のため、
``train.best_metric`` 以外の基準で最良だったエポックのチェックポイントは残らない）。
``False`` にすると個別保存を行わず、従来どおり ``checkpoint_last.pt`` /
``checkpoint_best.pt`` のみを保存する。

## チェックポイントからの再開

最上位の ``resume_from`` に再開元のチェックポイント（通常は ``checkpoint_last.pt``）を書く
（既定 ``None`` で再開しない）。

- ファイルが無ければ誤りとして止める（書き間違いで黙って最初から学習しないため）。
- ``optimizer_state`` を含むなら、モデルと最適化器（Adam）の状態を読み込み、エポック番号は
  再開元の ``epoch`` の続きから数える。上限 ``max_epochs`` は通算である。再開元が既に
  上限に達していれば誤りとして止める。学習率と重み減衰は**設定の値**で上書きし、
  最適化器の状態に保存された値と違えばその旨をログに書く。スケジューラは使っていないので
  扱う状態は無い（ログにもそう書く）。
- ``optimizer_state`` を含まないなら、再開せず最初から（乱数初期化・エポック1から）学習し、
  その旨と理由をログに書く。
- モデル構造（チェックポイントの ``model_config`` と今回の構造）と正規化（チェックポイントに
  保存された平均・標準偏差、無ければ正規化ファイルのパスとモード）が一致しなければ、
  データの読み込みより前に ``ResumeError`` で止める。
- 再開元が今回の出力先（``runs/<実験ID>/``）の中にある場合は止める。再開元の
  ``checkpoint_last.pt`` や ``metrics.jsonl`` を上書きしてしまうためである。

**再開時の最良値（早期終了と ``checkpoint_best.pt`` の基準）は引き継がず、測り直す**。
再開後の最初のエポックの dev 指標を最初の基準とし、そこから比較を始める。理由:

1. 再開は条件を変えて学習を続けるために使う（exp003/exp004 では学習データに無音サンプル
   ``silence_samples`` が加わる）。条件の変わった学習の最良は、その条件で学習した
   エポックの中から選ぶべきである。
2. 無音サンプルの効果（無音で0を出す）は dev の MAE に現れにくく、加えた直後に dev の MAE が
   わずかに悪化しうる。再開元の最良値を引き継ぐと、それを一度も下回らないまま
   ``patience`` で止まり、今回の出力先に ``checkpoint_best.pt`` が一度も書かれない。
3. dev の件数（``max_dev_clips``）が再開元と違えば値は比べられない。

再開元の最良値（チェックポイントの ``metrics`` にある ``best_metric`` の値）は参考として
ログと ``run_summary.json`` に残すが、比較には使わない。改善なしの連続回数も0から数える。

**乱数の扱い**: 再開時も ``seed`` で乱数を初期化し直す（大域の乱数状態はチェックポイントに
保存していない）。バッチの並び（``LengthBucketBatchSampler``）、拡張、無音サンプルの中身は
``(seed, エポック番号, 件の番号)`` から作るので、エポック番号を続きから数えれば、
続けて学習した場合と同じ系列になる。大域の乱数はモデルの初期化（再開では読み込んだ
重みで上書き）と DataLoader のワーカーの種にしか使っておらず、学習データの乱数には
効かない。ただし MPS の計算の非決定性などにより、ビット単位での一致は保証しない。

## 所要時間とデータ読み込み律速の計測

エポックごとに次を ``metrics.jsonl`` と ``log.txt`` に書く（単位は秒）。

| 列 | 中身 |
| --- | --- |
| ``epoch_seconds`` | エポック全体（学習ループ＋dev評価。チェックポイントの保存は含まない） |
| ``train_seconds`` | 学習ループ全体（従来からある列） |
| ``train_step_seconds`` | 学習ステップの計算（デバイスへの転送・順伝播・逆伝播・更新。mps は同期してから止める）の合計 |
| ``data_wait_seconds`` | DataLoader から次のバッチを取り出すまでの待ち時間の合計（ワーカーの起動を含む） |
| ``data_wait_ratio`` | ``data_wait_seconds / train_seconds`` |
| ``loader_end_seconds`` | 最後のバッチの後、ループを抜けるまで（DataLoader の終端処理とワーカーの終了待ち）。``train_seconds`` は計算・待ち・これの和 |
| ``dev_eval_seconds`` | dev 評価の所要時間 |

``data_wait_ratio`` が ``DATA_BOUND_RATIO``（0.5）以上なら、学習ループの半分以上を
データ待ちに使っているので「データ読み込み律速」とログに書く。ワーカーが計算に
追いついていれば待ち時間はほぼ0になる。

``loader_end_seconds`` は比率の分子に含めない（次のバッチを待つ時間ではないため）。
macOS（spawn）で ``num_workers>0`` のとき、ワーカーが終了要求にすぐ応じず、PyTorch の
終了待ちの上限（ワーカー1つあたり約5秒）まで待つことがある。これが大きい場合は
データ読み込みの律速とは別の固定費として読む。

## テストセット

検証には ``configs/splits/dev.json`` のみを使う。``configs/splits/test.json`` は
docs/PLAN.md 第10段階まで使用禁止であり、``spkrate.data.splits.load_split`` が
読み出しを拒否する。この学習ループには ``stage10_approved`` を渡す口を作らない。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from spkrate.data.augment import (
    AUGMENTATIONS,
    AugmentConfig,
    AugmentSetupError,
    describe_application_rates,
    describe_augment_config,
    estimate_application_rates,
    validate_augment_config,
)
from spkrate.eval.metrics import SpeedRateMetrics, compute_metrics
from spkrate.eval.runner import git_commit_info
from spkrate.models.cnn import CnnConfig, SpeechRateCNN, model_summary
from spkrate.train.data import (
    Batch,
    FeatureClipDataset,
    LengthBucketBatchSampler,
    Normalizer,
    WaveformClipDataset,
    collate_clips,
    load_normalization,
    select_clip_records,
)
from spkrate.train.silence import (
    SilenceSettings,
    SilenceSetupError,
    TrainWithSilence,
    build_silence_dataset,
    check_silence_setup,
    describe_silence_counts,
    is_silence_clip,
)

__all__ = [
    "AugmentSettings",
    "DataSettings",
    "DATA_BOUND_RATIO",
    "LOSSES",
    "POISSON_EPS",
    "ResumeError",
    "MpsFallbackWatcher",
    "TrainConfig",
    "TrainSettings",
    "TrainingOutcome",
    "build_loss",
    "build_datasets",
    "check_augment_setup",
    "evaluate_dev",
    "load_checkpoint",
    "load_train_config",
    "resolve_device",
    "run_training",
    "save_checkpoint",
    "write_config_snapshot",
]

LOSSES: tuple[str, ...] = ("mse", "poisson")

# ポアソン損失の log(λ) を守る下駄。docstring「ポアソン損失の実装と数値安定性」を参照。
POISSON_EPS = 1e-8

LOGGER_NAME = "spkrate.train"

# データ待ちの比率がこれ以上なら「データ読み込み律速」とログに書く。
# docstring「所要時間とデータ読み込み律速の計測」を参照。
DATA_BOUND_RATIO = 0.5

STOP_EARLY = "early_stopping"
STOP_MAX_EPOCHS = "max_epochs"


class ResumeError(ValueError):
    """再開元のチェックポイントが今回の設定と合わない（構造・正規化の不一致など）。"""


# --------------------------------------------------------------------------------------
# 設定


def _reject_unknown(cls: type, mapping: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"{cls.__name__} に未知の設定項目: {unknown}")
    return dict(mapping)


@dataclass(frozen=True)
class DataSettings:
    """データの出どころ（docs/decisions/006-augmentation.md 1節の2経路）。

    Attributes:
        source: 学習データの経路。``features``（事前計算済み）か ``waveform``（都度計算）。
            ``None`` なら拡張の有無から決める（拡張ありなら waveform）。
        dev_source: 検証データの経路。既定は ``features``（検証に拡張は掛けない）。
        features_dir: 事前計算特徴量の親ディレクトリ。直下に ``train`` / ``dev`` がある。
        clips_jsonl: クリップ一覧。waveform 経路で使う。
        audio_root: ``ClipRecord.audio_path`` を解決する基準ディレクトリ。clips.jsonl の
            ``audio_path`` はリポジトリ直下からの相対（``data/common_voice_ja/clips/…``）なので、
            既定はリポジトリ直下（``.``、実行時のカレント）とする。評価側
            （``build_eval_segments(..., ROOT)``）と同じ解決の仕方。
        train_split / dev_split: 分割ファイル。test.json は渡せない。
        max_train_clips / max_dev_clips: 件数の上限（試走用）。
        max_frames: これを超える長さのクリップを学習から外す（``None`` で無制限）。
        normalization: configs/normalization.yaml のパス。
        normalization_mode: ``per_mel`` か ``global``。``None`` なら yaml の既定。
    """

    source: str | None = None
    dev_source: str = "features"
    features_dir: str = "data/processed/features"
    clips_jsonl: str = "data/processed/clips.jsonl"
    audio_root: str = "."
    train_split: str = "configs/splits/train.json"
    dev_split: str = "configs/splits/dev.json"
    max_train_clips: int | None = None
    max_dev_clips: int | None = None
    max_frames: int | None = None
    normalization: str = "configs/normalization.yaml"
    normalization_mode: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "DataSettings":
        return cls(**_reject_unknown(cls, mapping or {}))


@dataclass(frozen=True)
class AugmentSettings:
    """データ拡張の設定（docs/decisions/006-augmentation.md）。

    ``enabled`` が False のときは拡張を一切掛けない。``params`` は
    ``spkrate.data.augment.AugmentConfig`` の項目をそのまま受ける。
    ``musan_root`` を与えると雑音重畳に MUSAN の noise サブセットを使う。
    """

    enabled: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    musan_root: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "AugmentSettings":
        return cls(**_reject_unknown(cls, mapping or {}))

    def build(self) -> AugmentConfig | None:
        """``AugmentConfig`` を作る。無効なら ``None``。"""
        if not self.enabled:
            return None
        return AugmentConfig.from_mapping(self.params)


@dataclass(frozen=True)
class TrainSettings:
    """最適化と実行の設定。

    Attributes:
        epochs: エポック数。``max_epochs`` を書かないときの上限。
        max_epochs: 上限エポック数（再開時は通算）。``None`` なら ``epochs`` を使う。
            ``epochs`` と同時には書けない。
        early_stopping_patience: 改善しないエポックがこの回数続いたら終了する。
            ``None`` で早期終了しない（既定）。
        early_stopping_min_delta: 改善とみなす最小の幅（0以上）。既定0で厳密な改善。
        batch_size: バッチの件数。
        loss: ``mse`` か ``poisson``（docs/spec.md「損失」）。
        learning_rate / weight_decay: Adam の設定。
        grad_clip: 勾配のノルムの上限。0以下で無効。
        num_workers: DataLoader のワーカー数。
        bucketing: 長さの近いものを同じバッチに集めるか（data.py の docstring を参照）。
        bucket_pool_batches: 長さで整列する塊の大きさ（バッチ数）。
        best_metric: 最良のチェックポイントを決める指標名。``metrics.jsonl`` の列名。
        best_mode: ``min`` か ``max``。``None`` なら指標名から決める。
        log_interval: 学習中に損失を書き出す間隔（バッチ数）。
        record_metrics_csv: 学習の最後に results/metrics.csv へ1行追記するか。
        metrics_csv: 追記先。
        save_every_epoch: 各エポックの終わりに ``run_dir/checkpoints/epoch_{epoch:03d}.pt``
            へ個別にチェックポイントを保存するか。既定 ``True``。``checkpoint_last.pt`` /
            ``checkpoint_best.pt`` の保存とは独立（docstring「エポックごとの個別
            チェックポイント保存」を参照）。
    """

    epochs: int = 10
    max_epochs: int | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    batch_size: int = 16
    loss: str = "mse"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    num_workers: int = 0
    bucketing: bool = True
    bucket_pool_batches: int = 20
    best_metric: str = "mae_moras_per_sec"
    best_mode: str | None = None
    log_interval: int = 50
    record_metrics_csv: bool = False
    metrics_csv: str = "results/metrics.csv"
    save_every_epoch: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "TrainSettings":
        payload = _reject_unknown(cls, mapping or {})
        if "epochs" in payload and payload.get("max_epochs") is not None:
            raise ValueError(
                "train.epochs と train.max_epochs は同時に書けない（上限はどちらか一方で書く）"
            )
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.resolved_max_epochs < 1:
            raise ValueError(f"上限エポック数は1以上: {self.resolved_max_epochs}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience は1以上か null: {self.early_stopping_patience}"
            )
        if not (self.early_stopping_min_delta >= 0.0):
            raise ValueError(
                f"early_stopping_min_delta は0以上: {self.early_stopping_min_delta}"
            )

    @property
    def resolved_max_epochs(self) -> int:
        """上限エポック数（通算）。"""
        return int(self.max_epochs if self.max_epochs is not None else self.epochs)

    @property
    def resolved_best_mode(self) -> str:
        if self.best_mode is not None:
            if self.best_mode not in {"min", "max"}:
                raise ValueError(f"best_mode は min か max: {self.best_mode}")
            return self.best_mode
        # 誤差と損失は小さいほど良い。相関は大きいほど良い。
        return "max" if "correlation" in self.best_metric else "min"


@dataclass(frozen=True)
class TrainConfig:
    """1回の学習の設定一式。yaml の最上位がそのままこの形である。"""

    experiment_id: str
    seed: int = 20260921
    device: str = "mps"
    runs_dir: str = "runs"
    model_config: str = "configs/model/cnn_base.yaml"
    model_overrides: dict[str, Any] = field(default_factory=dict)
    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    augment: AugmentSettings = field(default_factory=AugmentSettings)
    # 正解モーラ数0の無音・雑音サンプルの追加（spkrate.train.silence、docs/spec.md「学習データ」）。
    # 拡張とは独立に有効化できる。
    silence_samples: SilenceSettings = field(default_factory=SilenceSettings)
    # 再開元のチェックポイント（モジュール docstring「チェックポイントからの再開」）。
    resume_from: str | None = None
    notes: str = ""
    config_path: str = ""

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TrainConfig":
        payload = dict(mapping)
        data = DataSettings.from_mapping(payload.pop("data", None))
        train = TrainSettings.from_mapping(payload.pop("train", None))
        augment = AugmentSettings.from_mapping(payload.pop("augment", None))
        silence = SilenceSettings.from_mapping(payload.pop("silence_samples", None))
        payload = _reject_unknown(cls, payload)
        if "experiment_id" not in payload:
            raise ValueError("experiment_id が設定に無い")
        return cls(
            **payload, data=data, train=train, augment=augment, silence_samples=silence
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_changes(self, **changes: Any) -> "TrainConfig":
        """一部の項目だけを差し替えた設定を作る（コマンドラインの上書き用）。"""
        from dataclasses import replace as _replace

        return _replace(self, **changes)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.experiment_id

    @property
    def train_source(self) -> str:
        """学習データの経路。未指定なら拡張の有無から決める（006の1節）。"""
        if self.data.source is not None:
            return self.data.source
        return "waveform" if self.augment.enabled else "features"

    def build_model_config(self) -> CnnConfig:
        """モデル構造の設定を読み、``model_overrides`` を重ねる。"""
        from spkrate.models.cnn import load_config

        base = load_config(self.model_config)
        if not self.model_overrides:
            return base
        return CnnConfig.from_dict({**base.as_dict(), **self.model_overrides})


def load_train_config(path: str | Path) -> TrainConfig:
    """学習の設定ファイル（yaml）を読む。未知の項目は誤りとして弾く。"""
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"設定ファイルの最上位が辞書でない: {path}")
    payload.setdefault("config_path", str(path))
    return TrainConfig.from_mapping(payload)


# --------------------------------------------------------------------------------------
# 損失


def _mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """二乗誤差（クリップごとのモーラ数に対する平均）。"""
    return torch.nn.functional.mse_loss(prediction, target)


class PoissonLoss(nn.Module):
    """ポアソン損失。モジュール docstring「ポアソン損失の実装と数値安定性」を参照。

    モデルの出力 λ（softplus の総和、常に正）をポアソン分布の平均とみなし、

        損失 = λ - k·log(λ + eps)

    を最小化する。``log_input=False`` を使うのは出力が λ そのものだからであり、
    ``eps``（既定 ``POISSON_EPS`` = 1e-8）は λ が float32 で0に丸まったときの
    ``log(0) = -inf`` を防ぐ。``full=False`` で落としている Stirling 項は
    パラメータに依存しない定数で、勾配には影響しない。
    """

    def __init__(self, eps: float = POISSON_EPS) -> None:
        super().__init__()
        self.eps = float(eps)
        self._loss = nn.PoissonNLLLoss(
            log_input=False, full=False, eps=self.eps, reduction="mean"
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if torch.any(prediction < 0):
            raise ValueError("ポアソン損失には非負の予測が必要（softplus の総和を想定）")
        return self._loss(prediction, target)

    def extra_repr(self) -> str:
        return f"log_input=False, full=False, eps={self.eps}"


def build_loss(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    """設定の文字列から損失関数を作る（docs/spec.md「損失」）。

    Args:
        name: ``mse``（二乗誤差）か ``poisson``（ポアソン損失）。

    Raises:
        ValueError: 未知の名前のとき。設定の書き間違いを黙って既定値で通さない。
    """
    if name == "mse":
        return _mse_loss
    if name == "poisson":
        return PoissonLoss()
    raise ValueError(f"loss は {LOSSES} のいずれか: {name}")


# --------------------------------------------------------------------------------------
# デバイスとMPSのフォールバック


def resolve_device(name: str, logger: logging.Logger | None = None) -> torch.device:
    """設定のデバイス名を ``torch.device`` にする。

    CLAUDE.md の規定では ``mps`` を使う。``mps`` が使えない環境では警告を出して
    ``cpu`` に落とす（黙って落とすと、どこで学習したのか後から分からなくなるため
    必ずログに残す）。``auto`` は使える方を選ぶ。
    """
    log = logger or logging.getLogger(LOGGER_NAME)
    available = torch.backends.mps.is_available()
    if name == "auto":
        return torch.device("mps" if available else "cpu")
    if name == "mps" and not available:
        log.warning("mps が使えないため cpu で実行する（CLAUDE.md の規定は mps）")
        return torch.device("cpu")
    return torch.device(name)


class MpsFallbackWatcher:
    """MPS未対応の演算によるCPUフォールバックを捕まえてログに残す。

    PyTorch は未対応の演算で ``UserWarning: The operator 'aten::xxx' is not currently
    supported on the MPS backend and will fall back to run on the CPU.`` を出す
    （環境変数 ``PYTORCH_ENABLE_MPS_FALLBACK=1`` のとき）。この警告を横取りし、
    演算子の名前と発生箇所（リポジトリ内のPythonの呼び出し位置）を記録する。
    同じ演算子は最初の1回だけ記録する。

    環境変数が未設定のときはフォールバックせず ``NotImplementedError`` になるため、
    警告は出ない。その場合は例外がそのまま学習を止め、内容がログに残る。
    """

    _MARKERS = ("fall back to run on the CPU", "not currently supported on the MPS")

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.events: list[dict[str, str]] = []
        self._seen: set[str] = set()
        self._previous: Any = None

    def __enter__(self) -> "MpsFallbackWatcher":
        self._previous = warnings.showwarning
        warnings.showwarning = self._show  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        warnings.showwarning = self._previous

    def _show(self, message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        text = str(message)
        if any(marker in text for marker in self._MARKERS):
            key = text.split("'")[1] if "'" in text else text[:60]
            if key not in self._seen:
                self._seen.add(key)
                # 呼び出し元のうち、このリポジトリのコードだけを残す。
                stack = [
                    f"{frame.filename}:{frame.lineno} {frame.name}"
                    for frame in traceback.extract_stack()[:-1]
                    if "spkrate" in frame.filename or "scripts" in frame.filename
                ]
                location = stack[-1] if stack else f"{filename}:{lineno}"
                self.events.append({"operator": key, "location": location, "message": text})
                self.logger.warning(
                    "MPSのCPUフォールバック: 演算子=%s 発生箇所=%s", key, location
                )
        if self._previous is not None:
            self._previous(message, category, filename, lineno, file, line)


# --------------------------------------------------------------------------------------
# ログ


def setup_logger(run_dir: Path, *, name: str = LOGGER_NAME) -> logging.Logger:
    """``runs/<実験ID>/log.txt`` と標準出力の両方へ書くロガーを作る。

    バックグラウンド実行（CLAUDE.md「学習はバックグラウンドで実行しログをruns/以下に
    ファイル出力する」）を前提に、ファイル側は行ごとに flush する。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


# --------------------------------------------------------------------------------------
# 設定の記録


def _json_safe(value: Any) -> Any:
    """yaml / json に書ける形へ直す（tuple → list、Path → str など）。

    ``str`` や ``int`` の**派生型**も素の型に直す。``yaml.safe_dump`` は派生型を
    ``cannot represent an object`` で拒否するためである（``torch.__version__`` は
    ``str`` を継承した ``torch.torch_version.TorchVersion`` である）。
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _json_safe(value.item())
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def write_config_snapshot(
    config: TrainConfig,
    model_config: CnnConfig,
    *,
    run_dir: Path,
    device: torch.device,
    extra: dict[str, Any] | None = None,
) -> Path:
    """``runs/<実験ID>/config_snapshot.yaml`` を書く（PLAN.md 5-2 の要求）。

    gitのコミットハッシュ、設定ファイルの中身（元のテキストと解決後の値）、モデル構造と
    その要約、乱数シード、デバイス、ライブラリの版を残す。後からこのファイルだけで
    実行条件が分かるようにするためである。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit_info()
    model = SpeechRateCNN(model_config)

    snapshot: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git": {"commit_hash": commit.commit_hash, "dirty": commit.dirty},
        "seed": config.seed,
        "device": str(device),
        "config_path": config.config_path,
        "config": _json_safe(config.as_dict()),
        "train_source": config.train_source,
        "model_config_path": config.model_config,
        "model": _json_safe(model_config.as_dict()),
        "model_summary": _json_safe(model_summary(model)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "pytorch_enable_mps_fallback": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK", ""
            ),
        },
    }
    if config.config_path and Path(config.config_path).is_file():
        snapshot["config_file_text"] = Path(config.config_path).read_text(encoding="utf-8")
    if Path(config.model_config).is_file():
        snapshot["model_config_file_text"] = Path(config.model_config).read_text(
            encoding="utf-8"
        )
    if extra:
        snapshot.update(_json_safe(extra))

    path = run_dir / "config_snapshot.yaml"
    path.write_text(
        yaml.safe_dump(_json_safe(snapshot), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# チェックポイント


def save_checkpoint(
    path: str | Path,
    model: SpeechRateCNN,
    *,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    config: TrainConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    normalizer: Normalizer | None = None,
) -> Path:
    """チェックポイントを保存する。

    重みだけでなくモデル構造の設定も入れる。``load_checkpoint`` はこの設定から
    モデルを組み直すので、読み込み側が構造を知らなくても復元できる。
    ``normalizer`` を渡すと正規化の平均・標準偏差も入れる（再開時の一致の検査に使う）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": _json_safe(model.config.as_dict()),
        "epoch": int(epoch),
        "metrics": _json_safe(metrics or {}),
        "train_config": _json_safe(config.as_dict()) if config is not None else {},
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if normalizer is not None:
        payload["normalization"] = {
            "mode": normalizer.mode,
            "mean": np.ravel(normalizer.mean).astype(float).tolist(),
            "std": np.ravel(normalizer.std).astype(float).tolist(),
        }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[SpeechRateCNN, dict[str, Any]]:
    """チェックポイントからモデルを組み直して返す。

    Returns:
        ``(モデル, 保存した中身)``。モデルは ``eval`` 状態ではなく、保存時のまま。
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = SpeechRateCNN(CnnConfig.from_dict(dict(payload["model_config"])))
    model.load_state_dict(payload["model_state"])
    return model, payload


# --------------------------------------------------------------------------------------
# データの組み立て


def build_datasets(
    config: TrainConfig, logger: logging.Logger | None = None
) -> tuple[Dataset, Dataset, Normalizer]:
    """設定から学習用・検証用のデータセットを作る（006の1節の2経路）。

    ``silence_samples.enabled`` なら、学習データ（どちらの経路でも）の後ろに正解モーラ数0の
    無音・雑音サンプルを連結する（``TrainWithSilence``）。検証には加えない。

    検証は常に拡張なしである。拡張ありで ``features`` 経路を選ぶ組み合わせは、
    拡張が波形に掛かる以上ありえないので ``ValueError`` で弾く。
    """
    log = logger or logging.getLogger(LOGGER_NAME)
    normalizer = load_normalization(
        config.data.normalization, mode=config.data.normalization_mode
    )
    source = config.train_source
    if source not in {"features", "waveform"}:
        raise ValueError(f"data.source は features か waveform: {source}")
    if config.augment.enabled and source == "features":
        raise ValueError(
            "拡張ありでは事前計算特徴量を使えない（docs/decisions/006-augmentation.md 1節: "
            "拡張は波形に適用する）。data.source を waveform にすること"
        )
    if config.data.dev_source != "features":
        raise ValueError(
            "検証は事前計算特徴量（data/processed/features/dev）を使う。"
            f"dev_source={config.data.dev_source} は想定していない"
        )

    features_dir = Path(config.data.features_dir)
    if source == "features":
        from spkrate.data.splits import load_split

        train_dataset: Dataset = FeatureClipDataset(
            features_dir / "train",
            client_ids=load_split(config.data.train_split),
            normalizer=normalizer,
            limit=config.data.max_train_clips,
            max_frames=config.data.max_frames,
        )
    else:
        records = select_clip_records(
            config.data.clips_jsonl,
            config.data.train_split,
            limit=config.data.max_train_clips,
            max_duration_sec=(
                None
                if config.data.max_frames is None
                else config.data.max_frames / 100.0
            ),
        )
        train_dataset = WaveformClipDataset(
            records,
            config.data.audio_root,
            normalizer=normalizer,
            augment=config.augment.build(),
            noise_source=_build_noise_source(config, log),
            seed=config.seed,
        )

    if config.silence_samples.enabled:
        num_speech = len(train_dataset)  # type: ignore[arg-type]
        silence_dataset = build_silence_dataset(
            config.silence_samples,
            train_dataset,
            normalizer=normalizer,
            seed=config.seed,
            # 事前計算特徴量は float16 保存なので、同じ丸めを通す（silence.py の docstring）。
            match_feature_storage=(source == "features"),
        )
        train_dataset = TrainWithSilence(train_dataset, silence_dataset)
        log.info("%s", describe_silence_counts(num_speech, silence_dataset.counts))

    from spkrate.data.splits import load_split as _load_split

    dev_dataset: Dataset = FeatureClipDataset(
        features_dir / "dev",
        client_ids=_load_split(config.data.dev_split),
        normalizer=normalizer,
        limit=config.data.max_dev_clips,
    )
    log.info(
        "データ: 学習=%d件（経路=%s、拡張=%s） 検証=%d件（経路=features、拡張なし）",
        len(train_dataset),  # type: ignore[arg-type]
        source,
        "あり" if config.augment.enabled else "なし",
        len(dev_dataset),  # type: ignore[arg-type]
    )
    return train_dataset, dev_dataset, normalizer


def check_augment_setup(
    config: TrainConfig, logger: logging.Logger | None = None
) -> list[str]:
    """拡張の設定を学習開始前に検査し、ログに書く一覧を返す。

    データの読み込みやモデルの構築より前に呼ぶ（``run_training`` の冒頭）。
    拡張が有効なのに、ある拡張が黙って実行されない設定を ``AugmentSetupError`` で止める。
    検出する条件と、止めない条件の一覧は results/augment_validation.md。

    ``augment.enabled=false`` なのに ``augment.params`` や ``augment.musan_root`` が
    書かれている設定は止めず、``logger`` に警告を出す（docs/questions.md 2026-09-24 回答1）。
    ``run_training`` は学習ログ（log.txt に書くロガー）を渡す。

    Returns:
        ``log.txt`` に書く行。無効なら ``["拡張=なし"]``。
    """
    log = logger or logging.getLogger(LOGGER_NAME)
    if not config.augment.enabled:
        ignored = []
        if config.augment.params:
            ignored.append(f"augment.params（{', '.join(sorted(config.augment.params))}）")
        if config.augment.musan_root:
            ignored.append(f"augment.musan_root={config.augment.musan_root}")
        if ignored:
            log.warning(
                "augment.enabled=false のため、書かれている %s は使われない（学習は続ける）",
                "・".join(ignored),
            )
        return ["拡張=なし"]

    source = config.train_source
    if source != "waveform":
        raise AugmentSetupError(
            f"augment.enabled=true だが data.source={source} で、拡張が掛からない"
            "（拡張は波形に適用する。docs/decisions/006-augmentation.md 1節）。"
            "data.source を waveform にすること"
        )

    augment_config = config.augment.build()
    assert augment_config is not None
    validate_augment_config(augment_config)
    noise_description = "（使わない）"
    if augment_config.noise_enabled:
        noise_description = _check_musan(config.augment.musan_root)
    lines = describe_augment_config(augment_config, noise_description=noise_description)
    # 設計上の確率と実際の適用率の両方を残す（docs/questions.md 2026-09-24 回答4）。
    # ここまで来れば、雑音重畳が有効なら雑音源はある。
    rates = estimate_application_rates(augment_config, noise_available=True)
    return lines + describe_application_rates(rates)


def _check_musan(musan_root: str | None) -> str:
    """雑音重畳が有効なときの雑音源の検査。ログに書く雑音源の説明を返す。"""
    if not musan_root:
        raise AugmentSetupError(
            "augment.enabled=true だが augment.musan_root が未設定で、雑音重畳が一度も"
            "行われない。MUSAN の配置先（data/DATASETS.md、通常は data/musan）を書くこと"
            "（雑音重畳を外すなら augment.params に noise_enabled: false と書く）"
        )
    if not Path(musan_root).is_dir():
        raise AugmentSetupError(
            f"augment.musan_root={musan_root} がディレクトリとして存在しない。"
            "配置先は data/DATASETS.md を参照する"
        )
    from spkrate.data.augment import MusanNoiseSource

    noise = MusanNoiseSource(musan_root)
    try:
        noise_paths = noise.paths
    except FileNotFoundError as error:
        raise AugmentSetupError(
            f"augment.musan_root={musan_root} に雑音ファイルが無く、雑音重畳が一度も"
            f"行われない（{error}）"
        ) from error
    subsets = "、".join(str(Path(musan_root) / subset) for subset in noise.subsets)
    return f"{subsets}（{len(noise_paths)}ファイル）"


def _build_noise_source(config: TrainConfig, logger: logging.Logger) -> Any:
    """MUSAN の noise サブセットを雑音源として用意する（設定にあれば）。"""
    if not config.augment.enabled or not config.augment.musan_root:
        return None
    augment_config = config.augment.build()
    if augment_config is not None and not augment_config.noise_enabled:
        return None
    from spkrate.data.augment import MusanNoiseSource

    source = MusanNoiseSource(config.augment.musan_root)
    logger.info("雑音源: MUSAN noise %d ファイル", len(source.paths))
    return source


def _frame_counts(dataset: Dataset) -> list[int] | None:
    counts = getattr(dataset, "frame_counts", None)
    if counts is None:
        return None
    return [int(value) for value in counts]


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    settings: TrainSettings,
    seed: int,
) -> DataLoader:
    """DataLoader を作る。長さでまとめる設定なら専用のバッチ分けを使う。"""
    counts = _frame_counts(dataset)
    common: dict[str, Any] = {
        "collate_fn": collate_clips,
        "num_workers": settings.num_workers,
        "pin_memory": False,
    }
    if settings.bucketing and counts is not None:
        sampler = LengthBucketBatchSampler(
            counts,
            batch_size,
            shuffle=shuffle,
            pool_batches=settings.bucket_pool_batches,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)


# --------------------------------------------------------------------------------------
# 学習と検証


@dataclass
class TrainingOutcome:
    """学習の結果（run_training の戻り値）。"""

    run_dir: Path
    best_epoch: int
    best_metric_value: float
    best_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    mps_fallbacks: list[dict[str, str]]
    # 終了理由（``early_stopping`` か ``max_epochs``）、到達エポック（通算）、開始エポック。
    stop_reason: str = STOP_MAX_EPOCHS
    last_epoch: int = 0
    start_epoch: int = 1
    resume: dict[str, Any] = field(default_factory=dict)


def describe_epoch_augment_counts(
    epoch: int, counts: dict[str, int], num_clips: int
) -> str:
    """エポック終了時に書く、学習中に実際に掛かった拡張の回数（1行）。

    数えるのは ``AugmentResult.effective`` と周波数マスクが実際に掛かった回である
    （無響になった残響、無音で雑音を足さなかった回などは含まない）。
    """
    parts = []
    for name, label in AUGMENTATIONS:
        count = int(counts.get(name, 0))
        rate = count / num_clips if num_clips else 0.0
        parts.append(f"{label}={count}({rate:.3f})")
    return f"エポック{epoch} 拡張の実適用回数（学習{num_clips}件中）: " + " ".join(parts)


def _train_one_epoch(
    model: SpeechRateCNN,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    settings: TrainSettings,
    epoch: int,
    logger: logging.Logger,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_clips = 0
    total_abs_error = 0.0
    augment_counts: dict[str, int] = {}
    silence_clips = 0
    data_wait_seconds = 0.0
    step_seconds = 0.0
    started = time.perf_counter()
    # 次のバッチを取り出し始めた時刻。最初の取り出しにはワーカーの起動も含まれる。
    fetch_started = started
    for step, batch in enumerate(loader, start=1):
        step_started = time.perf_counter()
        data_wait_seconds += step_started - fetch_started
        silence_clips += sum(1 for clip_id in batch.clip_ids if is_silence_clip(clip_id))
        for names in getattr(batch, "augment_applied", ()):
            for name in names:
                augment_counts[name] = augment_counts.get(name, 0) + 1
        batch = batch.to(device)
        prediction = model(batch.features, batch.lengths)
        loss = loss_fn(prediction, batch.moras)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if settings.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        _synchronize(device)

        count = len(batch)
        total_loss += float(loss.detach().cpu()) * count
        total_clips += count
        with torch.no_grad():
            rate_error = torch.abs(
                prediction.detach() / batch.durations - batch.moras / batch.durations
            )
            total_abs_error += float(rate_error.sum().cpu())
        if settings.log_interval > 0 and step % settings.log_interval == 0:
            logger.info(
                "エポック%d 途中 %dバッチ 損失=%.4f", epoch, step, total_loss / total_clips
            )
        fetch_started = time.perf_counter()
        step_seconds += fetch_started - step_started
    # 最後のバッチの後、ループを抜けるまで（DataLoader の終端とワーカーの終了待ち）。
    loader_end_seconds = time.perf_counter() - fetch_started
    if total_clips == 0:
        raise RuntimeError("学習データが空である")
    train_seconds = time.perf_counter() - started
    return {
        "train_loss": total_loss / total_clips,
        "train_mae_moras_per_sec": total_abs_error / total_clips,
        "train_seconds": train_seconds,
        # 所要時間の内訳（モジュール docstring「所要時間とデータ読み込み律速の計測」）。
        "train_step_seconds": step_seconds,
        "data_wait_seconds": data_wait_seconds,
        "data_wait_ratio": data_wait_seconds / train_seconds if train_seconds > 0 else 0.0,
        "loader_end_seconds": loader_end_seconds,
        "train_clips": total_clips,
        # うち無音サンプル（spkrate.train.silence）の件数。拡張は掛けないので、
        # 拡張の実適用率の分母からは除く。
        "train_silence_clips": silence_clips,
        # 学習中に実際に掛かった拡張の回数。metrics.jsonl には書かず、ログにだけ出す。
        "augment_counts": augment_counts,
    }


@torch.no_grad()
def evaluate_dev(
    model: SpeechRateCNN,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
) -> tuple[SpeedRateMetrics, dict[str, float], dict[str, float]]:
    """検証セットで全指標を計算する。

    モデルの出力はクリップのモーラ数なので、``compute_metrics`` へは
    「正解モーラ数・推定モーラ数・クリップ長」をそのまま渡す（毎秒モーラ数への換算は
    ``compute_metrics`` の内部で行われる）。

    Returns:
        ``(指標, 付随する測定値, クリップIDから推定毎秒モーラ数への辞書)``。
    """
    model.eval()
    true_moras: list[float] = []
    pred_moras: list[float] = []
    durations: list[float] = []
    predictions: dict[str, float] = {}
    total_loss = 0.0
    total_clips = 0
    forward_seconds = 0.0

    for batch in loader:
        batch = batch.to(device)
        started = time.perf_counter()
        prediction = model(batch.features, batch.lengths)
        if device.type == "mps":
            torch.mps.synchronize()
        forward_seconds += time.perf_counter() - started
        loss = loss_fn(prediction, batch.moras)
        total_loss += float(loss.detach().cpu()) * len(batch)
        total_clips += len(batch)

        pred = prediction.detach().cpu().numpy().astype(np.float32)
        truth = batch.moras.detach().cpu().numpy().astype(np.float32)
        length = batch.durations.detach().cpu().numpy().astype(np.float32)
        for clip_id, p, t, d in zip(batch.clip_ids, pred, truth, length, strict=True):
            pred_moras.append(float(p))
            true_moras.append(float(t))
            durations.append(float(d))
            predictions[clip_id] = float(p) / float(d)

    metrics = compute_metrics(true_moras, pred_moras, durations)
    extras = {
        "val_loss": total_loss / total_clips if total_clips else float("nan"),
        "val_clips": float(total_clips),
        "val_forward_ms_per_clip": (
            forward_seconds / total_clips * 1000.0 if total_clips else float("nan")
        ),
    }
    return metrics, extras, predictions


def _synchronize(device: torch.device) -> None:
    """非同期のデバイスで計算の完了を待つ（計時を正しくするため）。"""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _set_seed(seed: int) -> None:
    """乱数シードを固定する（設定の ``seed``。config_snapshot.yaml にも残る）。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _is_better(value: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    """``value`` が ``best`` より ``min_delta`` を超えて良いか（早期終了の改善の判定）。"""
    if not math.isfinite(value):
        return False
    if not math.isfinite(best):
        return True
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


def _normalization_mismatch(
    payload: dict[str, Any], config: TrainConfig, current: Normalizer | None
) -> str | None:
    """再開元と今回の正規化が違えば、その説明を返す。一致すれば ``None``。"""
    saved = payload.get("normalization")
    if saved is not None and current is not None:
        if saved.get("mode") != current.mode:
            return f"正規化のモードが違う（再開元={saved.get('mode')} 今回={current.mode}）"
        mean = np.asarray(saved.get("mean", []), dtype=np.float32)
        std = np.asarray(saved.get("std", []), dtype=np.float32)
        if mean.shape != np.ravel(current.mean).shape or not (
            np.allclose(mean, np.ravel(current.mean)) and np.allclose(std, np.ravel(current.std))
        ):
            return "正規化の平均・標準偏差の値が違う"
        return None
    # 値が保存されていない古いチェックポイント（exp001・exp002）はパスとモードで比べる。
    data = dict(payload.get("train_config", {}).get("data", {}) or {})
    if not data:
        return "再開元に正規化の情報（値も設定も）が無く、一致を確かめられない"
    saved_path = data.get("normalization")
    if saved_path is None or Path(saved_path) != Path(config.data.normalization):
        return (
            f"正規化ファイルが違う（再開元={saved_path} 今回={config.data.normalization}）"
        )
    saved_mode = data.get("normalization_mode")
    current_mode = current.mode if current is not None else config.data.normalization_mode
    if saved_mode is not None and current_mode is not None and saved_mode != current_mode:
        return f"正規化のモードが違う（再開元={saved_mode} 今回={current_mode}）"
    return None


@dataclass
class ResumeState:
    """再開の判断の結果。``resumed`` が偽なら最初から学習する。"""

    source: str | None = None
    resumed: bool = False
    source_epoch: int = 0
    reason: str = ""
    source_best_value: float | None = None
    payload: dict[str, Any] | None = None

    @property
    def start_epoch(self) -> int:
        return self.source_epoch + 1 if self.resumed else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "resume_from": self.source,
            "resumed": self.resumed,
            "source_epoch": self.source_epoch if self.resumed else None,
            "start_epoch": self.start_epoch,
            "reason": self.reason,
            "source_best_metric_value": self.source_best_value,
            "best_value_policy": "remeasure" if self.resumed else None,
        }


def prepare_resume(
    config: TrainConfig,
    model_config: CnnConfig,
    normalizer: Normalizer | None,
    logger: logging.Logger,
) -> ResumeState:
    """再開元のチェックポイントを検査し、再開するかを決める（データの読み込みより前）。

    モジュール docstring「チェックポイントからの再開」を参照。
    """
    if not config.resume_from:
        return ResumeState(reason="resume_from が未設定のため最初から学習する")
    path = Path(config.resume_from)
    if not path.is_file():
        raise ResumeError(f"再開元のチェックポイントが無い: {path}")
    run_dir = config.run_dir.resolve()
    if run_dir == path.resolve().parent or run_dir in path.resolve().parents:
        raise ResumeError(
            f"再開元 {path} が今回の出力先 {config.run_dir} の中にある。再開元の記録を"
            "上書きしてしまうため、別の experiment_id で再開すること"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer_state" not in payload:
        state = ResumeState(
            source=str(path),
            reason=f"{path} に optimizer_state が無いため再開せず、最初から学習する",
        )
        logger.warning("再開: %s", state.reason)
        return state

    saved_model = _json_safe(dict(payload.get("model_config", {})))
    current_model = _json_safe(model_config.as_dict())
    if saved_model != current_model:
        diff = sorted(
            key
            for key in set(saved_model) | set(current_model)
            if saved_model.get(key) != current_model.get(key)
        )
        raise ResumeError(f"再開元とモデル構造が違う（違う項目: {diff}）: {path}")
    mismatch = _normalization_mismatch(payload, config, normalizer)
    if mismatch:
        raise ResumeError(f"再開元と正規化が違う: {mismatch}: {path}")

    source_epoch = int(payload.get("epoch", 0))
    max_epochs = config.train.resolved_max_epochs
    if source_epoch >= max_epochs:
        raise ResumeError(
            f"再開元は既にエポック{source_epoch}まで学習済みで、上限（通算）"
            f"{max_epochs} に達している: {path}"
        )
    source_value = dict(payload.get("metrics", {}) or {}).get(config.train.best_metric)
    return ResumeState(
        source=str(path),
        resumed=True,
        source_epoch=source_epoch,
        reason=f"{path} に optimizer_state があるため、モデルと最適化器の状態を読み込んで再開する",
        source_best_value=None if source_value is None else float(source_value),
        payload=payload,
    )


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
                logger.warning(
                    "再開: 最適化器の状態の %s=%g を設定の値 %g で上書きする",
                    key,
                    float(group[key]),
                    float(value),
                )
            group[key] = value


def _try_load_normalizer(config: TrainConfig) -> Normalizer | None:
    path = Path(config.data.normalization)
    if not path.is_file():
        return None
    return load_normalization(path, mode=config.data.normalization_mode)


def run_training(
    config: TrainConfig,
    *,
    train_dataset: Dataset | None = None,
    dev_dataset: Dataset | None = None,
    logger: logging.Logger | None = None,
) -> TrainingOutcome:
    """学習を実行する。

    ``train_dataset`` / ``dev_dataset`` を渡すと設定からのデータ構築を省略する
    （単体テストで合成データを使うための口。本番では ``None`` にして
    ``build_datasets`` に任せる）。

    出力はすべて ``runs/<実験ID>/`` に置く。エポックごとに検証セットの全指標を計算して
    ``metrics.jsonl`` に1行追記し、最良のエポックで ``checkpoint_best.pt`` を更新する。
    """
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logger or setup_logger(run_dir)

    # 設定の記録と拡張の検査は、データの読み込み・モデルの構築より前に行う。
    commit = git_commit_info()
    log.info(
        "コミット=%s%s 設定=%s",
        commit.commit_hash,
        "（未コミットの変更あり）" if commit.dirty else "",
        config.config_path or "（なし）",
    )
    try:
        augment_lines = check_augment_setup(config, log)
    except AugmentSetupError as error:
        log.error("拡張の設定の誤りで停止する: %s", error)
        raise
    for line in augment_lines:
        log.info("%s", line)
    try:
        silence_lines = check_silence_setup(config.silence_samples)
    except SilenceSetupError as error:
        log.error("無音サンプルの設定の誤りで停止する: %s", error)
        raise
    for line in silence_lines:
        log.info("%s", line)

    if config.train.loss not in LOSSES:
        raise ValueError(f"loss は {LOSSES} のいずれか: {config.train.loss}")
    config.train.validate()
    max_epochs = config.train.resolved_max_epochs
    patience = config.train.early_stopping_patience
    min_delta = float(config.train.early_stopping_min_delta)
    best_mode = config.train.resolved_best_mode
    _set_seed(config.seed)
    device = resolve_device(config.device, log)
    log.info(
        "実験ID=%s デバイス=%s 損失=%s シード=%d 上限エポック数（通算）=%d バッチ=%d",
        config.experiment_id,
        device,
        config.train.loss,
        config.seed,
        max_epochs,
        config.train.batch_size,
    )
    if patience is None:
        log.info("早期終了: なし（上限エポック数まで学習する）")
    else:
        log.info(
            "早期終了: あり 監視=%s（%s） patience=%d 最小改善幅=%g"
            "（%sのとき改善とみなす）",
            config.train.best_metric,
            "小さいほど良い" if best_mode == "min" else "大きいほど良い",
            patience,
            min_delta,
            "値 < 最良 − 最小改善幅" if best_mode == "min" else "値 > 最良 + 最小改善幅",
        )

    model_config = config.build_model_config()
    normalizer = _try_load_normalizer(config)
    # 再開元の検査（構造・正規化の不一致）はデータの読み込みより前に行う。
    try:
        resume = prepare_resume(config, model_config, normalizer, log)
    except ResumeError as error:
        log.error("再開元のチェックポイントの誤りで停止する: %s", error)
        raise
    model = SpeechRateCNN(model_config).to(device)
    if resume.resumed:
        assert resume.payload is not None
        model.load_state_dict(resume.payload["model_state"])
        log.info(
            "再開: 再開元=%s 再開元のエポック=%d → エポック%dから通算%dまで学習する。理由: %s",
            resume.source,
            resume.source_epoch,
            resume.start_epoch,
            max_epochs,
            resume.reason,
        )
        log.info(
            "再開: 最良値は引き継がず測り直す（再開後の最初のエポックの dev 指標を最初の基準に"
            "する）。再開元の %s=%s は参考値で、比較には使わない。理由: 再開後は学習の条件"
            "（無音サンプルの追加など）が変わるため、その条件で学習したエポックの中から最良を"
            "選ぶ。引き継ぐと条件変更直後の一時的な悪化で checkpoint_best.pt が一度も書かれない"
            "まま早期終了しうる。dev の件数が再開元と違う場合も値を比べられない",
            config.train.best_metric,
            "不明" if resume.source_best_value is None else f"{resume.source_best_value:.4f}",
        )
        log.info(
            "再開: 乱数は seed=%d で初期化し直す（大域の乱数状態は保存していない）。"
            "バッチの並び・拡張・無音サンプルは (seed, エポック番号, 件の番号) から作るので、"
            "エポック番号を続きから数えることで続けて学習した場合と同じ系列になる",
            config.seed,
        )
        log.info("再開: 学習率スケジューラは使っていないため、読み込む状態は無い")
    elif config.resume_from:
        log.info("再開: 再開元=%s 理由: %s（エポック1から）", resume.source, resume.reason)
    summary = model_summary(model)
    log.info(
        "モデル: パラメータ数=%d 受容野=%dフレーム(%.2f秒) float32=%.2fMiB",
        summary["num_parameters"],
        summary["receptive_field_frames"],
        summary["receptive_field_seconds"],
        summary["float32_mib"],
    )

    if train_dataset is None or dev_dataset is None:
        built_train, built_dev, _ = build_datasets(config, log)
        train_dataset = train_dataset or built_train
        dev_dataset = dev_dataset or built_dev

    train_loader = _make_loader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        settings=config.train,
        seed=config.seed,
    )
    dev_loader = _make_loader(
        dev_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        settings=config.train,
        seed=config.seed,
    )

    write_config_snapshot(
        config,
        model_config,
        run_dir=run_dir,
        device=device,
        extra={
            "dataset": {
                "num_train_clips": len(train_dataset),  # type: ignore[arg-type]
                "num_dev_clips": len(dev_dataset),  # type: ignore[arg-type]
                # 無音サンプルの件数（num_train_clips に含まれる）。無効なら null。
                "silence_counts": getattr(train_dataset, "silence_counts", None),
            },
            "resume": resume.as_dict(),
        },
    )

    loss_fn = build_loss(config.train.loss)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    if resume.resumed:
        assert resume.payload is not None
        _load_optimizer_state(optimizer, resume.payload["optimizer_state"], config.train, log)
        resume.payload = None  # 大きな重みを持ち続けない

    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")  # 実行ごとに作り直す
    best_value = math.inf if best_mode == "min" else -math.inf
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    # 早期終了の基準（再開時も測り直す。docstring「チェックポイントからの再開」）。
    stop_best = math.inf if best_mode == "min" else -math.inf
    epochs_without_improvement = 0
    stop_reason = STOP_MAX_EPOCHS
    last_epoch = resume.start_epoch - 1

    with MpsFallbackWatcher(log) as watcher:
        for epoch in range(resume.start_epoch, max_epochs + 1):
            epoch_started = time.perf_counter()
            for target in (train_loader.batch_sampler, train_dataset):
                if hasattr(target, "set_epoch"):
                    target.set_epoch(epoch)  # type: ignore[union-attr]

            train_stats = _train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                settings=config.train,
                epoch=epoch,
                logger=log,
            )
            augment_counts = train_stats.pop("augment_counts", {})
            if config.augment.enabled:
                log.info(
                    "%s",
                    describe_epoch_augment_counts(
                        epoch,
                        augment_counts,
                        int(train_stats["train_clips"])
                        - int(train_stats["train_silence_clips"]),
                    ),
                )
            dev_started = time.perf_counter()
            metrics, extras, predictions = evaluate_dev(model, dev_loader, loss_fn, device)
            dev_eval_seconds = time.perf_counter() - dev_started
            epoch_seconds = time.perf_counter() - epoch_started

            row: dict[str, Any] = {
                "experiment_id": config.experiment_id,
                "epoch": epoch,
                "timestamp": datetime.now(timezone.utc)
                .astimezone()
                .isoformat(timespec="seconds"),
                "loss": config.train.loss,
                "learning_rate": config.train.learning_rate,
                **{key: _json_safe(value) for key, value in train_stats.items()},
                **{key: _json_safe(value) for key, value in extras.items()},
                **_json_safe(metrics.as_dict()),
                "dev_eval_seconds": dev_eval_seconds,
                "epoch_seconds": epoch_seconds,
            }
            value = row.get(config.train.best_metric)
            if value is None:
                raise ValueError(
                    f"best_metric が metrics.jsonl の列に無い: {config.train.best_metric}"
                )
            is_best = _is_better(float(value), best_value, best_mode)
            row["is_best"] = is_best
            if _is_better(float(value), stop_best, best_mode, min_delta):
                stop_best = float(value)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row["epochs_without_improvement"] = epochs_without_improvement
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            history.append(row)

            log.info(
                "エポック%d 学習損失=%.4f 検証損失=%.4f 毎秒モーラ数MAE=%.4f 相関=%.4f "
                "帯別MAE(<4/4-6/6-8/>=8)=%.3f/%.3f/%.3f/%.3f %.1f秒%s",
                epoch,
                train_stats["train_loss"],
                extras["val_loss"],
                metrics.mae_moras_per_sec,
                metrics.correlation,
                metrics.mae_band_under4,
                metrics.mae_band_4to6,
                metrics.mae_band_6to8,
                metrics.mae_band_over8,
                train_stats["train_seconds"],
                "（最良）" if is_best else "",
            )
            wait_ratio = float(train_stats["data_wait_ratio"])
            log.info(
                "エポック%d 所要時間: 全体=%.1f秒 学習ループ=%.1f秒（計算=%.1f秒 "
                "データ待ち=%.1f秒 比率=%.3f 終端処理=%.1f秒） dev評価=%.1f秒 → %s",
                epoch,
                epoch_seconds,
                train_stats["train_seconds"],
                train_stats["train_step_seconds"],
                train_stats["data_wait_seconds"],
                wait_ratio,
                train_stats["loader_end_seconds"],
                dev_eval_seconds,
                "データ読み込み律速"
                if wait_ratio >= DATA_BOUND_RATIO
                else "データ読み込み律速ではない",
            )

            save_checkpoint(
                run_dir / "checkpoint_last.pt",
                model,
                epoch=epoch,
                metrics=row,
                config=config,
                optimizer=optimizer,
                normalizer=normalizer,
            )
            if config.train.save_every_epoch:
                save_checkpoint(
                    run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    model,
                    epoch=epoch,
                    metrics=row,
                    config=config,
                    normalizer=normalizer,
                )
            if is_best:
                best_value = float(value)
                best_epoch = epoch
                best_metrics = row
                save_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    model,
                    epoch=epoch,
                    metrics=row,
                    config=config,
                    normalizer=normalizer,
                )
                (run_dir / "predictions_best.json").write_text(
                    json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
                )
            last_epoch = epoch
            if patience is not None and epochs_without_improvement >= patience:
                stop_reason = STOP_EARLY
                log.info(
                    "早期終了: %s が %dエポック連続で改善しなかった（エポック%dで終了）",
                    config.train.best_metric,
                    epochs_without_improvement,
                    epoch,
                )
                break

    if watcher.events:
        log.warning("MPSのCPUフォールバックが%d種類の演算で発生した", len(watcher.events))
    else:
        log.info("MPSのCPUフォールバックは検出されなかった")

    summary_payload = {
        "experiment_id": config.experiment_id,
        "stop_reason": stop_reason,
        "last_epoch": last_epoch,
        "start_epoch": resume.start_epoch,
        "epochs_this_run": len(history),
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "best_metric": config.train.best_metric,
        "best_mode": best_mode,
        "best_epoch": best_epoch,
        "best_metric_value": best_value,
        "epochs_without_improvement": epochs_without_improvement,
        "resume": resume.as_dict(),
        "total_epoch_seconds": sum(float(row["epoch_seconds"]) for row in history),
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info(
        "学習を終了した。終了理由=%s 到達エポック=%d（通算、上限%d） 最良エポック=%d %s=%.4f 出力=%s",
        "早期終了" if stop_reason == STOP_EARLY else "上限到達",
        last_epoch,
        max_epochs,
        best_epoch,
        config.train.best_metric,
        best_value,
        run_dir,
    )

    if config.train.record_metrics_csv and best_metrics:
        _append_metrics_csv(config, best_metrics, run_dir, log)

    return TrainingOutcome(
        run_dir=run_dir,
        best_epoch=best_epoch,
        best_metric_value=best_value,
        best_metrics=best_metrics,
        history=history,
        mps_fallbacks=watcher.events,
        stop_reason=stop_reason,
        last_epoch=last_epoch,
        start_epoch=resume.start_epoch,
        resume=resume.as_dict(),
    )


def _append_metrics_csv(
    config: TrainConfig, best: dict[str, Any], run_dir: Path, logger: logging.Logger
) -> None:
    """results/metrics.csv に最良エポックの検証結果を1行追記する。

    ``latency_ms_per_inference`` はここでは**バッチ処理での1クリップあたりの前向き計算
    時間**であり、docs/spec.md の「1推論あたりの処理時間」（2.0秒窓を1件ずつ推論した値）
    とは測り方が違う。厳密な値は第5段階5-3の評価で ``spkrate.eval.runner`` から測る。
    """
    from spkrate.eval.runner import MetricsRow, append_metrics_row, model_size_bytes

    metrics = SpeedRateMetrics(
        **{
            key: best[key]
            for key in SpeedRateMetrics.__dataclass_fields__
            if key in best
        }
    )
    append_metrics_row(
        MetricsRow(
            experiment_id=config.experiment_id,
            method="cnn",
            config_path=config.config_path or config.model_config,
            split="dev",
            metrics=metrics,
            latency_ms_per_inference=float(best.get("val_forward_ms_per_clip", float("nan"))),
            model_size_bytes=model_size_bytes(run_dir / "checkpoint_best.pt"),
        ),
        csv_path=config.train.metrics_csv,
    )
    logger.info("results/metrics.csv に追記した: %s", config.train.metrics_csv)


# --------------------------------------------------------------------------------------
# コマンドライン


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="話速推定CNNの学習（docs/PLAN.md 第5段階 5-2）"
    )
    parser.add_argument("--config", required=True, help="学習設定のyaml")
    parser.add_argument("--experiment-id", default=None, help="実験IDを上書きする")
    parser.add_argument(
        "--epochs", type=int, default=None, help="上限エポック数（通算）を上書きする"
    )
    parser.add_argument("--device", default=None, help="デバイスを上書きする（mps/cpu）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シードを上書きする")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_train_config(args.config)
    if args.experiment_id:
        config = config.with_changes(experiment_id=args.experiment_id)
    if args.device:
        config = config.with_changes(device=args.device)
    if args.epochs is not None:
        # 上限を max_epochs で書いた設定では max_epochs を上書きする。
        key = "max_epochs" if config.train.max_epochs is not None else "epochs"
        config = config.with_changes(
            train=TrainSettings(**{**asdict(config.train), key: args.epochs})
        )
    if args.seed is not None:
        config = config.with_changes(seed=args.seed)
    run_training(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
