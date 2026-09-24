"""学習ループの単体テスト（docs/PLAN.md 第5段階 5-2）。

**実データには依存しない。** 特徴量は乱数で作り、事前計算特徴量を読む経路も
この場で小さなシャードを書いて試す。デバイスは cpu を使う（mps の有無に依存させない）。

固定する性質。

- 損失の切り替えが設定で効くこと（``mse`` / ``poisson``、未知の名前は誤り）
- 詰め物付きのバッチでも損失が1件ずつ計算した値と一致すること
- チェックポイントの保存と読み込みで重みと構造が戻ること
- ``config_snapshot.yaml`` と ``metrics.jsonl`` が所定の形式で書かれること

## 許容誤差

詰め物の有無で値が一致することを見るテストは ``rtol=1e-4, atol=1e-4`` とする。
理由は tests/test_cnn.py の docstring と同じで、畳み込みの実装が系列長によって
別の足し合わせ順序を選ぶため float32 の丸めの差が残りうるからである。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from spkrate.eval.metrics import METRIC_COLUMNS
from spkrate.models.cnn import CnnConfig, SpeechRateCNN
from spkrate.train.data import (
    ClipItem,
    FeatureClipDataset,
    LengthBucketBatchSampler,
    Normalizer,
    collate_clips,
    load_normalization,
)
from spkrate.train.train import (
    LOSSES,
    POISSON_EPS,
    MpsFallbackWatcher,
    TrainConfig,
    build_loss,
    load_checkpoint,
    load_train_config,
    run_training,
    save_checkpoint,
    write_config_snapshot,
)

N_MELS = 80


# --------------------------------------------------------------------------------------
# 合成データ


def _small_model_config() -> CnnConfig:
    """小さなモデル（テストを速くするため。構造は本番と同じ形）。"""
    return CnnConfig(
        n_mels=N_MELS,
        freq_channels=(4, 4),
        freq_strides=(4, 4),
        freq_kernel=(3, 3),
        temporal_channels=(8, 8),
        dilations=(1, 2),
        temporal_kernel=3,
    )


class SyntheticClipDataset(torch.utils.data.Dataset):
    """乱数の特徴量と、それと相関のあるモーラ数を返すデータセット。"""

    def __init__(self, num_clips: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.items: list[ClipItem] = []
        for index in range(num_clips):
            n_frames = int(rng.integers(120, 400))
            duration = n_frames / 100.0
            rate = float(rng.uniform(2.0, 9.0))
            self.items.append(
                ClipItem(
                    features=rng.standard_normal((n_frames, N_MELS)).astype(np.float32),
                    mora=round(rate * duration),
                    duration_sec=duration,
                    clip_id=f"synthetic_{index:04d}",
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ClipItem:
        return self.items[index]

    @property
    def frame_counts(self) -> list[int]:
        return [int(item.features.shape[0]) for item in self.items]


def _write_normalization(path: Path) -> Path:
    payload = {
        "mode": "per_mel",
        "global": {"mean": -7.0, "std": 4.0},
        "per_mel": {"n_mels": N_MELS, "mean": [-7.0] * N_MELS, "std": [4.0] * N_MELS},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_feature_shard(directory: Path, *, num_clips: int = 6) -> Path:
    """事前計算特徴量（docs/decisions/004-feature-storage.md の形式）を小さく作る。"""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    frames = [int(rng.integers(30, 90)) for _ in range(num_clips)]
    array = rng.standard_normal((sum(frames), N_MELS)).astype(np.float16)
    np.save(directory / "shard_0000.npy", array)
    clips = []
    offset = 0
    for index, n_frames in enumerate(frames):
        clips.append(
            {
                "clip_id": f"clip_{index}",
                "client_id": "speaker_a" if index % 2 == 0 else "speaker_b",
                "offset": offset,
                "n_frames": n_frames,
                "mora": index + 5,
                "duration_sec": n_frames / 100.0,
                "audio_duration_sec": n_frames / 100.0,
            }
        )
        offset += n_frames
    (directory / "shard_0000.json").write_text(
        json.dumps({"complete": True, "array": "shard_0000.npy", "clips": clips}),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "split": "dev",
                "num_clips": num_clips,
                "shards": [{"array": "shard_0000.npy", "index": "shard_0000.json"}],
            }
        ),
        encoding="utf-8",
    )
    return directory


def _write_train_config(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "experiment_id": "test_run",
        "seed": 123,
        "device": "cpu",
        "runs_dir": str(tmp_path / "runs"),
        "model_config": "configs/model/cnn_base.yaml",
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "loss": "mse",
            "learning_rate": 0.001,
            "num_workers": 0,
            "log_interval": 0,
        },
    }
    payload.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)  # 試験ごとに別の下位ディレクトリを渡せるように
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# 損失の切り替え


def test_build_loss_returns_requested_loss() -> None:
    prediction = torch.tensor([3.0, 5.0])
    target = torch.tensor([2.0, 6.0])

    mse = build_loss("mse")(prediction, target)
    poisson = build_loss("poisson")(prediction, target)

    expected_mse = ((3.0 - 2.0) ** 2 + (5.0 - 6.0) ** 2) / 2
    expected_poisson = (
        (3.0 - 2.0 * np.log(3.0 + POISSON_EPS)) + (5.0 - 6.0 * np.log(5.0 + POISSON_EPS))
    ) / 2
    assert mse.item() == pytest.approx(expected_mse, rel=1e-6)
    assert poisson.item() == pytest.approx(expected_poisson, rel=1e-6)
    assert mse.item() != pytest.approx(poisson.item())


def test_build_loss_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="loss"):
        build_loss("huber")
    assert LOSSES == ("mse", "poisson")


def test_poisson_loss_is_finite_when_prediction_is_zero() -> None:
    """λ が0に潰れても ``eps`` で ``log(0)`` を避ける（数値安定性）。"""
    loss = build_loss("poisson")(torch.tensor([0.0]), torch.tensor([3.0]))
    assert torch.isfinite(loss)


def test_loss_switch_comes_from_config(tmp_path: Path) -> None:
    """設定ファイルの ``train.loss`` がそのまま使われること。"""
    for name in LOSSES:
        path = _write_train_config(
            tmp_path / name,
            train={"epochs": 1, "batch_size": 4, "loss": name, "num_workers": 0},
        )
        config = load_train_config(path)
        assert config.train.loss == name
        assert type(build_loss(config.train.loss)) is type(build_loss(name))


def test_unknown_loss_in_config_stops_training(tmp_path: Path) -> None:
    path = _write_train_config(
        tmp_path, train={"epochs": 1, "batch_size": 2, "loss": "mae", "num_workers": 0}
    )
    config = load_train_config(path)
    with pytest.raises(ValueError, match="loss"):
        run_training(
            config,
            train_dataset=SyntheticClipDataset(4),
            dev_dataset=SyntheticClipDataset(4, seed=1),
        )


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = _write_train_config(tmp_path, tarin={"epochs": 1})
    with pytest.raises(ValueError, match="未知の設定項目"):
        load_train_config(path)


# --------------------------------------------------------------------------------------
# 詰め物付きのバッチ


def test_collate_pads_and_reports_lengths() -> None:
    items = [
        ClipItem(np.ones((10, N_MELS), np.float32), 5.0, 0.1, "a"),
        ClipItem(np.ones((30, N_MELS), np.float32), 9.0, 0.3, "b"),
    ]
    batch = collate_clips(items)

    assert batch.features.shape == (2, 30, N_MELS)
    assert batch.features.dtype == torch.float32
    assert torch.equal(batch.lengths, torch.tensor([10, 30]))
    assert torch.all(batch.features[0, 10:] == 0.0)
    assert batch.clip_ids == ("a", "b")


@pytest.mark.parametrize("loss_name", ["mse", "poisson"])
def test_loss_on_padded_batch_matches_per_clip_loss(loss_name: str) -> None:
    """詰め物付きのバッチで計算した損失が、1件ずつ計算した平均と一致すること。

    詰め物のフレームが総和に混ざると予測が水増しされ、損失がずれる。
    この一致が、``lengths`` によるマスクが効いていることの担保になる。
    """
    torch.manual_seed(0)
    model = SpeechRateCNN(_small_model_config()).eval()
    loss_fn = build_loss(loss_name)
    rng = np.random.default_rng(3)
    items = [
        ClipItem(
            rng.standard_normal((n_frames, N_MELS)).astype(np.float32),
            mora=float(mora),
            duration_sec=n_frames / 100.0,
            clip_id=f"c{index}",
        )
        for index, (n_frames, mora) in enumerate([(40, 8), (120, 25), (77, 15)])
    ]
    batch = collate_clips(items)

    with torch.no_grad():
        batched = model(batch.features, batch.lengths)
        per_clip = torch.stack(
            [
                model(torch.from_numpy(item.features).unsqueeze(0)).squeeze(0)
                for item in items
            ]
        )
        batched_loss = loss_fn(batched, batch.moras)
        per_clip_loss = torch.stack(
            [
                loss_fn(per_clip[index : index + 1], batch.moras[index : index + 1])
                for index in range(len(items))
            ]
        ).mean()

    torch.testing.assert_close(batched, per_clip, rtol=1e-4, atol=1e-4)
    assert batched_loss.item() == pytest.approx(per_clip_loss.item(), rel=1e-4, abs=1e-4)


def test_padding_does_not_inflate_prediction() -> None:
    """詰め物を足しても予測モーラ数が増えないこと（005の4.2節）。"""
    torch.manual_seed(1)
    model = SpeechRateCNN(_small_model_config()).eval()
    rng = np.random.default_rng(11)
    short = ClipItem(
        rng.standard_normal((50, N_MELS)).astype(np.float32), 10.0, 0.5, "short"
    )
    long = ClipItem(
        rng.standard_normal((400, N_MELS)).astype(np.float32), 80.0, 4.0, "long"
    )
    with torch.no_grad():
        alone = model(torch.from_numpy(short.features).unsqueeze(0))
        batch = collate_clips([short, long])
        together = model(batch.features, batch.lengths)
    torch.testing.assert_close(alone[0], together[0], rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------------
# 長さでまとめるバッチ分け


def test_bucket_sampler_covers_every_index_once() -> None:
    counts = [10, 500, 20, 300, 15, 280, 12, 450, 60, 70]
    sampler = LengthBucketBatchSampler(counts, batch_size=3, shuffle=True, pool_batches=2, seed=5)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == list(range(len(counts)))
    assert len(batches) == len(sampler)


def test_bucket_sampler_groups_similar_lengths() -> None:
    """長さでまとめると詰め物の総量が無作為な組み方より減ること。"""
    rng = np.random.default_rng(0)
    counts = [int(value) for value in rng.integers(110, 1330, size=120)]

    def padded_frames(batches: list[list[int]]) -> int:
        return sum(max(counts[i] for i in batch) * len(batch) for batch in batches)

    bucketed = list(
        LengthBucketBatchSampler(counts, 8, shuffle=True, pool_batches=8, seed=1)
    )
    shuffled = list(
        LengthBucketBatchSampler(counts, 8, shuffle=True, pool_batches=1, seed=1)
    )
    assert padded_frames(bucketed) < padded_frames(shuffled)


# --------------------------------------------------------------------------------------
# 正規化と事前計算特徴量の読み出し


def test_normalizer_uses_fixed_values(tmp_path: Path) -> None:
    normalizer = load_normalization(_write_normalization(tmp_path / "norm.yaml"))
    feature = np.full((3, N_MELS), -3.0, dtype=np.float32)
    normalized = normalizer(feature)

    assert normalizer.mode == "per_mel"
    assert normalized.dtype == np.float32
    assert normalized == pytest.approx(np.full((3, N_MELS), 1.0, dtype=np.float32))


def test_normalizer_rejects_zero_std() -> None:
    with pytest.raises(ValueError, match="標準偏差"):
        Normalizer(np.zeros(N_MELS), np.zeros(N_MELS), mode="per_mel")


def test_feature_dataset_reads_shards_and_filters_speakers(tmp_path: Path) -> None:
    directory = _write_feature_shard(tmp_path / "features" / "dev")
    normalizer = load_normalization(_write_normalization(tmp_path / "norm.yaml"))

    dataset = FeatureClipDataset(directory, client_ids=["speaker_a"], normalizer=normalizer)
    item = dataset[0]

    assert len(dataset) == 3  # 偶数番号の3件だけが speaker_a
    assert item.features.dtype == np.float32
    assert item.features.shape[1] == N_MELS
    assert item.features.shape[0] == dataset.frame_counts[0]
    assert item.duration_sec > 0.0
    assert item.mora == pytest.approx(5.0)


def test_feature_dataset_limit(tmp_path: Path) -> None:
    directory = _write_feature_shard(tmp_path / "features" / "dev")
    assert len(FeatureClipDataset(directory, limit=2)) == 2


# --------------------------------------------------------------------------------------
# チェックポイント


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _small_model_config()
    model = SpeechRateCNN(config)
    path = save_checkpoint(
        tmp_path / "checkpoint_best.pt", model, epoch=4, metrics={"mae_moras_per_sec": 0.5}
    )

    restored, payload = load_checkpoint(path)

    assert payload["epoch"] == 4
    assert payload["metrics"]["mae_moras_per_sec"] == pytest.approx(0.5)
    assert restored.config == config
    for (name, original), (other_name, loaded) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(original, loaded)

    features = torch.randn(1, 60, N_MELS)
    with torch.no_grad():
        torch.testing.assert_close(model(features), restored(features))


# --------------------------------------------------------------------------------------
# config_snapshot.yaml


def test_config_snapshot_has_commit_and_config(tmp_path: Path) -> None:
    config = load_train_config(_write_train_config(tmp_path))
    model_config = _small_model_config()

    path = write_config_snapshot(
        config, model_config, run_dir=tmp_path / "run", device=torch.device("cpu")
    )
    snapshot = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert path.name == "config_snapshot.yaml"
    assert snapshot["experiment_id"] == "test_run"
    assert snapshot["seed"] == config.seed
    assert snapshot["device"] == "cpu"
    assert set(snapshot["git"]) == {"commit_hash", "dirty"}
    assert snapshot["config"]["train"]["loss"] == "mse"
    assert snapshot["model"]["n_mels"] == N_MELS
    assert snapshot["model_summary"]["receptive_field_frames"] >= 1
    assert "config_file_text" in snapshot
    assert snapshot["environment"]["torch"] == torch.__version__


# --------------------------------------------------------------------------------------
# 学習の完走と metrics.jsonl


def _run_short_training(tmp_path: Path, **overrides: object) -> tuple[TrainConfig, object]:
    config = load_train_config(_write_train_config(tmp_path, **overrides))
    config = config.with_changes(model_overrides=_small_model_config().as_dict())
    outcome = run_training(
        config,
        train_dataset=SyntheticClipDataset(12, seed=0),
        dev_dataset=SyntheticClipDataset(8, seed=1),
    )
    return config, outcome


def test_training_writes_metrics_jsonl_and_checkpoints(tmp_path: Path) -> None:
    config, outcome = _run_short_training(tmp_path)
    run_dir = Path(config.runs_dir) / config.experiment_id

    assert run_dir == outcome.run_dir
    assert (run_dir / "log.txt").is_file()
    assert (run_dir / "config_snapshot.yaml").is_file()
    assert (run_dir / "checkpoint_last.pt").is_file()
    assert (run_dir / "checkpoint_best.pt").is_file()

    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == config.train.epochs
    rows = [json.loads(line) for line in lines]
    for epoch, row in enumerate(rows, start=1):
        assert row["epoch"] == epoch
        assert row["experiment_id"] == config.experiment_id
        assert row["loss"] == config.train.loss
        # 検証セットの全指標が入っていること（docs/spec.md「評価指標」）
        for column in METRIC_COLUMNS:
            assert column in row
        assert row["num_segments"] == 8
        assert isinstance(row["is_best"], bool)
        assert row["train_loss"] >= 0.0

    assert any(row["is_best"] for row in rows)
    assert outcome.best_epoch >= 1
    assert outcome.best_metrics["epoch"] == outcome.best_epoch


def test_training_best_checkpoint_matches_best_epoch(tmp_path: Path) -> None:
    config, outcome = _run_short_training(tmp_path)
    _, payload = load_checkpoint(outcome.run_dir / "checkpoint_best.pt")

    assert payload["epoch"] == outcome.best_epoch
    assert payload["metrics"]["mae_moras_per_sec"] == pytest.approx(
        outcome.best_metric_value
    )
    assert payload["train_config"]["experiment_id"] == config.experiment_id


def test_training_with_poisson_loss_runs(tmp_path: Path) -> None:
    _, outcome = _run_short_training(
        tmp_path,
        train={"epochs": 1, "batch_size": 4, "loss": "poisson", "num_workers": 0, "log_interval": 0},
    )
    assert len(outcome.history) == 1
    assert outcome.history[0]["loss"] == "poisson"
    assert np.isfinite(outcome.history[0]["train_loss"])


def test_training_reports_no_mps_fallback_on_cpu(tmp_path: Path) -> None:
    _, outcome = _run_short_training(tmp_path)
    assert outcome.mps_fallbacks == []


# --------------------------------------------------------------------------------------
# MPSのCPUフォールバックの記録


def test_mps_fallback_watcher_records_operator() -> None:
    import warnings

    with MpsFallbackWatcher() as watcher:
        warnings.warn(
            "The operator 'aten::_fft_r2c' is not currently supported on the MPS backend "
            "and will fall back to run on the CPU.",
            UserWarning,
            stacklevel=1,
        )
        warnings.warn("関係のない警告", UserWarning, stacklevel=1)

    assert len(watcher.events) == 1
    assert watcher.events[0]["operator"] == "aten::_fft_r2c"
    assert watcher.events[0]["location"]


# --------------------------------------------------------------------------------------
# テストセットの拒否


def test_training_config_cannot_point_dev_split_at_test_json(tmp_path: Path) -> None:
    """dev_split に test.json を書いても読み出しで止まること（docs/PLAN.md 禁止事項）。"""
    from spkrate.train.data import select_clip_records

    with pytest.raises(ValueError, match="テストセット"):
        select_clip_records(
            tmp_path / "clips.jsonl", Path("configs/splits/test.json"), limit=1
        )


# --------------------------------------------------------------------------------------
# 早期終了・再開・所要時間の計測


def _run_with(tmp_path: Path, name: str, *, train: dict, **overrides: object):
    """合成データで学習する。``train`` は train 節をまるごと与える。"""
    base_train = {"batch_size": 4, "loss": "mse", "num_workers": 0, "log_interval": 0}
    config = load_train_config(
        _write_train_config(
            tmp_path / name, experiment_id=name, train={**base_train, **train}, **overrides
        )
    )
    config = config.with_changes(
        runs_dir=str(tmp_path / "runs"), model_overrides=_small_model_config().as_dict()
    )
    return config, run_training(
        config,
        train_dataset=SyntheticClipDataset(12, seed=0),
        dev_dataset=SyntheticClipDataset(8, seed=1),
    )


def _log_text(outcome) -> str:
    return (outcome.run_dir / "log.txt").read_text(encoding="utf-8")


def test_early_stopping_stops_after_patience(tmp_path: Path) -> None:
    # 学習率0なら重みが変わらず dev の値は毎エポック同じ。エポック1が基準になり、
    # エポック2・3で改善なしが2回続くので patience=2 ならエポック3で止まる。
    _, outcome = _run_with(
        tmp_path,
        "es",
        train={"max_epochs": 10, "learning_rate": 0.0, "early_stopping_patience": 2},
    )
    assert outcome.stop_reason == "early_stopping"
    assert outcome.last_epoch == 3
    assert [row["epoch"] for row in outcome.history] == [1, 2, 3]
    assert [row["epochs_without_improvement"] for row in outcome.history] == [0, 1, 2]
    assert outcome.best_epoch == 1
    summary = json.loads((outcome.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "early_stopping"
    assert summary["last_epoch"] == 3
    assert summary["best_epoch"] == 1
    assert "終了理由=早期終了" in _log_text(outcome)


def test_min_delta_counts_small_gains_as_no_improvement(tmp_path: Path) -> None:
    # 最小改善幅を非常に大きくすれば、学習で値が下がってもエポック2以降は改善とみなさない。
    _, outcome = _run_with(
        tmp_path,
        "delta",
        train={
            "max_epochs": 5,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 1e6,
        },
    )
    assert outcome.stop_reason == "early_stopping"
    assert outcome.last_epoch == 2


def test_training_stops_at_max_epochs(tmp_path: Path) -> None:
    _, outcome = _run_with(
        tmp_path,
        "maxep",
        train={"max_epochs": 3, "learning_rate": 0.0, "early_stopping_patience": 5},
    )
    assert outcome.stop_reason == "max_epochs"
    assert outcome.last_epoch == 3
    assert len(outcome.history) == 3
    assert "終了理由=上限到達" in _log_text(outcome)


def test_epochs_and_max_epochs_cannot_both_be_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="同時に書けない"):
        load_train_config(
            _write_train_config(tmp_path, train={"epochs": 3, "max_epochs": 5})
        )


def test_new_settings_default_to_disabled() -> None:
    config = TrainConfig.from_mapping({"experiment_id": "x", "train": {"epochs": 6}})
    assert config.train.early_stopping_patience is None
    assert config.train.early_stopping_min_delta == 0.0
    assert config.train.resolved_max_epochs == 6
    assert config.resume_from is None


def _adam_steps(path: Path) -> list[float]:
    _, payload = load_checkpoint(path)
    return [float(state["step"]) for state in payload["optimizer_state"]["state"].values()]


def test_resume_continues_epoch_numbers_and_optimizer_state(tmp_path: Path) -> None:
    _, first = _run_with(tmp_path, "src", train={"max_epochs": 2, "learning_rate": 0.001})
    source = first.run_dir / "checkpoint_last.pt"
    # 12件・バッチ4で1エポック3ステップ。2エポックで Adam の step は6。
    assert set(_adam_steps(source)) == {6.0}

    _, resumed = _run_with(
        tmp_path,
        "dst",
        train={"max_epochs": 4, "learning_rate": 0.001},
        resume_from=str(source),
    )
    assert resumed.start_epoch == 3
    assert [row["epoch"] for row in resumed.history] == [3, 4]
    assert resumed.last_epoch == 4
    assert resumed.stop_reason == "max_epochs"
    # 最適化器の状態を引き継いだので、step は 6 + 2エポック×3 = 12。
    assert set(_adam_steps(resumed.run_dir / "checkpoint_last.pt")) == {12.0}
    # 最良値は測り直す（再開後の最初のエポックが基準）。
    assert resumed.history[0]["is_best"] is True
    assert resumed.history[0]["epochs_without_improvement"] == 0
    log = _log_text(resumed)
    assert f"再開元={source}" in log
    assert "再開元のエポック=2" in log
    assert "最良値は引き継がず測り直す" in log
    assert "乱数は seed=" in log
    summary = json.loads((resumed.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["resume"]["resumed"] is True
    assert summary["resume"]["source_epoch"] == 2
    assert summary["resume"]["best_value_policy"] == "remeasure"


def test_resume_loads_model_weights(tmp_path: Path) -> None:
    _, first = _run_with(tmp_path, "src", train={"max_epochs": 1, "learning_rate": 0.001})
    source = first.run_dir / "checkpoint_last.pt"
    # 学習率0で再開すれば重みは変わらないので、再開元と同じ重みが保存される。
    _, resumed = _run_with(
        tmp_path, "dst", train={"max_epochs": 2, "learning_rate": 0.0}, resume_from=str(source)
    )
    before, _ = load_checkpoint(source)
    after, _ = load_checkpoint(resumed.run_dir / "checkpoint_last.pt")
    for key, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[key]), key


def test_resume_without_optimizer_state_starts_from_scratch(tmp_path: Path) -> None:
    _, first = _run_with(tmp_path, "src", train={"max_epochs": 1, "learning_rate": 0.001})
    # checkpoint_best.pt は optimizer_state を含まない。
    source = first.run_dir / "checkpoint_best.pt"
    _, payload = load_checkpoint(source)
    assert "optimizer_state" not in payload

    _, outcome = _run_with(
        tmp_path, "dst", train={"max_epochs": 2, "learning_rate": 0.001}, resume_from=str(source)
    )
    assert outcome.start_epoch == 1
    assert [row["epoch"] for row in outcome.history] == [1, 2]
    assert outcome.resume["resumed"] is False
    log = _log_text(outcome)
    assert "optimizer_state が無いため再開せず、最初から学習する" in log


def test_resume_rejects_model_structure_mismatch(tmp_path: Path) -> None:
    from spkrate.train.train import ResumeError

    _, first = _run_with(tmp_path, "src", train={"max_epochs": 1})
    other = {**_small_model_config().as_dict(), "temporal_channels": [16, 16]}
    config = load_train_config(
        _write_train_config(
            tmp_path / "dst",
            experiment_id="dst",
            resume_from=str(first.run_dir / "checkpoint_last.pt"),
        )
    ).with_changes(runs_dir=str(tmp_path / "runs"), model_overrides=other)
    with pytest.raises(ResumeError, match="モデル構造が違う"):
        run_training(
            config,
            train_dataset=SyntheticClipDataset(12, seed=0),
            dev_dataset=SyntheticClipDataset(8, seed=1),
        )


def test_resume_rejects_normalization_mismatch(tmp_path: Path) -> None:
    from spkrate.train.train import ResumeError

    _, first = _run_with(tmp_path, "src", train={"max_epochs": 1})
    norm = _write_normalization(tmp_path / "other_norm.yaml")
    with pytest.raises(ResumeError, match="正規化"):
        _run_with(
            tmp_path,
            "dst",
            train={"max_epochs": 2},
            resume_from=str(first.run_dir / "checkpoint_last.pt"),
            data={"normalization": str(norm)},
        )


def test_resume_rejects_missing_file_and_finished_source(tmp_path: Path) -> None:
    from spkrate.train.train import ResumeError

    with pytest.raises(ResumeError, match="無い"):
        _run_with(tmp_path, "a", train={"max_epochs": 2}, resume_from=str(tmp_path / "no.pt"))
    _, first = _run_with(tmp_path, "src", train={"max_epochs": 2})
    with pytest.raises(ResumeError, match="上限"):
        _run_with(
            tmp_path,
            "b",
            train={"max_epochs": 2},
            resume_from=str(first.run_dir / "checkpoint_last.pt"),
        )
    # 再開元が今回の出力先の中にあるときも止める。
    with pytest.raises(ResumeError, match="出力先"):
        _run_with(
            tmp_path,
            "src",
            train={"max_epochs": 4},
            resume_from=str(first.run_dir / "checkpoint_last.pt"),
        )


def test_timing_columns_are_written(tmp_path: Path) -> None:
    _, outcome = _run_with(tmp_path, "timing", train={"max_epochs": 1})
    row = json.loads(
        (outcome.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    for key in (
        "epoch_seconds",
        "train_seconds",
        "train_step_seconds",
        "data_wait_seconds",
        "data_wait_ratio",
        "loader_end_seconds",
        "dev_eval_seconds",
    ):
        assert key in row and row[key] >= 0.0, key
    assert 0.0 <= row["data_wait_ratio"] <= 1.0
    parts = row["train_step_seconds"] + row["data_wait_seconds"] + row["loader_end_seconds"]
    assert parts == pytest.approx(row["train_seconds"], abs=1e-3)
    assert row["epoch_seconds"] >= row["train_seconds"] + row["dev_eval_seconds"] - 1e-6
    log = _log_text(outcome)
    assert "所要時間: 全体=" in log and "データ待ち=" in log and "dev評価=" in log
    assert "データ読み込み律速" in log


# --------------------------------------------------------------------------------------
# エポックごとの個別チェックポイント保存（train.save_every_epoch）


def test_save_every_epoch_defaults_to_true() -> None:
    config = TrainConfig.from_mapping({"experiment_id": "x", "train": {"epochs": 6}})
    assert config.train.save_every_epoch is True


def test_save_every_epoch_writes_one_checkpoint_per_epoch(tmp_path: Path) -> None:
    _, outcome = _run_with(
        tmp_path, "everyep", train={"max_epochs": 4, "save_every_epoch": True}
    )
    checkpoints_dir = outcome.run_dir / "checkpoints"
    files = sorted(p.name for p in checkpoints_dir.glob("epoch_*.pt"))
    assert files == ["epoch_001.pt", "epoch_002.pt", "epoch_003.pt", "epoch_004.pt"]

    # checkpoint_last.pt / checkpoint_best.pt の既存の保存は変わらず行われる。
    assert (outcome.run_dir / "checkpoint_last.pt").is_file()
    assert (outcome.run_dir / "checkpoint_best.pt").is_file()

    # 保存の中身は save_checkpoint の既定形式（model_state, model_config, epoch,
    # metrics, train_config）に合わせる。checkpoint_best.pt と同様に optimizer_state は
    # 含まない。
    model, payload = load_checkpoint(checkpoints_dir / "epoch_003.pt")
    assert payload["epoch"] == 3
    assert payload["metrics"]["epoch"] == 3
    assert payload["train_config"]["experiment_id"] == "everyep"
    assert "model_config" in payload
    assert "optimizer_state" not in payload
    assert isinstance(model, SpeechRateCNN)

    # 各エポックの中身がそのエポックの記録と一致すること（上書きではなく個別に保存されている）。
    rows = [
        json.loads(line)
        for line in (outcome.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        _, epoch_payload = load_checkpoint(checkpoints_dir / f"epoch_{row['epoch']:03d}.pt")
        assert epoch_payload["metrics"]["mae_moras_per_sec"] == pytest.approx(
            row["mae_moras_per_sec"]
        )


def test_save_every_epoch_false_skips_individual_checkpoints(tmp_path: Path) -> None:
    _, outcome = _run_with(
        tmp_path, "noeveryep", train={"max_epochs": 3, "save_every_epoch": False}
    )
    assert not (outcome.run_dir / "checkpoints").exists()
    # 既存の保存には影響しない。
    assert (outcome.run_dir / "checkpoint_last.pt").is_file()
    assert (outcome.run_dir / "checkpoint_best.pt").is_file()
