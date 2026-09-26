"""デバイスの選択・精度の設定・記録のテスト（docs/directives/2026-09-26-rtx3060.md タスク2）。

固定する性質。

- 設定の ``device`` に書けるのは mps・cuda・cpu だけで、使えないデバイスを指定すると開始前に
  ``DeviceUnavailableError`` で止まる（黙って cpu に落とさない）。学習・アライメントも同じ
- cuda のときだけ TF32 を無効にし（``allow_tf32`` を False）、mps・cpu では設定を変えない
- DataLoader の ``pin_memory`` は cuda のときだけ有効になる
- config_snapshot.yaml と log.txt にホスト名・デバイス名・torch と CUDA の版が書かれる

cuda が要るテストは cuda の無い環境（Mac）では飛ばす。cuda の有無を偽るテストは
``torch.cuda.is_available`` などを monkeypatch で差し替える。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import yaml

from spkrate import device as device_utils
from spkrate.device import (
    DeviceUnavailableError,
    configure_backends,
    dataloader_device_kwargs,
    describe_environment,
    environment_info,
    host_label,
    resolve_device,
    setup_device,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda が無い")
requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="mps が無い")


@pytest.fixture
def restore_tf32():
    """TF32 の設定を試験の前の値に戻す（大域の設定なので他の試験に漏らさない）。"""
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    saved_precision = {
        name: target.fp32_precision
        for name, target in (
            ("matmul", torch.backends.cuda.matmul),
            ("conv", getattr(torch.backends.cudnn, "conv", None)),
            ("rnn", getattr(torch.backends.cudnn, "rnn", None)),
        )
        if target is not None and hasattr(target, "fp32_precision")
    }
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
    targets = {
        "matmul": torch.backends.cuda.matmul,
        "conv": getattr(torch.backends.cudnn, "conv", None),
        "rnn": getattr(torch.backends.cudnn, "rnn", None),
    }
    for name, value in saved_precision.items():
        targets[name].fp32_precision = value


def _pretend_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# --------------------------------------------------------------------------------------
# デバイスの選択


def test_resolve_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(" CPU ") == torch.device("cpu")


@pytest.mark.parametrize("name", ["auto", "gpu", "tpu", "xla", "cuda:x", ""])
def test_resolve_rejects_unknown_names(name: str) -> None:
    with pytest.raises(ValueError, match="device"):
        resolve_device(name)


@pytest.mark.parametrize("name", ["cuda", "cuda:0"])
def test_resolve_stops_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _pretend_no_cuda(monkeypatch)
    with pytest.raises(DeviceUnavailableError, match="cuda が使えない"):
        resolve_device(name)


def test_resolve_stops_when_cuda_index_is_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    assert resolve_device("cuda") == torch.device("cuda")
    with pytest.raises(DeviceUnavailableError, match="1 個"):
        resolve_device("cuda:1")


def test_resolve_stops_when_mps_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="mps が使えない"):
        resolve_device("mps")


@requires_mps
def test_resolve_mps_when_available() -> None:
    assert resolve_device("mps") == torch.device("mps")


@requires_cuda
def test_resolve_cuda_when_available() -> None:
    assert resolve_device("cuda").type == "cuda"


def test_training_stops_before_start_when_device_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spkrate.train.train import load_train_config, run_training

    _pretend_no_cuda(monkeypatch)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "no_cuda",
                "device": "cuda",
                "runs_dir": str(tmp_path / "runs"),
                "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "log_interval": 0},
            }
        ),
        encoding="utf-8",
    )
    config = load_train_config(path)
    with pytest.raises(DeviceUnavailableError):
        run_training(config, train_dataset=[], dev_dataset=[])  # データに触れる前に止まる
    run_dir = tmp_path / "runs" / "no_cuda"
    assert "cuda が使えない" in (run_dir / "log.txt").read_text(encoding="utf-8")
    assert not (run_dir / "config_snapshot.yaml").exists()
    assert not (run_dir / "metrics.jsonl").exists()


def test_aligner_stops_before_loading_model_when_device_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spkrate.labels import alignment

    _pretend_no_cuda(monkeypatch)

    def must_not_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("デバイスの検査より前にモデルを読んだ")

    monkeypatch.setattr("transformers.Wav2Vec2ForCTC.from_pretrained", must_not_load)
    with pytest.raises(DeviceUnavailableError):
        alignment.HiraganaAligner("cuda")


# --------------------------------------------------------------------------------------
# 精度の設定（TF32）


def test_configure_backends_disables_tf32_for_cuda(restore_tf32) -> None:  # noqa: ANN001
    # 設定の書き換えだけなので cuda の無い環境でも確かめられる。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    values = configure_backends(torch.device("cuda"))
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert values["cuda_matmul_allow_tf32"] is False
    assert values["cudnn_allow_tf32"] is False
    assert values["float32_matmul_precision"] == "highest"
    for key in ("cuda_matmul_fp32_precision", "cudnn_conv_fp32_precision",
                "cudnn_rnn_fp32_precision"):
        if key in values:
            assert values[key] == "ieee"


@pytest.mark.parametrize("name", ["cpu", "mps"])
def test_configure_backends_leaves_other_devices_alone(restore_tf32, name: str) -> None:  # noqa: ANN001
    torch.backends.cudnn.allow_tf32 = True
    assert configure_backends(torch.device(name)) == {}
    assert torch.backends.cudnn.allow_tf32 is True


@requires_cuda
def test_setup_device_on_cuda_logs_tf32_and_gpu(restore_tf32, caplog) -> None:  # noqa: ANN001
    logger = logging.getLogger("test_device_cuda")
    with caplog.at_level(logging.INFO, logger="test_device_cuda"):
        device, record = setup_device("cuda", logger)
    assert device.type == "cuda"
    assert record["device_name"] == torch.cuda.get_device_name(device)
    assert record["cuda"] == torch.version.cuda
    assert record["backends"]["cuda_matmul_allow_tf32"] is False
    assert record["backends"]["cudnn_allow_tf32"] is False
    text = caplog.text
    assert "TF32 を無効にした" in text
    assert torch.cuda.get_device_name(device) in text
    # float32 のまま計算できる（float64・混合精度は使わない）
    x = torch.randn(4, 8, device=device)
    assert (x @ x.T).dtype == torch.float32


# --------------------------------------------------------------------------------------
# DataLoader


def test_pin_memory_only_for_cuda() -> None:
    assert dataloader_device_kwargs(torch.device("cpu")) == {"pin_memory": False}
    assert dataloader_device_kwargs(torch.device("mps")) == {"pin_memory": False}
    assert dataloader_device_kwargs(torch.device("cuda")) == {"pin_memory": True}


def test_training_loader_pins_memory_only_for_cuda() -> None:
    from spkrate.train.train import TrainSettings, _make_loader

    settings = TrainSettings(num_workers=0, bucketing=False)
    dataset = list(range(4))
    kwargs = {"batch_size": 2, "shuffle": False, "settings": settings, "seed": 0}
    assert _make_loader(dataset, **kwargs).pin_memory is False
    assert _make_loader(dataset, device=torch.device("cpu"), **kwargs).pin_memory is False
    assert _make_loader(dataset, device=torch.device("mps"), **kwargs).pin_memory is False
    assert _make_loader(dataset, device=torch.device("cuda"), **kwargs).pin_memory is True


# --------------------------------------------------------------------------------------
# 記録


def test_environment_info_for_cpu() -> None:
    info = environment_info(torch.device("cpu"))
    assert info["hostname"]
    assert info["host_label"] == host_label()
    assert info["device"] == "cpu"
    assert info["device_type"] == "cpu"
    assert info["device_name"]
    assert info["torch"] == str(torch.__version__)
    assert info["cuda"] == torch.version.cuda
    line = describe_environment(info)
    assert info["hostname"] in line
    assert "torch=" in line and "CUDA=" in line


def test_host_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(device_utils.HOST_ENV_VAR, raising=False)
    monkeypatch.setattr(device_utils.sys, "platform", "darwin")
    assert host_label() == "mac"
    monkeypatch.setattr(device_utils.sys, "platform", "linux")
    monkeypatch.setattr(device_utils.socket, "gethostname", lambda: "xps")
    assert host_label() == "ubuntu-desktop"
    monkeypatch.setattr(device_utils.socket, "gethostname", lambda: "other.example.lan")
    assert host_label() == "other"
    monkeypatch.setenv(device_utils.HOST_ENV_VAR, "custom")
    assert host_label() == "custom"


def test_training_records_host_and_device(tmp_path: Path) -> None:
    from spkrate.models.cnn import CnnConfig
    from spkrate.train.train import load_train_config, write_config_snapshot

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"experiment_id": "rec", "device": "cpu", "runs_dir": str(tmp_path)}),
        encoding="utf-8",
    )
    config = load_train_config(path)
    snapshot_path = write_config_snapshot(
        config,
        CnnConfig(
            n_mels=80,
            freq_channels=(4, 4),
            freq_strides=(4, 4),
            freq_kernel=(3, 3),
            temporal_channels=(8, 8),
            dilations=(1, 2),
            temporal_kernel=3,
        ),
        run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    host = snapshot["host"]
    assert host["hostname"] == environment_info(torch.device("cpu"))["hostname"]
    assert host["device"] == "cpu"
    assert host["device_name"]
    assert host["torch"] == str(torch.__version__)
    assert "cuda" in host and "cuda" in snapshot["environment"]


@requires_cuda
def test_training_loader_pins_batches_on_cuda() -> None:
    from spkrate.train.data import ClipItem, collate_clips
    from spkrate.train.train import TrainSettings, _make_loader

    items = [
        ClipItem(features=torch.randn(10 + i, 80).numpy(), mora=float(i), duration_sec=1.0,
                 clip_id=f"c{i}")
        for i in range(4)
    ]
    loader = _make_loader(
        items, batch_size=2, shuffle=False, settings=TrainSettings(num_workers=0, bucketing=False),
        seed=0, device=torch.device("cuda"),
    )
    batch = next(iter(loader))
    assert batch.features.is_pinned() and batch.moras.is_pinned()
    assert batch.clip_ids == ("c0", "c1")
    assert collate_clips(items[:2]).features.is_pinned() is False
