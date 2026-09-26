"""計算デバイスの選択・設定・記録（mps・cuda・cpu）。

docs/directives/2026-09-26-rtx3060.md の0節2・3とタスク2による。学習デバイスは mps（Mac）
または cuda（ubuntu-desktop、RTX 3060）である。学習・評価・アライメント・無発話判定は、
このモジュールの関数でデバイスを選び、設定し、記録する。

## デバイスの選択（``resolve_device``）

- 書けるのは ``mps``・``cuda``（``cuda:0`` のような番号付きも可）・``cpu`` だけである
- 指定したデバイスが使えない場合は ``DeviceUnavailableError`` で**開始前に止める**。
  黙って cpu に落とさない（どの計算機・デバイスで学習したかが後から分からなくなるため）。
  以前の ``auto``（使える方を選ぶ）と「mps が無ければ cpu」の挙動は廃止した
- mps が使える環境で ``mps`` を指定したときの挙動は従来と同じである

## 精度の条件（``configure_backends``）

cuda では float64・``torch.compile``・混合精度（float16・bfloat16）を使わず、TF32 も無効にする
（mps の float32 と精度の条件をそろえるため。0節3）。``configure_backends`` は
``torch.backends.cuda.matmul.allow_tf32`` と ``torch.backends.cudnn.allow_tf32`` を False にし、
新しい API（``fp32_precision``）がある版ではそれも ``ieee`` にそろえる。設定した後の値を
読み直して返すので、呼び出し側はそれを log.txt に書く。mps・cpu では何も変えない。

## 記録（``environment_info``）

ホスト名、metrics.csv に書くホストの呼び名（``host_label``）、デバイス、デバイス名
（cuda なら GPU の名前）、torch・CUDA・cuDNN の版を辞書で返す。config_snapshot.yaml と
log.txt に書く。

**ホストの呼び名**: metrics.csv は Public リポジトリにコミットされるため、Mac の
ホスト名（利用者の名前を含みうる）をそのまま書かず、呼び名を書く。環境変数
``SPKRATE_HOST`` があればその値、なければ macOS は ``mac``、既知のホスト名
（``KNOWN_HOST_LABELS``。ubuntu-desktop の ``xps``）は対応する呼び名、それ以外は
ホスト名の最初の要素である。生のホスト名は runs/ 以下（git の追跡外）の
config_snapshot.yaml と log.txt にだけ書く。
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from typing import Any

import torch

__all__ = [
    "DeviceUnavailableError",
    "KNOWN_HOST_LABELS",
    "SUPPORTED_DEVICE_TYPES",
    "configure_backends",
    "dataloader_device_kwargs",
    "describe_backends",
    "describe_environment",
    "device_name",
    "environment_info",
    "host_label",
    "resolve_device",
    "setup_device",
    "synchronize",
]

SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("mps", "cuda", "cpu")

# ホスト名（最初の要素）→ metrics.csv に書く呼び名。docs/decisions/010-compute-environment.md。
KNOWN_HOST_LABELS: dict[str, str] = {"xps": "ubuntu-desktop"}

HOST_ENV_VAR = "SPKRATE_HOST"


class DeviceUnavailableError(RuntimeError):
    """指定したデバイスがこの計算機で使えない（開始前に止めるための例外）。"""


def resolve_device(name: str | torch.device) -> torch.device:
    """設定のデバイス名を検査して ``torch.device`` にする。

    Raises:
        ValueError: mps・cuda・cpu 以外の名前（``auto`` を含む）。
        DeviceUnavailableError: 指定したデバイスがこの計算機で使えない。
    """
    text = str(name).strip().lower()
    try:
        device = torch.device(text)
    except RuntimeError as error:
        raise ValueError(
            f"device は {SUPPORTED_DEVICE_TYPES} のいずれか（cuda:0 のような番号付きも可）: {name!r}"
        ) from error
    if device.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"device は {SUPPORTED_DEVICE_TYPES} のいずれか: {name!r}")
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceUnavailableError(
                f"device={text} を指定したが mps が使えない"
                f"（torch.backends.mps.is_built()={torch.backends.mps.is_built()}）。"
                "cpu には落とさずに止める"
            )
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"device={text} を指定したが cuda が使えない"
                f"（torch {torch.__version__}、torch.version.cuda={torch.version.cuda}）。"
                "cpu には落とさずに止める"
            )
        count = torch.cuda.device_count()
        if device.index is not None and device.index >= count:
            raise DeviceUnavailableError(
                f"device={text} を指定したが cuda のデバイスは {count} 個しかない"
            )
    return device


def configure_backends(device: torch.device) -> dict[str, Any]:
    """デバイスに応じた精度の設定を適用し、適用後の値を返す（モジュール docstring）。

    cuda のときだけ TF32 を無効にする。mps・cpu では何も変えず空の辞書を返す。
    """
    if device.type != "cuda":
        return {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 新しい API（torch 2.9 以降）でも同じ意味の値にそろえる。
    for target in (
        torch.backends.cuda.matmul,
        getattr(torch.backends.cudnn, "conv", None),
        getattr(torch.backends.cudnn, "rnn", None),
    ):
        if target is not None and hasattr(target, "fp32_precision"):
            target.fp32_precision = "ieee"
    return describe_backends()


def describe_backends() -> dict[str, Any]:
    """TF32 と float32 の行列積の精度に関わる現在の設定値。"""
    values: dict[str, Any] = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for key, target in (
        ("cuda_matmul_fp32_precision", torch.backends.cuda.matmul),
        ("cudnn_conv_fp32_precision", getattr(torch.backends.cudnn, "conv", None)),
        ("cudnn_rnn_fp32_precision", getattr(torch.backends.cudnn, "rnn", None)),
    ):
        if target is not None and hasattr(target, "fp32_precision"):
            values[key] = str(target.fp32_precision)
    return values


def dataloader_device_kwargs(device: torch.device) -> dict[str, Any]:
    """DataLoader に渡すデバイス向けの設定。``pin_memory`` は cuda のときだけ有効にする。"""
    return {"pin_memory": device.type == "cuda"}


def synchronize(device: torch.device) -> None:
    """非同期のデバイスで計算の完了を待つ（計時を正しくするため）。cpu では何もしない。"""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_brand() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def device_name(device: torch.device) -> str:
    """デバイスの名前。cuda は GPU の名前、mps と cpu はチップ・CPU の名前。"""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return _cpu_brand()


def host_label() -> str:
    """metrics.csv の host 列に書く計算機の呼び名（モジュール docstring）。"""
    override = os.environ.get(HOST_ENV_VAR, "").strip()
    if override:
        return override
    if sys.platform == "darwin":
        return "mac"
    short = socket.gethostname().split(".")[0]
    return KNOWN_HOST_LABELS.get(short, short)


def environment_info(device: torch.device) -> dict[str, Any]:
    """実行した計算機とデバイスの記録（config_snapshot.yaml と log.txt に書く）。"""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "host_label": host_label(),
        "device": str(device),
        "device_type": device.type,
        "device_name": device_name(device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": None,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        properties = torch.cuda.get_device_properties(index)
        info["cudnn"] = torch.backends.cudnn.version()
        info["cuda_device_index"] = int(index)
        info["cuda_capability"] = f"{major}.{minor}"
        info["cuda_total_memory_mib"] = int(properties.total_memory // (1024 * 1024))
    return info


def describe_environment(info: dict[str, Any]) -> str:
    """``environment_info`` の1行の要約（log.txt 用）。"""
    return (
        f"計算機: ホスト名={info['hostname']} 呼び名={info['host_label']} "
        f"デバイス={info['device']} デバイス名={info['device_name']} "
        f"torch={info['torch']} CUDA={info['cuda'] or 'なし'} cuDNN={info['cudnn'] or 'なし'}"
    )


def setup_device(name: str | torch.device, logger=None) -> tuple[torch.device, dict[str, Any]]:  # noqa: ANN001
    """デバイスを選び、精度の設定を適用し、記録を log に書く（学習・評価の開始時に呼ぶ）。

    Returns:
        ``(device, record)``。``record`` は ``environment_info`` に ``backends``
        （``configure_backends`` の結果。mps・cpu では空）を足した辞書。
    """
    device = resolve_device(name)
    backends = configure_backends(device)
    record = environment_info(device)
    record["backends"] = backends
    if logger is not None:
        logger.info("%s", describe_environment(record))
        if device.type == "cuda":
            logger.info(
                "TF32 を無効にした（精度の条件を mps の float32 とそろえる）: %s",
                " ".join(f"{key}={value}" for key, value in backends.items()),
            )
    return device, record
