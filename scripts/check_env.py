"""環境確認スクリプト。

Python/torchのバージョン、MPSの利用可否、MPS上でのConv2d順伝播、
pyopenjtalkによる読み変換が動作するかを出力する。
"""

import sys

import torch


def check_python_version() -> None:
    print(f"Python version: {sys.version}")


def check_torch_version() -> None:
    print(f"torch version: {torch.__version__}")


def check_mps_available() -> bool:
    available = torch.backends.mps.is_available()
    print(f"torch.backends.mps.is_available(): {available}")
    return available


def check_mps_conv2d(mps_available: bool) -> None:
    if not mps_available:
        print("mps Conv2d forward: skipped (mps unavailable)")
        return
    try:
        device = torch.device("mps")
        conv = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3).to(device)
        x = torch.randn(1, 1, 16, 16, device=device)
        y = conv(x)
        torch.mps.synchronize()
        print(f"mps Conv2d forward: OK (output shape={tuple(y.shape)})")
    except Exception as e:
        print(f"mps Conv2d forward: FAILED ({e})")


def check_pyopenjtalk() -> None:
    try:
        import pyopenjtalk

        text = "今日は快晴です"
        result = pyopenjtalk.g2p(text, kana=True)
        print(f"pyopenjtalk g2p(kana=True) for '{text}': {result}")
    except Exception as e:
        print(f"pyopenjtalk check: FAILED ({e})")


def main() -> None:
    check_python_version()
    check_torch_version()
    mps_available = check_mps_available()
    check_mps_conv2d(mps_available)
    check_pyopenjtalk()


if __name__ == "__main__":
    main()
