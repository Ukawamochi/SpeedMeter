"""Chrome 拡張機能（extension/）に、リポジトリに入れない実行時のファイルを置く。

置くもの:
  - extension/models/<名前>.onnx と models.json : ONNX に書き出したモデル（既定は exp005 と exp016。窓の設定で切り替える）
  - extension/vendor/           : onnxruntime-web 1.30.0 の wasm 一式（拡張機能は外部の CDN を読めないので同梱する）
  - extension/icons/            : 拡張機能のアイコン（話速バーの絵）
extension/lib/ の dsp.js・realtime.js・recorder-worklet.js は web/ のものと同じ（コピーのずれは --check で検出する）。

使い方:
  uv run python scripts/build_extension.py [--models exp005 exp016]
その後 chrome://extensions で「デベロッパーモード」→「パッケージ化されていない拡張機能を読み込む」→ extension/ を選ぶ。
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "extension"
ORT_VERSION = "1.30.0"
ORT_BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/"
ORT_FILES = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
SHARED_WITH_WEB = ["dsp.js", "realtime.js", "recorder-worklet.js"]


def png(size: int) -> bytes:
    """紺黒の地に、話速バーを模した横棒（薄い青）と時間バー（灰）を描いた PNG。"""
    ink, track, pace, sub = (20, 26, 43), (35, 43, 68), (143, 184, 255), (154, 164, 196)
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            px = ink
            if size * 0.2 <= x < size * 0.8:
                if size * 0.32 <= y < size * 0.58:
                    px = pace if x < size * 0.58 else track
                elif size * 0.68 <= y < size * 0.78:
                    px = sub if x < size * 0.7 else track
            row += bytes(px)
        rows.append(b"\x00" + bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["exp005", "exp016"],
                        help="runs/onnx_<名前>/model_fp32.onnx（無ければ web/models/<名前>.onnx）を置く。先頭が既定のモデル")
    parser.add_argument("--check", action="store_true", help="extension/lib が web/ のコピーと一致するかだけ確かめる")
    args = parser.parse_args()

    mismatched = [n for n in SHARED_WITH_WEB if (EXT / "lib" / n).read_bytes() != (ROOT / "web" / n).read_bytes()]
    if mismatched:
        raise SystemExit(f"extension/lib と web/ が一致しない: {mismatched}（web/ を直したら extension/lib にコピーする）")
    if args.check:
        print("extension/lib は web/ と一致")
        return

    (EXT / "models").mkdir(exist_ok=True)
    for old in (EXT / "models").glob("*.onnx"):
        old.unlink()
    manifest = []
    for name in args.models:
        candidates = [ROOT / "runs" / f"onnx_{name}" / "model_fp32.onnx", ROOT / "web" / "models" / f"{name}.onnx"]
        source = next((c for c in candidates if c.exists()), None)
        if source is None:
            raise SystemExit(f"モデルが無い: {name}")
        shutil.copyfile(source, EXT / "models" / f"{name}.onnx")
        manifest.append({"id": name, "file": f"{name}.onnx"})
        print(f"モデル: {source.relative_to(ROOT)} -> extension/models/{name}.onnx")
    (EXT / "models" / "models.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    (EXT / "vendor").mkdir(exist_ok=True)
    for name in ORT_FILES:
        dest = EXT / "vendor" / name
        if not dest.exists():
            print(f"取得: {name}")
            urllib.request.urlretrieve(ORT_BASE + name, dest)

    (EXT / "icons").mkdir(exist_ok=True)
    for size in (16, 32, 48, 128):
        (EXT / "icons" / f"{size}.png").write_bytes(png(size))
    print("完了。chrome://extensions で extension/ を読み込む。")


if __name__ == "__main__":
    main()
