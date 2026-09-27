"""確認用ページ（web/）が読むモデルを web/models/ に置く。

``runs/onnx_exp005/model_fp32.onnx`` を ``web/models/model_fp32.onnx`` にコピーする（``--symlink`` なら
シンボリックリンク）。web/models/ は .gitignore の対象で、モデルはリポジトリに入れない。
``export_meta.json`` があれば SHA-256 を照合する。

実行（リポジトリ直下から）:
    uv run python scripts/setup_web_model.py [--source runs/onnx_exp005] [--symlink]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "model_fp32.onnx"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=ROOT / "runs" / "onnx_exp005")
    parser.add_argument("--dest", type=Path, default=ROOT / "web" / "models")
    parser.add_argument("--symlink", action="store_true", help="コピーせずにシンボリックリンクを作る")
    args = parser.parse_args(argv)

    source = (args.source / MODEL_NAME).resolve()
    if not source.is_file():
        raise SystemExit(f"モデルが無い: {source}")
    meta_path = args.source / "export_meta.json"
    if meta_path.is_file():
        expected = json.loads(meta_path.read_text(encoding="utf-8"))["files"][MODEL_NAME]["sha256"]
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"SHA-256 が export_meta.json と異なる: {actual} != {expected}")
    args.dest.mkdir(parents=True, exist_ok=True)
    dest = args.dest / MODEL_NAME
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if args.symlink:
        dest.symlink_to(source)
    else:
        shutil.copyfile(source, dest)
    print(f"{'linked' if args.symlink else 'copied'} {source} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
