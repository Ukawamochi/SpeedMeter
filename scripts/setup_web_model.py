"""確認用ページ（web/）が読むモデルを web/models/ に置く。

``--source`` に渡した書き出し先（``runs/onnx_<名前>``。複数可）ごとに ``model_fp32.onnx`` を
``web/models/<名前>.onnx`` にコピーし（``--symlink`` ならシンボリックリンク）、ページが読む一覧
``web/models/models.json`` を書く。名前は書き出し先のディレクトリ名から先頭の ``onnx_`` を除いたもの
（``runs/onnx_exp005`` → ``exp005``）。一覧の順は ``--source`` の順で、ページのチェックボックスと色の順になる。
web/models/ は .gitignore の対象で、モデルはリポジトリに入れない。
``export_meta.json`` があれば SHA-256 を照合する。

計算の照合ページ（check.html）は参照値が exp005 のモデルで作られているので ``exp005.onnx`` を読む。

実行（リポジトリ直下から）:
    uv run python scripts/setup_web_model.py [--source runs/onnx_exp005 runs/onnx_exp009 ...] [--symlink]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "model_fp32.onnx"
MANIFEST_NAME = "models.json"
SOURCE_PREFIX = "onnx_"
MAX_MODELS = 8  # ページの系列の色の数（web/realtime.js SERIES_COLORS）


def model_id(source: Path) -> str:
    name = source.resolve().name
    return name[len(SOURCE_PREFIX):] if name.startswith(SOURCE_PREFIX) else name


def checked_sha256(source_dir: Path, model: Path) -> str:
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    meta_path = source_dir / "export_meta.json"
    if meta_path.is_file():
        expected = json.loads(meta_path.read_text(encoding="utf-8"))["files"][MODEL_NAME]["sha256"]
        if actual != expected:
            raise SystemExit(f"SHA-256 が export_meta.json と異なる: {model}: {actual} != {expected}")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, nargs="+", default=[ROOT / "runs" / "onnx_exp005"])
    parser.add_argument("--dest", type=Path, default=ROOT / "web" / "models")
    parser.add_argument("--symlink", action="store_true", help="コピーせずにシンボリックリンクを作る")
    args = parser.parse_args(argv)

    if len(args.source) > MAX_MODELS:
        raise SystemExit(f"モデルは {MAX_MODELS} 個まで（系列の色の数）: {len(args.source)} 個")
    ids = [model_id(s) for s in args.source]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"モデルの名前が重複している: {ids}")
    entries = []
    for source_dir, mid in zip(args.source, ids):
        source = (source_dir / MODEL_NAME).resolve()
        if not source.is_file():
            raise SystemExit(f"モデルが無い: {source}")
        entries.append({"id": mid, "file": f"{mid}.onnx", "source": source, "sha256": checked_sha256(source_dir, source)})

    args.dest.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        dest = args.dest / entry["file"]
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        if args.symlink:
            dest.symlink_to(entry["source"])
        else:
            shutil.copyfile(entry["source"], dest)
        print(f"{'linked' if args.symlink else 'copied'} {entry['source']} -> {dest}")
    manifest = [{"id": e["id"], "file": e["file"], "sha256": e["sha256"]} for e in entries]
    manifest_path = args.dest / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({', '.join(ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
