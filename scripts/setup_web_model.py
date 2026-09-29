"""確認用ページ（web/）が読むモデルを web/models/ に置く。

表示するモデルは ``web/models.txt``（Git で管理する一覧。1行に1つ、リポジトリ直下からの書き出し先
``runs/onnx_<名前>``。``#`` 以降と空行は無視）で決める。``--source`` を渡すとその一覧の代わりに使う。

書き出し先ごとに ``model_fp32.onnx`` を ``web/models/<名前>.onnx`` にコピーし（``--symlink`` なら
シンボリックリンク）、ページが読む一覧 ``web/models/models.json`` を書く。名前は書き出し先のディレクトリ名から
先頭の ``onnx_`` を除いたもの（``runs/onnx_exp005`` → ``exp005``）。一覧の順がページのチェックボックスと色の順になる。
一覧に無くなった ``web/models/*.onnx`` は消す（このスクリプトが置いた複製なので）。
web/models/ は .gitignore の対象で、モデルはリポジトリに入れない。
``export_meta.json`` があれば SHA-256 を照合する。

計算の照合ページ（check.html）は参照値が exp005 のモデルで作られているので ``exp005.onnx`` を読む。
一覧に exp005 が無ければ警告する。

実行（リポジトリ直下から）:
    uv run python scripts/setup_web_model.py [--list web/models.txt | --source runs/onnx_exp005 ...] [--symlink]
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
DEFAULT_LIST = ROOT / "web" / "models.txt"
CHECK_MODEL_ID = "exp005"  # check.html の参照値を作ったモデル


def read_model_list(path: Path, root: Path = ROOT) -> list[Path]:
    """一覧のファイルを読み、書き出し先のパス（``root`` からの相対パスは絶対パスにする）を順に返す。"""
    sources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            source = Path(entry).expanduser()
            sources.append(source if source.is_absolute() else root / source)
    return sources


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", type=Path, default=DEFAULT_LIST, help="表示するモデルの一覧（既定 web/models.txt）")
    group.add_argument("--source", type=Path, nargs="+", help="一覧の代わりに書き出し先を直接並べる")
    parser.add_argument("--dest", type=Path, default=ROOT / "web" / "models")
    parser.add_argument("--symlink", action="store_true", help="コピーせずにシンボリックリンクを作る")
    args = parser.parse_args(argv)

    sources = args.source if args.source else read_model_list(args.list)
    if not sources:
        raise SystemExit(f"モデルが1つも無い: {args.list}")
    if len(sources) > MAX_MODELS:
        raise SystemExit(f"モデルは {MAX_MODELS} 個まで（系列の色の数）: {len(sources)} 個")
    ids = [model_id(s) for s in sources]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"モデルの名前が重複している: {ids}")
    if CHECK_MODEL_ID not in ids:
        print(f"警告: {CHECK_MODEL_ID} が一覧に無いので check.html（計算の照合）は動かない")
    entries = []
    for source_dir, mid in zip(sources, ids):
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
    kept = {e["file"] for e in entries}
    for stale in sorted(args.dest.glob("*.onnx")):
        if stale.name not in kept:
            stale.unlink()
            print(f"removed {stale}（一覧に無い）")
    manifest = [{"id": e["id"], "file": e["file"], "sha256": e["sha256"]} for e in entries]
    manifest_path = args.dest / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({', '.join(ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
