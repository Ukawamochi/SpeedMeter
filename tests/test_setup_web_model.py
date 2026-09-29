"""scripts/setup_web_model.py（確認用ページのモデルの配置）のテスト。本物のモデルは使わない。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from setup_web_model import MAX_MODELS, main, model_id, read_model_list  # noqa: E402


def _export_dir(root: Path, name: str, content: bytes) -> Path:
    directory = root / "runs" / name
    directory.mkdir(parents=True)
    (directory / "model_fp32.onnx").write_bytes(content)
    return directory


def test_model_id_drops_onnx_prefix(tmp_path: Path) -> None:
    assert model_id(tmp_path / "onnx_exp005") == "exp005"
    assert model_id(tmp_path / "custom") == "custom"


def test_read_model_list_skips_comments_and_resolves_relative(tmp_path: Path) -> None:
    listing = tmp_path / "models.txt"
    listing.write_text("# 見出し\n\nruns/onnx_exp005   # 基準\n  # runs/onnx_exp007\n/abs/onnx_exp009\n", encoding="utf-8")
    assert read_model_list(listing, root=tmp_path) == [tmp_path / "runs" / "onnx_exp005", Path("/abs/onnx_exp009")]


def test_repository_model_list_is_valid() -> None:
    sources = read_model_list(ROOT / "web" / "models.txt")
    ids = [model_id(s) for s in sources]
    assert 1 <= len(ids) <= MAX_MODELS
    assert len(set(ids)) == len(ids)
    assert "exp005" in ids  # check.html の照合に使う


def test_main_places_models_in_list_order_and_removes_stale(tmp_path: Path) -> None:
    a = _export_dir(tmp_path, "onnx_exp005", b"a")
    b = _export_dir(tmp_path, "onnx_exp009", b"b")
    listing = tmp_path / "models.txt"
    listing.write_text(f"{b}\n{a}\n", encoding="utf-8")
    dest = tmp_path / "web_models"
    dest.mkdir()
    (dest / "old.onnx").write_bytes(b"x")
    assert main(["--list", str(listing), "--dest", str(dest)]) == 0
    manifest = json.loads((dest / "models.json").read_text(encoding="utf-8"))
    assert [m["id"] for m in manifest] == ["exp009", "exp005"]
    assert (dest / "exp009.onnx").read_bytes() == b"b"
    assert not (dest / "old.onnx").exists()


def test_main_rejects_too_many_and_duplicates(tmp_path: Path) -> None:
    sources = [str(_export_dir(tmp_path, f"onnx_m{i}", b"m")) for i in range(MAX_MODELS + 1)]
    with pytest.raises(SystemExit, match="個まで"):
        main(["--source", *sources, "--dest", str(tmp_path / "d1")])
    other = tmp_path / "other" / "onnx_m0"
    other.mkdir(parents=True)
    (other / "model_fp32.onnx").write_bytes(b"m")
    with pytest.raises(SystemExit, match="重複"):
        main(["--source", sources[0], str(other), "--dest", str(tmp_path / "d2")])


def test_main_rejects_missing_model(tmp_path: Path) -> None:
    (tmp_path / "runs" / "onnx_exp005").mkdir(parents=True)
    with pytest.raises(SystemExit, match="モデルが無い"):
        main(["--source", str(tmp_path / "runs" / "onnx_exp005"), "--dest", str(tmp_path / "d")])
