"""評価の入口の test 分割対応（第10段階、2026-10-02 の人間の指示）の検証。

確かめること:
- 既定は dev のまま（プロファイルの値、metrics.csv の split 列の名前、既定の設定が test を拒むこと）
- ``--split test`` は configs/splits/test.json の話者だけを使い、dev・train の話者と重ならない
- test 用の設定（test_window・test_noisy・test_fast）が dev と同じ規則・同じ値で、
  対象の分割・出力先だけが違う
- 評価の入口のスクリプトが ``--split`` を受け付け、既定が dev である
実データの大きなファイル（clips.jsonl）は使わず、合成データで確かめる。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from spkrate.eval.dev_window import DevWindowConfig, load_window_config
from spkrate.eval.split_profile import SPLITS, get_profile, load_split_ids

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "configs" / "splits"
EVAL_DIR = ROOT / "configs" / "eval"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- 既定は dev のまま -------------------------------------------------------


def test_dev_profile_is_the_original_values():
    p = get_profile("dev")
    assert get_profile() is p
    assert p.split_json == "configs/splits/dev.json"
    assert p.window_config == "configs/eval/dev_window.yaml"
    assert p.fast_config == "configs/eval/dev_fast.yaml"
    assert p.noisy_config == "configs/eval/dev_noisy.yaml"
    assert p.suspect_tsv == "results/no_speech_suspect_dev.tsv"
    assert p.features_dir == "data/processed/features/dev"
    assert p.alignments == "data/processed/alignments/dev.jsonl"
    assert not p.is_test


def test_dev_split_names_are_unchanged():
    p = get_profile("dev")
    assert p.clean_split == "dev"
    assert p.noisy_split(5) == "dev_noisy_snr5"
    assert p.noisy_split(10.0) == "dev_noisy_snr10"
    assert p.noisy_pooled_split == "dev_noisy_all"
    assert p.window_split == "dev_window"
    assert p.window_noisy_split("snr15") == "dev_window_noisy_snr15"
    assert p.window_noisy_pooled_split == "dev_window_noisy_all"
    assert p.fast_split_prefix == "dev_fast"


def test_test_split_names_say_test():
    p = get_profile("test")
    assert p.is_test
    assert p.split_json == "configs/splits/test.json"
    assert p.clean_split == "test"
    assert p.noisy_split(5) == "test_noisy_snr5"
    assert p.noisy_pooled_split == "test_noisy_all"
    assert p.window_split == "test_window"
    assert p.window_noisy_split("snr5") == "test_window_noisy_snr5"
    assert p.window_noisy_pooled_split == "test_window_noisy_all"
    assert p.fast_split_prefix == "test_fast"
    # 出力先も dev と分ける
    dev = get_profile("dev")
    for field in ("features_dir", "alignments", "no_speech_metrics", "suspect_tsv"):
        assert getattr(p, field) != getattr(dev, field)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError):
        get_profile("train")
    assert SPLITS == ("dev", "test")


def test_default_window_config_still_rejects_test_split():
    assert not load_window_config(EVAL_DIR / "dev_window.yaml").allow_test_split
    with pytest.raises(ValueError):
        DevWindowConfig.from_mapping({**_yaml(EVAL_DIR / "dev_window.yaml"), "dev_split": "configs/splits/test.json"})


# --- test の話者だけを使う -----------------------------------------------------


def test_test_split_needs_explicit_permission():
    with pytest.raises(ValueError):
        load_split_ids(SPLIT_DIR / "test.json")  # 既定は従来どおり拒否
    assert load_split_ids(SPLIT_DIR / "dev.json") == load_split_ids(SPLIT_DIR / "dev.json", allow_test=True)


def test_test_speakers_do_not_overlap_dev_or_train():
    test = set(load_split_ids(SPLIT_DIR / "test.json", allow_test=True))
    dev = set(load_split_ids(SPLIT_DIR / "dev.json"))
    train = set(load_split_ids(SPLIT_DIR / "train.json"))
    assert len(test) == 1711
    assert not test & dev
    assert not test & train


def _write_synthetic(tmp_path: Path) -> tuple[Path, Path]:
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in {"train": ["a", "b"], "dev": ["c"], "test": ["d", "e"]}.items():
        (splits / f"{name}.json").write_text(
            json.dumps({"split": name, "client_ids": ids}), encoding="utf-8")
    clips = tmp_path / "clips.jsonl"
    rows = []
    for i, client in enumerate("abcdeabcde"):
        rows.append({"clip_id": f"clip{i}", "audio_path": f"x/clip{i}.mp3", "client_id": client,
                     "sentence": "あ", "kana": "ア", "mora": 1, "duration_sec": 1.0, "mora_per_second": 1.0})
    clips.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return splits, clips


def test_precompute_features_test_uses_only_test_speakers(tmp_path, monkeypatch):
    module = _load_script("precompute_features")
    splits, clips = _write_synthetic(tmp_path)
    monkeypatch.setattr(module, "SPLITS_DIR", splits)
    monkeypatch.setattr(module, "CLIPS_JSONL", clips)
    monkeypatch.setattr(module, "_TASK_CACHE", {})
    # 既定の分割は train と dev のまま。test は明示したときだけ
    assert module.ALLOWED_SPLITS == ("train", "dev")
    assert "test" in module.SELECTABLE_SPLITS
    tasks = module.load_split_tasks("test")
    assert {t.client_id for t in tasks} == {"d", "e"}
    assert len(tasks) == 4
    assert {t.client_id for t in module.load_split_tasks("dev")} == {"c"}
    assert {t.client_id for t in module.load_split_tasks("train")} == {"a", "b"}
    with pytest.raises(ValueError):
        module.load_split_tasks("val")


def test_select_dev_clips_test_only_with_permission(tmp_path):
    from spkrate.labels.alignment import select_dev_clips

    splits, clips = _write_synthetic(tmp_path)
    with pytest.raises(ValueError):
        select_dev_clips(clips, splits / "test.json")
    rows = select_dev_clips(clips, splits / "test.json", allow_test=True)
    assert {r["client_id"] for r in rows} == {"d", "e"}
    assert {r["client_id"] for r in select_dev_clips(clips, splits / "dev.json")} == {"c"}


# --- test 用の設定は dev と同じ規則 ----------------------------------------------


def test_test_window_config_differs_from_dev_only_in_target():
    dev = _yaml(EVAL_DIR / "dev_window.yaml")
    test = _yaml(EVAL_DIR / "test_window.yaml")
    differing = {k for k in set(dev) | set(test) if dev.get(k) != test.get(k)}
    assert differing == {"dev_split", "allow_test_split", "alignments", "known_no_speech_list", "output_dir"}
    assert test["dev_split"] == "configs/splits/test.json"
    assert test["alignments"] == "data/processed/alignments/test.jsonl"
    assert test["output_dir"] == "data/processed/test_window"
    assert test["known_no_speech_list"] is None
    config = load_window_config(EVAL_DIR / "test_window.yaml")
    assert config.allow_test_split
    assert config.noisy == load_window_config(EVAL_DIR / "dev_window.yaml").noisy


def test_test_noisy_config_is_dev_noisy():
    assert _yaml(EVAL_DIR / "test_noisy.yaml") == _yaml(EVAL_DIR / "dev_noisy.yaml")
    assert _yaml(EVAL_DIR / "test_window.yaml")["noisy"] == _yaml(EVAL_DIR / "dev_window.yaml")["noisy"]
    assert _yaml(EVAL_DIR / "test_noisy.yaml")["musan_noise_split"] == "configs/splits/musan_noise.json"


def test_test_fast_config_differs_from_dev_only_in_target():
    dev = _yaml(EVAL_DIR / "dev_fast.yaml")
    test = _yaml(EVAL_DIR / "test_fast.yaml")
    differing = {k for k in set(dev) | set(test) if dev.get(k) != test.get(k)}
    assert differing == {"dev_window_config", "output_dir"}
    assert test["dev_window_config"] == "configs/eval/test_window.yaml"
    assert test["output_dir"] == "data/processed/test_fast"


# --- 評価の入口が --split を受け付け、既定が dev ----------------------------------


def _parse(module, monkeypatch, argv, names=("cmd_predict", "cmd_summarize")):
    """main に引数を渡し、実行はせずに解釈された引数だけを返す。"""
    captured = []
    for name in names:
        monkeypatch.setattr(module, name, lambda args: captured.append(args) or 0)
    assert module.main(argv) == 0
    return captured[0]


def test_eval_dev_window_split_default_and_test(monkeypatch):
    module = _load_script("eval_dev_window")
    for sub in ("predict", "summarize"):
        default = _parse(module, monkeypatch, [sub])
        test = _parse(module, monkeypatch, [sub, "--split", "test"])
        assert default.split == "dev" and test.split == "test"
        # 既定の出力先は変わらない。test は別の場所
        assert default.out_dir == module.OUT_DIR == "runs/window_eval"
        assert test.out_dir == "runs/test_window_eval"
    assert _parse(module, monkeypatch, ["summarize"]).metrics_csv == "results/metrics.csv"


def test_eval_fast_speech_split_default(monkeypatch):
    module = _load_script("eval_fast_speech")
    base = ["summarize", "--skip-window", "--skip-fast", "--model-key", "m", "--out-dir", "o"]
    assert _parse(module, monkeypatch, base).split == "dev"
    assert _parse(module, monkeypatch, base + ["--split", "test"]).split == "test"


def test_eval_fast_speech_fast_config_follows_split():
    module = _load_script("eval_fast_speech")
    import argparse

    ns = argparse.Namespace(split="test", fast_config=None, conditions=None)
    cfg, names = module.fast_conditions(ns)
    assert ns.fast_config == "configs/eval/test_fast.yaml"
    assert cfg["output_dir"] == "data/processed/test_fast" and names == ["x1.5", "x2"]
    ns = argparse.Namespace(split="dev", fast_config=None, conditions=None)
    module.fast_conditions(ns)
    assert ns.fast_config == "configs/eval/dev_fast.yaml"


@pytest.mark.parametrize("name", ["align_dev", "detect_no_speech", "window_diagnostics", "eval_dev_full",
                                  "analyze_no_speech", "eval_dev_window", "eval_fast_speech"])
def test_scripts_accept_split_option_with_dev_default(name):
    _load_script(name)  # 構文・import が通る
    text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert '"--split"' in text
    assert 'default="dev"' in text


@pytest.mark.parametrize("name", ["precompute_features", "build_dev_window", "build_dev_fast"])
def test_prepare_scripts_are_importable(name):
    _load_script(name)


def test_run_test_eval_script_uses_test_split_and_cuda():
    text = (ROOT / "scripts" / "run_test_eval.sh").read_text(encoding="utf-8")
    assert text.count("--split test") >= 6
    for needle in ("--frozen", "status.txt", "configs/eval/test_window.yaml", "configs/eval/test_fast.yaml",
                   "--metrics-csv", "DEVICE=cuda", "configs/eval/test_noisy.yaml"):
        assert needle in text
    assert "--device mps" not in text and "DEVICE=mps" not in text
