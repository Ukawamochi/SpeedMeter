"""評価の実行と metrics.csv への追記の検証（src/spkrate/eval/runner.py）。

本物の results/metrics.csv は一切書き換えない。追記の検証は tmp_path で行う。
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from spkrate.eval.audio import TARGET_SAMPLE_RATE, load_audio, resample, to_mono
from spkrate.eval.metrics import compute_metrics
from spkrate.eval.runner import (
    HOST_DEVICE_COLUMNS,
    LATENCY_COLUMNS,
    LEGACY_METRICS_CSV_COLUMNS,
    METRICS_CSV_COLUMNS,
    PRE_HOST_METRICS_CSV_COLUMNS,
    CommitInfo,
    EvalSegment,
    MetricsRow,
    append_metrics_row,
    build_eval_segments,
    evaluate,
    git_commit_info,
    load_eval_client_ids,
    model_size_bytes,
    run_and_record,
    upgrade_metrics_csv,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_METRICS_CSV = ROOT / "results" / "metrics.csv"
REAL_TEST_SPLIT = ROOT / "configs" / "splits" / "test.json"


# --- 音声の読み込みと再標本化 -------------------------------------------------


def _write_wav(path: Path, sample_rate: int, seconds: float, channels: int = 1) -> Path:
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float32) / sample_rate
    wave = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    data = wave if channels == 1 else np.stack([wave] * channels, axis=1)
    sf.write(str(path), data, sample_rate)
    return path


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert to_mono(stereo).tolist() == [2.0, 3.0]
    assert to_mono(stereo).dtype == np.float32


def test_resample_halves_length_from_32k_to_16k():
    samples = np.zeros(32000, dtype=np.float32)
    out = resample(samples, 32000, 16000)
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 1


def test_resample_is_identity_for_same_rate():
    samples = np.arange(8, dtype=np.float32)
    assert resample(samples, 16000, 16000).tolist() == samples.tolist()


def test_load_audio_returns_16k_mono_float32(tmp_path):
    path = _write_wav(tmp_path / "stereo32k.wav", 32000, 1.0, channels=2)
    samples, sample_rate = load_audio(path)
    assert sample_rate == TARGET_SAMPLE_RATE
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert abs(len(samples) - 16000) <= 1


# --- 推定器の評価 -------------------------------------------------------------


def _fake_loader(_path):
    """音声ファイルを読まずに2秒分の無音を返す（区間長は EvalSegment 側で与える）。"""
    return np.zeros(2 * TARGET_SAMPLE_RATE, dtype=np.float32), TARGET_SAMPLE_RATE


def _segments():
    # 区間長2秒、正解モーラ数 6 と 16 -> 正解の毎秒モーラ数 3 と 8。
    return [
        EvalSegment("a", Path("a.wav"), true_mora=6.0, duration_sec=2.0),
        EvalSegment("b", Path("b.wav"), true_mora=16.0, duration_sec=2.0),
    ]


def test_evaluate_uses_rate_interface():
    """推定器は毎秒モーラ数を返す。内部でモーラ数へ戻して指標を計算する。"""
    result = evaluate(lambda samples, sr: 5.0, _segments(), audio_loader=_fake_loader)
    # 誤差は |5-3| = 2 と |5-8| = 3 -> 平均 2.5。
    assert result.metrics.mae_moras_per_sec == pytest.approx(2.5)
    assert result.metrics.mae_band_under4 == pytest.approx(2.0)
    assert result.metrics.mae_band_over8 == pytest.approx(3.0)
    assert result.predictions == {"a": 5.0, "b": 5.0}
    assert result.total_audio_sec == pytest.approx(4.0)
    assert result.latency_ms_per_inference >= 0.0
    assert math.isnan(result.metrics.correlation)  # 推定が定数なので分散0


def test_evaluate_matches_compute_metrics_directly():
    result = evaluate(lambda samples, sr: 4.0, _segments(), audio_loader=_fake_loader)
    expected = compute_metrics([6.0, 16.0], [8.0, 8.0], [2.0, 2.0])
    assert result.metrics.as_dict() == expected.as_dict()


def test_evaluate_falls_back_to_audio_length(tmp_path):
    """duration_sec が None なら読み込んだ音声の長さを区間長に使う。"""
    path = _write_wav(tmp_path / "clip.wav", 32000, 2.0)
    segment = EvalSegment("c", path, true_mora=10.0, duration_sec=None)
    result = evaluate(lambda samples, sr: 5.0, [segment])
    assert result.total_audio_sec == pytest.approx(2.0, abs=1e-3)
    assert result.metrics.mae_moras_per_sec == pytest.approx(0.0, abs=1e-3)


def test_evaluate_rejects_non_finite_prediction():
    with pytest.raises(ValueError):
        evaluate(lambda samples, sr: float("nan"), _segments(), audio_loader=_fake_loader)


def test_evaluate_on_empty_segments():
    result = evaluate(lambda samples, sr: 5.0, [], audio_loader=_fake_loader)
    assert result.metrics.num_segments == 0
    assert math.isnan(result.latency_ms_per_inference)


# --- metrics.csv への追記 -----------------------------------------------------


def _row(experiment_id="000-dummy", **overrides):
    metrics = overrides.pop(
        "metrics", compute_metrics([6.0, 16.0], [8.0, 18.0], [2.0, 2.0])
    )
    params = {
        "experiment_id": experiment_id,
        "method": "dummy",
        "config_path": "configs/dummy.yaml",
        "split": "dev",
        "metrics": metrics,
        "latency_ms_per_inference": 1.25,
        "model_size_bytes": 4096,
        "commit": CommitInfo("0123456789abcdef", False),
        "timestamp": "2026-09-21T12:00:00+09:00",
    }
    params.update(overrides)
    return MetricsRow(**params)


def test_append_writes_header_then_appends(tmp_path):
    csv_path = tmp_path / "results" / "metrics.csv"
    append_metrics_row(_row("001-a"), csv_path=csv_path)
    append_metrics_row(_row("002-b"), csv_path=csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == METRICS_CSV_COLUMNS
    assert len(rows) == 3  # ヘッダ1行 + データ2行
    assert rows[1][0] == "001-a"
    assert rows[2][0] == "002-b"

    with csv_path.open(encoding="utf-8", newline="") as handle:
        first = next(csv.DictReader(handle))
    assert first["commit_hash"] == "0123456789abcdef"
    assert first["commit_dirty"] == "clean"
    assert first["method"] == "dummy"
    assert first["config_path"] == "configs/dummy.yaml"
    assert first["split"] == "dev"
    assert first["timestamp"] == "2026-09-21T12:00:00+09:00"
    assert float(first["mae_moras_per_sec"]) == pytest.approx(1.0)
    assert first["latency_ms_per_inference"] == "1.25"
    assert first["model_size_bytes"] == "4096"


def test_append_records_dirty_worktree(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    append_metrics_row(_row(commit=CommitInfo("abc", True)), csv_path=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["commit_dirty"] == "dirty"


def test_append_writes_nan_metric_as_empty_cell(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    # 全区間が 4to6 帯なので、他の帯の平均絶対誤差は NaN になる。
    metrics = compute_metrics([10.0, 11.0], [10.0, 12.0], [2.0, 2.0])
    append_metrics_row(_row(metrics=metrics, model_size_bytes=None), csv_path=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["mae_band_under4"] == ""
    assert row["n_band_under4"] == "0"
    assert row["model_size_bytes"] == ""


def test_append_rejects_unexpected_header(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("experiment_id,timestamp\n", encoding="utf-8")
    with pytest.raises(ValueError):
        append_metrics_row(_row(), csv_path=csv_path)


def test_append_rejects_missing_or_unknown_columns(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with pytest.raises(ValueError):
        append_metrics_row({"experiment_id": "x"}, csv_path=csv_path)
    full = _row().as_row()
    full["unexpected"] = 1
    with pytest.raises(ValueError):
        append_metrics_row(full, csv_path=csv_path)
    assert not csv_path.exists()


# --- 推論時間の新方式の列（latency_session_id など） -----------------------------


def test_latency_columns_are_appended_after_legacy_columns():
    assert METRICS_CSV_COLUMNS[: len(LEGACY_METRICS_CSV_COLUMNS)] == LEGACY_METRICS_CSV_COLUMNS
    assert PRE_HOST_METRICS_CSV_COLUMNS == (*LEGACY_METRICS_CSV_COLUMNS, *LATENCY_COLUMNS)
    assert METRICS_CSV_COLUMNS == (*PRE_HOST_METRICS_CSV_COLUMNS, *HOST_DEVICE_COLUMNS)
    assert LATENCY_COLUMNS == ("latency_session_id", "latency_ms_median", "latency_ms_max")
    assert HOST_DEVICE_COLUMNS == ("host", "device")


def test_accuracy_row_leaves_latency_session_columns_empty(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    append_metrics_row(_row(), csv_path=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["latency_ms_per_inference"] == "1.25"
    assert row["latency_session_id"] == ""
    assert row["latency_ms_median"] == ""
    assert row["latency_ms_max"] == ""


# 列を足す前の csv に upgrade_metrics_csv が足す列（推論時間の新方式と host・device）。
ADDED_TO_LEGACY = (*LATENCY_COLUMNS, *HOST_DEVICE_COLUMNS)


def _write_legacy_csv(path: Path) -> list[list[str]]:
    """列を足す前の形式の csv を、既存の書き方（csv モジュールの既定）で書く。"""
    rows = [
        ["001-a", "2026-09-21T00:39:21+09:00", "abc", "clean", "configs/a.yaml",
         "envelope (音量包絡, 帯域通過)", "dev", *["0.5"] * (len(LEGACY_METRICS_CSV_COLUMNS) - 9),
         "0.4156061607290245", ""],
        ["002-b", "2026-09-22T00:00:00+09:00", "def", "dirty", "configs/b.yaml;configs/c.yaml",
         "cnn", "dev_noisy_all", *[""] * (len(LEGACY_METRICS_CSV_COLUMNS) - 9),
         "1.7455354149569757", "2005357"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LEGACY_METRICS_CSV_COLUMNS)
        writer.writerows(rows)
    return rows


def test_upgrade_metrics_csv_keeps_existing_rows(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    legacy_rows = _write_legacy_csv(csv_path)
    before = csv_path.read_bytes()

    assert upgrade_metrics_csv(csv_path) is True
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == METRICS_CSV_COLUMNS
    assert len(rows) == 1 + len(legacy_rows)
    for old, new in zip(legacy_rows, rows[1:]):
        assert new[: len(old)] == old  # 既存の列の値は文字列のまま
        assert new[len(old):] == [""] * len(ADDED_TO_LEGACY)  # 新しい列は空欄
    # 既存行の各行は「元の行 + 空欄の列」になっている（引用符・行末も変わらない）。
    old_lines = before.splitlines(keepends=True)
    new_lines = csv_path.read_bytes().splitlines(keepends=True)
    assert len(new_lines) == len(old_lines)
    for old_line, new_line in zip(old_lines[1:], new_lines[1:]):
        body = old_line.rstrip(b"\r\n")
        assert new_line == body + b"," * len(ADDED_TO_LEGACY) + old_line[len(body):]

    # 2回目は何もしない。移行後は追記できる。
    assert upgrade_metrics_csv(csv_path) is False
    append_metrics_row(_row("003-c"), csv_path=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert [r[0] for r in rows[1:]] == ["001-a", "002-b", "003-c"]


def test_upgrade_metrics_csv_keeps_mixed_line_endings(tmp_path):
    """行末が \\n と \\r\\n で混在していても行ごとの行末を保つ（本物の csv がそうである）。"""
    csv_path = tmp_path / "metrics.csv"
    _write_legacy_csv(csv_path)
    raw = csv_path.read_bytes().split(b"\r\n")
    mixed = raw[0] + b"\n" + raw[1] + b"\n" + raw[2] + b"\r\n"
    csv_path.write_bytes(mixed)
    assert upgrade_metrics_csv(csv_path) is True
    lines = csv_path.read_bytes().splitlines(keepends=True)
    assert [line.endswith(b"\r\n") for line in lines] == [False, False, True]
    assert lines[1] == raw[1] + b",,,,,\n"
    assert lines[0].endswith(b",latency_session_id,latency_ms_median,latency_ms_max,host,device\n")


def test_append_to_legacy_csv_is_refused_until_upgraded(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_legacy_csv(csv_path)
    before = csv_path.read_bytes()
    with pytest.raises(ValueError, match="upgrade_metrics_csv"):
        append_metrics_row(_row(), csv_path=csv_path)
    assert csv_path.read_bytes() == before


# --- 実行した計算機の列（host・device） -----------------------------------------


def _write_pre_host_csv(path: Path) -> list[list[str]]:
    """host・device の列を足す前の形式（今の results/metrics.csv と同じ列構成）の csv を書く。"""
    rows = [
        ["001-a", "2026-09-21T00:39:21+09:00", "abc", "clean", "configs/a.yaml", "envelope", "dev",
         *["0.5"] * (len(LEGACY_METRICS_CSV_COLUMNS) - 9), "0.41", "", "", "", ""],
        ["002-b", "2026-09-24T10:00:00+09:00", "def", "dirty", "configs/b.yaml", "cnn", "latency",
         *[""] * (len(LEGACY_METRICS_CSV_COLUMNS) - 9), "", "2005357", "lat-1", "2.5", "3.1"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PRE_HOST_METRICS_CSV_COLUMNS)
        writer.writerows(rows)
    return rows


def test_upgrade_pre_host_csv_adds_empty_host_and_device(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    old_rows = _write_pre_host_csv(csv_path)
    before = csv_path.read_bytes().splitlines(keepends=True)

    assert upgrade_metrics_csv(csv_path) is True
    after = csv_path.read_bytes().splitlines(keepends=True)
    assert after[0] == before[0].rstrip(b"\r\n") + b",host,device" + b"\r\n"
    for old_line, new_line in zip(before[1:], after[1:]):
        assert new_line == old_line.rstrip(b"\r\n") + b",," + b"\r\n"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == METRICS_CSV_COLUMNS
    for old, new in zip(old_rows, rows[1:]):
        assert new == [*old, "", ""]  # 既存の列と値は変えず、host・device は空欄
    assert upgrade_metrics_csv(csv_path) is False


def test_append_to_pre_host_csv_is_refused_until_upgraded(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_pre_host_csv(csv_path)
    before = csv_path.read_bytes()
    with pytest.raises(ValueError, match="upgrade_metrics_csv"):
        append_metrics_row(_row(), csv_path=csv_path)
    assert csv_path.read_bytes() == before


def test_append_writes_host_and_device(tmp_path, monkeypatch):
    from spkrate import device as device_utils

    monkeypatch.setenv(device_utils.HOST_ENV_VAR, "ubuntu-desktop")
    csv_path = tmp_path / "metrics.csv"
    append_metrics_row(_row("001-a", device="cuda"), csv_path=csv_path)
    append_metrics_row(_row("002-b", host="mac", device="mps"), csv_path=csv_path)
    append_metrics_row(_row("003-c"), csv_path=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["host"], r["device"]) for r in rows] == [
        ("ubuntu-desktop", "cuda"), ("mac", "mps"), ("ubuntu-desktop", "")]


def test_upgrade_metrics_csv_rejects_unknown_header_and_broken_rows(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("experiment_id,timestamp\r\nx,y\r\n", encoding="utf-8")
    with pytest.raises(ValueError):
        upgrade_metrics_csv(csv_path)

    broken = tmp_path / "broken.csv"
    _write_legacy_csv(broken)
    with broken.open("a", encoding="utf-8", newline="") as handle:
        handle.write("only,two\r\n")
    before = broken.read_bytes()
    with pytest.raises(ValueError, match="列数"):
        upgrade_metrics_csv(broken)
    assert broken.read_bytes() == before

    multiline = tmp_path / "multiline.csv"
    _write_legacy_csv(multiline)
    with multiline.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["改行\nを含む", *[""] * (len(LEGACY_METRICS_CSV_COLUMNS) - 1)])
    before = multiline.read_bytes()
    with pytest.raises(ValueError, match="改行"):
        upgrade_metrics_csv(multiline)
    assert multiline.read_bytes() == before


def test_run_and_record_appends_one_row(tmp_path):
    """ダミーの推定器で、評価から metrics.csv への追記まで通ることを確かめる。"""
    csv_path = tmp_path / "metrics.csv"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x" * 128)
    result = run_and_record(
        lambda samples, sr: 5.0,
        _segments(),
        experiment_id="000-dummy",
        method="constant-5.0",
        config_path="configs/dummy.yaml",
        split="dev",
        model_path=model,
        csv_path=csv_path,
        repo_dir=ROOT,
        audio_loader=_fake_loader,
    )
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["method"] == "constant-5.0"
    assert rows[0]["model_size_bytes"] == "128"
    assert float(rows[0]["mae_moras_per_sec"]) == pytest.approx(
        result.metrics.mae_moras_per_sec
    )
    assert rows[0]["commit_dirty"] in {"clean", "dirty"}


def test_model_size_bytes(tmp_path):
    assert model_size_bytes(None) is None
    assert model_size_bytes(tmp_path / "missing.onnx") is None
    path = tmp_path / "m.bin"
    path.write_bytes(b"abc")
    assert model_size_bytes(path) == 3


def test_git_commit_info_on_this_repo():
    info = git_commit_info(ROOT)
    assert len(info.commit_hash) == 40
    assert info.dirty_flag in {"clean", "dirty"}


def test_git_commit_info_outside_repo(tmp_path):
    info = git_commit_info(tmp_path)
    assert info.commit_hash == "unknown"
    assert info.dirty is True  # 不明な場合は汚れている側に倒す


# --- テスト分割の使用禁止 -----------------------------------------------------


def _write_split(path: Path, name: str, client_ids):
    path.write_text(
        json.dumps({"split": name, "client_ids": list(client_ids)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_load_eval_client_ids_refuses_test_split(tmp_path):
    path = _write_split(tmp_path / "test.json", "test", ["s1"])
    with pytest.raises(RuntimeError):
        load_eval_client_ids(path)
    # 第10段階の明示的な承認がある場合のみ読める。
    assert load_eval_client_ids(path, stage10_approved=True) == ["s1"]


def test_load_eval_client_ids_allows_dev_split(tmp_path):
    path = _write_split(tmp_path / "dev.json", "dev", ["s1", "s2"])
    assert load_eval_client_ids(path) == ["s1", "s2"]


def test_load_eval_client_ids_refuses_file_named_test_json(tmp_path):
    """中身の split 名が違っても、ファイル名が test.json なら拒否する。"""
    path = _write_split(tmp_path / "test.json", "dev", ["s1"])
    with pytest.raises(RuntimeError):
        load_eval_client_ids(path)


def test_build_eval_segments_refuses_test_split(tmp_path):
    clips = tmp_path / "clips.jsonl"
    clips.write_text(
        json.dumps(
            {
                "clip_id": "c1",
                "audio_path": "clips/c1.mp3",
                "client_id": "s1",
                "sentence": "あ",
                "kana": "ア",
                "mora": 1,
                "duration_sec": 1.0,
                "mora_per_second": 1.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    test_split = _write_split(tmp_path / "test.json", "test", ["s1"])
    with pytest.raises(RuntimeError):
        build_eval_segments(clips, test_split, tmp_path)

    dev_split = _write_split(tmp_path / "dev.json", "dev", ["s1"])
    segments = build_eval_segments(clips, dev_split, tmp_path)
    assert len(segments) == 1
    assert segments[0].segment_id == "c1"
    assert segments[0].audio_path == tmp_path / "clips/c1.mp3"
    assert segments[0].true_mora == 1.0


@pytest.mark.skipif(not REAL_TEST_SPLIT.exists(), reason="configs/splits/test.json が無い")
def test_real_test_split_is_refused():
    with pytest.raises(RuntimeError):
        load_eval_client_ids(REAL_TEST_SPLIT)


@pytest.mark.skipif(not REAL_METRICS_CSV.exists(), reason="results/metrics.csv が無い")
def test_real_metrics_csv_header_matches_columns():
    """本物の metrics.csv のヘッダが列定義と一致していること（読むだけ）。"""
    with REAL_METRICS_CSV.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    assert tuple(header) == METRICS_CSV_COLUMNS
