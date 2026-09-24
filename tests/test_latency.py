"""推論時間の新方式の測定（src/spkrate/eval/latency.py、scripts/measure_latency.py）の検証。

CPU 上の小さなモデルで測る。本物の results/metrics.csv は書き換えない（tmp_path を使う）。
"""

import csv
import importlib.util
import statistics
from pathlib import Path

import numpy as np
import pytest
import torch

from spkrate.eval.latency import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    LATENCY_SPLIT,
    WINDOW_SAMPLES,
    LatencySessionMismatchError,
    check_same_session,
    compare_latency,
    latency_csv_row,
    make_cnn_inference,
    make_window,
    measure_session,
    measurement_order,
    new_session_id,
    read_latency_rows,
)
from spkrate.eval.runner import (
    LEGACY_METRICS_CSV_COLUMNS,
    METRICS_CSV_COLUMNS,
    append_metrics_row,
    upgrade_metrics_csv,
)
from spkrate.models.cnn import CnnConfig, SpeechRateCNN
from spkrate.train.data import Normalizer

ROOT = Path(__file__).resolve().parents[1]

SMALL = CnnConfig(freq_channels=(4,), freq_strides=(2,), temporal_channels=(8, 8), dilations=(1, 2))


def _small_inference(seed: int = 0):
    torch.manual_seed(seed)
    model = SpeechRateCNN(SMALL).eval()
    normalizer = Normalizer(np.zeros(80), np.ones(80), mode="per_mel")
    return make_cnn_inference(model, normalizer, torch.device("cpu"))


def _row(session_id: str, experiment_id: str, median: float, *, timestamp: str) -> dict:
    from spkrate.eval.latency import LatencyResult

    result = LatencyResult(experiment_id, (median, median, median * 2), 0)
    return latency_csv_row(result, session_id, experiment_id=experiment_id, method="cnn",
                           config_path="configs/x.yaml", model_size_bytes=123,
                           commit_hash="abc", commit_dirty="clean", timestamp=timestamp)


# --- 測定 ---------------------------------------------------------------------


def test_defaults():
    assert DEFAULT_REPEATS == 100
    assert DEFAULT_WARMUP >= 1
    window = make_window()
    assert window.shape == (WINDOW_SAMPLES,) == (32000,)
    assert window.dtype == np.float32
    np.testing.assert_array_equal(window, make_window())  # 同じ種なら同じ入力


def test_measure_small_cnn_on_cpu_100_times():
    inferences = {"a": _small_inference(0), "b": _small_inference(1)}
    session = measure_session(inferences, repeats=100, warmup=3)
    assert session.session_id.startswith("lat-")
    assert session.interleave is True
    for name in ("a", "b"):
        result = session.results[name]
        assert result.repeats == 100
        assert result.warmup == 3
        assert all(t > 0 for t in result.times_ms)
        assert result.median_ms == pytest.approx(statistics.median(result.times_ms))
        assert result.max_ms == max(result.times_ms)
        assert result.median_ms <= result.max_ms


def test_warmup_calls_are_made_and_not_timed():
    calls = {"a": 0, "b": 0}

    def make(name):
        def infer(_samples):
            calls[name] += 1
        return infer

    session = measure_session({"a": make("a"), "b": make("b")}, np.zeros(4, np.float32),
                              repeats=100, warmup=7)
    assert calls == {"a": 107, "b": 107}
    assert session.results["a"].repeats == 100


def test_median_and_max_with_fake_clock():
    """時計を差し替え、各回の所要時間を決め打ちして中央値と最大値を確かめる。"""
    durations = iter([1.0, 5.0, 2.0, 9.0, 3.0])  # ms
    now = [0.0]
    pending = []

    def clock():
        if pending:  # 計時の終わり
            now[0] += pending.pop() / 1000.0
        else:  # 計時の始まり
            pending.append(next(durations))
        return now[0]

    session = measure_session({"m": lambda _s: None}, np.zeros(4, np.float32),
                              repeats=5, warmup=0, clock=clock)
    result = session.results["m"]
    assert result.times_ms == pytest.approx((1.0, 5.0, 2.0, 9.0, 3.0))
    assert result.median_ms == pytest.approx(3.0)
    assert result.max_ms == pytest.approx(9.0)


def test_synchronize_before_and_after_each_timed_call():
    events = []
    measure_session({"a": lambda _s: events.append("a"), "b": lambda _s: events.append("b")},
                    np.zeros(4, np.float32), repeats=2, warmup=1,
                    synchronize=lambda: events.append("sync"))
    # ウォームアップ（a, b）→ 同期1回 → 以降は各回の前後で同期。
    assert events[:3] == ["a", "b", "sync"]
    timed = events[3:]
    assert len(timed) == 2 * 2 * 3
    for chunk in (timed[i:i + 3] for i in range(0, len(timed), 3)):
        assert chunk[0] == "sync" and chunk[2] == "sync" and chunk[1] in {"a", "b"}


def test_interleaved_order_rotates_start():
    names = ["a", "b", "c"]
    assert measurement_order(names, 0) == ("a", "b", "c")
    assert measurement_order(names, 1) == ("b", "c", "a")
    assert measurement_order(names, 3) == ("a", "b", "c")
    assert measurement_order(names, 1, interleave=False) == ("a", "b", "c")

    order = []
    session = measure_session({n: (lambda n: lambda _s: order.append(n))(n) for n in names},
                              np.zeros(4, np.float32), repeats=6, warmup=0)
    assert order[:6] == ["a", "b", "c", "b", "c", "a"]
    # 各モデルが各順位を同じ回数（6周 / 3モデル = 2回）経験する。
    for position in range(3):
        counts = {n: sum(1 for r in session.rounds if r[position] == n) for n in names}
        assert set(counts.values()) == {2}


def test_non_interleaved_measures_each_model_in_a_block():
    order = []
    measure_session({n: (lambda n: lambda _s: order.append(n))(n) for n in "ab"},
                    np.zeros(4, np.float32), repeats=3, warmup=0, interleave=False)
    assert order == ["a", "a", "a", "b", "b", "b"]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        measure_session({"a": lambda _s: None}, repeats=0)
    with pytest.raises(ValueError):
        measure_session({"a": lambda _s: None}, warmup=-1)
    with pytest.raises(ValueError):
        measure_session({})


def test_session_ids_are_unique():
    ids = {new_session_id() for _ in range(200)}
    assert len(ids) == 200


# --- metrics.csv の行とセッションの照合 --------------------------------------


def test_session_rows_share_id_and_fill_latency_columns(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    session = measure_session({"exp-a": _small_inference(0), "exp-b": _small_inference(1)},
                              repeats=100, warmup=2)
    for name, result in session.results.items():
        append_metrics_row(latency_csv_row(result, session.session_id, experiment_id=name, method="cnn",
                                           config_path="configs/x.yaml", model_size_bytes=10,
                                           commit_hash="abc", commit_dirty="clean"),
                           csv_path=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == METRICS_CSV_COLUMNS
        rows = list(reader)
    assert [r["experiment_id"] for r in rows] == ["exp-a", "exp-b"]
    assert {r["latency_session_id"] for r in rows} == {session.session_id}
    for row in rows:
        result = session.results[row["experiment_id"]]
        assert row["split"] == LATENCY_SPLIT
        assert float(row["latency_ms_median"]) == pytest.approx(result.median_ms)
        assert float(row["latency_ms_max"]) == pytest.approx(result.max_ms)
        assert row["latency_ms_per_inference"] == ""  # 旧方式の列は空欄
        assert row["mae_moras_per_sec"] == ""
    sid, values = compare_latency(csv_path, ["exp-a", "exp-b"])
    assert sid == session.session_id
    assert values["exp-a"]["median_ms"] == pytest.approx(session.results["exp-a"].median_ms)


def test_append_latency_rows_to_upgraded_legacy_csv_keeps_old_rows(tmp_path):
    """列を足す前の csv を移行してから新方式の行を足しても、既存行は変わらない。"""
    csv_path = tmp_path / "metrics.csv"
    old = ["005-first-model", "2026-09-21T19:47:00+09:00", "6f2d", "dirty", "configs/exp001.yaml",
           "cnn (方式A, クリップ全体)", "dev", *["0.5"] * (len(LEGACY_METRICS_CSV_COLUMNS) - 9),
           "1.7455354149569757", "2005357"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LEGACY_METRICS_CSV_COLUMNS)
        writer.writerow(old)
    old_bytes = csv_path.read_bytes().splitlines(keepends=True)

    upgrade_metrics_csv(csv_path)
    append_metrics_row(_row("lat-1", "exp-a", 2.0, timestamp="2026-09-24T10:00:00+09:00"), csv_path=csv_path)

    new_bytes = csv_path.read_bytes().splitlines(keepends=True)
    assert new_bytes[1] == old_bytes[1].rstrip(b"\r\n") + b",,," + b"\r\n"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [rows[0][c] for c in LEGACY_METRICS_CSV_COLUMNS] == old
    assert rows[0]["latency_session_id"] == ""
    assert rows[1]["latency_session_id"] == "lat-1"
    # 旧方式の行は新方式の読み出しに入らない。
    assert [r["experiment_id"] for r in read_latency_rows(csv_path)] == ["exp-a"]


def test_compare_latency_refuses_values_from_different_sessions(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    append_metrics_row(_row("lat-1", "exp-a", 2.0, timestamp="2026-09-24T10:00:00+09:00"), csv_path=csv_path)
    append_metrics_row(_row("lat-2", "exp-b", 3.0, timestamp="2026-09-24T11:00:00+09:00"), csv_path=csv_path)
    with pytest.raises(LatencySessionMismatchError, match="同じセッション"):
        compare_latency(csv_path, ["exp-a", "exp-b"])
    with pytest.raises(LatencySessionMismatchError):
        compare_latency(csv_path, ["exp-a", "exp-b"], session_id="lat-1")
    # 片方だけなら比較相手がいないので可。
    assert compare_latency(csv_path, ["exp-a"])[0] == "lat-1"


def test_compare_latency_picks_latest_common_session(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    for sid, ts, (a, b) in (("lat-old", "2026-09-24T10:00:00+09:00", (2.0, 3.0)),
                            ("lat-new", "2026-09-24T12:00:00+09:00", (4.0, 5.0))):
        append_metrics_row(_row(sid, "exp-a", a, timestamp=ts), csv_path=csv_path)
        append_metrics_row(_row(sid, "exp-b", b, timestamp=ts), csv_path=csv_path)
    sid, values = compare_latency(csv_path, ["exp-a", "exp-b"])
    assert sid == "lat-new"
    assert values == {"exp-a": {"median_ms": 4.0, "max_ms": 8.0}, "exp-b": {"median_ms": 5.0, "max_ms": 10.0}}
    sid, values = compare_latency(csv_path, ["exp-a", "exp-b"], session_id="lat-old")
    assert values["exp-a"]["median_ms"] == 2.0


def test_check_same_session():
    assert check_same_session([{"latency_session_id": "x"}, {"latency_session_id": "x"}]) == "x"
    with pytest.raises(LatencySessionMismatchError):
        check_same_session([{"latency_session_id": "x"}, {"latency_session_id": "y"}])
    with pytest.raises(LatencySessionMismatchError, match="旧方式"):
        check_same_session([{"latency_session_id": "x"}, {"latency_session_id": ""}])


# --- 入口 scripts/measure_latency.py -----------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("measure_latency", ROOT / "scripts" / "measure_latency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_measures_checkpoints_in_one_session(tmp_path, monkeypatch):
    checkpoints = []
    for index, name in enumerate(("expA", "expB")):
        torch.manual_seed(index)
        model = SpeechRateCNN(SMALL)
        run_dir = tmp_path / name
        run_dir.mkdir()
        path = run_dir / "checkpoint_best.pt"
        torch.save({"model_config": SMALL.as_dict(), "model_state": model.state_dict(), "epoch": 1}, path)
        checkpoints.append(path)
    csv_path = tmp_path / "out" / "metrics.csv"
    json_path = tmp_path / "out" / "latency.json"
    monkeypatch.chdir(ROOT)  # configs/normalization.yaml を読むため
    script = _load_script()
    assert script.main(["--checkpoint", str(checkpoints[0]), "--checkpoint", str(checkpoints[1]),
                        "--device", "cpu", "--repeats", "100", "--warmup", "2",
                        "--metrics-csv", str(csv_path), "--output-json", str(json_path)]) == 0
    rows = read_latency_rows(csv_path)
    assert [r["experiment_id"] for r in rows] == ["expA", "expB"]
    assert len({r["latency_session_id"] for r in rows}) == 1
    assert all(r["model_size_bytes"] for r in rows)
    import json

    detail = json.loads(json_path.read_text(encoding="utf-8"))
    assert detail["repeats"] == 100 and detail["warmup"] == 2
    assert [len(v) for v in detail["times_ms"].values()] == [100, 100]
    assert detail["latency_session_id"] == rows[0]["latency_session_id"]
    compare_latency(csv_path, ["expA", "expB"])  # 同じセッションなので比較できる
