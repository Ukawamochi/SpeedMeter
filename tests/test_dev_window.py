"""窓単位の評価セット dev_window の生成の検証（src/spkrate/eval/dev_window.py）。

実データは使わず、合成のアライメントと波形で確かめる。
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from spkrate.data.augment import ArrayNoiseSource
from spkrate.eval.dev_window import (
    KIND_CONCAT,
    KIND_SINGLE,
    DevWindowAudio,
    WindowSource,
    build_concat_groups,
    build_dev_window,
    isolation_flags,
    load_dev_window,
    load_known_no_speech,
    load_window_config,
    save_dev_window,
    source_moras,
    window_mora,
    window_starts,
)
from spkrate.eval.noisy import clip_key, clip_rng, degrade_waveform, load_noisy_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval" / "dev_window.yaml"
SR = 16000


def _config(seed: int | None = None):
    config = load_window_config(CONFIG_PATH)
    return config if seed is None else replace(config, seed=seed)


def _alignment(duration: float, times: list[float], width: float = 0.02) -> dict:
    return {
        "ok": True,
        "duration_sec": duration,
        "moras": [{"kana": "ア", "start": t, "end": t + width} for t in times],
    }


def _dataset():
    """話者 A（7件）・B（4件）・C（2件）の合成アライメント。"""
    alignments: dict[str, dict] = {}
    clip_to_client: dict[str, str] = {}
    for speaker, count in (("A", 7), ("B", 4), ("C", 2)):
        for n in range(count):
            clip_id = f"{speaker}{n}"
            duration = 2.5 + 0.5 * n
            times = list(np.arange(0.5, duration - 0.4, 0.1))
            alignments[clip_id] = _alignment(duration, [float(t) for t in times])
            clip_to_client[clip_id] = speaker
    return alignments, clip_to_client


# ---------------------------------------------------------------------------------------
# 設定


def test_config_values_match_directive_and_dev_noisy():
    config = _config()
    assert config.window_sec == 2.0 and config.hop_sec == 0.25
    assert config.window_samples == 32000 and config.hop_samples == 4000
    assert config.concat.group_size == (3, 5)
    assert config.concat.gap_sec == (0.3, 1.5)
    assert config.isolation_threshold_sec == 1.0
    assert "test" not in config.dev_split
    assert config.noisy == load_noisy_config(ROOT / "configs" / "eval" / "dev_noisy.yaml")


# ---------------------------------------------------------------------------------------
# 窓と按分


def test_window_starts_drop_tail_and_short_clips():
    assert window_starts(31999, 32000, 4000).size == 0
    assert window_starts(32000, 32000, 4000).tolist() == [0]
    # 2.3秒: 開始 0, 0.25 まで（0.5 だと 2.5 秒ではみ出す）
    assert window_starts(36800, 32000, 4000).tolist() == [0, 4000]


def test_window_mora_prorates_edge_moras():
    starts = np.array([0.1, 1.95, 2.5], dtype=np.float32)
    ends = np.array([0.2, 2.05, 2.6], dtype=np.float32)
    labels = window_mora(starts, ends, np.array([0.0, 0.25, 3.0], dtype=np.float32), 2.0)
    # [0,2): 1 + 0.5（1.95〜2.05 の半分）。[0.25,2.25): 0.1〜0.2 は窓外、1.95〜2.05 は全部入る → 1。
    # [3,5): 0
    np.testing.assert_allclose(labels, [1.5, 1.0, 0.0], atol=1e-5)
    part = window_mora(
        np.array([1.9], dtype=np.float32), np.array([2.2], dtype=np.float32),
        np.array([0.0], dtype=np.float32), 2.0,
    )
    np.testing.assert_allclose(part, [1.0 / 3.0], atol=1e-5)


def test_window_mora_point_spans():
    starts = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    ends = np.array([1.0, 1.9, 0.0], dtype=np.float32)  # いずれも end <= start
    labels = window_mora(starts, ends, np.array([0.0, 0.5, 1.5], dtype=np.float32), 2.0)
    # [0,2): 1.0 と 0.0 → 2（2.0 は入らない）。[0.5,2.5): 1.0 と 2.0 → 2。[1.5,3.5): 2.0 → 1
    np.testing.assert_allclose(labels, [2.0, 2.0, 1.0])


def test_window_mora_total_is_preserved_across_tiling():
    rng = np.random.default_rng(0)
    starts = np.sort(rng.uniform(0.0, 7.9, size=40)).astype(np.float32)
    ends = starts + rng.uniform(0.0, 0.1, size=40).astype(np.float32)
    tiles = window_mora(starts, ends, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32), 2.0)
    assert tiles.sum() == pytest.approx(40.0, abs=1e-3)


def test_zero_windows_are_kept():
    alignments = {"X": _alignment(6.0, [0.5, 0.6, 0.7])}
    ws = build_dev_window(_config(), alignments=alignments, clip_to_client={"X": "s"})
    assert len(ws) == 17  # (6.0 − 2.0) / 0.25 + 1
    zero = ws.mora == 0
    assert zero.sum() > 0
    assert ws.meta["counts"]["single_windows_zero"] == int(zero.sum())
    # 開始 1.0 秒以降の窓は発話を含まない
    starts = ws.start_sample / SR
    assert np.all(ws.mora[starts >= 0.72] == 0)
    assert np.all(ws.mora[starts <= 0.5] == pytest.approx(3.0))


# ---------------------------------------------------------------------------------------
# 除外


def test_isolation_flags_threshold_is_strict():
    base = [{"start": 0.5}, {"start": 0.6}, {"start": 0.7}, {"start": 0.8}]
    assert isolation_flags(base, 1.0) == (False, False)
    head = [{"start": 0.0}, {"start": 1.01}, {"start": 1.1}, {"start": 1.2}]
    assert isolation_flags(head, 1.0) == (True, False)
    exact = [{"start": 0.0}, {"start": 1.0}, {"start": 1.1}]
    assert isolation_flags(exact, 1.0) == (False, False)
    tail = [{"start": 0.0}, {"start": 0.1}, {"start": 2.0}]
    assert isolation_flags(tail, 1.0) == (False, True)
    assert isolation_flags([{"start": 0.0}], 1.0) == (False, False)


def test_excluded_clips_are_not_in_single_or_concat():
    alignments, clip_to_client = _dataset()
    alignments["A0"]["moras"][0]["start"] = 0.0  # 先頭2モーラの差 1.2 秒
    alignments["A0"]["moras"][0]["end"] = 0.02
    alignments["A0"]["moras"][1]["start"] = 1.2
    alignments["A1"]["moras"][-1]["start"] += 1.5
    alignments["A1"]["moras"][-1]["end"] += 1.5
    alignments["A1"]["duration_sec"] += 1.5
    alignments["B0"] = {"clip_id": "B0", "ok": False, "reason": "align_failed"}
    ws = build_dev_window(_config(), alignments=alignments, clip_to_client=clip_to_client)
    used = {c for s in ws.sources for c in s.clip_ids}
    assert not used & {"A0", "A1", "B0"}
    counts = ws.meta["counts"]
    assert counts["excluded_head_only"] == 1
    assert counts["excluded_tail_only"] == 1
    assert counts["excluded_failed"] == 1
    assert counts["excluded_total"] == 3


# ---------------------------------------------------------------------------------------
# 連結


def test_concat_groups_are_same_speaker_and_in_range():
    alignments, clip_to_client = _dataset()
    ws = build_dev_window(_config(), alignments=alignments, clip_to_client=clip_to_client)
    concat = [s for s in ws.sources if s.kind == KIND_CONCAT]
    assert concat
    seen: list[str] = []
    for source in concat:
        assert 3 <= len(source.clip_ids) <= 5
        assert {clip_to_client[c] for c in source.clip_ids} == {source.client_id}
        for gap in source.gap_samples:
            assert 0.3 * SR <= gap <= 1.5 * SR
            assert gap % 160 == 0  # 0.01 秒単位
        seen.extend(source.clip_ids)
    assert len(seen) == len(set(seen))  # 各クリップは高々1回
    assert "C" not in {s.client_id for s in concat}  # 2件の話者は組にならない


def test_concat_offsets_shift_mora_times():
    alignments = {
        "p": _alignment(2.5, [0.5, 1.0]),
        "q": _alignment(3.0, [0.2]),
        "r": _alignment(2.0, [1.5]),
    }
    source = WindowSource(
        source_id="g", kind=KIND_CONCAT, client_id="s", clip_ids=("p", "q", "r"),
        clip_samples=(40000, 48000, 32000), gap_samples=(8000, 16000),
    )
    assert source.offsets == (0, 48000, 112000)
    assert source.num_samples == 40000 + 8000 + 48000 + 16000 + 32000
    starts, ends = source_moras(source, alignments, SR)
    np.testing.assert_allclose(starts, [0.5, 1.0, 3.0 + 0.2, 7.0 + 1.5], atol=1e-5)
    np.testing.assert_allclose(ends - starts, 0.02, atol=1e-5)


def test_generation_is_deterministic_and_seed_changes_groups():
    alignments, clip_to_client = _dataset()
    a = build_dev_window(_config(), alignments=alignments, clip_to_client=clip_to_client)
    b = build_dev_window(_config(), alignments=alignments, clip_to_client=clip_to_client)
    assert a.sources == b.sources
    for key in ("source_index", "start_sample", "mora", "kind", "known_no_speech"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    c = build_dev_window(_config(seed=1), alignments=alignments, clip_to_client=clip_to_client)
    groups = lambda ws: [(s.clip_ids, s.gap_samples) for s in ws.sources if s.kind == KIND_CONCAT]  # noqa: E731
    assert groups(a) != groups(c)
    # 単一クリップの窓は種に依存しない
    singles = lambda ws: [s for s in ws.sources if s.kind == KIND_SINGLE]  # noqa: E731
    assert singles(a) == singles(c)


def test_groups_do_not_depend_on_other_speakers():
    spec = _config().concat
    full, _ = build_concat_groups({"A": list("abcdefg"), "B": list("hijk")}, seed=5, spec=spec, sample_rate=SR)
    only_a, _ = build_concat_groups({"A": list("abcdefg")}, seed=5, spec=spec, sample_rate=SR)
    assert [g for g in full if g[0] == "A"] == only_a


def test_known_no_speech_marks_but_keeps(tmp_path):
    tsv = tmp_path / "to_listen.tsv"
    tsv.write_text("# 注記\nrank\tclip_id\n1\tA2\n2\tZZ\n", encoding="utf-8")
    known = load_known_no_speech(tsv)
    assert known == {"A2", "ZZ"}
    alignments, clip_to_client = _dataset()
    ws = build_dev_window(_config(), alignments=alignments, clip_to_client=clip_to_client, known_no_speech=known)
    marked = [s for s in ws.sources if s.known_no_speech]
    assert any(s.kind == KIND_SINGLE and s.clip_ids == ("A2",) for s in marked)
    assert all("A2" in s.clip_ids for s in marked)
    assert ws.meta["counts"]["known_no_speech_single_sources"] == 1
    assert ws.known_no_speech.sum() == sum(
        int((ws.source_index == i).sum()) for i, s in enumerate(ws.sources) if s.known_no_speech
    )


def test_save_and_load_roundtrip(tmp_path):
    alignments, clip_to_client = _dataset()
    ws = build_dev_window(_config(), alignments=alignments, clip_to_client=clip_to_client)
    save_dev_window(ws, tmp_path / "dw")
    back = load_dev_window(tmp_path / "dw")
    assert back.sources == ws.sources
    np.testing.assert_array_equal(back.mora, ws.mora)
    np.testing.assert_array_equal(back.start_sample, ws.start_sample)
    assert json.loads((tmp_path / "dw" / "meta.json").read_text())["counts"] == ws.meta["counts"]


# ---------------------------------------------------------------------------------------
# 波形の再生成


def _fake_audio(alignments):
    def loader(path: str) -> np.ndarray:
        clip_id = Path(path).stem
        n = int(round(alignments[clip_id]["duration_sec"] * SR))
        rng = np.random.default_rng(clip_key(clip_id))
        return (0.1 * rng.standard_normal(n)).astype(np.float32) + np.float32(0.01)

    paths = {c: f"{c}.wav" for c in alignments}
    return loader, paths


def test_waveform_regeneration_concat_and_noise():
    alignments, clip_to_client = _dataset()
    config = _config()
    ws = build_dev_window(config, alignments=alignments, clip_to_client=clip_to_client)
    loader, paths = _fake_audio(alignments)
    noise = ArrayNoiseSource([np.random.default_rng(1).standard_normal(50000).astype(np.float32)])
    audio = DevWindowAudio(config, paths, loader=loader, noise_source=noise)
    source = next(s for s in ws.sources if s.kind == KIND_CONCAT)
    whole = audio.source_waveform(source)
    assert whole.size == source.num_samples
    first_end = source.clip_samples[0]
    gap = whole[first_end : first_end + source.gap_samples[0]]
    assert gap.size == source.gap_samples[0] and np.all(gap == 0)
    np.testing.assert_array_equal(whole[:first_end], loader(paths[source.clip_ids[0]]))
    index = int(np.flatnonzero(ws.source_index == ws.sources.index(source))[3])
    win = audio.window_at(ws, index)
    start = int(ws.start_sample[index])
    np.testing.assert_array_equal(win, whole[start : start + 32000])
    # 雑音は音源全体に掛けてから切る（dev_noisy と同じ加工・鍵は noise_key）
    noisy = DevWindowAudio(config, paths, loader=loader, noise_source=noise).window_at(ws, index, 5.0)
    expected = degrade_waveform(whole, source.noise_key, 5.0, config=config.noisy, noise_source=noise)
    np.testing.assert_array_equal(noisy, expected[start : start + 32000])
    again = DevWindowAudio(config, paths, loader=loader, noise_source=noise).window_at(ws, index, 5.0)
    np.testing.assert_array_equal(noisy, again)


def test_single_clip_noise_matches_dev_noisy_key():
    source = WindowSource("c1", KIND_SINGLE, "s", ("c1",), (32000,), ())
    assert source.noise_key == "c1"
    concat = WindowSource("g", KIND_CONCAT, "s", ("a", "b", "c"), (1, 1, 1), (1, 1))
    assert concat.noise_key == "concat:a+b+c"
    # dev_noisy と同じ乱数生成器になる
    cfg = _config().noisy
    assert clip_rng(cfg.seed, source.noise_key).integers(0, 10**9) == clip_rng(cfg.seed, "c1").integers(0, 10**9)


def test_loader_length_mismatch_stops():
    config = _config()
    source = WindowSource("c1", KIND_SINGLE, "s", ("c1",), (32001,), ())
    audio = DevWindowAudio(config, {"c1": "c1.wav"}, loader=lambda p: np.zeros(32000, np.float32))
    with pytest.raises(ValueError, match="長さ"):
        audio.window(source, 0)


def test_noisy_version_reads_eval_noise_only(tmp_path):
    import soundfile as sf

    from spkrate.data.musan_split import build_musan_noise_split, list_noise_files
    from spkrate.eval.noisy import make_noise_source

    root = tmp_path / "musan"
    rng = np.random.default_rng(0)
    for group in ("free-sound", "sound-bible"):
        directory = root / "noise" / group
        directory.mkdir(parents=True)
        for index in range(5):
            sf.write(directory / f"{group}-{index}.wav", (0.1 * rng.standard_normal(8000)).astype(np.float32), SR)
    payload = build_musan_noise_split(list_noise_files(root), seed=3)
    split = tmp_path / "musan_noise.json"
    split.write_text(json.dumps(payload), encoding="utf-8")

    mapping = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert mapping["noisy"]["musan_noise_split"] == "configs/splits/musan_noise.json"
    mapping["noisy"].update(musan_root=str(root), musan_noise_split=str(split))
    path = tmp_path / "dev_window.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    config = load_window_config(path)
    source_noise = make_noise_source(config.noisy)
    loaded: list[Path] = []
    original = source_noise.load
    source_noise.load = lambda p, _o=original: (loaded.append(Path(p)), _o(p))[1]

    alignments, clip_to_client = _dataset()
    ws = build_dev_window(config, alignments=alignments, clip_to_client=clip_to_client)
    loader, paths = _fake_audio(alignments)
    audio = DevWindowAudio(config, paths, loader=loader, noise_source=source_noise)
    for i in range(0, len(ws), 7):
        audio.window_at(ws, i, 10.0)
    relative = {p.relative_to(root).as_posix() for p in loaded}
    assert relative and relative <= set(payload["eval"])
    assert not relative & set(payload["train"])


def test_noisy_without_split_stops(tmp_path):
    from spkrate.data.musan_split import MusanSplitError

    mapping = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mapping["noisy"].pop("musan_noise_split")
    path = tmp_path / "dev_window.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    with pytest.raises(MusanSplitError, match="未設定"):
        load_window_config(path)
