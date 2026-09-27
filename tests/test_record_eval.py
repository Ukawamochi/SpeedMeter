"""録音の道具（scripts/record_eval.py、src/spkrate/eval/record.py）のうちマイクを使わない部分の試験。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from spkrate.eval.record import (
    TARGET_SR,
    Sentence,
    build_filename,
    load_sentences,
    run_session,
    save_wav,
    to_mono_16k,
    validate_condition,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("record_eval", ROOT / "scripts" / "record_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- 文の読み込み ----


def test_load_sentences_tsv(tmp_path):
    path = tmp_path / "s.tsv"
    path.write_text(
        "sentence_id\ttext\tkana\tmora\n"
        "BASIC5000_0001\t水をマレーシアから買わなくてはならないのです。\tミズヲ\t3\n"
        "BASIC5000_0002\t木曜日。\tモクヨービ\t5\n",
        encoding="utf-8",
    )
    s = load_sentences(path)
    assert [x.index for x in s] == [1, 2]
    assert s[0].sentence_id == "BASIC5000_0001"
    assert s[0].text == "水をマレーシアから買わなくてはならないのです。"
    assert s[1].kana == "モクヨービ"


def test_load_sentences_txt(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("A_1:一つ目の文。\n\nB_2:二つ目:コロンを含む文。\n", encoding="utf-8")
    s = load_sentences(path)
    assert [(x.index, x.sentence_id, x.text, x.kana) for x in s] == [
        (1, "A_1", "一つ目の文。", ""),
        (2, "B_2", "二つ目:コロンを含む文。", ""),
    ]


@pytest.mark.parametrize(
    "content",
    ["", "区切りのない行\n", "A:一\nA:二\n", "A B:空白を含むID\n"],
)
def test_load_sentences_rejects(tmp_path, content):
    path = tmp_path / "s.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_sentences(path)


def test_load_sentences_tsv_missing_column(tmp_path):
    path = tmp_path / "s.tsv"
    path.write_text("id\ttext\nA\t文\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sentences(path)


# ---- ファイル名と条件 ----


def test_build_filename():
    s = Sentence(index=3, sentence_id="BASIC5000_0036", text="文")
    assert build_filename("builtin", "normal", s) == "builtin_normal_03_BASIC5000_0036.wav"
    assert build_filename("far", "veryfast", s) == "far_veryfast_03_BASIC5000_0036.wav"


def test_filenames_unique_over_nine_conditions():
    s = [Sentence(index=i, sentence_id=f"ID_{i}", text="文") for i in (1, 2)]
    names = {
        build_filename(m, r, x)
        for m in ("builtin", "earphone", "far")
        for r in ("normal", "fast", "veryfast")
        for x in s
    }
    assert len(names) == 18


@pytest.mark.parametrize("mic,rate", [("desk", "normal"), ("builtin", "slow"), ("", "")])
def test_validate_condition_rejects(mic, rate):
    with pytest.raises(ValueError):
        validate_condition(mic, rate)


def test_parse_args_ok():
    mod = _load_script()
    args = mod.parse_args(["--mic", "earphone", "--rate", "fast", "--device", "3", "--start", "5"])
    assert (args.mic, args.rate, args.device, args.start) == ("earphone", "fast", 3, 5)
    assert args.out_dir == ROOT / "data" / "eval_real"
    assert args.sentences == ROOT / "data" / "eval_scripts" / "sentences_labeled.tsv"


def test_parse_args_list_devices_needs_no_condition():
    mod = _load_script()
    assert mod.parse_args(["--list-devices"]).list_devices


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--mic", "builtin"],
        ["--rate", "normal"],
        ["--mic", "table", "--rate", "normal"],
        ["--mic", "builtin", "--rate", "slow"],
        ["--mic", "builtin", "--rate", "normal", "--start", "0"],
    ],
)
def test_parse_args_rejects(argv):
    mod = _load_script()
    with pytest.raises(SystemExit):
        mod.parse_args(argv)


def test_default_out_dir_is_gitignored():
    import subprocess

    target = "data/eval_real/builtin_normal_01_X.wav"
    result = subprocess.run(
        ["git", "check-ignore", "-q", target], cwd=ROOT, capture_output=True
    )
    assert result.returncode == 0


# ---- 16kHz・モノラルでの書き出し ----


def test_to_mono_16k_downmix_and_resample():
    sr = 48000
    t = np.arange(sr) / sr
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, left], axis=1)
    out = to_mono_16k(stereo, sr)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == TARGET_SR
    assert np.max(np.abs(out)) == pytest.approx(0.5, abs=0.02)


def test_to_mono_16k_passthrough():
    x = np.linspace(-0.5, 0.5, 1600).astype(np.float32)
    np.testing.assert_array_equal(to_mono_16k(x, TARGET_SR), x)


def test_to_mono_16k_44100():
    out = to_mono_16k(np.zeros(44100, np.float32), 44100)
    assert out.size == TARGET_SR


def test_save_wav_format(tmp_path):
    path = tmp_path / "sub" / "a.wav"
    save_wav(path, np.full((48000, 2), 0.25, np.float32), 48000)
    info = sf.info(path)
    assert (info.samplerate, info.channels, info.subtype, info.format) == (16000, 1, "PCM_16", "WAV")
    assert info.frames == 16000
    assert not list(tmp_path.glob("**/*.part"))


def test_save_wav_overwrites(tmp_path):
    path = tmp_path / "a.wav"
    save_wav(path, np.full(32000, 0.1, np.float32), 16000)
    save_wav(path, np.full(8000, -0.2, np.float32), 16000)
    data, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and data.size == 8000
    assert float(data.mean()) == pytest.approx(-0.2, abs=1e-3)


# ---- 録音の進行（偽のマイク） ----


class FakeRecorder:
    """start ごとに違う長さ・値の音声を返す。"""

    def __init__(self, sr=48000):
        self.sr = sr
        self.calls = 0
        self.recording = False

    def start(self):
        assert not self.recording
        self.recording = True
        self.calls += 1

    def stop(self):
        assert self.recording
        self.recording = False
        n = self.sr * self.calls  # 1回目は1秒、2回目は2秒…
        return np.full(n, 0.01 * self.calls, np.float32), self.sr


def _inputs(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def _sentences():
    return [Sentence(index=i, sentence_id=f"S_{i}", text=f"文{i}") for i in (1, 2, 3)]


def test_run_session_records_all(tmp_path):
    rec = FakeRecorder()
    # 各文: 開始 Enter、終了 Enter、次へ Enter
    saved = run_session(
        _sentences(), "builtin", "normal", tmp_path, rec,
        input_fn=_inputs(*[""] * 9), print_fn=lambda s: None,
    )
    assert [p.name for p in saved] == [
        "builtin_normal_01_S_1.wav",
        "builtin_normal_02_S_2.wav",
        "builtin_normal_03_S_3.wav",
    ]
    assert rec.calls == 3
    for p in saved:
        assert sf.info(p).samplerate == 16000


def test_run_session_redo_overwrites_same_file(tmp_path):
    rec = FakeRecorder()
    answers = [
        "", "", "r",  # 文1: 録音して録り直しを選ぶ
        "", "", "",  # 文1: 録り直して次へ
        "", "", "q",  # 文2: 録音して終了
    ]
    saved = run_session(
        _sentences(), "far", "fast", tmp_path, rec,
        input_fn=_inputs(*answers), print_fn=lambda s: None,
    )
    assert [p.name for p in saved] == ["far_fast_01_S_1.wav", "far_fast_02_S_2.wav"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["far_fast_01_S_1.wav", "far_fast_02_S_2.wav"]
    data, _ = sf.read(tmp_path / "far_fast_01_S_1.wav", dtype="float32")
    assert data.size == 2 * 16000  # 2回目の録音（2秒）で上書きされている
    assert float(data.mean()) == pytest.approx(0.02, abs=1e-3)
    assert rec.calls == 3


def test_run_session_quit_before_recording(tmp_path):
    rec = FakeRecorder()
    saved = run_session(
        _sentences(), "earphone", "veryfast", tmp_path, rec,
        input_fn=_inputs("q"), print_fn=lambda s: None,
    )
    assert saved == [] and rec.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_run_session_start(tmp_path):
    rec = FakeRecorder()
    saved = run_session(
        _sentences(), "builtin", "normal", tmp_path, rec,
        input_fn=_inputs(*[""] * 3), print_fn=lambda s: None, start=3,
    )
    assert [p.name for p in saved] == ["builtin_normal_03_S_3.wav"]
    with pytest.raises(ValueError):
        run_session(_sentences(), "builtin", "normal", tmp_path, rec, start=4)
