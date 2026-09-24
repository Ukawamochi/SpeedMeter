"""雑音下評価セット dev_noisy の生成と記録の検証（src/spkrate/eval/noisy.py、runner.py）。

本物の results/metrics.csv は書き換えない。追記の検証は tmp_path で行う。
MUSAN が無い環境（worktree など）では、実データを使う検査だけを飛ばす。
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from spkrate.data.augment import ArrayNoiseSource, SAMPLE_RATE
from spkrate.eval.noisy import (
    NoisyDevConfig,
    ReverbSpec,
    clip_key,
    degrade_waveform,
    fixed_rir,
    load_noisy_config,
    make_noise_source,
    _fixed_rir_cached,
)
from spkrate.eval.runner import (
    METRICS_CSV_COLUMNS,
    NOISY_POOLED_SPLIT,
    EvalSegment,
    evaluate,
    noisy_config_path,
    noisy_split_name,
    pool_metrics,
    run_and_record_clean_and_noisy,
)
from spkrate.eval.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval" / "dev_noisy.yaml"
CLIP_IDS = [f"common_voice_ja_{n}" for n in (1, 22, 333, 4444, 55555)]


def _speech_like(clip_id: str, seconds: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(clip_key(clip_id) % (2**32))
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t)).astype(np.float32)
    carrier = rng.standard_normal(n).astype(np.float32)
    return (0.1 * envelope * carrier).astype(np.float32)


def _noise_bank() -> ArrayNoiseSource:
    rng = np.random.default_rng(0)
    return ArrayNoiseSource(
        [rng.standard_normal(length).astype(np.float32) for length in (40000, 70000, 100000)]
    )


def _config(seed: int | None = None) -> NoisyDevConfig:
    config = load_noisy_config(CONFIG_PATH)
    if seed is None:
        return config
    return NoisyDevConfig(seed=seed, snr_db=config.snr_db, reverb=config.reverb,
                          musan_root=config.musan_root, musan_subsets=config.musan_subsets)


def _generate(config: NoisyDevConfig) -> dict[tuple[str, float], np.ndarray]:
    """全クリップ × 全SNR を生成する。毎回、残響のキャッシュと雑音源を作り直す。"""
    _fixed_rir_cached.cache_clear()
    source = _noise_bank()
    return {
        (clip_id, snr): degrade_waveform(_speech_like(clip_id), clip_id, snr,
                                         config=config, noise_source=source)
        for clip_id in CLIP_IDS
        for snr in config.snr_db
    }


# --- 設定 ---------------------------------------------------------------------


def test_config_file_matches_task_definition():
    config = load_noisy_config(CONFIG_PATH)
    assert config.snr_db == (5.0, 10.0, 15.0)
    assert config.musan_subsets == ("noise",)
    assert config.musan_root == "data/musan"
    assert isinstance(config.seed, int)
    # 残響は固定値（範囲ではない）で記録されている
    assert len(config.reverb.room_dim_m) == 3
    assert config.reverb.rt60_sec > 0


def test_reverb_spec_rejects_positions_outside_room():
    with pytest.raises(ValueError):
        ReverbSpec(room_dim_m=(4.0, 4.0, 3.0), rt60_sec=0.4, source_pos_m=(5.0, 1.0, 1.0),
                   mic_pos_m=(1.0, 1.0, 1.0), max_order=10)


# --- 決定性 -------------------------------------------------------------------


def test_same_seed_is_bit_identical():
    first = _generate(_config())
    second = _generate(_config())
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].dtype == np.float32
        assert first[key].tobytes() == second[key].tobytes(), key


def test_different_seed_differs():
    base = _config()
    first = _generate(base)
    other = _generate(_config(base.seed + 1))
    differing = [key for key in first if first[key].tobytes() != other[key].tobytes()]
    # 雑音はどれもクリップより長く、切り出し位置も乱数で決まる。雑音ファイルが
    # たまたま同じになっても位置まで一致することはまず無いので、全件が異なることを求める
    assert len(differing) == len(first)


def test_fixed_rir_is_deterministic_and_normalized():
    spec = _config().reverb
    _fixed_rir_cached.cache_clear()
    first = np.array(fixed_rir(spec))
    _fixed_rir_cached.cache_clear()
    second = np.array(fixed_rir(spec))
    assert first.tobytes() == second.tobytes()
    assert first.dtype == np.float32
    assert float(np.max(np.abs(first))) == pytest.approx(1.0)
    assert first.size > 1  # 無響ではない


def test_noise_choice_does_not_depend_on_snr_or_order():
    """同じクリップでは SNR が違っても同じ雑音を使い、評価順にも依存しない。"""
    config = _config()
    samples = _speech_like(CLIP_IDS[0])
    source = _noise_bank()
    reverberant = degrade_waveform(samples, CLIP_IDS[0], 1000.0, config=config,
                                   noise_source=source)  # 雑音をほぼ無視できるSNR
    residual = {}
    for snr in config.snr_db:
        noisy = degrade_waveform(samples, CLIP_IDS[0], snr, config=config, noise_source=source)
        residual[snr] = (noisy - reverberant).astype(np.float32)
    # 雑音成分は SNR で振幅だけが変わる（向きは同じ）
    a, b = residual[5.0], residual[15.0]
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.999
    # 別のクリップを先に処理しても結果は変わらない
    before = degrade_waveform(samples, CLIP_IDS[0], 10.0, config=config, noise_source=source)
    degrade_waveform(_speech_like(CLIP_IDS[1]), CLIP_IDS[1], 10.0, config=config,
                     noise_source=source)
    after = degrade_waveform(samples, CLIP_IDS[0], 10.0, config=config, noise_source=source)
    assert before.tobytes() == after.tobytes()


def test_degraded_waveform_keeps_length_and_hits_snr():
    config = _config()
    samples = _speech_like(CLIP_IDS[2])
    source = _noise_bank()
    clean_reverb = degrade_waveform(samples, CLIP_IDS[2], 1000.0, config=config,
                                    noise_source=source)
    for snr in config.snr_db:
        noisy = degrade_waveform(samples, CLIP_IDS[2], snr, config=config, noise_source=source)
        assert noisy.shape == samples.shape
        noise = noisy - clean_reverb
        measured = 10 * np.log10(np.mean(clean_reverb**2) / np.mean(noise**2))
        assert measured == pytest.approx(snr, abs=0.05)


def test_clip_key_is_stable():
    # Python の hash と違いプロセスごとに変わらない（SHA-256 の先頭8バイト）
    assert clip_key("common_voice_ja_1") == int.from_bytes(
        __import__("hashlib").sha256(b"common_voice_ja_1").digest()[:8], "big"
    )


@pytest.mark.skipif(not (ROOT / "data" / "musan" / "noise").is_dir(),
                    reason="MUSAN が配置されていない")
def test_real_musan_generation_is_bit_identical():
    config = _config()
    samples = _speech_like(CLIP_IDS[0], seconds=3.0)
    outputs = []
    for _ in range(2):
        source = make_noise_source(config, ROOT)
        outputs.append(degrade_waveform(samples, CLIP_IDS[0], 5.0, config=config,
                                        noise_source=source))
    assert outputs[0].tobytes() == outputs[1].tobytes()


# --- 記録 ---------------------------------------------------------------------


def test_split_names_and_config_path():
    assert noisy_split_name(5) == "dev_noisy_snr5"
    assert noisy_split_name(10.0) == "dev_noisy_snr10"
    assert NOISY_POOLED_SPLIT == "dev_noisy_all"
    assert noisy_config_path("configs/exp001.yaml", "configs/eval/dev_noisy.yaml") == (
        "configs/exp001.yaml;configs/eval/dev_noisy.yaml"
    )


def test_pool_metrics_concatenates_conditions():
    parts = [([10.0, 20.0], [11.0, 18.0], [2.0, 4.0]), ([10.0, 20.0], [9.0, 25.0], [2.0, 4.0])]
    pooled = pool_metrics(parts)
    direct = compute_metrics([10, 20, 10, 20], [11, 18, 9, 25], [2, 4, 2, 4])
    assert pooled == direct


def test_run_and_record_clean_and_noisy_writes_five_rows(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    segments = [
        EvalSegment(segment_id=clip_id, audio_path=Path(clip_id), true_mora=9.0,
                    duration_sec=1.5)
        for clip_id in CLIP_IDS
    ]

    def loader(path):
        return _speech_like(str(path)), SAMPLE_RATE

    def estimator(samples, sample_rate):
        # 実効値に応じて変わる推定器（雑音で値が変わることを確かめる）
        return float(np.sqrt(np.mean(samples**2)) * 50.0)

    outputs = run_and_record_clean_and_noisy(
        estimator, segments,
        noisy_config_file=CONFIG_PATH,
        experiment_id="000-dummy", method="rms", config_path="configs/dummy.yaml",
        csv_path=csv_path, repo_dir=ROOT, audio_loader=loader, noise_source=_noise_bank(),
    )
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == METRICS_CSV_COLUMNS
        rows = list(reader)
    assert [row["split"] for row in rows] == [
        "dev", "dev_noisy_snr5", "dev_noisy_snr10", "dev_noisy_snr15", "dev_noisy_all",
    ]
    assert rows[0]["config_path"] == "configs/dummy.yaml"
    assert all(row["config_path"] == f"configs/dummy.yaml;{CONFIG_PATH}" for row in rows[1:])
    assert rows[-1]["num_segments"] == str(3 * len(segments))
    assert {row["latency_ms_per_inference"] for row in rows} == {
        rows[0]["latency_ms_per_inference"]
    }
    clean_mae = float(rows[0]["mae_moras_per_sec"])
    snr5_mae = float(rows[1]["mae_moras_per_sec"])
    assert snr5_mae != pytest.approx(clean_mae)
    per_condition = [float(row["mae_moras_per_sec"]) for row in rows[1:4]]
    assert float(rows[-1]["mae_moras_per_sec"]) == pytest.approx(np.mean(per_condition))
    # 雑音下の評価はクリップごとに決定的（同じ入力なら同じ推定値）
    again = evaluate(estimator, segments, audio_loader=loader)
    assert again.metrics == outputs["dev"].metrics
