"""拡張ありの経路（WaveformClipDataset）で既定の audio_root から音声を開けるか。

clips.jsonl の ``audio_path`` はリポジトリ直下からの相対（``data/common_voice_ja/clips/…``）
であり、``DataSettings`` の既定 ``audio_root`` と結合して実在するパスになる必要がある
（results/augmentation_check.md 3.2節「音声パス」）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from spkrate.data.augment import AugmentConfig
from spkrate.data.common_voice import ClipRecord
from spkrate.train.data import WaveformClipDataset
from spkrate.train.train import DataSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_JSONL = REPO_ROOT / "data" / "processed" / "clips.jsonl"
NUM_REAL_CLIPS = 3


def _record(clip_id: str, audio_path: str, duration_sec: float) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        audio_path=audio_path,
        client_id="c1",
        sentence="テスト",
        kana="テスト",
        mora=3,
        duration_sec=duration_sec,
        mora_per_second=3.0 / duration_sec,
    )


def _load_real_records(limit: int) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with CLIPS_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not (REPO_ROOT / row["audio_path"]).is_file():
                continue
            records.append(
                ClipRecord(**{key: row[key] for key in ClipRecord.__dataclass_fields__})
            )
            if len(records) >= limit:
                break
    return records


def test_default_audio_root_resolves_synthetic_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clips.jsonl と同じ形式の相対パスが、既定の audio_root で実在ファイルに解決される。"""
    audio_path = "data/common_voice_ja/clips/synthetic_0.wav"
    target = tmp_path / audio_path
    target.parent.mkdir(parents=True)
    sample_rate = 16000
    duration_sec = 1.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    sf.write(target, 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32), sample_rate)

    monkeypatch.chdir(tmp_path)
    audio_root = DataSettings().audio_root
    assert (Path(audio_root) / audio_path).resolve() == target.resolve()

    dataset = WaveformClipDataset(
        [_record("synthetic_0", audio_path, duration_sec)],
        audio_root,
        augment=AugmentConfig(),
        seed=0,
    )
    item = dataset[0]
    assert item.clip_id == "synthetic_0"
    assert item.features.ndim == 2 and item.features.shape[0] > 0
    assert np.isfinite(item.features).all()
    assert item.duration_sec > 0


@pytest.mark.skipif(not CLIPS_JSONL.is_file(), reason="data/processed/clips.jsonl が無い")
def test_default_audio_root_loads_real_clips_with_augment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """実データ（data/common_voice_ja 以下）の数件を拡張ありで既定 audio_root のまま読める。"""
    records = _load_real_records(NUM_REAL_CLIPS)
    if not records:
        pytest.skip("data/common_voice_ja 以下に実在するクリップが無い")

    monkeypatch.chdir(REPO_ROOT)
    dataset = WaveformClipDataset(
        records, DataSettings().audio_root, augment=AugmentConfig(), seed=0
    )
    for index, record in enumerate(records):
        item = dataset[index]
        assert item.clip_id == record.clip_id
        assert item.features.ndim == 2 and item.features.shape[0] > 0
        assert np.isfinite(item.features).all()
        assert item.mora == float(record.mora)
