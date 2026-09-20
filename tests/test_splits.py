"""話者単位の分割の検証。

主眼は「同一話者が複数の分割に現れないこと」である。次の2通りで確かめる。

1. 生成済みの configs/splits/*.json を読み、話者の重複が無いことを確かめる
   （ファイルが無い環境では skip する）
2. 小さな合成データで、分割関数が話者単位であること・決定的であることを確かめる
   （実データに依存しない）
"""

import json
from pathlib import Path

import pytest

from spkrate.data.common_voice import ClipRecord
from spkrate.data.splits import (
    DEFAULT_RATIOS,
    SPLIT_NAMES,
    SPLIT_SEED,
    SpeakerStats,
    aggregate_speakers,
    assign_speakers,
    build_split_payload,
    find_overlaps,
    load_clip_records,
    load_split,
    load_test_split,
    summarize_splits,
    write_split_files,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "configs" / "splits"
CLIPS_JSONL = ROOT / "data" / "processed" / "clips.jsonl"


# --- 1. 生成済みの実ファイルに対する検証 ---------------------------------


def load_generated_splits():
    """configs/splits/*.json を読む。1つでも欠けていれば skip する。"""
    paths = {name: SPLIT_DIR / f"{name}.json" for name in SPLIT_NAMES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip(f"分割ファイルが未生成: {', '.join(missing)}")
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }


def test_generated_splits_have_no_shared_speaker():
    """生成済みの分割で、同一話者が複数の分割に現れない。"""
    payloads = load_generated_splits()
    assignment = {name: payload["client_ids"] for name, payload in payloads.items()}
    assert find_overlaps(assignment) == {}


def test_generated_splits_have_unique_ids_within_split():
    """各分割の中で client_id が重複しない。"""
    payloads = load_generated_splits()
    for name, payload in payloads.items():
        client_ids = payload["client_ids"]
        assert len(client_ids) == len(set(client_ids)), name
        assert client_ids == sorted(client_ids), f"{name} は昇順であること"
        assert payload["unit"] == "client_id"
        assert payload["seed"] == SPLIT_SEED
        assert payload["num_speakers"] == len(client_ids)


def test_generated_test_split_records_usage_restriction():
    """test.json に使用禁止の注記がある。"""
    payloads = load_generated_splits()
    assert "第10段階" in payloads["test"]["usage_restriction"]


def test_generated_splits_cover_all_speakers_in_clips():
    """クリップ一覧の全話者が、ちょうど1つの分割に入る。"""
    if not CLIPS_JSONL.exists():
        pytest.skip(f"クリップ一覧が未生成: {CLIPS_JSONL}")
    payloads = load_generated_splits()
    assigned: list[str] = []
    for payload in payloads.values():
        assigned.extend(payload["client_ids"])
    assert len(assigned) == len(set(assigned))
    speakers = {record.client_id for record in load_clip_records(CLIPS_JSONL)}
    assert set(assigned) == speakers


# --- 2. 合成データに対する検証 -------------------------------------------


def make_records(spec):
    """spec は (client_id, クリップ数, 1件あたりの秒数) の列。"""
    records = []
    for client_id, count, seconds in spec:
        for index in range(count):
            mora = 20
            records.append(
                ClipRecord(
                    clip_id=f"{client_id}-{index}",
                    audio_path=f"clips/{client_id}-{index}.mp3",
                    client_id=client_id,
                    sentence="今日はいい天気です",
                    kana="キョーワイイテンキデス",
                    mora=mora,
                    duration_sec=seconds,
                    mora_per_second=mora / seconds,
                )
            )
    return records


@pytest.fixture
def synthetic_records():
    """クリップ数が偏った30人分の合成データ。"""
    spec = []
    for index in range(30):
        client_id = f"spk{index:02d}"
        count = 1 + (index * 7) % 40
        seconds = 3.0 + (index % 5)
        spec.append((client_id, count, seconds))
    return make_records(spec)


def test_aggregate_speakers_counts_clips_and_duration():
    records = make_records([("a", 3, 2.0), ("b", 1, 5.0)])
    speakers = aggregate_speakers(records)
    assert [speaker.client_id for speaker in speakers] == ["a", "b"]
    assert speakers[0].num_clips == 3
    assert speakers[0].total_duration_sec == pytest.approx(6.0)
    assert speakers[1].total_duration_sec == pytest.approx(5.0)


def test_assign_speakers_is_speaker_level(synthetic_records):
    """1人の話者はちょうど1つの分割にだけ現れる。"""
    speakers = aggregate_speakers(synthetic_records)
    assignment = assign_speakers(speakers)

    assert find_overlaps(assignment) == {}
    assigned = [client_id for ids in assignment.values() for client_id in ids]
    assert sorted(assigned) == sorted(speaker.client_id for speaker in speakers)

    # クリップ単位で見ても、各話者のクリップは1つの分割に閉じている。
    owner = {
        client_id: name for name, ids in assignment.items() for client_id in ids
    }
    seen: dict[str, set[str]] = {}
    for record in synthetic_records:
        seen.setdefault(record.client_id, set()).add(owner[record.client_id])
    assert all(len(names) == 1 for names in seen.values())


def test_assign_speakers_is_deterministic(synthetic_records):
    """同じシード・同じ入力なら何度実行しても同じ分割になる。"""
    speakers = aggregate_speakers(synthetic_records)
    first = assign_speakers(speakers)
    second = assign_speakers(speakers)
    assert first == second


def test_assign_speakers_ignores_input_order(synthetic_records):
    """入力の並び順が変わっても結果は変わらない。"""
    speakers = aggregate_speakers(synthetic_records)
    shuffled = list(reversed(speakers))
    assert assign_speakers(speakers) == assign_speakers(shuffled)


def test_assign_speakers_uses_seed_for_ties():
    """クリップ数が同数の話者の並びはシードで決まる。

    貪欲法はクリップ数の多い順に置くため、乱数が効くのは同数の話者どうしの
    並びだけである。実データには同数（特に1件）の話者が多数あるので、ここでは
    全員同数の合成データで乱数が効いていることを確かめる。
    """
    records = make_records([(f"spk{index:02d}", 4, 3.0) for index in range(30)])
    speakers = aggregate_speakers(records)
    assert assign_speakers(speakers) != assign_speakers(speakers, seed=SPLIT_SEED + 1)
    # シードが同じなら同数の話者しかいなくても結果は変わらない。
    assert assign_speakers(speakers) == assign_speakers(speakers)


def test_assign_speakers_matches_clip_ratio(synthetic_records):
    """クリップ数の比率が目標（8:1:1）に概ね合う。"""
    speakers = aggregate_speakers(synthetic_records)
    assignment = assign_speakers(speakers)
    counts = {name: 0 for name in assignment}
    for speaker in speakers:
        for name, ids in assignment.items():
            if speaker.client_id in ids:
                counts[name] += speaker.num_clips
    total = sum(counts.values())
    for name, ratio in DEFAULT_RATIOS.items():
        assert counts[name] / total == pytest.approx(ratio, abs=0.03)


def test_assign_speakers_rejects_unknown_split_name():
    speakers = [SpeakerStats("a", 1, 1.0)]
    with pytest.raises(ValueError):
        assign_speakers(speakers, ratios={"train": 0.9, "valid": 0.1})


def test_find_overlaps_reports_shared_speaker():
    overlaps = find_overlaps({"train": ["a", "b"], "dev": ["b"], "test": ["c"]})
    assert overlaps == {"train&dev": {"b"}}


def test_summarize_splits_reports_size_and_rate(synthetic_records):
    speakers = aggregate_speakers(synthetic_records)
    assignment = assign_speakers(speakers)
    summaries = summarize_splits(synthetic_records, assignment)
    assert sum(s.num_clips for s in summaries.values()) == len(synthetic_records)
    for name, summary in summaries.items():
        assert summary.num_speakers == len(assignment[name])
        assert summary.total_duration_sec > 0.0
        assert summary.rate.minimum <= summary.rate.median <= summary.rate.maximum


def test_write_split_files_roundtrip(tmp_path, synthetic_records):
    """書き出した json を読み戻すと同じ話者集合になる。"""
    speakers = aggregate_speakers(synthetic_records)
    assignment = assign_speakers(speakers)
    written = write_split_files(tmp_path, assignment)

    assert set(written) == set(SPLIT_NAMES)
    for name in ("train", "dev"):
        assert sorted(load_split(written[name])) == sorted(assignment[name])
    assert sorted(
        load_test_split(written["test"], stage10_approved=True)
    ) == sorted(assignment["test"])


# --- 3. テストセットの使用禁止 -------------------------------------------


def test_load_test_split_refuses_without_stage10_flag(tmp_path, synthetic_records):
    """test.json は明示的な承認なしには読めない。"""
    speakers = aggregate_speakers(synthetic_records)
    written = write_split_files(tmp_path, assign_speakers(speakers))
    with pytest.raises(RuntimeError, match="第10段階"):
        load_test_split(written["test"])


def test_load_split_refuses_test_payload(tmp_path, synthetic_records):
    """汎用の load_split では test.json を読めない。"""
    speakers = aggregate_speakers(synthetic_records)
    written = write_split_files(tmp_path, assign_speakers(speakers))
    with pytest.raises(ValueError, match="第10段階"):
        load_split(written["test"])


def test_build_split_payload_marks_only_test_split():
    assert "usage_restriction" not in build_split_payload("train", ["a"])
    assert "usage_restriction" not in build_split_payload("dev", ["a"])
    assert "usage_restriction" in build_split_payload("test", ["a"])
