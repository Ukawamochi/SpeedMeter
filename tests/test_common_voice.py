"""合成した小さなtsvで、クリップ一覧の除外条件が働くことを確認する。

実データ（data/common_voice_ja/）には依存させない。
"""

import json

import pytest

from spkrate.data.common_voice import (
    ClipRecord,
    analyse_sentence,
    build_clip_index,
    read_clip_durations,
    summarize_mora_per_second,
    write_records,
)

COLUMNS = ["client_id", "path", "sentence", "up_votes", "down_votes"]


def write_validated(path, rows):
    """合成 validated.tsv を書く。rowsは (client_id, path, sentence, up, down)。"""
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_durations(path, pairs):
    """合成 clip_durations.tsv を書く。pairsは (ファイル名, ミリ秒)。"""
    lines = ["clip\tduration[ms]"]
    for name, ms in pairs:
        lines.append(f"{name}\t{ms}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def base_case(tmp_path):
    """条件ごとに1件ずつ含む合成データ。残るのは a.mp3 のみ。"""
    rows = [
        # 残る: 「きょうはいいてんきです」相当、モーラ数に対し妥当な長さ
        ("spk1", "a.mp3", "今日はいい天気です", 2, 0),
        # 1. down_votes >= up_votes
        ("spk2", "b.mp3", "今日はいい天気です", 2, 2),
        # 2. sentence が空
        ("spk3", "c.mp3", "", 3, 0),
        # 3b. 3桁以上の数字列
        ("spk4", "d.mp3", "電話番号は0312345678です", 3, 0),
        # 3a. 読み変換後にカタカナ以外が残る
        ("spk5", "e.mp3", "これは♬です", 3, 0),
        # 4. 音声長が無い
        ("spk6", "f.mp3", "今日はいい天気です", 3, 0),
        # 5a. 毎秒モーラ数が小さすぎる（1未満）
        ("spk7", "g.mp3", "今日はいい天気です", 3, 0),
        # 5b. 毎秒モーラ数が大きすぎる（12超）
        ("spk8", "h.mp3", "今日はいい天気です", 3, 0),
    ]
    validated = write_validated(tmp_path / "validated.tsv", rows)
    durations = write_durations(
        tmp_path / "clip_durations.tsv",
        [
            ("a.mp3", 3000),
            ("b.mp3", 3000),
            ("c.mp3", 3000),
            ("d.mp3", 3000),
            ("e.mp3", 3000),
            # f.mp3 は意図的に入れない
            ("g.mp3", 600000),
            ("h.mp3", 300),
        ],
    )
    return validated, durations


def test_each_exclusion_is_counted_once(base_case):
    validated, durations = base_case
    records, stats = build_clip_index(validated, durations)

    assert stats.total_rows == 8
    assert stats.excluded_down_votes == 1
    assert stats.excluded_empty_sentence == 1
    assert stats.excluded_unconverted == 1
    assert stats.excluded_long_digit_run == 1
    assert stats.missing_duration == 1
    assert stats.excluded_outlier == 2
    assert stats.kept == 1
    assert [record.clip_id for record in records] == ["a"]


def test_kept_record_fields(base_case):
    validated, durations = base_case
    records, _ = build_clip_index(validated, durations)
    record = records[0]

    assert isinstance(record, ClipRecord)
    assert record.clip_id == "a"
    assert record.audio_path == "data/common_voice_ja/clips/a.mp3"
    assert record.client_id == "spk1"
    assert record.sentence == "今日はいい天気です"
    assert record.kana and all("ァ" <= ch <= "ヺ" or ch == "ー" for ch in record.kana)
    assert record.mora > 0
    assert record.duration_sec == pytest.approx(3.0)
    assert record.mora_per_second == pytest.approx(record.mora / 3.0)


def test_down_votes_equal_to_up_votes_is_excluded(tmp_path):
    validated = write_validated(
        tmp_path / "validated.tsv", [("spk", "x.mp3", "今日はいい天気です", 2, 2)]
    )
    durations = write_durations(tmp_path / "d.tsv", [("x.mp3", 3000)])
    records, stats = build_clip_index(validated, durations)
    assert records == []
    assert stats.excluded_down_votes == 1


def test_down_votes_below_up_votes_is_kept(tmp_path):
    validated = write_validated(
        tmp_path / "validated.tsv", [("spk", "x.mp3", "今日はいい天気です", 3, 2)]
    )
    durations = write_durations(tmp_path / "d.tsv", [("x.mp3", 3000)])
    records, stats = build_clip_index(validated, durations)
    assert stats.kept == 1
    assert records[0].clip_id == "x"


def test_whitespace_only_sentence_is_treated_as_empty(tmp_path):
    validated = write_validated(
        tmp_path / "validated.tsv", [("spk", "x.mp3", "   ", 3, 0)]
    )
    durations = write_durations(tmp_path / "d.tsv", [("x.mp3", 3000)])
    _, stats = build_clip_index(validated, durations)
    assert stats.excluded_empty_sentence == 1


def test_outlier_bounds_are_inclusive(tmp_path):
    """しきい値ちょうどの毎秒モーラ数は残す（範囲は両端を含む）。

    音声長はミリ秒の整数でしか与えられないため、下限・上限に「ちょうど」
    一致する値を作るには、与えた音声長から実際の毎秒モーラ数を計算し、
    その値をしきい値として渡す。こうすると境界そのものを厳密に検証できる。
    """
    validated = write_validated(
        tmp_path / "validated.tsv",
        [
            ("spk", "low.mp3", "今日はいい天気です", 3, 0),
            ("spk", "high.mp3", "今日はいい天気です", 3, 0),
        ],
    )
    mora = analyse_sentence("今日はいい天気です").mora
    low_ms = mora * 1000  # 毎秒モーラ数が 1.0 ちょうどになる長さ
    high_ms = round(mora / 12.0 * 1000)  # 12.0 の近傍（整数ミリ秒）
    durations = write_durations(
        tmp_path / "d.tsv", [("low.mp3", low_ms), ("high.mp3", high_ms)]
    )
    low_rate = mora / (low_ms / 1000.0)
    high_rate = mora / (high_ms / 1000.0)
    assert low_rate == pytest.approx(1.0)
    assert high_rate == pytest.approx(12.0, abs=0.01)

    records, stats = build_clip_index(
        validated,
        durations,
        min_mora_per_second=low_rate,
        max_mora_per_second=high_rate,
    )
    assert stats.excluded_outlier == 0
    assert {record.clip_id for record in records} == {"low", "high"}
    assert records[0].mora_per_second == pytest.approx(low_rate)
    assert records[1].mora_per_second == pytest.approx(high_rate)


def test_outlier_bounds_exclude_just_outside(tmp_path):
    """しきい値をわずかに外れる毎秒モーラ数は除外する。"""
    validated = write_validated(
        tmp_path / "validated.tsv",
        [
            ("spk", "low.mp3", "今日はいい天気です", 3, 0),
            ("spk", "high.mp3", "今日はいい天気です", 3, 0),
        ],
    )
    mora = analyse_sentence("今日はいい天気です").mora
    low_ms = mora * 1000
    high_ms = round(mora / 12.0 * 1000)
    durations = write_durations(
        tmp_path / "d.tsv", [("low.mp3", low_ms), ("high.mp3", high_ms)]
    )
    low_rate = mora / (low_ms / 1000.0)
    high_rate = mora / (high_ms / 1000.0)

    _, stats = build_clip_index(
        validated,
        durations,
        min_mora_per_second=low_rate + 1e-9,
        max_mora_per_second=high_rate - 1e-9,
    )
    assert stats.excluded_outlier == 2
    assert stats.kept == 0


def test_training_filter_can_be_disabled_for_eval_sets(base_case):
    """評価セット用に学習専用の除外を外せること。既定は学習用（適用する）。"""
    validated, durations = base_case
    records, stats = build_clip_index(
        validated, durations, apply_training_filter=False
    )
    assert stats.excluded_unconverted == 0
    assert stats.excluded_long_digit_run == 0
    assert stats.unconverted_kept == 1
    assert {record.clip_id for record in records} == {"a", "d", "e"}


def test_custom_outlier_range(base_case):
    validated, durations = base_case
    _, stats = build_clip_index(
        validated, durations, min_mora_per_second=0.0, max_mora_per_second=100.0
    )
    assert stats.excluded_outlier == 0
    assert stats.kept == 3


def test_same_sentence_is_converted_once(tmp_path, monkeypatch):
    """同一原文の読み変換結果を再利用すること。"""
    import spkrate.data.common_voice as module

    calls = []
    original = module.analyse_sentence

    def counting(text, **kwargs):
        calls.append(text)
        return original(text, **kwargs)

    monkeypatch.setattr(module, "analyse_sentence", counting)
    validated = write_validated(
        tmp_path / "validated.tsv",
        [
            ("spk1", "a.mp3", "今日はいい天気です", 3, 0),
            ("spk2", "b.mp3", "今日はいい天気です", 3, 0),
        ],
    )
    durations = write_durations(
        tmp_path / "d.tsv", [("a.mp3", 3000), ("b.mp3", 3000)]
    )
    _, stats = build_clip_index(validated, durations)
    assert calls == ["今日はいい天気です"]
    assert stats.unique_sentences == 1
    assert stats.kept == 2


def test_read_clip_durations_converts_milliseconds(tmp_path):
    path = write_durations(tmp_path / "d.tsv", [("a.mp3", 2484), ("b.mp3", 3348)])
    durations = read_clip_durations(path)
    assert durations == {"a.mp3": pytest.approx(2.484), "b.mp3": pytest.approx(3.348)}


def test_non_numeric_votes_are_excluded(tmp_path):
    validated = write_validated(
        tmp_path / "validated.tsv", [("spk", "x.mp3", "今日はいい天気です", "", "")]
    )
    durations = write_durations(tmp_path / "d.tsv", [("x.mp3", 3000)])
    _, stats = build_clip_index(validated, durations)
    assert stats.excluded_down_votes == 1


def test_write_records_falls_back_to_jsonl(tmp_path, monkeypatch):
    import spkrate.data.common_voice as module

    monkeypatch.setattr(module, "_has_pyarrow", lambda: False)
    record = ClipRecord(
        clip_id="a",
        audio_path="data/common_voice_ja/clips/a.mp3",
        client_id="spk",
        sentence="今日はいい天気です",
        kana="キョウハイイテンキデス",
        mora=11,
        duration_sec=3.0,
        mora_per_second=11 / 3.0,
    )
    written = write_records([record], tmp_path / "clips.parquet")
    assert written.suffix == ".jsonl"
    loaded = [json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()]
    assert loaded[0]["clip_id"] == "a"
    assert loaded[0]["sentence"] == "今日はいい天気です"


def test_summarize_mora_per_second():
    records = [
        ClipRecord("c%d" % i, "p", "spk", "s", "ア", i, 1.0, float(i))
        for i in (2, 4, 6, 8)
    ]
    summary = summarize_mora_per_second(records)
    assert summary.count == 4
    assert summary.minimum == 2.0
    assert summary.maximum == 8.0
    assert summary.mean == pytest.approx(5.0)
    assert summary.median == pytest.approx(5.0)
    assert summary.q1 == pytest.approx(3.5)
    assert summary.q3 == pytest.approx(6.5)
    assert sum(count for _, _, count in summary.histogram) == 4
    assert summary.total_duration_sec == pytest.approx(4.0)
