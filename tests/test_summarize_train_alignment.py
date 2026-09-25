"""scripts/summarize_train_alignment.py の結合テスト（合成の入力で実行する）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "summarize_train_alignment.py"

# 無発話疑いの規則（configs/eval/no_speech.yaml）に該当する・しない指標
SILENT = {"frame_db_p90": -40.0, "frame_db_range": 10.0, "cer": 1.0, "hyp": "", "ref": "あ"}
SPEECH = {"frame_db_p90": -15.0, "frame_db_range": 50.0, "cer": 0.1, "hyp": "あ", "ref": "あ"}


def _moras(starts):
    return [{"kana": "ア", "start": s, "end": s + 0.02} for s in starts]


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_summarize_counts_and_outputs(tmp_path):
    spk = "spkA"
    # (clip_id, mora_per_second, アライメント, 指標)。None は未処理
    cases = [
        ("ok1", 3.0, {"ok": True, "moras": _moras([0.5, 0.6, 0.7])}, SPEECH),
        ("fail", 5.0, {"ok": False, "reason": "align_failed", "detail": "x"}, SILENT),  # 失敗＋無発話
        ("mism", 5.0, {"ok": True, "moras": _moras([0.5, 0.6])}, SPEECH),  # 不一致
        ("head", 7.0, {"ok": True, "moras": _moras([0.0, 1.5, 1.6])}, SILENT),  # 先頭の孤立＋無発話
        ("tail", 9.0, {"ok": True, "moras": _moras([0.0, 0.1, 1.2])}, SPEECH),  # 末尾の孤立
        ("sil", 9.0, {"ok": True, "moras": _moras([0.5, 0.6, 0.7])}, SILENT),  # 無発話のみ
        ("todo", 3.0, {"ok": True, "moras": _moras([0.5, 0.6, 0.7])}, None),  # 指標が未処理
        ("other", 3.0, None, None),  # 別の話者（対象外）
    ]
    clips, align, metrics = [], [], []
    for cid, rate, a, m in cases:
        clips.append({"clip_id": cid, "audio_path": "", "client_id": "spkB" if cid == "other" else spk,
                      "sentence": "", "kana": "", "mora": 3, "duration_sec": 1.0, "mora_per_second": rate})
        if a is not None:
            align.append({"clip_id": cid, **a})
        if m is not None:
            metrics.append({"clip_id": cid, **m})
    _write_jsonl(tmp_path / "clips.jsonl", clips)
    _write_jsonl(tmp_path / "a.jsonl", align)
    _write_jsonl(tmp_path / "m.jsonl", metrics)
    (tmp_path / "split.json").write_text(json.dumps({"split": "train", "client_ids": [spk]}), encoding="utf-8")
    out = tmp_path / "sel"
    md = tmp_path / "r.md"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--alignments", str(tmp_path / "a.jsonl"), "--metrics", str(tmp_path / "m.jsonl"),
         "--clips", str(tmp_path / "clips.jsonl"), "--split", str(tmp_path / "split.json"),
         "--suspect-out", str(tmp_path / "suspect.tsv"), "--out-dir", str(out), "--md", str(md)],
        cwd=REPO, check=True, capture_output=True,
    )
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["train_clips"] == 7 and s["pending"] == 1 and not s["complete"]
    assert s["usable"] == 1 and s["excluded"] == 5
    assert s["reasons_with_overlap"] == {
        "align_failed": 1, "mora_mismatch": 1, "head_isolated": 1, "tail_isolated": 1, "no_speech_suspect": 3,
    }
    assert s["reasons_exclusive"] == {
        "align_failed": 1, "mora_mismatch": 1, "head_isolated": 1, "tail_isolated": 1, "no_speech_suspect": 1,
    }
    assert s["by_band"]["exclusive"]["no_speech_suspect"] == {"over8": 1}
    assert s["alignment_fail_codes"] == {"align_failed": 1}
    assert (out / "usable_clip_ids.txt").read_text().split() == ["ok1"]
    assert len((tmp_path / "suspect.tsv").read_text().splitlines()) == 1 + 3
    text = md.read_text(encoding="utf-8")
    assert "途中結果" in text and "重複を含む件数" in text and "排他の件数" in text


def test_summarize_rejects_test_split(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--split", "configs/splits/test.json", "--md", str(tmp_path / "x.md")],
        cwd=REPO, capture_output=True,
    )
    assert r.returncode != 0
