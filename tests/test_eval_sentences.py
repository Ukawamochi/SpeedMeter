"""実環境評価の文の選び方（src/spkrate/eval/eval_sentences.py）の試験。"""

from __future__ import annotations

import pytest

from spkrate.eval.eval_sentences import (
    Candidate,
    build_candidates,
    is_eligible,
    normalize_annotation,
    select,
)


def test_normalize_annotation():
    assert normalize_annotation("もくよーび、てーせんかいだんわ") == "モクヨービテーセンカイダンワ"


def test_build_candidates_counts_from_annotation():
    transcript = {"A": "木曜日", "B": "対応なし"}
    labels = {"A": {"kana_level0": "もくよーび"}}
    (c,) = build_candidates(transcript, labels)
    assert (c.utt_id, c.kana, c.mora) == ("A", "モクヨービ", 5)
    assert c.g2p_mora == 5


def _cand(i, mora, g2p_mora=None, g2p_kana=None):
    return Candidate(
        utt_id=f"ID_{i:04d}",
        text="文",
        kana="ア" * mora,
        mora=mora,
        g2p_kana=g2p_kana if g2p_kana is not None else "ア" * (g2p_mora or mora),
        g2p_mora=g2p_mora if g2p_mora is not None else mora,
    )


def test_is_eligible():
    assert is_eligible(_cand(1, 20), 20, 40)
    assert is_eligible(_cand(1, 40), 20, 40)
    assert not is_eligible(_cand(1, 19), 20, 40)
    assert not is_eligible(_cand(1, 41), 20, 40)
    assert not is_eligible(_cand(1, 25, g2p_mora=26), 20, 40)
    assert not is_eligible(_cand(1, 25, g2p_mora=25, g2p_kana="ア" * 24 + "A"), 20, 40)


def test_select_deterministic_and_sorted():
    cands = [_cand(i, 20 + i % 30) for i in range(200)]
    a = select(cands, 20, 7, 20, 40)
    b = select(cands, 20, 7, 20, 40)
    assert a == b
    assert len(a) == 20
    assert [c.utt_id for c in a] == sorted(c.utt_id for c in a)
    assert all(20 <= c.mora <= 40 for c in a)
    assert select(cands, 20, 8, 20, 40) != a


def test_select_too_few():
    with pytest.raises(ValueError):
        select([_cand(i, 25) for i in range(5)], 20, 0, 20, 40)
