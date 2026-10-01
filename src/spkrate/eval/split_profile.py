"""評価スクリプトの対象分割（dev・test）ごとの設定とファイルの場所の対応表。

第10段階（テストセットでの評価、2026-10-02 の人間の指示）で、dev だけを対象にしていた評価の入口
（scripts/eval_dev_window.py・eval_dev_full.py・eval_fast_speech.py など）を ``--split test`` でも
動かせるようにするための共通の定義。既定は dev で、dev の設定・出力先・metrics.csv の split 列の値は
従来と一字も変わらない（tests/test_split_profile.py）。

test の分割は configs/splits/test.json（話者単位で固定済み）。test 用の窓・速めた音声の設定は
configs/eval/test_window.yaml・test_fast.yaml・test_noisy.yaml で、dev と同じ値（同じ乱数の種、
同じ窓・雑音・WSOLA の規則）に、対象の分割と出力先だけを変えたもの。
無発話疑い（no_speech_suspect。configs/eval/no_speech.yaml の固定の規則）は test にも同じ規則を適用する
（scripts/detect_no_speech.py --split test → scripts/analyze_no_speech.py apply --split test）。
known_no_speech（人間が dev で聴いた37件）は test には無い。

metrics.csv の split 列は、dev の ``dev``・``dev_noisy_*``・``dev_window*``・``dev_fast_*`` の先頭の ``dev`` を
``test`` に置き換えた値（``test``・``test_noisy_snr5``・``test_window``・``test_window_noisy_snr5``・
``test_window_bin_6to7``・``test_fast_x1.5`` など）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from spkrate.data.splits import load_split, load_test_split

SPLITS = ("dev", "test")
DEFAULT_SPLIT = "dev"


@dataclass(frozen=True)
class SplitProfile:
    """1つの分割の評価に使う設定・出力先・split 列の接頭辞。"""

    split: str
    split_json: str
    window_config: str
    fast_config: str
    noisy_config: str
    suspect_tsv: str
    features_dir: str
    alignments: str
    no_speech_metrics: str

    @property
    def prefix(self) -> str:
        """metrics.csv の split 列の接頭辞（"dev" または "test"）。"""
        return self.split

    # --- metrics.csv の split 列 ---
    @property
    def clean_split(self) -> str:
        return self.split

    def noisy_split(self, snr_db: float) -> str:
        return f"{self.split}_noisy_snr{float(snr_db):g}"

    @property
    def noisy_pooled_split(self) -> str:
        return f"{self.split}_noisy_all"

    @property
    def window_split(self) -> str:
        return f"{self.split}_window"

    def window_noisy_split(self, condition: str) -> str:
        return f"{self.split}_window_noisy_{condition}"

    @property
    def window_noisy_pooled_split(self) -> str:
        return f"{self.split}_window_noisy_all"

    @property
    def fast_split_prefix(self) -> str:
        return f"{self.split}_fast"

    @property
    def is_test(self) -> bool:
        return self.split == "test"


_PROFILES = {
    "dev": SplitProfile(
        split="dev",
        split_json="configs/splits/dev.json",
        window_config="configs/eval/dev_window.yaml",
        fast_config="configs/eval/dev_fast.yaml",
        noisy_config="configs/eval/dev_noisy.yaml",
        suspect_tsv="results/no_speech_suspect_dev.tsv",
        features_dir="data/processed/features/dev",
        alignments="data/processed/alignments/dev.jsonl",
        no_speech_metrics="data/processed/no_speech/dev_metrics.jsonl",
    ),
    "test": SplitProfile(
        split="test",
        split_json="configs/splits/test.json",
        window_config="configs/eval/test_window.yaml",
        fast_config="configs/eval/test_fast.yaml",
        noisy_config="configs/eval/test_noisy.yaml",
        suspect_tsv="data/processed/no_speech/suspect_test.tsv",
        features_dir="data/processed/features/test",
        alignments="data/processed/alignments/test.jsonl",
        no_speech_metrics="data/processed/no_speech/test_metrics.jsonl",
    ),
}


def get_profile(split: str = DEFAULT_SPLIT) -> SplitProfile:
    if split not in _PROFILES:
        raise ValueError(f"分割は {SPLITS} のどれか: {split}")
    return _PROFILES[split]


def load_split_ids(path: str | Path, *, allow_test: bool = False) -> list[str]:
    """分割ファイルの client_id の一覧。test の分割は ``allow_test=True`` のときだけ読む。

    dev・train は従来どおり ``load_split``。test.json は ``load_split`` が拒否するので
    ``load_test_split(stage10_approved=True)`` を通す。これは第10段階（テストセットでの評価）を
    人間が指示した（2026-10-02）ことによる。``allow_test`` を True にするのは ``--split test`` を
    明示された評価・準備の入口だけにする（既定の dev の動作は変えない）。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split") == "test" or Path(path).name == "test.json":
        if not allow_test:
            return load_split(path)  # 従来どおり ValueError で拒否される
        return load_test_split(path, stage10_approved=True)
    return load_split(path)
