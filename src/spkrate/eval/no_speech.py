"""無発話クリップの自動検出（docs/directives/2026-09-25.md 追加指示（2026-09-26）の1）。

聴取で「発話が聞き取れない」と確定したクリップ（known_no_speech）は強制アライメントが
ok になるため、アライメントの検査では見つからない。ここではクリップごとに次の指標を求め、
単純な判定規則で ``no_speech_suspect`` の印を付ける。

- 認識: アライメントと同じひらがなCTCモデル（``spkrate.labels.alignment`` のコミット固定）で
  文を与えずに貪欲デコードし、正解のかな（ラベルのカタカナ読みを語彙に合わせて変換）との
  文字誤り率（CER）を求める。あわせて非blankフレームの割合、認識文字数とモーラ数の比、
  正解の読みを与えたときの CTC の負の対数尤度（1トークンあたり）
- 音量: 全体の実効値（dBFS）、ピーク、20ms フレームの実効値の分位点、動的範囲、
  エネルギー基準の発話区間の割合

判定規則は「指標 比較 閾値」の条件の論理積（``Rule``）。値は configs/eval/no_speech.yaml。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spkrate.labels.alignment import _HIRA_FALLBACK, kata_to_hira

__all__ = [
    "Condition",
    "Rule",
    "apply_rule",
    "cer",
    "ctc_greedy_decode",
    "edit_distance",
    "energy_metrics",
    "hyp_to_text",
    "load_rule",
    "normalize_reference",
    "windows_from_clips",
]

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20ms（CTC のフレーム長と同じ）
_EPS = 1e-10
# 語彙のうち文字として扱わない記号
_NON_CHAR_TOKENS = frozenset({"|", "[UNK]", "[PAD]"})


# ---------------------------------------------------------------- かなの正規化
def normalize_reference(kana: str, vocab: Mapping[str, int]) -> str:
    """ラベルのカタカナ読みをモデルの語彙の文字列（ひらがな）に変換する。

    各文字をひらがなに変え、語彙に無い「ゎゕゖゐゑ」はアライメントと同じ置き換えをする。
    それでも語彙に無い文字（記号など）と、区切り記号「|」は捨てる。「ー」は語彙にあるので残す。
    """
    out: list[str] = []
    for ch in kana:
        h = kata_to_hira(ch)
        if h not in vocab:
            h = _HIRA_FALLBACK.get(h, h)
        if h in vocab and h not in _NON_CHAR_TOKENS:
            out.append(h)
    return "".join(out)


def ctc_greedy_decode(frame_ids: Sequence[int], blank: int) -> list[int]:
    """フレームごとの最尤トークン列を CTC の規則で縮約する（連続の重複をまとめ blank を除く）。"""
    out: list[int] = []
    prev = None
    for t in frame_ids:
        t = int(t)
        if t != prev and t != blank:
            out.append(t)
        prev = t
    return out


def hyp_to_text(token_ids: Iterable[int], id_to_token: Mapping[int, str]) -> str:
    """トークン列を文字列にする。区切り「|」・[UNK]・[PAD] は捨てる。"""
    return "".join(
        id_to_token[i] for i in token_ids if id_to_token[i] not in _NON_CHAR_TOKENS
    )


# ---------------------------------------------------------------- 文字誤り率
def edit_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """レーベンシュタイン距離（置換・挿入・脱落の重みはすべて1）。"""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """文字誤り率 = 編集距離 ÷ 正解の文字数。正解が空なら、認識も空で0、そうでなければ1。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------- 音量の指標
def _db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.asarray(power, dtype=np.float64) + _EPS)


def energy_metrics(
    wav: np.ndarray,
    *,
    frame_samples: int = FRAME_SAMPLES,
    rel_db: float = 15.0,
    abs_dbfs: float = -45.0,
) -> dict[str, float]:
    """波形（16kHz、float32、振幅は ±1 が満尺）の音量に関する指標。

    - ``rms_dbfs``: 全体の実効値。``peak_dbfs``: 標本の絶対値の最大
    - ``frame_db_p10/p50/p90/p99``: 20ms フレーム（重なりなし、末尾の端数は捨てる）の実効値の分位点
    - ``frame_db_range``: p95 − p10（雑音だけのクリップでは小さいと期待される）
    - ``energy_speech_frac``: 実効値が p10（雑音の床の目安）より ``rel_db`` 以上大きいフレームの割合
    - ``abs_speech_frac``: 実効値が ``abs_dbfs`` を超えるフレームの割合
    """
    x = np.asarray(wav, dtype=np.float32)
    n = x.size // frame_samples
    if n == 0:
        frames = x[None, :] if x.size else np.zeros((1, 1), dtype=np.float32)
    else:
        frames = x[: n * frame_samples].reshape(n, frame_samples)
    fdb = _db(np.mean(frames.astype(np.float64) ** 2, axis=1))
    p10, p50, p90, p95, p99 = np.percentile(fdb, [10, 50, 90, 95, 99])
    return {
        "rms_dbfs": float(_db(np.mean(x.astype(np.float64) ** 2)) if x.size else _db(0.0)),
        "peak_dbfs": float(20.0 * math.log10(float(np.max(np.abs(x))) + 1e-10) if x.size else -200.0),
        "frame_db_p10": float(p10),
        "frame_db_p50": float(p50),
        "frame_db_p90": float(p90),
        "frame_db_p99": float(p99),
        "frame_db_range": float(p95 - p10),
        "energy_speech_frac": float(np.mean(fdb > p10 + rel_db)),
        "abs_speech_frac": float(np.mean(fdb > abs_dbfs)),
    }


# ---------------------------------------------------------------- 判定規則
_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


@dataclass(frozen=True)
class Condition:
    metric: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"未定義の比較: {self.op}")

    def __call__(self, row: Mapping[str, Any]) -> bool:
        value = row[self.metric]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        return bool(_OPS[self.op](float(value), self.threshold))

    def __str__(self) -> str:
        return f"{self.metric} {self.op} {self.threshold:g}"


@dataclass(frozen=True)
class Rule:
    """条件の論理積。すべての条件を満たすクリップを no_speech_suspect とする。"""

    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("条件が空の規則は定義しない")

    def __call__(self, row: Mapping[str, Any]) -> bool:
        return all(c(row) for c in self.conditions)

    def __str__(self) -> str:
        return " かつ ".join(str(c) for c in self.conditions)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Rule":
        return cls(
            tuple(
                Condition(str(c["metric"]), str(c["op"]), float(c["threshold"]))
                for c in mapping["conditions"]
            )
        )


def apply_rule(rows: Iterable[Mapping[str, Any]], rule: Rule) -> list[str]:
    """規則に該当するクリップの clip_id を入力の順に返す。"""
    return [r["clip_id"] for r in rows if rule(r)]


def load_rule(path: str | Path) -> Rule:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return Rule.from_mapping(yaml.safe_load(handle)["rule"])


# ---------------------------------------------------------------- dev_window との対応
def windows_from_clips(window_set: Any, clip_ids: Iterable[str]) -> np.ndarray:
    """dev_window の各窓が、指定の clip_id を1件でも含む音源に由来するかの真偽配列。

    known_no_speech と同じ付け方（単一クリップはそのクリップ、連結は組の1件でも該当すれば
    組のすべての窓）。窓の定義は変えない。
    """
    target = set(clip_ids)
    per_source = np.array(
        [any(c in target for c in s.clip_ids) for s in window_set.sources], dtype=bool
    )
    return per_source[np.asarray(window_set.source_index, dtype=np.int64)]


# ---------------------------------------------------------------- クリップごとの指標
def ctc_metrics(
    emission: Any,
    reference: str,
    vocab: Mapping[str, int],
    blank: int,
    mora: int,
) -> dict[str, Any]:
    """放出確率（1, フレーム数, 語彙数; log_softmax 済み; CPU の torch.Tensor）から認識の指標を求める。

    - ``hyp``: 貪欲デコードの結果。``cer``: 正解 ``reference``（``normalize_reference`` 済み）との文字誤り率
    - ``hyp_len``・``ref_len``・``hyp_mora_ratio``（認識文字数 ÷ ラベルのモーラ数）
    - ``ctc_nonblank_frac``: 最尤トークンが blank・区切り以外のフレームの割合
    - ``ctc_nll_per_token``: 正解の読みを与えたときの CTC 損失（負の対数尤度）÷ トークン数。
      フレームが足りず計算できなければ NaN
    """
    import torch
    import torch.nn.functional as F

    id_to_token = {i: t for t, i in vocab.items()}
    lp = emission[0].to(torch.float32)
    frame_ids = lp.argmax(dim=-1).tolist()
    hyp = hyp_to_text(ctc_greedy_decode(frame_ids, blank), id_to_token)
    delim = vocab.get("|")
    nonblank = sum(1 for t in frame_ids if t != blank and t != delim)
    num_frames = len(frame_ids)
    nll = float("nan")
    if reference:
        target = torch.tensor([[vocab[c] for c in reference]], dtype=torch.long)
        loss = F.ctc_loss(
            lp[:, None, :],
            target,
            input_lengths=torch.tensor([num_frames]),
            target_lengths=torch.tensor([len(reference)]),
            blank=blank,
            reduction="sum",
            zero_infinity=False,
        )
        value = float(loss)
        if math.isfinite(value):
            nll = value / len(reference)
    return {
        "hyp": hyp,
        "hyp_len": len(hyp),
        "ref_len": len(reference),
        "cer": cer(reference, hyp),
        "hyp_mora_ratio": len(hyp) / mora if mora > 0 else float("nan"),
        "ctc_frames": num_frames,
        "ctc_nonblank_frac": nonblank / num_frames if num_frames else 0.0,
        "ctc_nll_per_token": nll,
    }
