"""特殊なモーラの数え落としの分析（docs/directives/2026-09-29.md タスク1）。学習・推論はしない。

dev_window（clean・主指標）の各窓について、窓に含まれるモーラのうち特殊拍（長音「ー」・撥音「ン」・
促音「ッ」）と拗音（小書き文字と結合したモーラ）の数、窓内の語の平均モーラ数を求め、既存の予測
（``scripts/eval_dev_window.py predict`` が保存した ``pred_clean.npz``）の偏り（出力 − 正解）を、
正解の話速の帯ごとに集計する。

## 数え方

- モーラの種類は、アライメント（``data/processed/alignments/dev.jsonl``）の各モーラの ``kana``
  （``split_mora`` の単位）から決める（``classify_mora``）。
  - 特殊拍: ``ー``・``ン``・``ッ`` の1文字のモーラ
  - 拗音: 2文字のモーラ（直前のかなと小書き文字「ャュョァィゥェォヮ」の結合。キャ・シュのほか、
    ティ・ファ・クヮなども含む）。うち ``ャュョ`` との結合を ``yoon_ya``（狭い意味の拗音）として別に数える
- 窓端の按分: 正解（``dev_window.window_mora``）と同じ規則で、各モーラの窓内に入る割合 f
  （区間 [start, end] と窓 [t, t + 2.0) の重なり ÷ (end − start)。end <= start は点として 0/1）を求め、
  種類ごとに f を足す。したがって特殊拍の数などは小数になり、種類ごとの和は窓の正解に一致する
  （一致を確認してから集計する）
- 特殊なモーラの割合 = 種類の f の和 ÷ 窓の正解（f の総和）
- 語: pyopenjtalk の ``run_frontend`` の各ノード（形態素。助詞・助動詞も1語）。読みが0モーラの
  ノード（記号）は語に数えない。語の読みを ``split_mora`` で分けて連結したものが、アライメントの
  モーラ列と一致するクリップだけ語の情報を使う（一致しないクリップの窓は語の集計から除く。
  pyopenjtalk が語の境界を小書き文字の直前に置いた場合などに起きる）。
  発話ごとに1回だけ処理し、``--word-cache`` にキャッシュする
- 窓内の語の平均モーラ数 = 窓の正解 ÷ 窓内の語数。窓内の語数は Σ f / L（L はそのモーラが属する語の
  モーラ数）で、窓の端で切れた語はその窓内の割合で数える

## 集計

- 対象: clean・主指標（dev_window と同じ除外: known_no_speech・no_speech_suspect）のうち、
  正解が ``--min-true-mora``（既定 1.0 モーラ、0.5 mora/s）以上の窓（割合が定義できない・極端に
  なる正解0付近の窓を除く）
- 帯: 正解の毎秒モーラ数の4帯（4未満・4〜6・6〜8・8以上）。各帯の中で割合・語の平均モーラ数の
  区間ごとに、窓数・正解の平均・偏り・「話速を揃えた偏り」を出す
- 話速を揃えた偏り: 各窓の偏りから、同じモデルの、正解の話速の 0.5 mora/s 刻みの区間の平均偏りを
  引いた値（帯の中でも正解の話速が高いほど偏りが負になるため、その分を除く）
- 傾き: 帯ごと（と1刻みの区間ごと）に、偏り ~ 割合 + 正解の話速 の最小二乗の割合の係数
  （割合 0.1 あたりの偏り、mora/s）。95% 区間は音源（単一クリップ・連結の組）を単位にした
  ブートストラップ（``--bootstrap`` 回、種 ``--seed``）。窓は 0.25 秒刻みで重なるので窓単位の
  区間は狭すぎる
- 同時の回帰: 帯ごとに 偏り ~ 特殊拍の割合 + 拗音の割合 + 語の平均モーラ数 + 正解の話速
- ``--exclude-latin``: 文にラテン文字を含むクリップ（英文を pyopenjtalk がアルファベット読み
  「エー」「ダブリュー」などにしたもの。長音を多く含み、ラベルの読みが話者の読みと違う疑いがある）に
  由来する窓を除いた感度分析

## 使い方

    # 1回目は語の分割（pyopenjtalk）と窓ごとの特徴量を作ってキャッシュする（CPU、1プロセス）
    nice -n 10 uv run python scripts/special_mora_analysis.py \\
        --model exp005=runs/exp005/window_eval --model exp015=runs/exp015_eval/window_eval \\
        --model exp007=runs/exp007_eval/window_eval --reference exp005 --noise-pair exp005,exp007 \\
        --out-dir runs/special_mora

    # ラテン文字を含む文を除いた感度分析（キャッシュを使うので速い）
    nice -n 10 uv run python scripts/special_mora_analysis.py ...（同じ --model） \
        --out-dir runs/special_mora --exclude-latin --tables-name special_mora_nolatin

``--model`` は ``名前=予測のディレクトリ[:npz のキー]``（キーの既定は名前）。何度でも指定できる。
出力は ``<out-dir>/special_mora_summary.json`` と ``special_mora_tables.md``。特徴量のキャッシュは
``<out-dir>/window_features.npz`` と ``--word-cache``（既定 ``<out-dir>/words_dev.jsonl``）。
data/ 以下は読むだけ。configs/splits/test.json は使わない。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

WINDOW_CONFIG = "configs/eval/dev_window.yaml"
SUSPECT_TSV = "results/no_speech_suspect_dev.tsv"
WINDOW_SEC = 2.0
SAMPLE_RATE = 16000

SPECIAL_KANA = frozenset("ーンッ")
YOON_YA = frozenset("ャュョ")
CATEGORIES = ("special", "chouon", "hatsuon", "sokuon", "yoon", "yoon_ya")

# 集計の区間（左閉右開）。最後は上限なし
RATIO_EDGES = {
    "special": (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
    "yoon": (0.0, 0.001, 0.05, 0.10, 0.20),
    "special_yoon": (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
}
WORDLEN_EDGES = (0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
BAND_EDGES = ((-np.inf, 4.0, "4未満"), (4.0, 6.0, "4〜6"), (6.0, 8.0, "6〜8"), (8.0, np.inf, "8以上"))
FINE_STEP = 0.5  # 話速を揃えた偏りの区間の幅（mora/s）


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# ------------------------------------------------------------------ かなの分類（テスト対象）


def classify_mora(mora: str) -> dict[str, bool]:
    """``split_mora`` の1単位のモーラの種類。

    - ``chouon``・``hatsuon``・``sokuon``: ``ー``・``ン``・``ッ``（1文字）。``special`` はそのいずれか
    - ``yoon``: 2文字のモーラ（小書き文字と結合したもの。キャ・ティ・クヮなど）
    - ``yoon_ya``: そのうち2文字目が ``ャュョ``
    """
    single = len(mora) == 1
    return {
        "chouon": single and mora == "ー",
        "hatsuon": single and mora == "ン",
        "sokuon": single and mora == "ッ",
        "special": single and mora in SPECIAL_KANA,
        "yoon": len(mora) == 2,
        "yoon_ya": len(mora) == 2 and mora[1] in YOON_YA,
    }


def count_special_morae(kana: str) -> dict[str, int]:
    """カタカナ読みから、モーラ数と種類ごとのモーラ数を数える（``split_mora`` と同じ単位）。"""
    from spkrate.labels.alignment import split_mora

    moras = split_mora(kana)
    counts = {key: 0 for key in CATEGORIES}
    for mora in moras:
        for key, hit in classify_mora(mora).items():
            counts[key] += int(hit)
    counts["mora"] = len(moras)
    return counts


def word_mora_lengths(nodes: list[dict]) -> tuple[list[int], list[str]]:
    """pyopenjtalk の ``run_frontend`` のノード列から、語（0モーラの記号を除く）ごとのモーラ数と読み。

    読みは ``pyopenjtalk.g2p(kana=True)`` と同じく記号は表記、それ以外は ``pron``（アクセント記号
    ``’`` を除く）を使い、``split_mora`` で分ける。
    """
    from spkrate.labels.alignment import split_mora

    lengths: list[int] = []
    prons: list[str] = []
    for node in nodes:
        pron = node["string"] if node["pos"] == "記号" else node["pron"]
        pron = pron.replace("’", "")
        count = len(split_mora(pron))
        if count == 0:
            continue
        lengths.append(count)
        prons.append("".join(split_mora(pron)))
    return lengths, prons


# ------------------------------------------------------------------ 語の分割（キャッシュ）


def build_word_cache(clip_ids: list[str], cache_path: Path, clips_jsonl: Path,
                     alignments: dict[str, dict]) -> dict[str, dict]:
    """clip_id → {ok, lengths}。キャッシュに無いクリップだけ pyopenjtalk で処理して追記する。"""
    cached: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    cached[row["clip_id"]] = row
    pending = [c for c in clip_ids if c not in cached]
    if not pending:
        return cached
    import unicodedata

    import pyopenjtalk

    from spkrate.labels.mora import _split_for_g2p

    wanted = set(pending)
    sentences: dict[str, str] = {}
    with open(clips_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["clip_id"] in wanted:
                    sentences[row["clip_id"]] = row["sentence"]
    log(f"語の分割: {len(pending)} クリップを pyopenjtalk で処理する（キャッシュ済み {len(cached)}）")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(cache_path, "a", encoding="utf-8") as out:
        for number, clip_id in enumerate(pending):
            normalized = unicodedata.normalize("NFKC", sentences[clip_id])
            nodes: list[dict] = []
            for chunk in _split_for_g2p(normalized):
                nodes.extend(pyopenjtalk.run_frontend(chunk))
            lengths, prons = word_mora_lengths(nodes)
            aligned = [m["kana"] for m in alignments[clip_id]["moras"]]
            ok = "".join(prons) == "".join(aligned) and sum(lengths) == len(aligned)
            latin = any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in normalized)
            row = {"clip_id": clip_id, "ok": ok, "latin": latin, "lengths": lengths, "prons": prons}
            cached[clip_id] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (number + 1) % 5000 == 0:
                log(f"  {number + 1}/{len(pending)} {time.perf_counter() - started:.0f}s")
    log(f"語の分割: 完了 {time.perf_counter() - started:.0f}s")
    return cached


# ------------------------------------------------------------------ 窓ごとの特徴量


def mora_fractions(starts: np.ndarray, ends: np.ndarray, window_start: np.ndarray,
                   window_sec: float) -> np.ndarray:
    """各窓（行）× 各モーラ（列）の窓内の割合。``dev_window.window_mora`` の和の前の値と同じ計算。"""
    s = np.asarray(starts, dtype=np.float32)[None, :]
    e = np.asarray(ends, dtype=np.float32)[None, :]
    t0 = np.asarray(window_start, dtype=np.float32)[:, None]
    t1 = t0 + np.float32(window_sec)
    length = e - s
    positive = length > 0
    overlap = np.clip(np.minimum(e, t1) - np.maximum(s, t0), 0.0, None)
    safe = np.where(positive, length, np.float32(1.0))
    fraction = np.where(positive, np.minimum(overlap / safe, np.float32(1.0)), np.float32(0.0))
    point = (~positive) & (s >= t0) & (s < t1)
    return np.where(point, np.float32(1.0), fraction)


def compute_window_features(ws, alignments: dict[str, dict], words: dict[str, dict]) -> dict[str, np.ndarray]:
    from spkrate.eval.window_eval import source_window_ranges

    n = len(ws)
    feats = {key: np.zeros(n, dtype=np.float32) for key in CATEGORIES}
    feats["mora_sum"] = np.zeros(n, dtype=np.float32)
    feats["word_count"] = np.zeros(n, dtype=np.float32)
    feats["word_ok"] = np.ones(n, dtype=bool)
    feats["latin"] = np.zeros(n, dtype=bool)
    ranges = source_window_ranges(ws)
    starts_all = np.asarray(ws.start_sample, dtype=np.int64)
    for number, source in enumerate(ws.sources):
        lo, hi = ranges[number]
        if lo == hi:
            continue
        m_start: list[float] = []
        m_end: list[float] = []
        kinds: list[dict[str, bool]] = []
        inv_len: list[float] = []
        ok = True
        for clip_id, offset in zip(source.clip_ids, source.offsets, strict=True):
            shift = offset / float(SAMPLE_RATE)
            moras = alignments[clip_id]["moras"]
            info = words.get(clip_id)
            if info is not None and info.get("latin"):
                feats["latin"][lo:hi] = True
            if info is None or not info["ok"]:
                ok = False
                per_mora = [np.nan] * len(moras)
            else:
                per_mora = [1.0 / length for length in info["lengths"] for _ in range(length)]
            for mora, inv in zip(moras, per_mora, strict=True):
                m_start.append(float(mora["start"]) + shift)
                m_end.append(float(mora["end"]) + shift)
                kinds.append(classify_mora(mora["kana"]))
                inv_len.append(inv)
        window_start = starts_all[lo:hi].astype(np.float32) / np.float32(SAMPLE_RATE)
        if not m_start:
            continue
        frac = mora_fractions(np.asarray(m_start, dtype=np.float32), np.asarray(m_end, dtype=np.float32),
                              window_start, WINDOW_SEC)
        feats["mora_sum"][lo:hi] = frac.sum(axis=1, dtype=np.float32)
        for key in CATEGORIES:
            mask = np.array([k[key] for k in kinds], dtype=bool)
            feats[key][lo:hi] = frac[:, mask].sum(axis=1, dtype=np.float32)
        if ok:
            feats["word_count"][lo:hi] = (frac * np.asarray(inv_len, dtype=np.float32)[None, :]).sum(axis=1)
        else:
            feats["word_ok"][lo:hi] = False
    return feats


def load_or_build_features(args: argparse.Namespace):
    from spkrate.eval.dev_window import load_alignments, load_dev_window, load_known_no_speech, load_window_config
    from spkrate.eval.no_speech import windows_from_clips

    wcfg = load_window_config(_resolve(WINDOW_CONFIG))
    ws = load_dev_window(_resolve(wcfg.output_dir))
    out_dir = _resolve(args.out_dir)
    feat_path = out_dir / "window_features.npz"
    flagged_ids = load_known_no_speech(_resolve(wcfg.known_no_speech_list)) | _load_suspect(_resolve(SUSPECT_TSV))
    keep = ~windows_from_clips(ws, flagged_ids)
    if feat_path.exists():
        with np.load(feat_path) as data:
            feats = {k: data[k] for k in data.files}
        log(f"特徴量のキャッシュを読む: {feat_path}")
    else:
        alignments = load_alignments(_resolve(wcfg.alignments))
        clip_ids = sorted({c for s in ws.sources for c in s.clip_ids})
        word_cache = _resolve(args.word_cache) if args.word_cache else out_dir / "words_dev.jsonl"
        words = build_word_cache(clip_ids, word_cache, _resolve(wcfg.clips_jsonl), alignments)
        bad = [c for c in clip_ids if not words[c]["ok"]]
        log(f"語の読みがアライメントと一致しないクリップ: {len(bad)}/{len(clip_ids)}")
        started = time.perf_counter()
        feats = compute_window_features(ws, alignments, words)
        log(f"窓の特徴量: {time.perf_counter() - started:.0f}s")
        feats["clip_word_mismatch"] = np.array([len(bad), len(clip_ids)], dtype=np.int64)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feat_path, **feats)
    diff = np.abs(feats["mora_sum"] - ws.mora.astype(np.float32))
    if float(diff.max()) > 1e-3:
        raise ValueError(f"種類ごとの按分の和が正解と一致しない（最大差 {float(diff.max())}）")
    return ws, feats, keep


def _load_suspect(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return {row["clip_id"] for row in csv.DictReader(lines, delimiter="\t")}


# ------------------------------------------------------------------ 集計


def _bin_labels(edges: tuple[float, ...], fmt: str) -> list[str]:
    labels = []
    for i, low in enumerate(edges):
        high = edges[i + 1] if i + 1 < len(edges) else None
        labels.append(f"{fmt.format(low)}以上" if high is None else f"{fmt.format(low)}〜{fmt.format(high)}")
    return labels


def _digitize(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.digitize(values, np.asarray(edges[1:], dtype=np.float64))


def band_index(true_rate: np.ndarray) -> np.ndarray:
    codes = np.full(true_rate.shape, -1, dtype=np.int64)
    for i, (low, high, _) in enumerate(BAND_EDGES):
        codes[(true_rate >= low) & (true_rate < high)] = i
    return codes


def adjusted_bias(true_rate: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """正解の話速の FINE_STEP 刻みの区間の平均偏りを引いた偏り。"""
    fine = np.floor(true_rate / FINE_STEP).astype(np.int64)
    fine -= fine.min()
    sums = np.bincount(fine, weights=bias)
    counts = np.bincount(fine)
    means = sums / np.maximum(counts, 1)
    return bias - means[fine]


def ols_slope(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """y ~ 1 + x + z の x の係数。"""
    if x.size < 10:
        return float("nan")
    design = np.stack([np.ones_like(x), x, z], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1])


def cluster_bootstrap_coefs(cols: list[np.ndarray], y: np.ndarray, cluster: np.ndarray, reps: int,
                            rng: np.random.Generator) -> np.ndarray:
    """y ~ 1 + cols の係数を、音源を単位に復元抽出して求めた (reps, 1 + len(cols)) の配列（十分統計量で計算）。"""
    design = [np.ones_like(y)] + list(cols)
    p = len(design)
    if y.size < 10 or reps <= 0:
        return np.full((0, p), np.nan)
    uniq, inv = np.unique(cluster, return_inverse=True)
    k = uniq.size
    xtx = np.zeros((k, p, p))
    xty = np.zeros((k, p))
    for i in range(p):
        xty[:, i] = np.bincount(inv, weights=design[i] * y, minlength=k)
        for j in range(i, p):
            v = np.bincount(inv, weights=design[i] * design[j], minlength=k)
            xtx[:, i, j] = v
            xtx[:, j, i] = v
    coefs = np.full((reps, p), np.nan)
    for r in range(reps):
        w = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(np.float64)
        try:
            coefs[r] = np.linalg.solve(np.tensordot(w, xtx, axes=1), w @ xty)
        except np.linalg.LinAlgError:
            pass
    return coefs


def cluster_bootstrap_slope(x, y, z, cluster, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """y ~ 1 + x + z の x の係数の、音源単位のブートストラップの 2.5%・97.5% 点。"""
    coefs = cluster_bootstrap_coefs([x, z], y, cluster, reps, rng)
    if coefs.shape[0] == 0:
        return float("nan"), float("nan")
    return float(np.nanquantile(coefs[:, 1], 0.025)), float(np.nanquantile(coefs[:, 1], 0.975))


def joint_regression(named: dict[str, np.ndarray], true_rate, bias, cluster, reps, rng) -> dict:
    """偏り ~ named の各列 + 正解の話速 の係数と、音源単位のブートストラップ95%区間。"""
    cols = list(named.values()) + [true_rate]
    design = np.stack([np.ones_like(bias)] + cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, bias, rcond=None)
    boot = cluster_bootstrap_coefs(cols, bias, cluster, reps, rng)
    out = {}
    for i, key in enumerate(named, start=1):
        ci = (float(np.nanquantile(boot[:, i], 0.025)), float(np.nanquantile(boot[:, i], 0.975))) if boot.size else (np.nan, np.nan)
        out[key] = {"coef": float(coef[i]), "ci": ci}
    return out


def group_table(true_rate, bias, adj, groups, n_groups) -> list[dict]:
    rows = []
    for g in range(n_groups):
        m = groups == g
        n = int(m.sum())
        rows.append({
            "n": n,
            "true_mean": float(true_rate[m].mean()) if n else float("nan"),
            "bias": float(bias[m].mean()) if n else float("nan"),
            "adj_bias": float(adj[m].mean()) if n else float("nan"),
        })
    return rows


def analyze_model(name: str, pred: np.ndarray, ws, feats: dict, sel: np.ndarray, args) -> dict:
    true = ws.mora.astype(np.float64)[sel]
    true_rate = true / WINDOW_SEC
    bias_all = (pred.astype(np.float64)[sel] - true) / WINDOW_SEC
    adj_all = adjusted_bias(true_rate, bias_all)
    bands = band_index(true_rate)
    cluster = np.asarray(ws.source_index, dtype=np.int64)[sel]
    ratios = {
        "special": feats["special"][sel] / true,
        "yoon": feats["yoon"][sel] / true,
        "special_yoon": (feats["special"][sel] + feats["yoon"][sel]) / true,
    }
    word_ok = feats["word_ok"][sel] & (feats["word_count"][sel] > 0)
    wordlen = np.where(word_ok, true / np.where(word_ok, feats["word_count"][sel], 1.0), np.nan)
    rng = np.random.default_rng(args.seed)
    out: dict = {"n": int(sel.sum()), "overall_bias": float(bias_all.mean()), "bands": []}
    fine_edges = np.arange(np.floor(true_rate.min()), np.ceil(true_rate.max()) + 1.0, 1.0)
    for b, (low, high, label) in enumerate(BAND_EDGES):
        mb = bands == b
        entry: dict = {"band": label, "n": int(mb.sum()), "bias": float(bias_all[mb].mean()), "by": {}}
        variables = dict(ratios)
        variables["wordlen"] = wordlen
        for var, values in variables.items():
            edges = WORDLEN_EDGES if var == "wordlen" else RATIO_EDGES[var]
            mv = mb & np.isfinite(values)
            groups = _digitize(values[mv], edges)
            table = group_table(true_rate[mv], bias_all[mv], adj_all[mv], groups, len(edges))
            x = values[mv]
            scale = 1.0 if var == "wordlen" else 0.1  # 割合は 0.1 あたり、語長は1モーラあたり
            slope = ols_slope(x / scale, bias_all[mv], true_rate[mv])
            ci = cluster_bootstrap_slope(x / scale, bias_all[mv], true_rate[mv], cluster[mv], args.bootstrap, rng)
            corr_rate = float(np.corrcoef(x, true_rate[mv])[0, 1]) if x.size > 2 else float("nan")
            entry["by"][var] = {"table": table, "slope": slope, "slope_ci": ci, "corr_with_rate": corr_rate,
                                "mean": float(x.mean()) if x.size else float("nan"), "n": int(mv.sum())}
        mj = mb & np.isfinite(wordlen)
        # 割合は 0.1 あたり、語の平均モーラ数は1モーラあたり
        entry["joint"] = joint_regression(
            {"special": ratios["special"][mj] / 0.1, "yoon": ratios["yoon"][mj] / 0.1, "wordlen": wordlen[mj]},
            true_rate[mj], bias_all[mj], cluster[mj], args.bootstrap, rng)
        entry["joint"]["n"] = int(mj.sum())
        entry["joint_parts"] = joint_regression(
            {k: feats[k][sel][mj] / true[mj] / 0.1 for k in ("chouon", "hatsuon", "sokuon", "yoon")}
            | {"wordlen": wordlen[mj]},
            true_rate[mj], bias_all[mj], cluster[mj], args.bootstrap, rng)
        out["bands"].append(entry)
    # 1 mora/s 刻みの区間ごとの傾き（特殊拍+拗音の割合、特殊拍の割合、語の平均モーラ数）
    fine = []
    for lo in fine_edges[:-1]:
        m = (true_rate >= lo) & (true_rate < lo + 1.0)
        if m.sum() < 200:
            continue
        row = {"low": float(lo), "n": int(m.sum()), "bias": float(bias_all[m].mean())}
        for var in ("special_yoon", "special", "wordlen"):
            values = wordlen if var == "wordlen" else ratios[var]
            mv = m & np.isfinite(values)
            scale = 1.0 if var == "wordlen" else 0.1
            row[var] = ols_slope(values[mv] / scale, bias_all[mv], true_rate[mv])
        fine.append(row)
    out["fine"] = fine
    return out


# ------------------------------------------------------------------ 表


def _f(value: float, digits: int = 3) -> str:
    return "—" if value != value else f"{value:+.{digits}f}".replace("-", "−")


def _p(value: float, digits: int = 3) -> str:
    return "—" if value != value else f"{value:.{digits}f}"


VAR_TITLES = {
    "special_yoon": ("特殊拍＋拗音の割合", RATIO_EDGES["special_yoon"], "{:.2f}"),
    "special": ("特殊拍（ー・ン・ッ）の割合", RATIO_EDGES["special"], "{:.2f}"),
    "yoon": ("拗音の割合", RATIO_EDGES["yoon"], "{:.3g}"),
    "wordlen": ("語の平均モーラ数", WORDLEN_EDGES, "{:.1f}"),
}


def render_tables(results: dict, models: list[str], feats_info: dict) -> list[str]:
    md = ["# 特殊なモーラと偏りの集計（scripts/special_mora_analysis.py）", "",
          f"対象: clean・主指標、正解 {feats_info['min_true_mora']} モーラ以上の窓 {feats_info['n']:,}窓。"
          f"語の情報のない窓 {feats_info['no_word']:,}（語の読みがアライメントと一致しないクリップ "
          f"{feats_info['clip_word_mismatch'][0]}/{feats_info['clip_word_mismatch'][1]}）。"
          f"ラテン文字を含む文のクリップに由来する窓 {feats_info['latin_windows']:,}"
          + ("（除いた）" if feats_info["exclude_latin"] else "（含む）") + "。偏り・話速を揃えた偏りは mora/s。", ""]
    for var, (title, edges, fmt) in VAR_TITLES.items():
        md += [f"## {title}", ""]
        labels = _bin_labels(edges, fmt)
        header = "| 帯 | 区間 | 窓数 | 正解の平均 | " + " | ".join(f"{m} 偏り | {m} 揃えた偏り" for m in models) + " |"
        md += [header, "| --- | --- | ---: | ---: | " + " | ".join("---: | ---:" for _ in models) + " |"]
        for b in range(len(BAND_EDGES)):
            base = results[models[0]]["bands"][b]
            for g, label in enumerate(labels):
                row = base["by"][var]["table"][g]
                if row["n"] == 0:
                    continue
                cells = []
                for m in models:
                    r = results[m]["bands"][b]["by"][var]["table"][g]
                    cells.append(f"{_f(r['bias'])} | {_f(r['adj_bias'])}")
                md.append(f"| {base['band']} | {label} | {row['n']:,} | {_p(row['true_mean'], 2)} | " + " | ".join(cells) + " |")
        md += ["", f"傾き（偏り ~ {title} + 正解の話速 の係数。"
               + ("1モーラあたり" if var == "wordlen" else "割合0.1あたり")
               + "、mora/s。括弧は音源単位のブートストラップ95%区間）", "",
               "| 帯 | 窓数 | 平均 | 話速との相関 | " + " | ".join(models) + " |",
               "| --- | ---: | ---: | ---: | " + " | ".join("---" for _ in models) + " |"]
        for b in range(len(BAND_EDGES)):
            base = results[models[0]]["bands"][b]["by"][var]
            cells = []
            for m in models:
                v = results[m]["bands"][b]["by"][var]
                cells.append(f"{_f(v['slope'])} ({_f(v['slope_ci'][0])}, {_f(v['slope_ci'][1])})")
            md.append(f"| {results[models[0]]['bands'][b]['band']} | {base['n']:,} | {_p(base['mean'])} | "
                      f"{_p(base['corr_with_rate'])} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## 同時の回帰（偏り ~ 特殊拍の割合 + 拗音の割合 + 語の平均モーラ数 + 正解の話速）", "",
           "係数は特殊拍・拗音が割合0.1あたり、語の平均モーラ数が1モーラあたり（mora/s）。括弧は音源単位のブートストラップ95%区間。", "",
           "| 帯 | 窓数 | " + " | ".join(f"{m} 特殊拍 | {m} 拗音 | {m} 語長" for m in models) + " |",
           "| --- | ---: | " + " | ".join("--- | --- | ---" for _ in models) + " |"]
    for b in range(len(BAND_EDGES)):
        cells = []
        for m in models:
            j = results[m]["bands"][b]["joint"]
            cells.append(" | ".join(f"{_f(j[k]['coef'])} ({_f(j[k]['ci'][0])}, {_f(j[k]['ci'][1])})"
                                    for k in ("special", "yoon", "wordlen")))
        md.append(f"| {results[models[0]]['bands'][b]['band']} | {results[models[0]]['bands'][b]['joint']['n']:,} | "
                  + " | ".join(cells) + " |")
    md.append("")
    parts = ("chouon", "hatsuon", "sokuon", "yoon", "wordlen")
    md += ["## 同時の回帰の内訳（偏り ~ 長音 + 撥音 + 促音 + 拗音の割合 + 語の平均モーラ数 + 正解の話速）", "",
           "係数の単位は上と同じ。", "",
           "| 帯 | モデル | 長音 ー | 撥音 ン | 促音 ッ | 拗音 | 語長 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for b in range(len(BAND_EDGES)):
        for m in models:
            j = results[m]["bands"][b]["joint_parts"]
            md.append(f"| {results[m]['bands'][b]['band']} | {m} | "
                      + " | ".join(f"{_f(j[k]['coef'])} ({_f(j[k]['ci'][0])}, {_f(j[k]['ci'][1])})" for k in parts) + " |")
    md.append("")
    md += ["## 1 mora/s 刻みの区間ごとの傾き（偏り ~ 変数 + 正解の話速）", "",
           "| 区間 | 窓数 | " + " | ".join(f"{m} 特殊拍＋拗音 | {m} 特殊拍 | {m} 語長" for m in models) + " |",
           "| --- | ---: | " + " | ".join("---: | ---: | ---:" for _ in models) + " |"]
    base = results[models[0]]["fine"]
    for i, row in enumerate(base):
        cells = []
        for m in models:
            r = results[m]["fine"][i]
            cells.append(f"{_f(r['special_yoon'])} | {_f(r['special'])} | {_f(r['wordlen'])}")
        md.append(f"| {row['low']:.0f}〜{row['low'] + 1:.0f} | {row['n']:,} | " + " | ".join(cells) + " |")
    md.append("")
    return md


# ------------------------------------------------------------------ 入口


def parse_model(spec: str) -> tuple[str, Path, str]:
    name, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"--model は 名前=ディレクトリ[:キー]: {spec}")
    directory, _, key = rest.partition(":")
    return name, _resolve(directory), key or name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="名前=予測のディレクトリ[:npz のキー]（pred_clean.npz を読む）。複数可")
    parser.add_argument("--out-dir", default="runs/special_mora")
    parser.add_argument("--word-cache", default=None, help="語の分割のキャッシュ（既定 <out-dir>/words_dev.jsonl）")
    parser.add_argument("--min-true-mora", type=float, default=1.0)
    parser.add_argument("--exclude-latin", action="store_true",
                        help="文にラテン文字を含むクリップ（英文をアルファベット読みにしたラベルなど）に由来する窓を除く")
    parser.add_argument("--tables-name", default="special_mora",
                        help="出力の名前（<out-dir>/<名前>_summary.json・<名前>_tables.md）")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    ws, feats, keep = load_or_build_features(args)
    sel = keep & (ws.mora.astype(np.float32) >= np.float32(args.min_true_mora))
    latin_windows = int((sel & feats["latin"]).sum())
    if args.exclude_latin:
        sel &= ~feats["latin"]
    log(f"主指標の窓 {int(keep.sum())}、うち正解 {args.min_true_mora} モーラ以上 {int(sel.sum())}")
    models = [parse_model(s) for s in args.model]
    results: dict = {}
    for name, directory, key in models:
        with np.load(directory / "pred_clean.npz") as data:
            pred = data[key].astype(np.float32)
        if pred.shape != ws.mora.shape or not np.isfinite(pred).all():
            raise ValueError(f"{name}: 予測の形か値が不正")
        results[name] = analyze_model(name, pred, ws, feats, sel, args)
        results[name]["source"] = str(directory)
        log(f"{name}: 集計 n={results[name]['n']} 偏り={results[name]['overall_bias']:+.4f}")
    word_ok = feats["word_ok"][sel] & (feats["word_count"][sel] > 0)
    info = {"n": int(sel.sum()), "min_true_mora": args.min_true_mora, "no_word": int((~word_ok).sum()),
            "clip_word_mismatch": [int(v) for v in feats["clip_word_mismatch"]],
            "exclude_latin": bool(args.exclude_latin), "latin_windows": latin_windows}
    # 帯ごとの窓内の種類の平均（参考）
    true = ws.mora.astype(np.float64)[sel]
    bands = band_index(true / WINDOW_SEC)
    info["composition"] = []
    for b, (_, _, label) in enumerate(BAND_EDGES):
        mb = bands == b
        row = {"band": label, "n": int(mb.sum())}
        for key in CATEGORIES:
            row[key] = float(feats[key][sel][mb].sum() / true[mb].sum())
        info["composition"].append(row)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tables_name}_summary.json").write_text(
        json.dumps({"info": info, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_tables(results, [m[0] for m in models], info)
    md += ["## 帯ごとの種類の割合（窓の按分の和の比）", "",
           "| 帯 | 窓数 | " + " | ".join(CATEGORIES) + " |", "| --- | ---: | " + " | ".join("---:" for _ in CATEGORIES) + " |"]
    for row in info["composition"]:
        md.append(f"| {row['band']} | {row['n']:,} | " + " | ".join(f"{row[k]:.3f}" for k in CATEGORIES) + " |")
    (out_dir / f"{args.tables_name}_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"書き出し: {out_dir}/{args.tables_name}_summary.json・{args.tables_name}_tables.md（{time.perf_counter() - started:.0f}s）")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
