"""Common Voice日本語データの内容を確認し、概要を results/common_voice_overview.md に書き出す。

docs/PLAN.md 第2段階「2-1 展開と内容確認」に対応する。

data/common_voice_ja/ は読み取りのみ行い、一切変更しない。

出力:
- results/common_voice_overview.md
"""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "data" / "common_voice_ja"
CLIPS_DIR = CV_DIR / "clips"
DURATIONS_TSV = CV_DIR / "clip_durations.tsv"
VALIDATED_TSV = CV_DIR / "validated.tsv"
OUT_MD = ROOT / "results" / "common_voice_overview.md"

# 無作為抽出の固定シード。再現のため変更しない。
SEED = 20260920
N_AUDIO_SAMPLES = 10
N_DURATION_SAMPLES = 1000

# TSVはタブ区切りで引用符を使わない（sentence中の " をそのまま保持するため）。
CSV_KWARGS = {"delimiter": "\t", "quoting": csv.QUOTE_NONE}


def count_newlines(path: Path) -> int:
    """ファイル中の改行数を数える。末尾に改行が無い行も1行として数える。"""
    total = 0
    last = b"\n"
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            total += chunk.count(b"\n")
            last = chunk[-1:]
    if last not in (b"\n", b""):
        total += 1
    return total


def count_csv_records(path: Path) -> int:
    """csvモジュールでレコード数を数える（ヘッダ行を除く）。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, **CSV_KWARGS)
        return max(sum(1 for _ in reader) - 1, 0)


def scan_clips(directory: Path) -> tuple[int, int, int]:
    """clips/ を1回走査し、ファイル数・合計バイト数・mp3以外の件数を返す。"""
    count = 0
    total_bytes = 0
    non_mp3 = 0
    with os.scandir(directory) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            count += 1
            total_bytes += entry.stat(follow_symlinks=False).st_size
            if not entry.name.endswith(".mp3"):
                non_mp3 += 1
    return count, total_bytes, non_mp3


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def load_durations_ms(path: Path) -> dict[str, int]:
    """clip_durations.tsv を {クリップ名: 長さ(ミリ秒)} として読む。"""
    durations: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, **CSV_KWARGS)
        header = next(reader)
        clip_i = header.index("clip")
        dur_i = header.index("duration[ms]")
        for row in reader:
            if len(row) <= max(clip_i, dur_i):
                continue
            durations[row[clip_i]] = int(row[dur_i])
    return durations


def read_validated(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, **CSV_KWARGS)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def git_ignored(paths: list[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for p in paths:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", p], cwd=ROOT, capture_output=True
        )
        result[p] = proc.returncode == 0
    return result


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    import random

    if not CV_DIR.is_dir():
        print(f"データが見つからない: {CV_DIR}", file=sys.stderr)
        return 1

    rng = random.Random(SEED)
    lines: list[str] = []

    # --- 1. 各tsvの行数 ---
    print("tsvの行数を数える", flush=True)
    tsv_paths = sorted(CV_DIR.glob("*.tsv"))
    tsv_stats = []
    for p in tsv_paths:
        newlines = count_newlines(p)
        records = count_csv_records(p)
        tsv_stats.append(
            {
                "name": p.name,
                "data_rows": max(newlines - 1, 0),
                "csv_records": records,
                "bytes": p.stat().st_size,
            }
        )
        print(f"  {p.name}: {max(newlines - 1, 0)}", flush=True)

    # --- 2. validated.tsv の中身 ---
    print("validated.tsv を読む", flush=True)
    header, rows = read_validated(VALIDATED_TSV)
    client_i = header.index("client_id")
    path_i = header.index("path")
    unique_clients = len({row[client_i] for row in rows if len(row) > client_i})
    head_rows = rows[:5]
    validated_paths = [row[path_i] for row in rows if len(row) > path_i]

    # --- 3. clips/ の総数と合計サイズ ---
    print("clips/ を走査する", flush=True)
    clip_count, clip_bytes, non_mp3 = scan_clips(CLIPS_DIR)
    print(f"  clips: {clip_count} files, {clip_bytes} bytes", flush=True)

    # --- 4. 無作為10件の音声の形式 ---
    print("無作為10件の音声を読む", flush=True)
    audio_samples = rng.sample(validated_paths, N_AUDIO_SAMPLES)
    audio_info = []
    for name in audio_samples:
        info = sf.info(str(CLIPS_DIR / name))
        audio_info.append(
            {
                "name": name,
                "samplerate": info.samplerate,
                "channels": info.channels,
                "format": f"{info.format}/{info.subtype}",
                "duration_sec": info.frames / info.samplerate,
            }
        )

    # --- 5. クリップ長（clip_durations.tsv を使い、実音声10件で突き合わせる）---
    print("clip_durations.tsv を読む", flush=True)
    durations_ms = load_durations_ms(DURATIONS_TSV)
    duration_samples = rng.sample(validated_paths, N_DURATION_SAMPLES)
    sampled_sec = [
        durations_ms[name] / 1000.0 for name in duration_samples if name in durations_ms
    ]
    missing = N_DURATION_SAMPLES - len(sampled_sec)

    # 実音声との突き合わせ
    cross = []
    for item in audio_info:
        tsv_sec = durations_ms.get(item["name"])
        cross.append(
            {
                "name": item["name"],
                "tsv_sec": None if tsv_sec is None else tsv_sec / 1000.0,
                "audio_sec": item["duration_sec"],
                "diff_sec": None
                if tsv_sec is None
                else abs(tsv_sec / 1000.0 - item["duration_sec"]),
            }
        )
    diffs = [c["diff_sec"] for c in cross if c["diff_sec"] is not None]
    max_diff = max(diffs) if diffs else None

    # --- 6. .gitignore の確認 ---
    ignored = git_ignored(["data/.archives/", "data/common_voice_ja/"])

    # --- 出力 ---
    lines.append("# Common Voice 日本語データ 内容確認")
    lines.append("")
    lines.append(f"- 対象ディレクトリ: `{CV_DIR.relative_to(ROOT)}`")
    lines.append("- 版: cv-corpus-27.0-2026-09-11（Common Voice 日本語）")
    lines.append(f"- 生成スクリプト: `scripts/{Path(__file__).name}`")
    lines.append(f"- 無作為抽出の固定シード: `{SEED}`（`random.Random({SEED})`）")
    lines.append("- data/ 以下は読み取りのみ行い、変更・削除はしていない")
    lines.append("")

    lines.append("## 1. 各tsvファイルの行数")
    lines.append("")
    lines.append("行数はいずれもヘッダ行を除いたデータ行数である。")
    lines.append(
        "「改行数-1」は改行を数えた値、「csvレコード数」は"
        "タブ区切り・引用符なし（QUOTE_NONE）で解析したレコード数から"
        "ヘッダ1行を引いた値。両者が一致すれば1レコード1行である。"
    )
    lines.append("")
    lines.append("| ファイル | データ行数（改行数-1） | csvレコード数 | サイズ |")
    lines.append("| --- | ---: | ---: | ---: |")
    for s in tsv_stats:
        lines.append(
            f"| {s['name']} | {s['data_rows']:,} | {s['csv_records']:,} | {human_bytes(s['bytes'])} |"
        )
    lines.append("")
    mismatched = [s["name"] for s in tsv_stats if s["data_rows"] != s["csv_records"]]
    if mismatched:
        lines.append(f"両者が一致しないファイル: {', '.join(mismatched)}")
    else:
        lines.append("すべてのファイルで両者が一致した（1レコード1行）。")
    lines.append("")

    lines.append("## 2. validated.tsv のカラム一覧と先頭5行")
    lines.append("")
    lines.append(f"カラム数: {len(header)}")
    lines.append("")
    for i, col in enumerate(header, 1):
        lines.append(f"{i}. `{col}`")
    lines.append("")
    lines.append("先頭5行（データ行）。client_id と sentence_id は長いため先頭12文字のみ示す。")
    lines.append("")
    show_cols = list(header)
    lines.append("| " + " | ".join(show_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in show_cols) + " |")
    for row in head_rows:
        cells = []
        for col, value in zip(header, row):
            if col in ("client_id", "sentence_id"):
                value = value[:12] + "…"
            cells.append(md_escape(value) if value else "")
        cells += [""] * (len(show_cols) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 3. 音声ファイル（clips/）")
    lines.append("")
    lines.append(f"- ファイル総数: {clip_count:,}")
    lines.append(f"- 合計サイズ: {human_bytes(clip_bytes)}（{clip_bytes:,} バイト）")
    lines.append(f"- 拡張子が .mp3 でないファイル: {non_mp3:,}")
    lines.append(
        f"- 参考: validated.tsv の行数 {len(rows):,} に対し、clips/ には "
        f"{clip_count:,} 件（other/invalidated 等も含む全クリップ）が存在する"
    )
    lines.append("")

    lines.append("## 4. validated.tsv の話者数")
    lines.append("")
    lines.append(f"- client_id のユニーク数: {unique_clients:,}")
    lines.append(f"- validated.tsv のデータ行数: {len(rows):,}")
    lines.append(f"- 1話者あたりの平均クリップ数: {len(rows) / unique_clients:.1f}")
    lines.append("")

    lines.append("## 5. 無作為10件の音声のサンプリングレートとチャンネル数")
    lines.append("")
    lines.append(
        f"validated.tsv の path 列から固定シード {SEED} で10件を抽出し、"
        "soundfile（libsndfile）で読み取った。"
    )
    lines.append("")
    lines.append("| クリップ | サンプリングレート(Hz) | チャンネル数 | 形式 | 長さ(秒) |")
    lines.append("| --- | ---: | ---: | --- | ---: |")
    for item in audio_info:
        lines.append(
            f"| {item['name']} | {item['samplerate']:,} | {item['channels']} | "
            f"{item['format']} | {item['duration_sec']:.3f} |"
        )
    lines.append("")
    rates = sorted({i["samplerate"] for i in audio_info})
    chans = sorted({i["channels"] for i in audio_info})
    lines.append(
        f"10件すべてで サンプリングレート {rates}、チャンネル数 {chans} であった。"
        if len(rates) == 1 and len(chans) == 1
        else f"サンプリングレートの種類: {rates}、チャンネル数の種類: {chans}"
    )
    lines.append("")

    lines.append("## 6. 無作為1000件のクリップ長")
    lines.append("")
    lines.append(
        "**使用した方法: clip_durations.tsv の値**（duration[ms] をミリ秒として読み、"
        "秒に換算した）。validated.tsv の path 列から固定シード "
        f"{SEED} で1000件を抽出した（上記10件の抽出のあと、同じ乱数列から抽出）。"
    )
    lines.append("")
    lines.append(f"- 対象件数: {len(sampled_sec):,}（clip_durations.tsv に無かった件数: {missing}）")
    lines.append(f"- 最小: {min(sampled_sec):.3f} 秒")
    lines.append(f"- 最大: {max(sampled_sec):.3f} 秒")
    lines.append(f"- 平均: {statistics.fmean(sampled_sec):.3f} 秒")
    lines.append(f"- 中央値: {statistics.median(sampled_sec):.3f} 秒")
    lines.append(f"- 合計: {sum(sampled_sec) / 3600:.2f} 時間")
    lines.append("")
    lines.append("### 実音声との突き合わせ")
    lines.append("")
    lines.append(
        "clip_durations.tsv の値が信用できるか確認するため、5節の10件について "
        "soundfile で読んだ実長（frames / samplerate）と比較した。"
    )
    lines.append("")
    lines.append("| クリップ | clip_durations.tsv(秒) | 実音声(秒) | 差(秒) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for c in cross:
        tsv_s = "なし" if c["tsv_sec"] is None else f"{c['tsv_sec']:.3f}"
        diff_s = "-" if c["diff_sec"] is None else f"{c['diff_sec']:.3f}"
        lines.append(f"| {c['name']} | {tsv_s} | {c['audio_sec']:.3f} | {diff_s} |")
    lines.append("")
    if max_diff is not None:
        lines.append(f"差の最大は {max_diff:.3f} 秒であった。")
    lines.append("")

    lines.append("## 7. .gitignore の確認")
    lines.append("")
    status = " / ".join(
        f"`{p}` は{'無視される' if ok else '無視されない'}" for p, ok in ignored.items()
    )
    lines.append(
        f"`.gitignore` の `/data/*`（例外は `/data/DATASETS.md` のみ）により、{status}"
        "（`git check-ignore` で確認）。"
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"書き出した: {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
