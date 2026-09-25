"""MUSAN noise をファイル単位で学習用・評価用に分け、configs/splits/musan_noise.json に固定する。

処理の本体は src/spkrate/data/musan_split.py にある。data/musan は読むだけで変更しない。
固定の種を使うため、同じ入力に対して何度実行しても同じ分割になる。

実行（リポジトリ直下から）:
    uv run python scripts/build_musan_noise_split.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.data.musan_split import (  # noqa: E402
    DEFAULT_EVAL_RATIO,
    MUSAN_SPLIT_SEED,
    build_musan_noise_split,
    list_noise_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan-root", default="data/musan")
    parser.add_argument("--seed", type=int, default=MUSAN_SPLIT_SEED)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--output", default="configs/splits/musan_noise.json")
    args = parser.parse_args(argv)

    musan_root = Path(args.musan_root)
    if not musan_root.is_absolute():
        musan_root = ROOT / musan_root
    groups = list_noise_files(musan_root)
    payload = build_musan_noise_split(groups, seed=args.seed, eval_ratio=args.eval_ratio)
    payload = {"musan_root": args.musan_root, **payload}

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for group, count in payload["counts"].items():
        print(f"{group}: 学習用 {count['train']} / 評価用 {count['eval']}（計 {count['total']}）")
    print(f"合計: 学習用 {payload['num_train']} / 評価用 {payload['num_eval']} → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
