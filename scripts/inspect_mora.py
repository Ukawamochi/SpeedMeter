"""モーラ数カウントの目視確認用スクリプト。

1行1文のテキストファイルを受け取り、各行について
「原文\tカタカナ読み\tモーラ数」をタブ区切りで標準出力に書き出す。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spkrate.labels.mora import count_mora, read_kana


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_file", type=Path, help="1行1文のテキストファイル")
    args = parser.parse_args()

    with args.text_file.open(encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            if text == "":
                continue
            kana = read_kana(text)
            mora = count_mora(kana)
            print(f"{text}\t{kana}\t{mora}")


if __name__ == "__main__":
    main()
