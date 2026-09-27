"""実環境評価用の録音の道具（docs/plan.md 9-2 の準備、docs/directives/2026-09-27.md 系統C）。

画面に1文ずつ表示し、Enter で録音を開始・終了して、文ごとに1つの wav
（16kHz・モノラル・16bit PCM）を保存する。r で同じ文を録り直す（上書き）。
条件（マイクの種類・話速）は引数で受け取り、ファイル名に入れる。

例:
    uv run python scripts/record_eval.py --mic builtin --rate normal
    uv run python scripts/record_eval.py --list-devices

手順は data/eval_scripts/README.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spkrate.eval.record import MICS, RATES, load_sentences, run_session  # noqa: E402

DEFAULT_SENTENCES = ROOT / "data" / "eval_scripts" / "sentences_labeled.tsv"
DEFAULT_OUT_DIR = ROOT / "data" / "eval_real"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="実環境評価の録音（1文ずつ、Enter で開始・終了、r で録り直し、q で終了）"
    )
    parser.add_argument("--mic", choices=list(MICS), help="マイクの種類: " + "、".join(f"{k}={v}" for k, v in MICS.items()))
    parser.add_argument("--rate", choices=list(RATES), help="話速: " + "、".join(f"{k}={v}" for k, v in RATES.items()))
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES, help="読み上げる文（既定: %(default)s）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="保存先（既定: %(default)s）")
    parser.add_argument("--start", type=int, default=1, help="この番号の文から始める（途中からの再開用）")
    parser.add_argument("--device", default=None, help="入力デバイスの番号または名前（既定: OS の既定の入力）")
    parser.add_argument("--list-devices", action="store_true", help="入力デバイスの一覧を表示して終わる")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        return args
    if args.mic is None or args.rate is None:
        parser.error("--mic と --rate を指定する（--list-devices の場合を除く）")
    if args.start < 1:
        parser.error("--start は1以上")
    if isinstance(args.device, str) and args.device.isdigit():
        args.device = int(args.device)
    return args


class SoundDeviceRecorder:
    """sounddevice でマイクから録音する。標本化周波数はデバイスの既定の値を使う。"""

    def __init__(self, device: int | str | None) -> None:
        import sounddevice as sd

        self._sd = sd
        self.device = device
        info = sd.query_devices(device, "input")
        self.name = info["name"]
        self.sr = int(info["default_samplerate"])
        self._chunks: list[np.ndarray] = []
        self._stream = None

    def _callback(self, indata, frames, time, status) -> None:  # noqa: ARG002
        if status:
            print(f"  （入力の状態: {status}）", file=sys.stderr)
        self._chunks.append(indata[:, 0].copy())

    def start(self) -> None:
        self._chunks = []
        self._stream = self._sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> tuple[np.ndarray, int]:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        return audio, self.sr


def list_devices() -> None:
    import sounddevice as sd

    default_in = sd.default.device[0]
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            mark = "*" if index == default_in else " "
            print(f"{mark} {index}: {info['name']}（{int(info['default_samplerate'])} Hz）")
    print("* は OS の既定の入力。--device に番号を渡すと切り替えられる。")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return
    if not args.sentences.is_file():
        sys.exit(f"文の一覧がない: {args.sentences}（scripts/select_eval_sentences.py で作る）")
    sentences = load_sentences(args.sentences)
    if args.start > len(sentences):
        sys.exit(f"--start は 1〜{len(sentences)}")
    recorder = SoundDeviceRecorder(args.device)
    print(f"入力デバイス: {recorder.name}（{recorder.sr} Hz で録音し、16000 Hz で保存）")
    try:
        run_session(sentences, args.mic, args.rate, args.out_dir, recorder, start=args.start)
    except (KeyboardInterrupt, EOFError):
        recorder.stop()
        print("\n中断した（録音中の文は保存していない）。")


if __name__ == "__main__":
    main()
