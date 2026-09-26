#!/usr/bin/env bash
# 遠隔機（既定: ubuntu-desktop）の ~/SpeedMeter/runs/<名前> を Mac の元のリポジトリの runs/ に戻す
# （docs/directives/2026-09-26-rtx3060.md タスク3、docs/decisions/010-compute-environment.md）。
#
# Mac の既存の runs/ は上書き・削除しない。同じ名前が Mac の runs/ にあれば何も送らずに停止する。
# 戻した後、通常ファイルの数と合計バイト数を遠隔機と Mac で比べて表示する。
#
# 使い方:
#   scripts/fetch_from_remote.sh [--host HOST] [--remote-dir DIR] [--dest DIR] [--retries N] <名前>...
#     <名前>        遠隔機の runs/ の下の実行名（ディレクトリ）またはファイル名。複数指定できる
#     --host        接続先（既定: ubuntu-desktop）
#     --remote-dir  遠隔機のリポジトリの置き場所。ホームからの相対パス（既定: SpeedMeter）
#     --dest        戻し先の runs/（既定: 元のリポジトリ（git の共通ディレクトリの親）の runs/）
#     --retries     ssh と rsync の再試行の回数（既定: 4）
#
# 例:
#   scripts/fetch_from_remote.sh exp000_method_b_smoke_cuda exp000_method_b_smoke_cuda.stdout.log
set -euo pipefail

HOST="ubuntu-desktop"
REMOTE_DIR="SpeedMeter"
DEST=""
RETRIES=4
NAMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) echo "不明な引数: $1" >&2; exit 2 ;;
    *) NAMES+=("$1"); shift ;;
  esac
done

if ((${#NAMES[@]} == 0)); then
  echo "実行名を1つ以上指定する（--help を参照）" >&2
  exit 2
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

retry() {
  local i
  for ((i = 1; i <= RETRIES; i++)); do
    if "$@"; then return 0; fi
    log "失敗（${i}/${RETRIES}回目）: $*" >&2
    ((i < RETRIES)) && sleep $((i * 5))
  done
  return 1
}

rssh() { ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" "$@"; }

# ssh の接続の失敗（終了コード255）だけを再試行する。コマンド自体の失敗は再試行しない
rssh_r() {
  local i rc
  for ((i = 1; i <= RETRIES; i++)); do
    rc=0
    rssh "$@" || rc=$?
    if ((rc != 255)); then return $rc; fi
    log "ssh の接続に失敗（${i}/${RETRIES}回目）" >&2
    ((i < RETRIES)) && sleep $((i * 5))
  done
  return 255
}

# 通常ファイルの数と合計バイト数、symlink の数（sync_to_remote.sh と同じ数え方）
COUNT_PL='use File::Find; my ($n,$s,$l)=(0,0,0);
find({no_chdir=>1, wanted=>sub { my @st=lstat($_); if (-l _) {$l++} elsif (-f _) {$n++; $s+=$st[7]} }}, $ARGV[0]);
print "$n $s $l\n";'

if [[ -z "$DEST" ]]; then
  COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
  DEST="$(cd "$COMMON_DIR/.." && pwd -P)/runs"
fi
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd -P)"
log "接続先=$HOST 遠隔機の runs=~/$REMOTE_DIR/runs 戻し先=$DEST"

# 先にすべての名前を確かめる（1つでも問題があれば何も送らない）
for name in "${NAMES[@]}"; do
  if [[ "$name" == */* || "$name" == "." || "$name" == ".." || -z "$name" ]]; then
    log "エラー: 名前は runs/ の直下の名前にする: $name"; exit 1
  fi
  if [[ -e "$DEST/$name" || -L "$DEST/$name" ]]; then
    log "エラー: Mac の $DEST/$name がすでにある。上書きしないので停止する"; exit 1
  fi
  if ! rssh_r "test -e '$REMOTE_DIR/runs/$name'"; then
    log "エラー: 遠隔機に ~/$REMOTE_DIR/runs/$name が無い"; exit 1
  fi
done

mismatch=0
for name in "${NAMES[@]}"; do
  if rssh_r "test -d '$REMOTE_DIR/runs/$name'"; then
    log "$name（ディレクトリ）を戻す"
    retry rsync -a --partial "$HOST:$REMOTE_DIR/runs/$name/" "$DEST/$name/"
  else
    log "$name（ファイル）を戻す"
    retry rsync -a --partial "$HOST:$REMOTE_DIR/runs/$name" "$DEST/$name"
  fi
  local_count="$(perl -e "$COUNT_PL" "$DEST/$name")"
  remote_count="$(rssh_r "perl -e '$COUNT_PL' '$REMOTE_DIR/runs/$name'")"
  if [[ "$local_count" == "$remote_count" ]]; then
    log "  一致 runs/$name: 遠隔機=[$remote_count] Mac=[$local_count]（通常ファイル数 バイト数 symlink数）"
  else
    log "  不一致 runs/$name: 遠隔機=[$remote_count] Mac=[$local_count]"
    mismatch=1
  fi
done

if ((mismatch)); then
  log "エラー: 一致しないものがある（遠隔機で書き込み中の可能性）"
  exit 1
fi
log "完了"
