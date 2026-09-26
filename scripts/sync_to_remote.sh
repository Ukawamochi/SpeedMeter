#!/usr/bin/env bash
# Mac のリポジトリを遠隔機（既定: ubuntu-desktop）の ~/SpeedMeter に送る
# （docs/directives/2026-09-26-rtx3060.md タスク3、docs/decisions/010-compute-environment.md）。
#
# 手順:
#   1. git の共通ディレクトリ（`git rev-parse --git-common-dir`、元の .git）を遠隔機の
#      <remote-dir>/.git に送る。worktree の .git はポインタのファイルなので送らない。
#      worktrees/（Mac の worktree の管理情報）と index は送らない。
#   2. 送り元の作業ツリーのファイルを、data/・runs/・.venv/・listening/・
#      results/error_cases/ と results/alignment_check/ の音声・__pycache__ などを除いて送る。
#      除外したものは遠隔機でも消さない（rsync の --delete は除外した場所に及ばない）。
#   3. 遠隔機で `git checkout --force --detach <コミット>` をして HEAD をそろえ、
#      送り元が clean なら遠隔機の `git status --porcelain` が空であることを確かめる。
#      遠隔機ではコミットしない（checkout だけを行う）。
#   4. --data を付けたときだけ data/common_voice_ja・data/musan・data/processed を送る。
#      送り元は元のリポジトリの data/ の実体（symlink を辿った先）。--delete は使わない。
#      送った後、ファイル数と合計バイト数を Mac と遠隔機で比べて表示する。
#
# 使い方:
#   scripts/sync_to_remote.sh [--host HOST] [--src DIR] [--commit REV] [--remote-dir DIR]
#                             [--data] [--data-src DIR] [--retries N]
#     --host        接続先（既定: ubuntu-desktop）
#     --src         送り元の作業ツリー（既定: このスクリプトを実行した場所の git の作業ツリーのトップ）
#     --commit      遠隔機で checkout するコミット（既定: 送り元の HEAD）
#     --remote-dir  遠隔機の置き場所。ホームからの相対パス（既定: SpeedMeter）
#     --data        データも送る
#     --data-src    データの送り元の data/（既定: 元のリポジトリ（共通ディレクトリの親）の data/）
#     --retries     ssh と rsync の再試行の回数（既定: 4）
#
# 例:
#   scripts/sync_to_remote.sh                                   # コードと .git だけ
#   scripts/sync_to_remote.sh --src ../SpeedMeter-wt-cuda --data
set -euo pipefail

HOST="ubuntu-desktop"
SRC=""
COMMIT=""
REMOTE_DIR="SpeedMeter"
WITH_DATA=0
DATA_SRC=""
RETRIES=4
DATA_DIRS=(common_voice_ja musan processed)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --data) WITH_DATA=1; shift ;;
    --data-src) DATA_SRC="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "不明な引数: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 失敗したら待って再試行する（ssh の初回の「No route to host」対策）
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

# 通常ファイルの数と合計バイト数、symlink の数を「files bytes symlinks」で出す。
# Mac と Linux の両方にある perl で同じ数え方をする。symlink は辿らない。
COUNT_PL='use File::Find; my ($n,$s,$l)=(0,0,0);
find({no_chdir=>1, wanted=>sub { my @st=lstat($_); if (-l _) {$l++} elsif (-f _) {$n++; $s+=$st[7]} }}, $ARGV[0]);
print "$n $s $l\n";'

# --- 送り元の決定 -------------------------------------------------------------
if [[ -z "$SRC" ]]; then
  SRC="$(git rev-parse --show-toplevel)"
fi
SRC="$(cd "$SRC" && pwd -P)"
COMMON_DIR="$(git -C "$SRC" rev-parse --path-format=absolute --git-common-dir)"
MAIN_REPO="$(cd "$COMMON_DIR/.." && pwd -P)"
SRC_HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [[ -z "$COMMIT" ]]; then
  COMMIT="$SRC_HEAD"
else
  COMMIT="$(git -C "$SRC" rev-parse --verify "${COMMIT}^{commit}")"
fi
SRC_DIRTY="$(git -C "$SRC" status --porcelain)"

log "接続先=$HOST 置き場所=~/$REMOTE_DIR"
log "送り元の作業ツリー=$SRC（HEAD=$SRC_HEAD）"
log "git の共通ディレクトリ=$COMMON_DIR"
log "遠隔機で checkout するコミット=$COMMIT"
if [[ -n "$SRC_DIRTY" ]]; then
  log "注意: 送り元に未コミットの変更がある。遠隔機では checkout で追跡中のファイルがコミットの状態に戻る"
fi

rssh_r "mkdir -p '$REMOTE_DIR/.git' '$REMOTE_DIR/data' '$REMOTE_DIR/runs'"

# --- 1. .git（共通ディレクトリ） ------------------------------------------------
log "1. .git を送る"
retry rsync -a --partial --delete \
  --exclude='/worktrees/' --exclude='/index' --exclude='*.lock' \
  "$COMMON_DIR/" "$HOST:$REMOTE_DIR/.git/"

# --- 2. 作業ツリー ------------------------------------------------------------
log "2. 作業ツリーを送る"
retry rsync -a --partial --delete \
  --exclude='/.git' \
  --exclude='/data/' \
  --exclude='/runs/' \
  --exclude='/.venv/' \
  --exclude='/listening/' \
  --exclude='/results/error_cases/*.mp3' \
  --exclude='/results/alignment_check/*.wav' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.DS_Store' \
  "$SRC/" "$HOST:$REMOTE_DIR/"

# --- 3. HEAD をそろえる ---------------------------------------------------------
log "3. 遠隔機で checkout する（コミットはしない）"
rssh_r "cd '$REMOTE_DIR' && git -c core.ignorecase=false checkout --quiet --force --detach '$COMMIT'"
REMOTE_HEAD="$(rssh_r "cd '$REMOTE_DIR' && git rev-parse HEAD")"
REMOTE_STATUS="$(rssh_r "cd '$REMOTE_DIR' && git -c core.ignorecase=false status --porcelain")"
log "遠隔機の HEAD=$REMOTE_HEAD"
if [[ "$REMOTE_HEAD" != "$COMMIT" ]]; then
  log "エラー: 遠隔機の HEAD が $COMMIT と一致しない"
  exit 1
fi
if [[ -z "$SRC_DIRTY" && "$COMMIT" == "$SRC_HEAD" ]]; then
  if [[ -n "$REMOTE_STATUS" ]]; then
    log "エラー: 送り元は clean だが遠隔機の git status --porcelain が空でない:"
    echo "$REMOTE_STATUS"
    exit 1
  fi
  log "遠隔機の git status --porcelain は空（一致）"
else
  log "遠隔機の git status --porcelain（送り元が clean でない、または別のコミットのため参考）:"
  echo "${REMOTE_STATUS:-（空）}"
fi

# --- 4. データ ------------------------------------------------------------------
if ((WITH_DATA)); then
  if [[ -z "$DATA_SRC" ]]; then
    DATA_SRC="$MAIN_REPO/data"
  fi
  DATA_SRC="$(cd "$DATA_SRC" && pwd -P)"
  log "4. データを送る（送り元=$DATA_SRC、--delete は使わない）"
  mismatch=0
  for d in "${DATA_DIRS[@]}"; do
    real="$(cd "$DATA_SRC/$d" && pwd -P)"   # symlink を辿った実体
    log "  $d（実体=$real）を送る"
    retry rsync -a --partial "$real/" "$HOST:$REMOTE_DIR/data/$d/"
  done
  log "  ファイル数と合計バイト数を比べる（通常ファイル数 バイト数 symlink数）"
  for d in "${DATA_DIRS[@]}"; do
    real="$(cd "$DATA_SRC/$d" && pwd -P)"
    local_count="$(perl -e "$COUNT_PL" "$real")"
    remote_count="$(rssh_r "perl -e '$COUNT_PL' '$REMOTE_DIR/data/$d'")"
    if [[ "$local_count" == "$remote_count" ]]; then
      log "  一致 data/$d: Mac=[$local_count] 遠隔機=[$remote_count]"
    else
      log "  不一致 data/$d: Mac=[$local_count] 遠隔機=[$remote_count]"
      mismatch=1
    fi
  done
  if ((mismatch)); then
    log "エラー: データが一致しない"
    exit 1
  fi
  log "データはすべて一致した"
fi

log "完了"
