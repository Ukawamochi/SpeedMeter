# 計算機の構成（ubuntu-desktop の確認）（docs/directives/2026-09-26-rtx3060.md タスク1）

- 確認日: 2026-09-26（遠隔機の時刻 23:14 JST 頃）
- 読んだ資料: `CLAUDE.md`、`docs/directives/2026-09-26-rtx3060.md`（0節・タスク1）、`src/spkrate/eval/runner.py`（`git_commit_info`）、`pyproject.toml`、`.python-version`
- 方法: Mac から `ssh ubuntu-desktop`（`~/.ssh/config` の設定。HostName 192.168.24.50、User xps）で読み取りのコマンドだけを実行した
- 遠隔機の環境には変更を加えていない（パッケージの導入、uv の導入もしていない）。書き込んだのは転送試験の一時ファイル（/tmp、測定後に削除）だけである

---

## 0. 結論

| 項目 | 結果 |
| --- | --- |
| 停止条件（nvidia-smi が動かない） | **非該当**。ドライバ 595.91.07、CUDA 13.2、RTX 3060 12GB を認識 |
| 停止条件（ディスク空き60GB未満） | **非該当**。`/home/xps` のあるファイルシステム（/、ext4）の空きは 432GB |
| 停止条件（rsync が無い） | **非該当**。rsync 3.4.1（protocol 32） |
| 停止条件（導入に管理者権限が要り人間の操作が要る） | 確認時は tmux と git が無く、導入には sudo（パスワード入力）が要った。**その後、人間が git と tmux を導入した**（git 2.53.0、tmux 3.6 を ssh で確認済み）。解消 |
| uv | 無い（`~/.local/bin`、`~/.cargo/bin` にも無い）。ユーザー権限で導入できる（タスク3） |
| 置き場所 | `~/SpeedMeter`（= `/home/xps/SpeedMeter`、未作成）。data/・runs/ も同じファイルシステムに置く |
| 転送速度 | 300MiB の乱数ファイルで 約 62〜67 MiB/s（3回、4.5〜4.9秒） |

## 1. 確認した値

### 1.1 OS・CPU・RAM

| 項目 | 値 |
| --- | --- |
| ホスト名 | xps |
| OS | Ubuntu 26.04.1 LTS（Resolute Raccoon）、カーネル 7.0.0-34-generic |
| CPU | 13th Gen Intel Core i7-13700、1ソケット、16コア（P8 + E8）、24スレッド（`nproc` = 24） |
| RAM | MemTotal 13,420,216 kB（約 12.8GiB、`free -h` で total 12Gi、available 10Gi）。スワップ 4.0GiB |
| 稼働状況 | 確認時の load average 0.01、GPU 使用は画面表示のプロセスだけ（138MiB） |

RAM は指示書の記載（16GB）より少なく見える（OS から見える量が約 12.8GiB）。原因（搭載量の違い、ハードウェアの予約など）は sudo が要る `dmidecode` で確かめられないため未確認である。0節の5（1台で同時に実行する学習は1つまで）の前提は変わらないが、タスク3の完走確認では num_workers とメモリ使用量に注意する。

### 1.2 ディスク

| マウント先 | 種別 | 大きさ | 使用 | 空き | 用途 |
| --- | --- | --- | --- | --- | --- |
| `/`（`/dev/nvme0n1p2`、`/home/xps` を含む） | ext4 | 468G | 13G | **432G** | リポジトリ・data/・runs/・.venv の置き場所（`~/SpeedMeter`） |
| `/tmp` | tmpfs | 6.4G | 3.9M | 6.4G | RAM 上。大きなファイルを置かない |
| `/run/media/xps/DATA`（`/dev/sda2`、BitLocker 暗号化） | — | 1.9T | 86G | 1.8T | 使わない（デスクトップのログイン時に開かれる取り外し可能な媒体で、ssh だけの運用では開かれない恐れがある） |

必要量の目安（Common Voice 約16GB、data/processed 約21GB、MUSAN、仮想環境、runs/）に対して十分である。

### 1.3 GPU（nvidia-smi）

```
NVIDIA-SMI 595.91.07      Driver Version: 595.91.07      CUDA Version: 13.2
GPU 0: NVIDIA GeForce RTX 3060  Bus-Id 00000000:01:00.0  Disp.A On
       12288MiB（使用 138MiB）  Perf P8  9W / 170W  Compute M. Default
compute_cap 8.6
```

- カーネルモジュールは NVIDIA Open Kernel Module 595.91.07
- `nvcc` は無い（CUDA Toolkit は未導入）。PyTorch の cuda 版の wheel は実行時ライブラリを同梱するため、学習には不要である。表示の「CUDA Version 13.2」はドライバが対応する上限であり、wheel の CUDA の版（例: cu12x、cu13x）がこれ以下なら動く
- 画面表示に同じ GPU を使っている（Disp.A On）。学習時に使える VRAM は 12GB から表示分（約 0.1GB）を引いた量になる

### 1.4 ソフトウェア

| 名前 | 有無 | 版・場所 |
| --- | --- | --- |
| python3 | 有 | Python 3.14.4（`/usr/bin/python3`）。`venv` は使えるが `ensurepip` が無い |
| git | 確認時は無 → 人間が導入 | 2.53.0（`/usr/bin/git`） |
| rsync | 有 | 3.4.1、protocol version 32（`/usr/bin/rsync`） |
| uv | **無** | `~/.local/bin/uv`、`~/.cargo/bin/uv` も無い |
| tmux | 確認時は無 → 人間が導入 | 3.6（`/usr/bin/tmux`） |
| screen | 有 | `/usr/bin/screen` |
| nohup / setsid | 有 | `/usr/bin/nohup`、`/usr/bin/setsid` |
| curl | 無 | — |
| wget | 有 | `/usr/bin/wget` |
| gcc / make | 無 | — |
| libsndfile1 | 有 | 1.2.2-4 |
| sudo | 所属グループに sudo はあるが、`sudo -n` はパスワードを求めて失敗する（非対話では使えない） |

Mac 側の rsync は openrsync（protocol version 29）である。

## 2. タスク3に影響する事項

1. **uv**: 無い。curl が無いので、ユーザー権限での導入は wget で配布元の導入スクリプトを取得して行う（`~/.local/bin` に入る）。`.python-version` は 3.12、`pyproject.toml` は `requires-python >= 3.12` であり、遠隔機の python3 は 3.14 なので、uv が管理する Python 3.12 を取得させる（システムの python3 は使わない）
2. **tmux**: 確認時は無かったが、人間が導入した（tmux 3.6）。学習の起動には tmux を使える（nohup と setsid で代える必要は無くなった）
3. **git**: 確認時は無く、導入には sudo が要った。git が無いと `src/spkrate/eval/runner.py` の `git_commit_info` が `commit_hash = "unknown"` を返し、`tests/test_runner.py::test_git_commit_info_on_this_repo` が失敗するところだったが、**人間が git を導入した**（git 2.53.0）ため解消した。リポジトリは .git を含めて送るので、遠隔機でもコミットハッシュが記録される（遠隔機ではコミットしない）
4. **置き場所**: `~/SpeedMeter`（`/home/xps/SpeedMeter`、ext4、空き 432GB）。現時点で未作成
5. **RAM**: OS から見える量は約 12.8GiB（1.1節）。num_workers の既定値のままでメモリが足りるかを完走確認で見る
6. **/tmp は tmpfs（6.4GB）**: 大きな一時ファイルや uv のキャッシュを /tmp に置かない（uv の既定のキャッシュは `~/.cache/uv` で問題ない）

## 3. 転送速度

- 方法: Mac のスクラッチパッドに `dd if=/dev/urandom` で 300MiB（314,572,800 バイト、圧縮が効かない）のファイルを作り、`rsync -a` で `ubuntu-desktop:/tmp/` に送った時間を測った。Mac 側は有線（USB-C Dock Ethernet、en5）、同じ LAN（192.168.24.0/24）
- 結果:

| 回 | 時間 | 速度 |
| --- | --- | --- |
| 1 | 4.48 秒 | 67.0 MiB/s |
| 2 | 4.87 秒 | 61.6 MiB/s |
| 3 | 4.65 秒 | 64.6 MiB/s |

- 送った後の SHA-256 は Mac と一致した。試験ファイルは遠隔機の /tmp と Mac のスクラッチパッドから削除した
- 目安: 約 40GB（Common Voice、data/processed、MUSAN）を大きなファイルとして送るなら約 10〜11 分。Common Voice の音声は小さなファイルが多いため、実際にはこれより長くかかる見込みである
- ping の往復は 69〜153 ms とばらついた（無線区間を含む可能性）。1回目の ssh 接続は「No route to host」で失敗し、直後の再試行で成功した。sync_to_remote.sh では接続失敗時の再試行を考慮するとよい

## 4. 完走確認の結果（タスク3）

（タスク3で追記する）
