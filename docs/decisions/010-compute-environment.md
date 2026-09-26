# 計算機の構成（ubuntu-desktop の確認）（docs/directives/2026-09-26-rtx3060.md タスク1。0aed609 の改訂版を含む）

- 確認日: 2026-09-26（遠隔機の時刻 23:14 JST 頃。改訂版タスク1による追加の確認と導入は 23:30〜23:33 頃）
- 読んだ資料: `CLAUDE.md`、`docs/directives/2026-09-26-rtx3060.md`（0節・タスク1）、`src/spkrate/eval/runner.py`（`git_commit_info`）、`pyproject.toml`、`.python-version`
- 方法: Mac から `ssh ubuntu-desktop`（`~/.ssh/config` の設定。HostName 192.168.24.50、User xps）で読み取りのコマンドだけを実行した
- 最初の確認（23:14 頃）では遠隔機の環境を変えていない。書き込んだのは転送試験の一時ファイル（/tmp、測定後に削除）だけである。その後の導入と管理者操作は5節にすべて記す

---

## 0. 結論

| 項目 | 結果 |
| --- | --- |
| 停止条件（nvidia-smi が動かない） | **非該当**。ドライバ 595.91.07、CUDA 13.2、RTX 3060 12GB を認識 |
| 停止条件（ディスク空き60GB未満） | **非該当**。`/home/xps` のあるファイルシステム（/、ext4）の空きは 432GB |
| 停止条件（rsync が無い） | **非該当**。rsync 3.4.1（protocol 32） |
| 停止条件（導入に管理者権限が要り人間の操作が要る） | 確認時は tmux と git が無く、導入には sudo（パスワード入力）が要った。**その後、人間が git と tmux を導入した**（git 2.53.0、tmux 3.6 を ssh で確認済み）。解消 |
| uv | 確認時は無かった。Agent がユーザー権限で導入した（uv 0.12.19、`~/.local/bin/uv`） |
| sudo | 改訂版タスク1の時点で `sudo -n true` は成功した（人間が `/etc/sudoers.d/90-claude-setup` でパスワードなしの sudo を許可した） |
| Secure Boot | `mokutil --sb-state` は「SecureBoot enabled」。ドライバはすでに動作している（nvidia-smi で確認）ので、ドライバの導入も再起動もしない。停止条件（MOK の登録が必要）には当たらない |
| 追加の導入 | Agent が `sudo apt-get install build-essential cmake` を行った（pyopenjtalk が sdist だけで、ビルドに C/C++ コンパイラと cmake が要るため）。5節 |
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

RAM は指示書の記載（16GB）より少なく見える（OS から見える量が約 12.8GiB）。原因（搭載量の違い、ハードウェアの予約など）は未確認である（最初の確認時は sudo が使えず `dmidecode` を実行できなかった。その後も環境構築に要らない操作なので実行していない）。0節の5（1台で同時に実行する学習は1つまで）の前提は変わらないが、タスク3の完走確認では num_workers とメモリ使用量に注意する。

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
| git | 確認時は無 → 人間が導入 | 2.53.0（`/usr/bin/git`、apt の 1:2.53.0-1ubuntu1） |
| rsync | 有 | 3.4.1、protocol version 32（`/usr/bin/rsync`） |
| uv | 確認時は無 → Agent が導入 | 0.12.19（`~/.local/bin/uv`、`~/.local/bin/uvx`） |
| tmux | 確認時は無 → 人間が導入 | 3.6（`/usr/bin/tmux`、apt の 3.6a-2ubuntu0.1） |
| screen | 有 | `/usr/bin/screen` |
| nohup / setsid | 有 | `/usr/bin/nohup`、`/usr/bin/setsid` |
| curl | 無 | — |
| wget | 有 | `/usr/bin/wget` |
| gcc / g++ / make / cmake | 確認時は無 → Agent が導入 | gcc・g++ 15.2.0、GNU Make 4.4.1、cmake 4.2.3（`build-essential`、`cmake`） |
| libsndfile1 | 有 | 1.2.2-4 |
| sudo | 最初の確認時は `sudo -n` がパスワードを求めて失敗した。その後、人間が `/etc/sudoers.d/90-claude-setup` を置き、`sudo -n true` は成功する（終了コード0） |
| mokutil | `mokutil --sb-state` → SecureBoot enabled |

Mac 側の rsync は openrsync（protocol version 29）である。

## 2. タスク3に影響する事項

1. **uv**: Agent がユーザー権限で導入した（0.12.19、`~/.local/bin`）。導入スクリプトが `~/.bashrc`・`~/.profile` の末尾に `. "$HOME/.local/bin/env"` を加え、`~/.zshrc` を新しく作った（5節）。Ubuntu の `~/.bashrc` は非対話のシェルでは先頭で戻るため、`ssh ubuntu-desktop '<コマンド>'` の形では PATH に入らない。スクリプトでは `~/.local/bin/uv` と絶対パスで呼ぶか `bash -lc` を使う。`.python-version` は 3.12 で、遠隔機の python3 は 3.14 なので、uv が管理する Python 3.12 を取得させる（システムの python3 は使わない）
2. **tmux**: 確認時は無かったが、人間が導入した（tmux 3.6）。学習の起動には tmux を使える（nohup と setsid で代える必要は無くなった）
3. **git**: 確認時は無く、導入には sudo が要った。git が無いと `src/spkrate/eval/runner.py` の `git_commit_info` が `commit_hash = "unknown"` を返し、`tests/test_runner.py::test_git_commit_info_on_this_repo` が失敗するところだったが、**人間が git を導入した**（git 2.53.0）ため解消した。リポジトリは .git を含めて送るので、遠隔機でもコミットハッシュが記録される（遠隔機ではコミットしない）
4. **置き場所**: `~/SpeedMeter`（`/home/xps/SpeedMeter`、ext4、空き 432GB）。現時点で未作成
5. **RAM**: OS から見える量は約 12.8GiB（1.1節）。num_workers の既定値のままでメモリが足りるかを完走確認で見る
6. **/tmp は tmpfs（6.4GB）**: 大きな一時ファイルや uv のキャッシュを /tmp に置かない（uv の既定のキャッシュは `~/.cache/uv` で問題ない）
7. **pyopenjtalk のビルド**: uv.lock の pyopenjtalk は sdist だけで wheel が無いため、uv sync で C/C++ のビルドが走る。そのため build-essential と cmake を導入した。システムの cmake は 4.2.3 で、古い `cmake_minimum_required` を書いた CMakeLists を拒否することがある。uv sync でビルドに失敗した場合は、ビルド時の依存の cmake の版、または `CMAKE_POLICY_VERSION_MINIMUM=3.5` の指定を確認する（タスク3）

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

実施日: 2026-09-27 0:00〜0:10 頃（JST）。遠隔機のコミットは feat/remote-sync の 9fbfbb9（main f7d147e に下記 4.1 のスクリプトを加えたもの）。完走確認の後、main に取り込んだ 86a8929 を送り直した（遠隔機の HEAD = 86a8929、`git status --porcelain` は空）。

### 4.1 送り方（scripts/sync_to_remote.sh・scripts/fetch_from_remote.sh）

- `scripts/sync_to_remote.sh [--host HOST] [--src DIR] [--commit REV] [--remote-dir DIR] [--data] [--data-src DIR]`（既定: ubuntu-desktop、`~/SpeedMeter`）
  1. git の共通ディレクトリ（`git rev-parse --git-common-dir`、元の .git）を遠隔機の `~/SpeedMeter/.git` に送る。`worktrees/`・`index`・`*.lock` は送らない
  2. 送り元の作業ツリー（既定は実行した場所の作業ツリー。`--src` で worktree を指定できる）を、`data/`・`runs/`・`.venv/`・`listening/`・`results/error_cases/*.mp3`・`results/alignment_check/*.wav`・`__pycache__`・`.pytest_cache`・`.DS_Store` を除いて送る（`--delete` は除外した場所に及ばない）
  3. 遠隔機で `git checkout --force --detach <送り元の HEAD>` をし、送り元が clean なら遠隔機の `git status --porcelain` が空であることを確かめる
  4. `--data` のときだけ data/common_voice_ja・musan・processed を元のリポジトリの data/ の実体から送る（`--delete` なし）。送った後、通常ファイルの数・合計バイト数・symlink の数を Mac と遠隔機で同じ perl の数え方で比べる
  - ssh は接続の失敗（終了コード255）だけ、rsync は失敗時に、最大4回まで再試行する
- `scripts/fetch_from_remote.sh [--host HOST] <名前>...`: 遠隔機の `~/SpeedMeter/runs/<名前>`（ディレクトリまたはファイル）を元のリポジトリの runs/ に戻す。Mac に同じ名前があれば何も送らずに停止する（確認済み）。戻した後に数とバイト数を比べる
- **統括が許可した例外**: 遠隔機では `git checkout --force --detach` だけを行う（コミットはしない）。0節の5「ubuntu-desktop ではコミットしない」はこれで守られる。遠隔機の HEAD は detached で、次の送り出しで上書きされる

### 4.2 データの一致

`--data` の送り出しは約11分（23:56〜00:07、Common Voice 約6分、processed 約4分）。

| data/ | 通常ファイル数（Mac / 遠隔機） | 合計バイト数（Mac / 遠隔機） | 結果 |
| --- | --- | --- | --- |
| common_voice_ja | 585,341 / 585,341 | 16,002,538,336 / 16,002,538,336 | 一致 |
| musan | 935 / 935 | 717,334,021 / 717,334,021 | 一致 |
| processed | 557 / 557 | 22,301,841,915 / 22,301,841,915 | 一致 |

symlink はどちらも0。送った後の遠隔機のディスクは使用 58G・空き 387G（.venv は 6.4G）。

### 4.3 uv sync・環境・pytest

- `~/.local/bin/uv sync --frozen`: 成功。pyopenjtalk 0.4.1 は sdist からビルドされ、`CMAKE_POLICY_VERSION_MINIMUM` の指定は要らなかった。Python 3.12.14（uv が管理するもの）
- torch 2.14.0+cu130、`torch.version.cuda` = 13.0、cuDNN 92400、`torch.cuda.is_available()` = True、GPU = NVIDIA GeForce RTX 3060（capability 8.6）
- `scripts/check_env.py`: cuda の Conv2d 順伝播 OK、TF32 の設定はすべて無効（`allow_tf32=False`、fp32 精度 `ieee`）。pyopenjtalk の読み変換 OK（初回に open_jtalk の辞書を取得した）
- `pytest`（全件）: **730 passed、3 skipped**（31.8秒）。skip は mps が要る2件（test_cnn.py:481、test_device.py:107）と、ひらがな CTC モデルの重みが未取得の1件（test_alignment.py:293）。cuda が要る試験（`requires_cuda`）は skip されずに通った。遠隔機でアライメントを行うなら CTC モデルの重みの取得が要る

### 4.4 完走確認（1エポック、device=cuda）

`uv run --frozen python -m spkrate.train.train --config configs/<設定>.yaml --device cuda --experiment-id <設定>_cuda` を tmux の中で実行した（設定ファイルは作っていない）。両方とも終了コード0で完走した。log.txt に「計算機: ホスト名=xps 呼び名=ubuntu-desktop デバイス=cuda デバイス名=NVIDIA GeForce RTX 3060 torch=2.14.0+cu130 CUDA=13.0 cuDNN=92400」と「TF32 を無効にした … cuda_matmul_allow_tf32=False cudnn_allow_tf32=False」の行がある。stderr に CPU フォールバックの警告は無い。

| 設定 | 計算機 | num_workers | 全体 | 学習ループ | 計算 | データ待ち | 比率 | 終端処理 | dev評価 | 実行時間（プロセス全体） | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| method_b_smoke | Mac（mps） | 2 | 23.9秒 | 12.9秒 | 1.1秒 | 1.8秒 | 0.136 | 10.0秒 | 11.0秒 | 27.0秒 | 3.2764 |
| method_b_smoke | ubuntu（cuda） | 2 | 2.5秒 | 2.4秒 | 0.8秒 | 1.6秒 | **0.672** | 0.0秒 | 0.1秒 | 6.8秒 | 3.2815 |
| method_b_smoke | ubuntu（cuda） | 4 | 1.6秒 | 1.5秒 | 0.6秒 | 0.9秒 | 0.578 | 0.0秒 | 0.1秒 | 5.8秒 | 3.2815 |
| method_b_smoke | ubuntu（cuda） | 8 | 1.6秒 | 1.5秒 | 0.6秒 | 0.8秒 | 0.553 | 0.0秒 | 0.1秒 | 5.9秒 | 3.2815 |
| method_a_selection_smoke | Mac（mps） | 2 | 24.5秒 | 13.5秒 | 2.1秒 | 1.4秒 | 0.106 | 10.0秒 | 10.9秒 | 26.7秒 | 56.4619 |
| method_a_selection_smoke | ubuntu（cuda） | 2 | 2.0秒 | 1.8秒 | 0.8秒 | 1.0秒 | **0.529** | 0.0秒 | 0.2秒 | 5.5秒 | 56.4580 |
| method_a_selection_smoke | ubuntu（cuda） | 4 | 1.6秒 | 1.4秒 | 0.9秒 | 0.5秒 | 0.357 | 0.0秒 | 0.2秒 | 5.1秒 | 56.4580 |
| method_a_selection_smoke | ubuntu（cuda） | 8 | 1.7秒 | 1.5秒 | 0.9秒 | 0.5秒 | 0.355 | 0.0秒 | 0.2秒 | 5.1秒 | 56.4580 |

- Mac の値は runs/exp000_method_b_smoke・runs/exp000_method_a_selection_smoke の log.txt と *.time.log（2026-09-26 14:30、exp005 の開始前）。遠隔機の値は runs/<設定>_cuda[_nwN]/log.txt と runs/<設定>_cuda.stderr.log・runs/nw_trial/*.stderr.log（`/usr/bin/time -v`）
- num_workers の比較は、データ待ちの比率が0.5以上だったので行った。configs/ の設定から num_workers だけを変えた写しを遠隔機の runs/nw_trial/ に作って使った（リポジトリには入れていない。runs/nw_trial/ ごと Mac に戻した）
- 学習クリップ256件（31〜16バッチ）の小さな試験なので、データ待ちにはワーカーの起動の時間が多く含まれ、本番の学習の比率の目安としては粗い。Mac の「終端処理=10.0秒」は Mac でのワーカーの終了待ちで、遠隔機では0.0秒だった
- MAE は Mac と遠隔機で小数第3位で異なる（3.2764 と 3.2815、56.4619 と 56.4580）。拡張の乱数は同じ系列で（拡張の実適用回数は同じ）、差は mps と cuda の数値計算の差と考えられる。num_workers を 2・4・8 と変えても遠隔機の MAE は同じだった（学習損失は小数第3〜4位で異なる）
- 参考: Mac の exp005（本番、num_workers 4、バッチ64）の直近のエポックはデータ待ちの比率 0.35〜0.39 だった。cuda では計算が速くなるので、本番でも比率は Mac より上がる見込みである

### 4.5 メモリ使用量（遠隔機）

`free -m` と `nvidia-smi` を1秒ごとに記録した（runs/<設定>_cuda.mem.log、runs/nw_trial/*.mem.log）。実行前の used は約 2.4GiB（画面表示を含む）、available は約 10.7GiB。

| 設定 | num_workers | used の最大 | available の最小 | GPU メモリの最大 | 主プロセスの最大 RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| method_b_smoke | 2 | 4,027 MiB | 9,078 MiB | 564 MiB | 1.75 GB |
| method_b_smoke | 4 | 3,633 MiB | 9,472 MiB | 402 MiB | 1.77 GB |
| method_b_smoke | 8 | 5,204 MiB | 7,901 MiB | 564 MiB | 1.77 GB |
| method_a_selection_smoke | 2 | 4,040 MiB | 9,065 MiB | 1,610 MiB | 1.82 GB |
| method_a_selection_smoke | 4 | 4,606 MiB | 8,499 MiB | 1,610 MiB | 1.83 GB |
| method_a_selection_smoke | 8 | 5,225 MiB | 7,879 MiB | 1,610 MiB | 1.87 GB |

- 1秒ごとの記録なので最大値を取り逃している可能性がある。ワーカー1つあたり概ね 0.2〜0.4GiB 増えた。小さな試験ではメモリ不足の兆しは無い

### 4.6 exp006（タスク4-2）の起動で注意すること

- num_workers: exp004・exp005 と同じ 4 を推奨する（比較の条件をそろえるため。num_workers を変えても指標はほぼ変わらないことは 4.4 で確かめたが、条件は変えない方がよい）。起動後の最初のエポックでデータ待ちの比率と `free` を確かめ、比率が0.5以上で available に余裕（例えば 4GiB 以上）があれば、統括の判断で増やす余地がある
- バッチ64・全学習クリップではワーカーと主プロセスのメモリが試験より増える。最初のエポックの間に `free -m` を見て、available が 2GiB を切るようなら num_workers を減らす
- 起動は `cd ~/SpeedMeter && tmux new-session -d -s exp006 '~/.local/bin/uv run --frozen python -m spkrate.train.train --config configs/exp006.yaml > runs/exp006.stdout.log 2> runs/exp006.stderr.log'` の形（非対話の ssh では uv が PATH に無い）。起動前に `scripts/sync_to_remote.sh` で exp/011-method-a-control のコミットを送り、遠隔機の HEAD と `git status --porcelain` が空であることを確かめる
- 遠隔機では Mac と比べて dev評価と終端処理が大幅に短い（小さな試験で dev評価 約11秒 → 0.1〜0.2秒）

## 5. 導入したパッケージと実行した管理者操作

時刻は遠隔機の時刻（JST）。apt の記録は `/var/log/apt/history.log` で確認した。

| 日時 | 実行者 | 権限 | 操作 | 入ったもの |
| --- | --- | --- | --- | --- |
| 2026-09-26 23:19 | 人間 | sudo | `apt install tmux git` | git 1:2.53.0-1ubuntu1、tmux 3.6a-2ubuntu0.1（依存の自動導入: git-man、liberror-perl、libevent-core-2.1-7t64） |
| 2026-09-26 23:29 | 人間 | root | `/etc/sudoers.d/90-claude-setup` を置く（パスワードなしの sudo の許可） | — |
| 2026-09-26 23:31 | Agent | ユーザー | `wget -qO- https://astral.sh/uv/install.sh \| sh`（公式の導入スクリプト） | `~/.local/bin/uv`・`uvx` 0.12.19、`~/.local/bin/env`・`env.fish`。`~/.bashrc`・`~/.profile` の末尾に `. "$HOME/.local/bin/env"` を追加、`~/.zshrc` を新規作成（同じ1行） |
| 2026-09-26 23:32 | Agent | sudo | `apt-get update` | パッケージ一覧の更新だけ |
| 2026-09-26 23:32 | Agent | sudo | `apt-get install -y build-essential cmake` | build-essential 12.12ubuntu2.26.04.2、gcc・g++ 15.2.0、make 4.4.1、cmake 4.2.3、dpkg-dev ほか依存（自動導入） |

- Agent が sudo で行ったのは上の2件（`apt-get update` と `apt-get install build-essential cmake`）だけである。読み取りの確認として `sudo -n true` を実行した
- NVIDIA ドライバの導入、再起動、システムの設定（ファイアウォール・ssh・利用者）の変更は行っていない
- タスク3（2026-09-27）では sudo を使っていない。遠隔機に作ったのは `~/SpeedMeter`（リポジトリ・.venv・data/・runs/）と uv のキャッシュ（`~/.cache/uv`）、pyopenjtalk の辞書（.venv の中）だけである
