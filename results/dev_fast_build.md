# 速めた音声の評価セット dev_fast の作成（指示書 2026-09-28-fast-speech タスク1 評価2）

- 定義: `configs/eval/dev_fast.yaml`。実装: `src/spkrate/eval/fast_speech.py`（単体テスト `tests/test_fast_speech.py`）、入口: `scripts/build_dev_fast.py`（作成・確認）、`scripts/eval_fast_speech.py`（推論・区間別の集計・metrics.csv への追記）
- 作成: 2026-09-28、Mac、コミット becdca8、所要259秒。ログ `runs/dev_fast_build/build.log`（Git 管理外）
- 出力: `data/processed/dev_fast/`（Git 管理外、1.6GB）。`selection.json`（選んだ音源）と条件ごとの `x1.5/`・`x2/`（`sources.jsonl`・`windows.npz`・`meta.json`・`audio.npy`）
- configs/splits/test.json は使っていない。音声は聴いていない

## 1. 作り方

- 音源: dev_window（`data/processed/dev_window/`、36,629音源）のうち、2倍速でも窓を1つ以上持つ音源（元の長さ4.0秒以上、20,460音源）を候補とし、`numpy.random.default_rng(202609281).permutation` の先頭2,000件を選んだ。単一クリップ1,278・連続発話の組722、クリップ4,108件、元の音声6.43時間。選んだ音源は2条件で共通
- 速める: 音源の clean の波形全体（連続発話は無音を含む連結音声の全体。`DevWindowAudio.source_waveform`）を1.5倍速・2倍速にする。長さは round(元の標本数 ÷ 速さ)
  - 方式: **WSOLA**（audiotsm 0.1.2 の `audiotsm.wsola`、numpy 2.5.3）。フレーム長1024標本（64ms）、合成の刻み512、分析の刻み 512 × 速さ（768・1024）、許容のずれ512標本。学習の時間伸縮（位相ボコーダ、librosa）とは別の方式
  - audiotsm は末尾の数フレームを出力しないため、入力の末尾に0を4096標本足してから速め、先頭から所定の長さを取る
- 時刻の変換: モーラ時刻 t（音源内。連結では各クリップの開始位置を足した時刻）を t ÷ 速さ にする。合成の信号で、WSOLA の出力の時刻が t ÷ 速さ と20ms以内で対応することを確かめた（`tests/test_fast_speech.py`）
- 窓と正解: dev_window と同じ（2.0秒窓・0.25秒ずらし・窓全体が収まるものだけ・按分の正解）
- 主指標の除外: dev_window と同じ（known_no_speech・no_speech_suspect のクリップを由来に持つ窓を集計で除く）

## 2. 窓数

| 条件 | 全窓 | 単一 | 連続 | 正解0 | 主指標（除外後） | 音声 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x1.5 | 46,761 | 12,902 | 33,859 | 529 | 45,797 | 4.29時間 |
| x2 | 31,346 | 7,290 | 24,056 | 61 | 30,684 | 3.22時間 |

主指標の窓の、正解の毎秒モーラ数の区間別の窓数（正解の平均は x1.5 で8.45、x2 で11.03）:

| 条件 | 6未満 | 6〜7 | 7〜8 | 8〜9 | 9〜10 | 10〜11 | 11〜12 | 12〜13 | 13〜14 | 14以上 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x1.5 | 13,584 | 3,348 | 3,165 | 3,466 | 3,644 | 3,822 | 4,031 | 3,819 | 2,866 | 4,052 |
| x2 | 4,994 | 2,173 | 2,109 | 2,139 | 2,026 | 1,986 | 1,878 | 1,879 | 1,812 | 9,688 |

## 3. 留保

- WSOLA は分析の窓を最大512標本（32ms）動かすので、速めた音声の中のモーラの時刻は t ÷ 速さ から局所的にずれうる。窓の端に掛かるモーラの按分にだけ影響する
- 速めた波形の最大振幅は1を超える（x1.5 で最大1.82・288標本、x2 で最大1.27・150標本）。float32 のまま保存しており、切り詰めはしていない
- 候補を2倍速でも窓を持つ音源（4.0秒以上）に限ったため、短いクリップは入らない

## 4. 遠隔機（ubuntu-desktop）で使う

- Mac から `scripts/sync_to_remote.sh --data` で送る（data/processed の下なので送られる）。送った後、遠隔機で `uv run python scripts/build_dev_fast.py --verify 20` を実行し、選んだ音源・窓の定義の一致と、先頭・末尾の20音源を作り直した波形のビット一致（`VERIFY_OK`）を確かめる。Mac では20/20でビット一致した
- 遠隔機で作り直す場合は、空の `data/processed/dev_fast/` に対して `uv run python scripts/build_dev_fast.py` を実行する（既存の dev_fast があれば上書きせずに止まる）

## 5. 評価の実行（測定の担当向け）

```
# 評価2の推論（mps または cuda）
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_fast_speech.py predict \
    --checkpoint runs/exp005/checkpoint_best.pt --model-key exp005 --device mps \
    --out-dir runs/exp005/fast_speech > runs/exp005/fast_speech_predict.log 2>&1
# 評価1（既存の dev_window の予測を再利用）と評価2の集計・metrics.csv への追記
uv run python scripts/eval_fast_speech.py summarize --model-key exp005 \
    --checkpoint runs/exp005/checkpoint_best.pt --model-config configs/exp005.yaml \
    --window-eval-dir runs/exp005/window_eval --out-dir runs/exp005/fast_speech \
    --experiment-id 013-fast-speech-exp005 --method-name "..." --append-metrics
```

- 評価1の予測: exp005 は `runs/exp005/window_eval`、exp007 は `runs/exp007_eval/window_eval`、exp010 は `runs/exp010_eval/window_eval`（`pred_clean.npz`、配列名はモデル名）
- 出力: `<out-dir>/fast_speech_summary.json`（主指標と全窓の、全体と区間別の値）と `fast_speech_tables.md`
- metrics.csv の行（主指標）: 評価1は split `dev_window_bin_6to7`〜`dev_window_bin_13to14`・`dev_window_bin_ge14`、評価2は `dev_fast_x1.5`・`dev_fast_x2`（全窓）と `dev_fast_x1.5_bin_<区間>`・`dev_fast_x2_bin_<区間>`。1モデルで29行。実験IDは `013-fast-speech-<モデル>` を推奨。偏り・出力の平均・10%点と90%点は列が無いので json と md にだけ出る
- `predict --max-sources N` は先頭 N 音源だけを通す動作確認（保存しない）
