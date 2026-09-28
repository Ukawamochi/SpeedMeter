# 実験ごとの計算機と状態

どの実験をどの計算機（mac・ubuntu-desktop）で実行しているかを管理する表。
統括Agentは学習を起動したとき（状態=実行中、開始日時を記入）と完了したとき（状態=完了、終了日時を記入）に、この表を更新する。
計算機の構成は [docs/decisions/010-compute-environment.md](decisions/010-compute-environment.md) を参照する。
日時は JST。開始日時は runs/<実験ID>/log.txt の先頭行、終了日時は最終行の時刻とする。

| 実験ID | 計算機 | デバイス | 状態 | 開始 | 終了 | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| exp005 | mac | mps | 完了 | 2026-09-26 14:34 | 2026-09-27 02:31 | configs/exp005.yaml（方式B）。上限30エポックで終了、最良はエポック26（検証 MAE 0.4737） |
| exp000_method_b_smoke_cuda | ubuntu-desktop | cuda | 完了 | 2026-09-27 00:08 | 2026-09-27 00:08 | 完走確認（1エポック）。010 の4.4節 |
| exp000_method_a_selection_smoke_cuda | ubuntu-desktop | cuda | 完了 | 2026-09-27 00:08 | 2026-09-27 00:08 | 完走確認（1エポック）。010 の4.4節 |
| exp006 | ubuntu-desktop | cuda | 中断 | 2026-09-27 00:16 | 2026-09-27 00:40 | configs/exp006.yaml（方式Aの対照）。exp/011-method-a-control の f70044e。計算機の移設のため、エポック3の途中で Ctrl-C で止めた。checkpoint_last.pt はエポック2の終わり（00:39:52）で、そこから再開する |
| exp006_resume | ubuntu-desktop | cuda | 完了 | 2026-09-27 00:50 | 2026-09-27 06:13 | configs/exp006_resume.yaml（exp/011-method-a-control の 35b8773）。exp006 のエポック2の checkpoint_last.pt からエポック3以降を続ける。最良値は測り直し。exp006 の結果は runs/exp006 と runs/exp006_resume の両方で見る。上限30エポックで終了、最良はエポック27（dev MAE 0.4343）。runs/ は Mac に戻した |
| exp007 | ubuntu-desktop | cuda | 完了 | 2026-09-27 14:27 | 2026-09-27 22:42 | configs/exp007.yaml（exp/012-seed-variance の f949ef5）。exp005 と同じ設定で seed だけ 20260929（exp005 は 20260921）。学習ごとのばらつきの測定（指示書 2026-09-27 A-1）。tmux のセッション exp007。上限30エポックで終了、最良はエポック27（検証 MAE 0.4779）。1エポック約980秒。runs/ は Mac に戻した |
| exp008 | ubuntu-desktop | cuda | 完了 | 2026-09-28 00:14 | 2026-09-28 08:29 | configs/exp008.yaml（exp/013-poisson-loss の 3914f67）。exp005 の損失だけを mse → poisson（条件の探索、精度の目的。指示書 2026-09-27 A-3、011 の exp008）。tmux のセッション exp008。前に完走確認 exp000_poisson_smoke_cuda（1ec6475 の設定、00:12〜00:13、267バッチで損失が有限のまま 74.9 → −7.4 に下がった）を行った。上限30エポックで終了、最良はエポック30（検証 MAE 0.4851）。1エポック約980〜995秒。runs/ は Mac に戻した |
| exp009 | mac | mps | 実行中 | 2026-09-28 00:14 | | configs/exp009.yaml（exp/014-freq-ch-half の 9e9de29）。exp005 から周波数段のチャネル数だけ [16, 32, 32] に変えた軽さの条件（指示書 2026-09-27 A-3、011 の exp009）。パラメータ数 334,305。worktree SpeedMeter-wt-c から nohup で起動し、runs_dir を元の作業ツリーの runs の絶対パスにして runs/exp009 に出力 |
| exp010 | ubuntu-desktop | cuda | 実行中 | 2026-09-28 09:19 | | configs/exp010.yaml（exp/015-stretch-log-uniform の 54490e8）。exp005 から時間伸縮の伸縮率の分布だけを一様 → 対数一様（範囲 0.7〜1.5・確率 0.5 は同じ。条件の探索、精度の目的。指示書 2026-09-27 A-3、011 の exp010）。tmux のセッション exp010。log.txt に「分布=log_uniform」の行があることを確かめた |
