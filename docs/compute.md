# 実験ごとの計算機と状態

どの実験をどの計算機（mac・ubuntu-desktop）で実行しているかを管理する表。
統括Agentは学習を起動したとき（状態=実行中、開始日時を記入）と完了したとき（状態=完了、終了日時を記入）に、この表を更新する。
計算機の構成は [docs/decisions/010-compute-environment.md](decisions/010-compute-environment.md) を参照する。
日時は JST。開始日時は runs/<実験ID>/log.txt の先頭行、終了日時は最終行の時刻とする。

| 実験ID | 計算機 | デバイス | 状態 | 開始 | 終了 | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| exp005 | mac | mps | 実行中 | 2026-09-26 14:34 | — | configs/exp005.yaml（方式B） |
| exp000_method_b_smoke_cuda | ubuntu-desktop | cuda | 完了 | 2026-09-27 00:08 | 2026-09-27 00:08 | 完走確認（1エポック）。010 の4.4節 |
| exp000_method_a_selection_smoke_cuda | ubuntu-desktop | cuda | 完了 | 2026-09-27 00:08 | 2026-09-27 00:08 | 完走確認（1エポック）。010 の4.4節 |
