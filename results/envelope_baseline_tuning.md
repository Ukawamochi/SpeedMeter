# 信号処理ベースラインの調整（docs/PLAN.md 第3段階 3-2）

## 調整の条件

- 実装: `src/spkrate/baselines/envelope.py`
- 実行: `scripts/run_envelope_baseline.py tune`
- 対象: 検証セット（dev）から固定シード `20260921` で無作為抽出した **2000件**
- テストセット（`configs/splits/test.json`）は一切使っていない
- 指標は毎秒モーラ数の平均絶対誤差（MAE）。小さいほど良い
- 比較の下敷き: 常に中央値 4.762 と答えた場合の MAE = **1.5246**

## 探索の方法

包絡の窓長25ミリ秒・移動10ミリ秒（docs/spec.md の特徴量と同じ刻み）は固定し、残りを段階ごとに順に決める座標降下法で探索した。全ての組み合わせを試すと候補数が数千になり、部分集合の復号を何度も繰り返すことになるためである。各段階では前段階までに選んだ値を固定し、その段階のパラメータだけを動かす。

「ピークを数える」と「正方向のゼロ交差を数える」は、どちらも最後まで別々に調整してから比べた。既定値のまま一度に比べると、片方に不利な初期値のまま優劣が決まってしまうためである。

換算係数 `mora_per_syllable` は各候補ごとに、その候補の音節数に対して平均絶対誤差を最小にする値を厳密に求めている（誤差は換算係数について区分線形かつ凸なので、重み付き中央値が最小点になる）。したがって下表の MAE はいずれも「その設定で最良の換算係数を使ったときの値」である。

## 数え方ごとの最終結果

| 数え方 | 換算係数 | 部分集合での MAE |
| --- | --- | --- |
| peak ←採用 | 1.5278 | 0.8675 |
| zero_cross | 1.8519 | 0.9284 |

## 探索した範囲と各候補の平均絶対誤差

### peak / 1巡目 / 帯域

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| band_low_hz=2.0, band_high_hz=10.0 ←採用 | 1.2759 | 1.1772 |
| band_low_hz=3.0, band_high_hz=10.0 | 1.2 | 1.1836 |
| band_low_hz=3.0, band_high_hz=8.0 | 1.3333 | 1.1878 |
| band_low_hz=2.0, band_high_hz=8.0 | 1.4286 | 1.1882 |
| band_low_hz=2.5, band_high_hz=10.0 | 1.2333 | 1.1936 |
| band_low_hz=2.0, band_high_hz=9.0 | 1.3333 | 1.1942 |
| band_low_hz=3.0, band_high_hz=9.0 | 1.2593 | 1.1990 |
| band_low_hz=2.5, band_high_hz=8.0 | 1.3824 | 1.2012 |
| band_low_hz=2.5, band_high_hz=9.0 | 1.3 | 1.2065 |

### peak / 1巡目 / 突出量と平滑化

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| peak_prominence=1.0, smoothing_sec=0.0 ←採用 | 1.2759 | 1.1772 |
| peak_prominence=0.5, smoothing_sec=0.0 | 1.2105 | 1.1775 |
| peak_prominence=0.25, smoothing_sec=0.0 | 1.1739 | 1.1823 |
| peak_prominence=0.5, smoothing_sec=0.03 | 1.25 | 1.1862 |
| peak_prominence=0.25, smoothing_sec=0.03 | 1.2 | 1.1871 |
| peak_prominence=1.0, smoothing_sec=0.02 | 1.2857 | 1.1872 |
| peak_prominence=0.1, smoothing_sec=0.0 | 1.1429 | 1.1876 |
| peak_prominence=0.5, smoothing_sec=0.02 | 1.2222 | 1.1880 |
| peak_prominence=0.05, smoothing_sec=0.0 | 1.1395 | 1.1889 |
| peak_prominence=0.5, smoothing_sec=0.05 | 1.3077 | 1.1902 |
| peak_prominence=0.25, smoothing_sec=0.02 | 1.1833 | 1.1915 |
| peak_prominence=0.1, smoothing_sec=0.03 | 1.18 | 1.1919 |
| peak_prominence=1.5, smoothing_sec=0.0 | 1.3333 | 1.1921 |
| peak_prominence=0.25, smoothing_sec=0.05 | 1.2727 | 1.1928 |
| peak_prominence=0.1, smoothing_sec=0.02 | 1.1667 | 1.1942 |
| peak_prominence=0.05, smoothing_sec=0.03 | 1.1667 | 1.1946 |
| peak_prominence=0.05, smoothing_sec=0.02 | 1.1481 | 1.1950 |
| peak_prominence=2.0, smoothing_sec=0.0 | 1.375 | 1.1959 |
| peak_prominence=4.0, smoothing_sec=0.02 | 1.55 | 1.1960 |
| peak_prominence=0.05, smoothing_sec=0.05 | 1.2222 | 1.1961 |
| peak_prominence=1.0, smoothing_sec=0.03 | 1.3077 | 1.1962 |
| peak_prominence=3.0, smoothing_sec=0.05 | 1.6 | 1.1965 |
| peak_prominence=2.0, smoothing_sec=0.05 | 1.5 | 1.1967 |
| peak_prominence=4.0, smoothing_sec=0.05 | 1.6842 | 1.1974 |
| peak_prominence=0.1, smoothing_sec=0.05 | 1.24 | 1.1975 |
| peak_prominence=1.5, smoothing_sec=0.02 | 1.3333 | 1.1976 |
| peak_prominence=1.5, smoothing_sec=0.03 | 1.3571 | 1.1976 |
| peak_prominence=4.0, smoothing_sec=0.0 | 1.5385 | 1.1979 |
| peak_prominence=1.0, smoothing_sec=0.05 | 1.375 | 1.1995 |
| peak_prominence=4.0, smoothing_sec=0.03 | 1.5833 | 1.1998 |
| peak_prominence=1.5, smoothing_sec=0.05 | 1.4286 | 1.2002 |
| peak_prominence=2.0, smoothing_sec=0.03 | 1.4 | 1.2010 |
| peak_prominence=3.0, smoothing_sec=0.03 | 1.5 | 1.2017 |
| peak_prominence=2.0, smoothing_sec=0.02 | 1.381 | 1.2017 |
| peak_prominence=3.0, smoothing_sec=0.0 | 1.45 | 1.2020 |
| peak_prominence=3.0, smoothing_sec=0.02 | 1.4667 | 1.2021 |
| peak_prominence=6.0, smoothing_sec=0.0 | 1.6923 | 1.2056 |
| peak_prominence=6.0, smoothing_sec=0.02 | 1.7143 | 1.2068 |
| peak_prominence=6.0, smoothing_sec=0.03 | 1.75 | 1.2134 |
| peak_prominence=6.0, smoothing_sec=0.05 | 1.8462 | 1.2157 |

### peak / 1巡目 / 無音の閾値

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| silence_floor_db=-20.0 ←採用 | 1.5714 | 0.8830 |
| silence_floor_db=-25.0 | 1.5 | 0.9099 |
| silence_floor_db=-15.0 | 1.68 | 0.9425 |
| silence_floor_db=-30.0 | 1.4255 | 0.9784 |
| silence_floor_db=-35.0 | 1.3478 | 1.0738 |
| silence_floor_db=-40.0 | 1.2759 | 1.1772 |
| silence_floor_db=-10.0 | 2.0 | 1.2314 |
| silence_floor_db=-45.0 | 1.1935 | 1.2680 |
| silence_floor_db=-50.0 | 1.1154 | 1.3430 |
| silence_floor_db=-5.0 | 3.375 | 2.1416 |

### peak / 1巡目 / 最小間隔

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| min_peak_distance_sec=0.08 ←採用 | 1.5714 | 0.8829 |
| min_peak_distance_sec=0.03 | 1.5714 | 0.8830 |
| min_peak_distance_sec=0.04 | 1.5714 | 0.8830 |
| min_peak_distance_sec=0.05 | 1.5714 | 0.8830 |
| min_peak_distance_sec=0.06 | 1.5714 | 0.8830 |
| min_peak_distance_sec=0.12 | 1.7692 | 0.8898 |
| min_peak_distance_sec=0.1 | 1.625 | 0.8961 |

### peak / 2巡目 / 帯域

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| band_low_hz=3.0, band_high_hz=10.0 ←採用 | 1.5 | 0.8771 |
| band_low_hz=2.0, band_high_hz=10.0 | 1.5714 | 0.8829 |
| band_low_hz=2.5, band_high_hz=10.0 | 1.5455 | 0.8900 |
| band_low_hz=3.0, band_high_hz=9.0 | 1.5714 | 0.8986 |
| band_low_hz=2.5, band_high_hz=8.0 | 1.7143 | 0.9014 |
| band_low_hz=2.5, band_high_hz=9.0 | 1.6154 | 0.9016 |
| band_low_hz=3.0, band_high_hz=8.0 | 1.6667 | 0.9021 |
| band_low_hz=2.0, band_high_hz=9.0 | 1.6667 | 0.9045 |
| band_low_hz=2.0, band_high_hz=8.0 | 1.7609 | 0.9086 |

### peak / 2巡目 / 突出量と平滑化

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| peak_prominence=0.5, smoothing_sec=0.05 ←採用 | 1.5366 | 0.8707 |
| peak_prominence=0.25, smoothing_sec=0.05 | 1.5 | 0.8710 |
| peak_prominence=0.1, smoothing_sec=0.05 | 1.4737 | 0.8733 |
| peak_prominence=0.5, smoothing_sec=0.02 | 1.4545 | 0.8736 |
| peak_prominence=1.0, smoothing_sec=0.02 | 1.5 | 0.8745 |
| peak_prominence=0.5, smoothing_sec=0.03 | 1.4737 | 0.8766 |
| peak_prominence=1.0, smoothing_sec=0.0 | 1.5 | 0.8771 |
| peak_prominence=0.05, smoothing_sec=0.0 | 1.3889 | 0.8782 |
| peak_prominence=0.5, smoothing_sec=0.0 | 1.4444 | 0.8783 |
| peak_prominence=0.25, smoothing_sec=0.02 | 1.4286 | 0.8784 |
| peak_prominence=0.05, smoothing_sec=0.05 | 1.463 | 0.8791 |
| peak_prominence=1.0, smoothing_sec=0.03 | 1.5278 | 0.8793 |
| peak_prominence=0.1, smoothing_sec=0.02 | 1.4 | 0.8794 |
| peak_prominence=0.1, smoothing_sec=0.0 | 1.4 | 0.8800 |
| peak_prominence=0.05, smoothing_sec=0.02 | 1.4 | 0.8800 |
| peak_prominence=0.25, smoothing_sec=0.0 | 1.4167 | 0.8819 |
| peak_prominence=0.25, smoothing_sec=0.03 | 1.4444 | 0.8828 |
| peak_prominence=1.0, smoothing_sec=0.05 | 1.6 | 0.8829 |
| peak_prominence=1.5, smoothing_sec=0.0 | 1.55 | 0.8857 |
| peak_prominence=0.1, smoothing_sec=0.03 | 1.4186 | 0.8861 |
| peak_prominence=0.05, smoothing_sec=0.03 | 1.4074 | 0.8862 |
| peak_prominence=1.5, smoothing_sec=0.02 | 1.5667 | 0.8892 |
| peak_prominence=1.5, smoothing_sec=0.03 | 1.5897 | 0.8895 |
| peak_prominence=1.5, smoothing_sec=0.05 | 1.6522 | 0.8954 |
| peak_prominence=2.0, smoothing_sec=0.03 | 1.625 | 0.8970 |
| peak_prominence=2.0, smoothing_sec=0.02 | 1.6061 | 0.8971 |
| peak_prominence=2.0, smoothing_sec=0.0 | 1.6 | 0.9009 |
| peak_prominence=2.0, smoothing_sec=0.05 | 1.6944 | 0.9038 |
| peak_prominence=3.0, smoothing_sec=0.05 | 1.7931 | 0.9134 |
| peak_prominence=3.0, smoothing_sec=0.02 | 1.6957 | 0.9171 |
| peak_prominence=3.0, smoothing_sec=0.0 | 1.6818 | 0.9182 |
| peak_prominence=3.0, smoothing_sec=0.03 | 1.7143 | 0.9183 |
| peak_prominence=4.0, smoothing_sec=0.02 | 1.7778 | 0.9378 |
| peak_prominence=4.0, smoothing_sec=0.03 | 1.8 | 0.9389 |
| peak_prominence=4.0, smoothing_sec=0.0 | 1.7647 | 0.9408 |
| peak_prominence=4.0, smoothing_sec=0.05 | 1.8667 | 0.9428 |
| peak_prominence=6.0, smoothing_sec=0.03 | 2.0 | 0.9971 |
| peak_prominence=6.0, smoothing_sec=0.02 | 1.9524 | 1.0030 |
| peak_prominence=6.0, smoothing_sec=0.0 | 1.9231 | 1.0057 |
| peak_prominence=6.0, smoothing_sec=0.05 | 2.0294 | 1.0111 |

### peak / 2巡目 / 無音の閾値

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| silence_floor_db=-20.0 ←採用 | 1.5366 | 0.8707 |
| silence_floor_db=-25.0 | 1.4545 | 0.9054 |
| silence_floor_db=-15.0 | 1.6667 | 0.9256 |
| silence_floor_db=-30.0 | 1.3846 | 0.9807 |
| silence_floor_db=-35.0 | 1.3158 | 1.0768 |
| silence_floor_db=-40.0 | 1.2353 | 1.2067 |
| silence_floor_db=-10.0 | 2.0 | 1.2152 |
| silence_floor_db=-45.0 | 1.1429 | 1.3268 |
| silence_floor_db=-50.0 | 1.0476 | 1.4108 |
| silence_floor_db=-5.0 | 3.2857 | 2.1251 |

### peak / 2巡目 / 最小間隔

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| min_peak_distance_sec=0.06 ←採用 | 1.5278 | 0.8675 |
| min_peak_distance_sec=0.03 | 1.5278 | 0.8676 |
| min_peak_distance_sec=0.04 | 1.5278 | 0.8676 |
| min_peak_distance_sec=0.05 | 1.5278 | 0.8676 |
| min_peak_distance_sec=0.08 | 1.5366 | 0.8707 |
| min_peak_distance_sec=0.12 | 1.6923 | 0.8774 |
| min_peak_distance_sec=0.1 | 1.5833 | 0.8793 |

### zero_cross / 1巡目 / 帯域

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| band_low_hz=2.5, band_high_hz=8.0 ←採用 | 1.6429 | 1.0903 |
| band_low_hz=2.0, band_high_hz=10.0 | 1.6667 | 1.1028 |
| band_low_hz=2.5, band_high_hz=9.0 | 1.5714 | 1.1058 |
| band_low_hz=2.0, band_high_hz=8.0 | 1.7778 | 1.1059 |
| band_low_hz=3.0, band_high_hz=8.0 | 1.5185 | 1.1097 |
| band_low_hz=2.5, band_high_hz=10.0 | 1.5263 | 1.1121 |
| band_low_hz=2.0, band_high_hz=9.0 | 1.7083 | 1.1140 |
| band_low_hz=3.0, band_high_hz=9.0 | 1.4545 | 1.1256 |
| band_low_hz=3.0, band_high_hz=10.0 | 1.4118 | 1.1365 |

### zero_cross / 1巡目 / 平滑化

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| smoothing_sec=0.0 ←採用 | 1.6429 | 1.0903 |
| smoothing_sec=0.02 | 1.6364 | 1.0975 |
| smoothing_sec=0.03 | 1.6486 | 1.0993 |
| smoothing_sec=0.05 | 1.6667 | 1.1152 |

### zero_cross / 1巡目 / 無音の閾値

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| silence_floor_db=-25.0 ←採用 | 1.9677 | 0.9418 |
| silence_floor_db=-30.0 | 1.8182 | 0.9423 |
| silence_floor_db=-35.0 | 1.7273 | 0.9954 |
| silence_floor_db=-20.0 | 2.1579 | 1.0027 |
| silence_floor_db=-40.0 | 1.6429 | 1.0903 |
| silence_floor_db=-15.0 | 2.5 | 1.2019 |
| silence_floor_db=-45.0 | 1.5417 | 1.2157 |
| silence_floor_db=-50.0 | 1.4286 | 1.3120 |
| silence_floor_db=-10.0 | 3.35 | 1.7243 |
| silence_floor_db=-5.0 | 7.0 | 2.9392 |

### zero_cross / 1巡目 / 最小間隔

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| min_peak_distance_sec=0.1 ←採用 | 2.0 | 0.9369 |
| min_peak_distance_sec=0.03 | 1.9677 | 0.9406 |
| min_peak_distance_sec=0.04 | 1.9677 | 0.9406 |
| min_peak_distance_sec=0.05 | 1.9677 | 0.9410 |
| min_peak_distance_sec=0.06 | 1.9677 | 0.9418 |
| min_peak_distance_sec=0.08 | 1.9783 | 0.9440 |
| min_peak_distance_sec=0.12 | 2.0 | 0.9463 |

### zero_cross / 2巡目 / 帯域

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| band_low_hz=3.0, band_high_hz=10.0 ←採用 | 1.75 | 0.9369 |
| band_low_hz=2.5, band_high_hz=8.0 | 2.0 | 0.9369 |
| band_low_hz=3.0, band_high_hz=8.0 | 1.8333 | 0.9431 |
| band_low_hz=2.5, band_high_hz=10.0 | 1.8571 | 0.9432 |
| band_low_hz=2.5, band_high_hz=9.0 | 1.9062 | 0.9477 |
| band_low_hz=3.0, band_high_hz=9.0 | 1.7778 | 0.9494 |
| band_low_hz=2.0, band_high_hz=10.0 | 2.0 | 0.9673 |
| band_low_hz=2.0, band_high_hz=9.0 | 2.0312 | 0.9844 |
| band_low_hz=2.0, band_high_hz=8.0 | 2.1351 | 0.9921 |

### zero_cross / 2巡目 / 平滑化

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| smoothing_sec=0.02 ←採用 | 1.7576 | 0.9347 |
| smoothing_sec=0.0 | 1.75 | 0.9369 |
| smoothing_sec=0.03 | 1.7778 | 0.9392 |
| smoothing_sec=0.05 | 1.8333 | 0.9398 |

### zero_cross / 2巡目 / 無音の閾値

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| silence_floor_db=-25.0 ←採用 | 1.7576 | 0.9347 |
| silence_floor_db=-30.0 | 1.6364 | 0.9429 |
| silence_floor_db=-20.0 | 1.9444 | 1.0108 |
| silence_floor_db=-35.0 | 1.5455 | 1.0189 |
| silence_floor_db=-40.0 | 1.4524 | 1.1384 |
| silence_floor_db=-15.0 | 2.2857 | 1.2574 |
| silence_floor_db=-45.0 | 1.3542 | 1.2800 |
| silence_floor_db=-50.0 | 1.24 | 1.3786 |
| silence_floor_db=-10.0 | 3.0833 | 1.7777 |
| silence_floor_db=-5.0 | 6.5 | 2.9734 |

### zero_cross / 2巡目 / 最小間隔

| 候補 | 換算係数 | MAE |
| --- | --- | --- |
| min_peak_distance_sec=0.12 ←採用 | 1.8519 | 0.9284 |
| min_peak_distance_sec=0.1 | 1.7576 | 0.9347 |
| min_peak_distance_sec=0.06 | 1.6957 | 0.9362 |
| min_peak_distance_sec=0.08 | 1.7097 | 0.9368 |
| min_peak_distance_sec=0.03 | 1.6923 | 0.9369 |
| min_peak_distance_sec=0.04 | 1.6923 | 0.9369 |
| min_peak_distance_sec=0.05 | 1.6923 | 0.9372 |

## 選んだ値

| パラメータ | 値 |
| --- | --- |
| `frame_length_sec` | 0.025 |
| `hop_sec` | 0.01 |
| `floor_db` | -60.0 |
| `smoothing_sec` | 0.05 |
| `band_low_hz` | 3.0 |
| `band_high_hz` | 10.0 |
| `filter_order` | 3 |
| `count_method` | peak |
| `peak_prominence` | 0.5 |
| `min_peak_distance_sec` | 0.06 |
| `silence_floor_db` | -20.0 |
| `mora_per_syllable` | 1.5278 |

- 部分集合での MAE = **0.8675**（常に中央値と答える場合は 1.5246）
- 音節が1つも検出されなかったクリップの割合 = 0.00%
- 設定ファイル: `configs/baselines/envelope.yaml`

## 参考: 指定より広い帯域（不採用）

帯域の候補は docs/PLAN.md 3-2 の「2Hzから10Hz」の内側に限って選んだ。その外側も測っておいたが、改善はわずかで、手法の定義を指定から外してまで得る価値は無いと判断して採用しなかった。

| 帯域 | 換算係数 | MAE | 採用値との差 |
| --- | --- | --- | --- |
| band_low_hz=2.0, band_high_hz=16.0 | 1.5 | 0.8607 | -0.0067 |
| band_low_hz=2.0, band_high_hz=14.0 | 1.5 | 0.8612 | -0.0063 |
| band_low_hz=2.0, band_high_hz=12.0 | 1.5312 | 0.8674 | -0.0000 |
| band_low_hz=1.25, band_high_hz=12.0 | 1.5714 | 0.8755 | +0.0080 |
| band_low_hz=1.0, band_high_hz=16.0 | 1.5333 | 0.8864 | +0.0189 |
| band_low_hz=1.5, band_high_hz=10.0 | 1.6364 | 0.8901 | +0.0226 |

## 最終評価の条件

- 実行: `scripts/run_envelope_baseline.py eval`（実験ID `003-envelope-baseline`）
- 対象: **検証セット（dev）の全件 29,518 クリップ**（37.10時間）。部分集合ではない
- 所要時間: 0.36分（音声の復号のみ8プロセスで並列化し、推定器は主プロセスで逐次実行した）
- 結果は `results/metrics.csv` の1行として記録した

| 指標 | 値 |
| --- | --- |
| 毎秒モーラ数の平均絶対誤差 | 0.8961 |
| 同（4未満、9,686件） | 0.8399 |
| 同（4以上6未満、11,135件） | 0.7284 |
| 同（6以上8未満、6,682件） | 0.8737 |
| 同（8以上、2,015件） | 2.1677 |
| 相関係数 | 0.7670 |
| 1推論あたりの処理時間 | 0.416 ミリ秒 |

調整に使った2000件での平均絶対誤差は 0.8675 で、dev 全件では 0.8961 だった。
部分集合で選んだ値が dev 全体でも同程度に働いている。

話速帯別に見ると、毎秒8モーラ以上の帯だけ誤差が 2.1677 と大きく、他の帯の
2倍以上ある。音量の山の数は速く話しても頭打ちになりやすく（モーラが1つの山に
まとまる）、固定の換算係数では速い話速を過小評価するためである。
