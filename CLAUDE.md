このリポジトリは日本語音声から話速（毎秒モーラ数）を推定するCNNを開発する。

## 規則
- 実装はPythonのみ。依存の追加はpyproject.tomlに記述しuvで同期する
- docs/spec.mdを仕様の正とする。仕様に疑問があれば実装せずdocs/questions.mdに記録して停止する
- 仕様を自分の判断で変更しない。ユーザーが仕様変更を明示した場合は、その指示に従ってdocs/spec.md・実装・テストを整合させる
- 1セッションで扱うのは指定されたタスク1件のみ。先の段階に進まない。この規則は実装・測定・分析・判断の各サブエージェントにのみ適用する。統括Agentには適用しない
- 統括Agentはdocs/PLAN.mdの停止条件（人間の作業がある段階、必要なデータが未配置、仕様変更の承認が必要、判断サブエージェントの結論への承認が必要）に該当するまで、段階をまたいで実行を継続する
- テストセットでの評価を行う前にconfigs/splits/test.jsonで分割を固定する。条件の選択には検証セットのみを使う。テストセットでの評価は明示的に指示された場合のみ行う
- 学習実行前にgit statusを確認し、未コミットの変更があれば先にコミットする
- 学習はバックグラウンドで実行しログをruns/以下にファイル出力する。ログ全文をセッションに読み込まず、最終行と指標のみ確認する
- 評価が終わったらresults/metrics.csvに1行追記する。列は実験ID、日時、コミットハッシュ、設定ファイルのパス、全指標
- セッション終了時にdocs/progress.mdへ5行以内で追記する
- 音声を聴く必要がある判断は行わない。該当する場合はdocs/questions.mdに記録して停止する
- データセットはユーザーが手動で取得・配置する。利用前に必要なファイルの存在と構成を確認する。配置先はdata/DATASETS.mdを参照する。取得中のファイルやアーカイブを無断で変更・削除しない
- data/以下のデータセット本体・アーカイブ（バックアップ含む）は、このCLAUDE.mdに削除の指示がある場合を除き、ユーザーの明示的な指示なしに削除しない

## 実行環境
- 計算機は2台。mac（作業の本拠地、デバイスはmps）とubuntu-desktop（Macから`ssh ubuntu-desktop`で接続する。RTX 3060、デバイスはcuda）
- 学習デバイスはmpsまたはcuda。指定したデバイスが使えない場合は開始前に停止し、cpuに落とさない
- float64は使わない。torch.compileは使わない。混合精度（float16・bfloat16での学習）は使わない
- cudaではTF32を無効にする（torch.backends.cuda.matmul.allow_tf32とtorch.backends.cudnn.allow_tf32をFalse）。mpsのfloat32と精度の条件をそろえるため
- MPS未対応演算でCPUフォールバックが起きたら発生箇所をログに記録して報告する
- ubuntu-desktopでは学習と推論の実行だけを行う。git、docs、results、metrics.csvの正本はMacのリポジトリだけに置く
- metrics.csv、progress.md、questions.md、docs/への書き込みとgitの操作（コミット）はMac上の統括Agentだけが行う。ubuntu-desktopではコミットしない。例外として、scripts/sync_to_remote.shが遠隔機で行う`git checkout --force --detach`だけを許す
- ubuntu-desktopへの送り出しはscripts/sync_to_remote.sh、runs/の結果の取り戻しはscripts/fetch_from_remote.shで行う。どちらの計算機で評価しても、結果はMacのresults/とmetrics.csvに書く
- 推論時間の測定はMacだけで行う（これまでの値と比べるため）
- 互いに依存しない学習は2台で同時に実行してよい。1台で同時に実行する学習は1つまで
- 評価（推論）は学習中でない計算機で実行してよい
- 実験ごとの計算機と状態はdocs/compute.mdで管理する

## gitの運用
- mainには動作確認済みの状態のみ置く
- 実験ごとにブランチを切る（exp/003-window-1sの形式）
- 1コミット1論点。設定ファイルの変更とコードの変更を混ぜない
- Publicリポジトリで運用しているため、機密情報をPushしない
