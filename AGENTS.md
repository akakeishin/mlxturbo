# AGENTS.md (Codex の入口)

このリポジトリは mlxturbo (Apple Silicon 向けの MLX 推論エンジン、投機デコード付き)。
2026-09-04 に Claude Code のセッションから Codex へ引き継いだ。読む順は次のとおり。

1. `CLAUDE.md` (リポジトリ直下)。**決まりは全部ここにある**: 計測の作法、触ると壊れるシーム、knob の既定値と棄却の記録、品質の受け入れ幅、代金ゼロの改善の扱い。名前が Claude 向けなだけで、Codex にもそのまま適用する。
2. `docs/HANDOFF-2026-09-04.md`。引き継ぎ時点の現在地、レーンの状態 (採った / 畳んだ / 未決)、次の一手の順、環境。
3. `docs/NEXT-SESSION-PROMPT.md` の末尾と `docs/research/SESSION-2026-09-02-CATCHUP.md` の末尾。時系列の記録 (数字の出典)。
4. `docs/research/NOTES-FROM-MEMORY-2026-09-04.md`。Claude 側の memory の写し (競合の位置づけ、ユーザーの決定、計測の罠 1〜20)。

## Claude 側の仕組みで、Codex では効かないもの

Claude 側では hook が機械的に止めていた。Codex ではこれらを決まりとして守る。

- 資格情報を含みうるファイル (`.env`、鍵、`secrets/`) を書かない。HF のトークンは huggingface_hub の既定の場所だけ。引数・ログ・環境変数に出さない。
- 直近のテストが失敗している状態で、テストのアサーションを減らす編集や skip の追加をしない。実装を直すか、テストが誤っている根拠を示す。
- 依頼していない改善を持ち込まない (コメントの書き換え、設定項目やエラー階層の追加など)。差分に依頼と紐づかない塊があれば削る。
- Markdown と txt は日本語で、機械翻訳調や定型の言い回しを避けて書く (記録は人が読み返すもの)。
- 「サブエージェントは Opus、判定と commit は親」の分業は Codex では成り立たない。**計測の判定 (既定に入れる / 畳む) は数字を表にしてユーザーに見せてから**にする。commit は依頼があったときだけ。

## GPU の使い方

- GPU を使うコマンドは全部 `tools/biglock.sh` で包む (`tools/biglock.sh .venv/bin/python ...`)。同時に 2 つのモデルを載せると片方が落ちる。
- 優先度は `BIGLOCK_PRIO=0/1/2` (0 が最優先)。札は `$TMPDIR/mlxturbo-biglock-prio/<pid>` にあり、中身を書き換えると順番が変わる。
- 1 プロセス内で交互に測る A/B の道具: Flash-Next は `tools/decode_ab.py`、qwen3_5 (27B) は `tools/decode_ab_generic.py`、常駐 worker は `tools/ab_daemon.py` (`tools/ab_submit.py` で投げる)。
- 計測中にダウンロードや別の GPU プロセスを並走させない。長い GPU ジョブは開始 10 分で進捗を見る。
- フルベンチ (`bench/self_snapshot.py` の full tier) と overnight は、ユーザーの指示があるまで走らせない。

## 記録の置き場

- 結果は `docs/research/SESSION-2026-09-02-CATCHUP.md` の末尾に節を足す (時刻は `date` で取る。推定で書かない)。
- 既定値を変えたら `CLAUDE.md` の knob 段落も直す。棄却した案は knob を残さず消し、記録だけ CATCHUP と `docs/BACKLOG.md` に残す。
- 未決の仕事は `docs/NEXT-SESSION-PROMPT.md` の末尾に「再開の 1 コマンド」付きで書く。チャットだけの申し送りは不可。
- 走行中の一覧は リポジトリ直下 `scratchpad/INFLIGHT-<date>.md` (untracked)。`bench/results/` は gitignore (JSON は手元だけ)。

## commit

- 1 commit 1 論点。メッセージは日本語で、数字 (文脈と %) と判定を 1 行目に入れる。
- `git add -p` でキーワード選別した hunk を commit しない (罠 20: 紛れた断片で prefill が 5 時間落ちた)。部分 stage の後は必ずその経路を実際に踏むテストを通す。
- push は `origin main` に直接 (ユーザー 2026-09-04「main を push しているならそのままでいい」)。
