# RELEASE — 公開時にやること

このリポジトリは現時点では非公開のローカル開発物。git remote は意図的に設定していない
(公開先は未決定)。実際に公開する際の手順をここに書いておく。

## 前提チェック

公開前に以下を確認する。

- [ ] `uv run pytest bench/test_server.py -q` が通る (2026-08-30 時点で 346 件)
- [ ] `uv run mlxturbo-serve --version` が `mlxturbo-serve 0.1.0` を表示する (改名時の
      置き残しは解消済み)
- [ ] `uv run mlxturbo --help` と `uv run mlxturbo-serve --help` が通る
- [ ] コードのコメントと docstring が英語化されている。ユーザー向けの文字列 (ログ、
      エラー、argparse の help) は日本語のままでよい
- [ ] `git grep -nE "/Users/|/Volumes/"` が `bench/results/` 以外で何も返さない
      (測定記録の中のパスは事実なので書き換えない)
- [ ] `pip install fastmlx` で本プロジェクトとは無関係な Prince Canuma (Blaizzy) 氏の既存
      パッケージが入ることを README 等で明記済みか再確認 (名前衝突の周知)
- [ ] LICENSE (Apache-2.0) と NOTICE (akakeishin, 2026／第三者帰属) が実態と合っているか確認
- [ ] README の実測値に「どのコマンドで再現できるか」が併記されているか確認

## 1. GitHub remote を追加する

```
git remote add origin git@github.com:<owner>/mlxturbo.git
git push -u origin main
```

`<owner>` と repo 名は公開時に決める。PyPI 名 (`mlxturbo`) と GitHub repo 名を揃えておくと
混乱が少ない。

## 2. GitHub 側の設定

- リポジトリの About に一言(「Apple Silicon 向け MLX 推論エンジン」等)と、対応表の要旨
  (flash_spec / spec / fallback の3経路)を書く
- Topics に `mlx`, `apple-silicon`, `llm-inference` 等を付ける
- `.github/workflows/test.yml` の CI が実際に GitHub Actions 上で動くか、push 後に確認する
  (`macos-14` ランナーで `mlx` が読み込めるかは実地でしか確認していない)

## 3. タグを打つ

```
git tag -a v0.1.0 -m "Initial public release"
git push origin v0.1.0
```

`pyproject.toml` の `version` とタグを一致させる。

## 4. PyPI へ公開する

`mlxturbo` という PyPI 名は事前に空きを確認済み (公開時点で再確認すること — 別の誰かが
先に登録している可能性はゼロではない)。

```
uv build
uv publish  # 初回は --token または ~/.pypirc の認証情報が必要
```

`uv publish` は `~/.pypirc` の `[pypi]` セクション、または `UV_PUBLISH_TOKEN` 環境変数を見る。
PyPI の Trusted Publishing (GitHub Actions からの OIDC 公開) を使う場合は、PyPI 側のプロジェクト
設定で GitHub リポジトリを signing publisher として登録してから、`uv publish` を CI 経由の
ワークフローに置き換える (このリポジトリには未実装)。

公開後:

```
pip install mlxturbo
mlxturbo-serve --version
```

で実際にインストールできることを確認する。

## 5. 公開後にやること

- READMEに実際のGitHub URL / PyPIバッジを足す
- Issue テンプレート・CONTRIBUTING は今回未整備。要望が出たら足す
- `docs/BACKLOG.md` の「サーバーの配布準備」節を見て、認証・公開範囲まわりで詰め切れていない
  ものが無いか再確認する (現状はローカル `127.0.0.1` 前提の設計)
