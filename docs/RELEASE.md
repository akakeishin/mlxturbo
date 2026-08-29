# RELEASE — 公開時にやること

このリポジトリは現時点では非公開のローカル開発物。git remote は意図的に設定していない
(公開先は未決定)。実際に公開する際の手順をここに書いておく。

## 前提チェック

公開前に以下を確認する。

- [ ] `uv run pytest bench/test_server.py -q` が通る (改名作業が完了し、`mlxturbo/server.py` /
      `mlxturbo/runner.py` / `mlxturbo/spec_flash.py` / `bench/test_server.py` 側の
      `fastmlx` 文字列置換が完了していること — 詳細は改名時の報告を参照)
- [ ] `uv run mlxturbo-serve --version` が `mlxturbo-serve <version>` を正しく表示する
      (現状は `_FASTMLX_VERSION`/`_fastmlx_version()` が `fastmlx` という古いパッケージ名で
      `importlib.metadata` を引いているため、`mlxturbo` へ直すまでは `0.0.0-unknown` 表示になる)
- [ ] `pip install fastmlx` で本プロジェクトとは無関係な Prince Canuma (Blaizzy) 氏の既存
      パッケージが入ることを README 等で明記済みか再確認 (名前衝突の周知)
- [ ] LICENSE (MIT, akakeishin, 2026) の年・氏名が実態と合っているか確認
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
