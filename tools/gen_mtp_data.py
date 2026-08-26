"""MTP ヘッド微調整用の自己生成データ作成 (D2 手順 2)。

実使用分布 (コード作成・コード編集・日本語技術文・日英混在) のプロンプトを
テンプレート合成し、モデル自身に temp 0.6 / top-p 0.95 で応答を生成させる。
編集タスクの素材はローカルの実コード (このリポジトリと .venv 内 mlx_lm) から
切り出す。出力はトークン id 入り JSONL で、学習側での再トークン化ずれを防ぐ。

途中終了しても JSONL は追記式なので、同じコマンドで再開できる
(--target-tokens 到達分は生成済みとして数える)。

使い方 (GPU 作業。長時間になるのでカーネル側セッションと排他を確認してから):
  uv run python tools/gen_mtp_data.py --target-tokens 5000000 \
      --out data/mtp_selfgen.jsonl
"""

import argparse
import json
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LANGS = ["Python", "JavaScript", "TypeScript", "Rust", "Go", "C++", "シェルスクリプト", "SQL"]

CODE_TASKS = [
    "ディレクトリ以下の全ファイルをSHA-256でハッシュ化して重複を検出するツール",
    "JSONLファイルを読み込んで指定キーで集計するスクリプト",
    "簡単なLRUキャッシュの実装",
    "並行ダウンローダ（同時数制限つき）",
    "CSVをパースして欠損値を補完する処理",
    "テキストからURLを抽出して有効性を確認するツール",
    "再帰的にシンボリックリンクを解決してループを検出する関数",
    "トークンバケット方式のレートリミッタ",
    "ログファイルを追跡して特定パターンを通知する監視スクリプト",
    "2つのディレクトリツリーの差分を報告するツール",
    "設定ファイル(TOML)を読んでバリデーションするローダ",
    "文字列の編集距離を計算する関数とそのテスト",
    "優先度つきタスクキューの実装",
    "画像ファイルのEXIFを読んで撮影日ごとに整理するスクリプト",
    "簡単なマークダウンパーサ（見出しとリストだけ対応）",
    "HTTPサーバのアクセスログから上位IPを集計する処理",
    "バイナリファイルの16進ダンプ表示ツール",
    "ファイル変更を監視して自動でテストを走らせるランナー",
    "依存関係グラフをトポロジカルソートする関数",
    "固定幅フォーマットのレコードをパースするライブラリ",
    "リトライとバックオフつきのAPIクライアントラッパ",
    "プロセスのメモリ使用量を定期記録するプロファイラ",
    "簡単なテンプレートエンジン（変数展開と条件分岐）",
    "ソケット通信のエコーサーバとクライアント",
    "行指向のdiffアルゴリズムの実装",
]

CODE_CONSTRAINTS = [
    "標準ライブラリだけで書いてください。",
    "型ヒントをつけてください。",
    "エラー処理を丁寧に入れてください。",
    "テストコードも添えてください。",
    "コメントは日本語で書いてください。",
    "",
]

EDIT_OPS = [
    "型ヒントとエラーハンドリングを追加してください",
    "この関数をリファクタリングして読みやすくしてください",
    "docstringを日本語で追加してください",
    "潜在的なバグがないかレビューして、あれば修正版を示してください",
    "この処理をより効率的に書き直してください",
    "ユニットテストを書いてください",
    "この関数を非同期(async)版に書き換えてください",
    "エッジケースへの対応を追加してください",
]

PROSE_TOPICS = [
    "分散システムにおける結果整合性と強整合性の違い",
    "GCのある言語とない言語のメモリ管理の考え方の違い",
    "TCPとUDPの使い分け",
    "データベースのインデックスが効く条件と効かない条件",
    "ハッシュテーブルの衝突解決の方式",
    "非同期I/Oとスレッドの使い分け",
    "静的型付けの利点と限界",
    "キャッシュの無効化が難しい理由",
    "ソフトウェアテストにおける単体テストと統合テストの境界",
    "コンパイラの最適化がプログラマの想定を裏切る例",
    "浮動小数点数の比較で起きる問題と対策",
    "文字コードの歴史とUTF-8が主流になった理由",
    "RESTとRPCの設計思想の違い",
    "並行処理におけるデッドロックの典型パターン",
    "量子化がニューラルネットワーク推論に与える影響",
    "GPUとCPUのメモリ帯域の違いが性能に効く理由",
    "投機的実行の考え方とその応用",
    "ロックフリーなデータ構造の基本的な考え方",
]

MIXED_PROMPTS = [
    "Explain the difference between `mx.eval` and lazy evaluation in MLX. 日本語で答えてください。",
    "What does `async_eval` do in a Metal-backed array framework? 具体例つきで日本語で。",
    "Summarize the key idea of speculative decoding in English, then 日本語で補足してください。",
    "このエラーの原因を英語のドキュメントを引用しながら日本語で説明してください: RuntimeError: attempting to eval an array during function transformations",
    "Write a git commit message in English for a change that fixes an off-by-one error in cache rollback, そして日本語で変更内容を説明してください。",
    "Translate this docstring to English and improve it: 「この関数は入力を正規化して返す。副作用はない。」",
]


def collect_snippets(rng, n):
    """ローカルの実コードから編集タスク用の切り出しを作る。"""
    roots = [REPO / "fastmlx", REPO / "bench", REPO / "tools"]
    venv_mlx = list((REPO / ".venv").glob("lib/python*/site-packages/mlx_lm"))
    roots.extend(venv_mlx)
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*.py") if p.stat().st_size > 500)
    snippets = []
    attempts = 0
    while len(snippets) < n and attempts < n * 10:
        attempts += 1
        path = rng.choice(files)
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        if len(lines) < 20:
            continue
        span = rng.randint(15, 60)
        start = rng.randint(0, max(0, len(lines) - span))
        chunk = "\n".join(lines[start : start + span]).strip()
        if len(chunk) < 200:
            continue
        snippets.append(chunk)
    return snippets


def build_prompts(rng, n):
    """カテゴリ比率: code 40% / edit 30% / prose 20% / mixed 10%。"""
    n_edit = int(n * 0.3)
    snippets = collect_snippets(rng, n_edit)
    prompts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.4:
            task = rng.choice(CODE_TASKS)
            lang = rng.choice(LANGS)
            c = rng.choice(CODE_CONSTRAINTS)
            text = f"{lang}で{task}を書いてください。{c}".strip()
            cat = "code"
        elif r < 0.7 and snippets:
            op = rng.choice(EDIT_OPS)
            sn = rng.choice(snippets)
            text = f"次のコードに対して、{op}。\n```python\n{sn}\n```"
            cat = "edit"
        elif r < 0.9:
            topic = rng.choice(PROSE_TOPICS)
            style = rng.choice(
                ["詳しく", "具体例を挙げながら", "初学者向けに", "簡潔に"]
            )
            text = f"{topic}について、{style}説明してください。"
            cat = "prose"
        else:
            text = rng.choice(MIXED_PROMPTS)
            cat = "mixed"
        prompts.append((cat, text))
    return prompts


def count_existing(out_path):
    done, tokens = 0, 0
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done += 1
                tokens += len(rec["gen_tokens"])
    return done, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit"
    )
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--out", default="data/mtp_selfgen.jsonl")
    args = parser.parse_args()

    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, gen_tokens = count_existing(out_path)
    print(f"resume: {done} samples / {gen_tokens} gen tokens already present")

    rng = random.Random(args.seed)
    # プール上限は目標トークン数から粗く見積もる (平均 700 tok/sample 想定)。
    pool = build_prompts(rng, max(2000, args.target_tokens // 700 * 2))
    pool = pool[done:]

    model, tok = load(args.model)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    idx = done
    t0 = time.time()
    with open(out_path, "a") as f:
        for i in range(0, len(pool), args.batch_size):
            if gen_tokens >= args.target_tokens:
                break
            batch = pool[i : i + args.batch_size]
            prompt_ids = [
                tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                )
                for _, text in batch
            ]
            res = batch_generate(
                model,
                tok,
                prompt_ids,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            for (cat, text), pids, out_text in zip(
                batch, prompt_ids, res.texts
            ):
                out_ids = tok.encode(out_text, add_special_tokens=False)
                rec = {
                    "id": idx,
                    "category": cat,
                    "prompt": text,
                    "prompt_tokens": pids,
                    "gen_tokens": out_ids,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                idx += 1
                gen_tokens += len(out_ids)
            f.flush()
            rate = gen_tokens / max(time.time() - t0, 1e-9)
            print(
                f"{idx} samples, {gen_tokens}/{args.target_tokens} tokens"
                f" ({rate:.0f} tok/s cumulative)",
                flush=True,
            )
    print(f"done: {idx} samples, {gen_tokens} gen tokens -> {out_path}")


if __name__ == "__main__":
    main()
