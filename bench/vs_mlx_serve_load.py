"""mlx-serve と mlxturbo を、**同時 N 要求下**で比べる (`vs_mlx_serve.py` の負荷版)。

## なぜ要る

mlx-serve の手書き Metal カーネルはソースを読んだ限り `B*S==1` でしか発火しない
(`do_sort = B*S > 1 || total_inds >= 64 || has_expert_bias`)。GDN の高速カーネルも
`seq_len>=64` 条件で decode 幅を除外する。つまり単一リクエストは相手の最良条件で、
`vs_mlx_serve.py` はずっとそこで測ってきた。**同時要求が来たときに非対称があるか**
をここで測る。

## `vs_mlx_serve.py` から流用したもの

サーバーの起動・停止 (`Server`)、`/v1/models` 待ち (`wait_ready`)、SSE のクライアント側
計時 (`stream_once` -- TTFT = 最初の content delta まで、decode = 以降のチャンク数 /
経過秒)、A→B→B→A の順序バイアス相殺、短いプロンプト定数 (`SHORT`)。**そのまま import
して使う (コピーしない)。**

## 揃えるもの / 揃わないもの

揃える: プロンプト集合 (同じ順序で両エンジンに割り当てる)、生成トークン数、
temperature=0、サーバーは 1 つずつ順に起動 (91GB を 2 つ同時には載せられない)。
揃わない: 重みが違う (`vs_mlx_serve.py` の注記と同じ)。

## プロンプトの多様性

全要求が同じプロンプトだと prefix キャッシュが効いて実態とずれるので、
`tools/_bench_text.py` の文書プールから ctx を複数振って重ならない窓を切り、
短い質問・中文書・長文書・非常に長い文書が混ざったプールを作る
(`build_prompt_pool`)。両エンジンには同じプール・同じ割り当て順序を使う
(セッションごとにカーソルを 0 から振り直す)ので、たまたま片方が短いプロンプトを
多く引く、という偏りは起きない。

## 測る指標

- **スループット**: そのラウンドの全要求の合計トークン (chunk 数の近似) / 壁時計
  (投入開始からその N 本が全部終わるまで)。
- **要求ごとの TTFT の分布**: 中央値と p95 (平均だけにしない -- 依頼のとおり)。
- **要求ごとの decode tok/s の分布**: 同上。

N ごとに「ウォームアップ (8 トークン、捨てる) → 計測 (`--tokens`、記録)」を 1 ラウンドとし、
`--rounds` で繰り返して標本を増やせる (既定 1 -- N 本の要求そのものが 1 ラウンドの標本になる)。

## N=1 の整合性チェック

N=1 は `run_batch` が 1 要求だけを `stream_once` で流すのと同じで、`vs_mlx_serve.py` が
呼んでいる関数と全く同じコードパス。**N=1 の decode tok/s 中央値が `vs_mlx_serve.py` の
単発計測 (近い文脈長・同じ `--tokens`) と大きくずれるなら、計時かサーバー設定がずれている
合図**であり、N を増やした結果を信じる前にそこを疑うこと。ただしスループット
(tok/s, 壁時計ベース) は TTFT を含むため、decode tok/s とは別物 (TTFT が短ければ近づく)。

## 同時実行機構についての注意 (解釈時に必ず読むこと)

`mlxturbo/batch_spec.py` (バッチ x MTP 投機) は**サーバーに配線されていない**
(`docs/BACKLOG.md` の「サーバー配線・admission・スケジューラは書いていない」)。
サーバーが実際に持つ継続バッチングは `mlxturbo/batch.py` (`--max-batch N`) だけで、
これは **`FallbackRunner` 限定** (`spec`/`flash_spec` ランナーは対象外、
`docs/OPERATIONS.md` 「制約」節)。推奨構成 (Flash-Next + MTP) は `flash_spec` で
解決されるので、**`--turbo-max-batch` を指定しても既定構成の同時要求はこのバッチングの
恩恵を受けない**。つまり `--turbo-max-batch` 無しで測る場合、mlxturbo 側の N>1 は
「サーバーの直列ロック (`--max-queue`、既定 8) 越しに順番待ちするだけ」になる可能性が高い。
それ自体が測るべき現実 (バッチ x 投機が実戦投入されていない、という事実) であり、
数字が悪くても実装のバグではない。

## 実行

91GB のモデルを 2 つ同時には載せられないので、サーバーは 1 つずつ順に起動する
(`Server` が担う)。**GPU を使うので `tools/biglock.sh` 経由で実行すること。**

    tools/biglock.sh .venv/bin/python bench/vs_mlx_serve_load.py \\
        --serve-bin ~/dev/mlx-serve/zig-out/bin/mlx-serve \\
        --serve-model ~/models/ddalcu-flashnext-serve-4bit \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep \\
        --ns 1,2,4,8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "bench"))

from vs_mlx_serve import (  # noqa: E402
    QUESTIONS, SHORT, Server, model_id, stream_once,
)

# vs_mlx_serve.py の QUESTIONS (2本) だけでは N=8 で使い回しが多すぎるので、
# 窓を増やすための追加の設問。窓の中身 (文書プールの非重複スライス) は
# 質問文の使い回しとは無関係に別物になる (tools/_bench_text.py の long_prompts
# 参照: 質問文は末尾に付くだけで、本文は index ごとの非重複スライス)。
LOAD_QUESTIONS = list(QUESTIONS) + [
    "上の文書の設計判断に対する反論を 1 つ挙げてください。",
    "上の文書を読んでいない人向けに 3 行で要約してください。",
]
# 中文書 / 長文書 / 非常に長い文書。SHORT (ctx=0 相当) と合わせて実利用の
# 分布 (短い質問から長い文脈まで) を模す。
LOAD_CTXS = (2000, 8000, 24000)


def build_prompt_pool(tok, need: int, long_prompts) -> list[str]:
    """長さの違う実文プロンプトを need 本以上そろえる。

    全要求が同じプロンプトだと prefix キャッシュが効いて実態とずれるので、
    ctx を複数振り、各 ctx で重ならない窓を複数切って混ぜる。素材が
    足りなければ `long_prompts` 自体が ValueError で止まる (繰り返しで
    埋めて受理率を偽装しない、というリポジトリの方針を継承する)。
    """
    per_ctx = -(-need // len(LOAD_CTXS))  # ceil
    questions = [LOAD_QUESTIONS[i % len(LOAD_QUESTIONS)] for i in range(per_ctx)]
    pool = [SHORT]
    for c in LOAD_CTXS:
        pool += long_prompts(tok, c, questions)
    assert len(pool) >= need, f"プール不足: {len(pool)} < {need}"
    return pool


def run_batch(port: int, prompts: list[str], n_tokens: int, model: str = "x"):
    """prompts を同時に流し、(各要求の (ttft, decode_s, n_chunks, reply) のリスト, 壁時計秒) を返す。

    ThreadPoolExecutor で全要求を同時に投げる (SSE 読み取りはブロッキング I/O
    なので GIL 越しでも並行に進む -- `bench/batch_throughput.py` と同じ考え方)。
    壁時計はこの N 本が全部そろうまでの経過時間で、実クライアントが体感する
    スループットの分母そのもの。
    """
    msgs_list = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [ex.submit(stream_once, port, m, n_tokens, model)
                for m in msgs_list]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    return results, wall


def summarize_round(results: list[tuple], wall: float) -> dict:
    """1 ラウンド (N 本) の結果を、スループット・TTFT 分布・decode tok/s 分布に集約する。"""
    ttfts = [r[0] for r in results if r[0] == r[0]]  # NaN (content 未到達) を除く
    decode_tps = []
    for ttft, dec, n, _reply in results:
        decode_tps.append((n - 1) / dec if dec > 0 and n > 1 else 0.0)
    total_tokens = sum(r[2] for r in results)
    throughput = total_tokens / wall if wall > 0 else 0.0
    return dict(
        throughput_tok_s=throughput,
        ttft_median_s=float(np.median(ttfts)) if ttfts else float("nan"),
        ttft_p95_s=float(np.quantile(ttfts, 0.95)) if ttfts else float("nan"),
        decode_tps_median=float(np.median(decode_tps)) if decode_tps else float("nan"),
        decode_tps_p95=float(np.quantile(decode_tps, 0.95)) if decode_tps else float("nan"),
        total_tokens=total_tokens,
        wall_s=wall,
        n_ok=len(ttfts),
        n_total=len(results),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-bin", required=True)
    ap.add_argument("--serve-model", required=True)
    ap.add_argument("--serve-port", type=int, default=11234)
    ap.add_argument("--model", required=True, help="mlxturbo 側のモデル")
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument(
        "--turbo-max-batch", type=int, default=None,
        help="mlxturbo 側に --max-batch を渡す (継続バッチング)。"
             "FallbackRunner 限定で spec/flash_spec には効かない"
             " (モジュール docstring の「同時実行機構についての注意」参照)。"
             "省略時は渡さない (既定 = 直列)")
    ap.add_argument("--ns", default="1,2,4,8", help="同時要求数を振る値 (カンマ区切り)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="各 N を何ラウンド繰り返すか (増やすと分布の標本が増える)")
    ap.add_argument("--tokens", type=int, default=128, help="要求ごとの生成トークン数")
    ap.add_argument("--out", default="bench/results/vs-mlx-serve-load.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.model))
    from _bench_text import long_prompts

    ns = [int(x) for x in args.ns.split(",")]
    if any(n < 1 for n in ns):
        raise ValueError("--ns はすべて 1 以上であること")
    need = 2 * args.rounds * sum(ns)  # ウォームアップぶんも含めて必要な本数
    pool = build_prompt_pool(tok, need, long_prompts)

    serve_argv = [os.path.expanduser(args.serve_bin), "--serve",
                  "--model", os.path.expanduser(args.serve_model),
                  "--host", "127.0.0.1", "--port", str(args.serve_port), "--mtp"]
    turbo_argv = [sys.executable, "-m", "mlxturbo.server",
                  "--model", os.path.expanduser(args.model),
                  "--host", "127.0.0.1", "--port", str(args.port)]
    if args.ngram:
        turbo_argv += ["--ngram", os.path.expanduser(args.ngram)]
    if args.mtp:
        turbo_argv += ["--mtp", os.path.expanduser(args.mtp)]
    if args.turbo_max_batch:
        turbo_argv += ["--max-batch", str(args.turbo_max_batch)]

    sides = {
        "mlx-serve": (Server, serve_argv, args.serve_port),
        "mlxturbo": (Server, turbo_argv, args.port),
    }

    print("判定基準はモジュール docstring のとおり (測る前に宣言済み)。")
    print(f"生成 {args.tokens} トークン、N={ns}、rounds={args.rounds}、順序は A B B A。")
    print("mlxturbo の flash_spec/spec ランナーは --max-batch の対象外 "
          "(docstring の「同時実行機構についての注意」参照)。\n")

    rows = []
    for name in ("mlx-serve", "mlxturbo", "mlxturbo", "mlx-serve"):
        cls, argv, port = sides[name]
        with cls(name, argv, port):
            cursor = 0
            # 起動直後の 1 本は必ず冷えているので捨てる (vs_mlx_serve.py と同じ約束)
            mid = model_id(port)
            stream_once(port, [{"role": "user", "content": SHORT}], 8, mid)
            for n in ns:
                for round_idx in range(args.rounds):
                    warm_prompts = pool[cursor:cursor + n]
                    cursor += n
                    run_batch(port, warm_prompts, 8, mid)  # ウォームアップ、捨てる

                    meas_prompts = pool[cursor:cursor + n]
                    cursor += n
                    results, wall = run_batch(port, meas_prompts, args.tokens, mid)
                    summary = summarize_round(results, wall)
                    row = dict(engine=name, n=n, round=round_idx, **summary)
                    rows.append(row)
                    tag = "  (単発と比較する基準)" if n == 1 else ""
                    print(f"  {name:10s} N={n:2d} round={round_idx}  "
                          f"スループット {summary['throughput_tok_s']:7.1f} tok/s (壁時計)  "
                          f"TTFT中央 {summary['ttft_median_s']:6.2f}s p95 {summary['ttft_p95_s']:6.2f}s  "
                          f"decode中央 {summary['decode_tps_median']:6.1f} tok/s"
                          f" p95 {summary['decode_tps_p95']:6.1f}"
                          f" ({summary['n_ok']}/{summary['n_total']} 応答){tag}", flush=True)

    print("\n=== まとめ (エンジンごとに 2 セッション x rounds の平均) ===")
    hdr = ("N", "スループット serve", "スループット turbo",
           "TTFT中央 serve", "TTFT中央 turbo",
           "decode中央 serve", "decode中央 turbo")
    print("".join(f"{h:>18s}" for h in hdr))
    for n in ns:
        def avg(engine, key):
            v = [r[key] for r in rows
                 if r["engine"] == engine and r["n"] == n
                 and r[key] == r[key]]  # NaN を除く
            return float(np.mean(v)) if v else float("nan")
        vals = (avg("mlx-serve", "throughput_tok_s"), avg("mlxturbo", "throughput_tok_s"),
                avg("mlx-serve", "ttft_median_s"), avg("mlxturbo", "ttft_median_s"),
                avg("mlx-serve", "decode_tps_median"), avg("mlxturbo", "decode_tps_median"))
        print(f"{n:>18d}" + "".join(f"{v:18.2f}" for v in vals))

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
