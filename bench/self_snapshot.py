#!/usr/bin/env python3
"""いまの mlxturbo が単体で何を出せるかを 1 枚にする (対戦なし)。

`bench/vs_mlx_serve.py` は 2 つのエンジンを立てて比べる道具で、相手が要る。
**こちらは自分だけ。**「いまの実力」を知りたいときに使う。

測るもの (文脈ごとに):

- **冷 TTFT**: そのプロンプトを初めて見たときの最初のトークンまで
- **温 TTFT**: 前のやり取り + 新しい発言をまるごと送り直したとき (実クライアントの
  2 ターン目)。接頭辞の再利用が効いているかがここに出る
- **decode**: 最初のトークン以降のチャンク数 / 経過秒

作法 (CLAUDE.md):

- **温めは測定と別のプロンプトで打つ。**同じものを投げると接頭辞キャッシュに
  当たって「冷」が冷えない (2026-09-02 に対戦ハーネスで踏んだ)
- **思考 (`reasoning_content`) も数える。**うちのサーバーは既定で思考を出すので、
  本文だけ数えると長文脈が 0 トークンになる (同じく 2026-09-02 に踏んだ)
- サーバーは 1 回だけ起動する (91GB のロードを繰り返さない)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from vs_mlx_serve import (  # noqa: E402
    QUESTIONS, SHORT, Server, install_term_handler, model_id, stream_once,
)


def main() -> int:
    install_term_handler()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--ctxs", default="0,4000,17000,50000")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2,
                    help="文脈ごとの繰り返し (中央値を取る)")
    ap.add_argument("--out", default="bench/results/self-snapshot.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from _bench_text import long_prompts

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.model))
    ctxs = [int(c) for c in args.ctxs.split(",")]
    # **文脈ごとに、かつ繰り返しごとに別のプロンプトを使う。**同じ本文を
    # 2 回送ると 2 回目が接頭辞キャッシュに当たり、「冷 TTFT」が冷えた 1 回と
    # キャッシュ当たり 1 回の平均になる。**2026-09-02 に実際にそれをやった**
    # (17k で 31.7s のはずが 16.89s = (31.7+2)/2 と出た)。
    # `long_prompts` は問いごとに違う窓を切るので、問いを変えれば本文も変わる。
    prompts: dict[int, list[str]] = {}
    for i, c in enumerate(ctxs):
        if c == 0:
            # 短文脈は本文が固定なので、末尾に通し番号を足して別物にする
            prompts[c] = [f"{SHORT} (#{r})" for r in range(args.reps)]
        else:
            qs = [QUESTIONS[(i * args.reps + r) % len(QUESTIONS)]
                  for r in range(args.reps)]
            prompts[c] = [long_prompts(tok, c, [q])[0] for q in qs]
            if len({p[:200] for p in prompts[c]}) < args.reps:
                print(f"警告: 文脈 {c} で {args.reps} 本の別プロンプトを作れなかった"
                      f" (問いが {len(QUESTIONS)} 本しかない)。--reps を下げること",
                      flush=True)

    argv = [sys.executable, "-m", "mlxturbo.server",
            "--model", os.path.expanduser(args.model),
            "--host", "127.0.0.1", "--port", str(args.port)]
    if args.ngram:
        argv += ["--ngram", os.path.expanduser(args.ngram)]
    if args.mtp:
        argv += ["--mtp", os.path.expanduser(args.mtp)]

    rows = []
    print(f"生成 {args.tokens} トークン、文脈 {ctxs}、各 {args.reps} 回の中央値\n")
    with Server("mlxturbo", argv, args.port):
        mid = model_id(args.port)
        # 温め: カーネルの初回コンパイルを済ませる。**測定と別のプロンプトで。**
        stream_once(args.port, [{"role": "user", "content": SHORT}], 8, mid)
        for c in ctxs:
            colds, warms, decs, ntok = [], [], [], []
            for r in range(args.reps):
                msgs = [{"role": "user", "content": prompts[c][r]}]
                t_cold, dec, n, reply = stream_once(args.port, msgs, args.tokens, mid)
                # 追記ターン: 実クライアントは履歴をまるごと送り直す
                msgs2 = msgs + [{"role": "assistant", "content": reply},
                                {"role": "user", "content": "続けて。"}]
                t_warm, _, _, _ = stream_once(args.port, msgs2, 8, mid)
                colds.append(t_cold)
                warms.append(t_warm)
                decs.append((n - 1) / dec if dec > 0 and n > 1 else 0.0)
                ntok.append(n)
            row = dict(ctx=c, cold_ttft=statistics.median(colds),
                       warm_ttft=statistics.median(warms),
                       decode_tps=statistics.median(decs),
                       n_tokens=statistics.median(ntok))
            rows.append(row)
            pt = len(tok.encode(prompts[c][0]))
            print(f"  文脈 {c:>6} ({pt:>6} tok)  冷 TTFT {row['cold_ttft']:7.2f}s  "
                  f"温 TTFT {row['warm_ttft']:6.2f}s  "
                  f"decode {row['decode_tps']:6.1f} tok/s  "
                  f"prefill {pt / row['cold_ttft']:7.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows, tokens=args.tokens, reps=args.reps), f,
                  ensure_ascii=False, indent=2)
    print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
