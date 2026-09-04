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
import shlex
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from vs_mlx_serve import (  # noqa: E402
    QUESTIONS, SHORT, Server, install_term_handler, model_id, stream_once,
)


def _output_token_count(tokenizer, reply: str, n_chunks: int) -> int:
    """SSE chunk数ではなく、生成本文を同じtokenizerで数え直す。"""
    return len(tokenizer.encode(reply)) if reply else n_chunks


def main() -> int:
    install_term_handler()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="mlxturbo 側のモデル。--serve-bin を指定したときは"
                         " プロンプトを組むトークナイザとしてのみ使う")
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--serve-bin", default=None,
                    help="指定すると mlxturbo ではなく mlx-serve を単独で測る。"
                         "**手順は完全に同じにする** — 対戦ハーネス (A→B→B→A) と"
                         "単独測定で 35%% 食い違ったので、その切り分けに使う")
    ap.add_argument("--serve-model", default=None)
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--ctxs", default="0,4000,17000,50000")
    ap.add_argument(
        "--offset-tokens", type=int, default=0,
        help="長文prompt池の開始offset。fullの特定cellだけを同じ本文で再生するときに使う",
    )
    ap.add_argument(
        "--question-index-base", type=int, default=0,
        help="QUESTIONSの開始index。fullの特定cellだけを同じ質問で再生するときに使う",
    )
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2,
                    help="文脈ごとの繰り返し (中央値を取る)")
    ap.add_argument("--out", default="bench/results/self-snapshot.json")
    ap.add_argument("--warm-long", type=int, default=4000,
                    help="温め 2 段目 (専門家重みと n-gram 行のページイン) の"
                         " 窓トークン数。0 で無効")
    ap.add_argument("--server-log", default=None,
                    help="指定するとサーバーの stdout/stderr をこのファイルに追記する"
                         " (既定は捨てる)")
    ap.add_argument("--serve-log-level", default="debug",
                    help="mlx-serve の --log-level に渡す値")
    ap.add_argument("--serve-extra", default=None,
                    help="mlx-serve の argv 末尾に足す文字列 (shlex.split)")
    ap.add_argument("--turbo-extra", default=None,
                    help="mlxturbo の argv 末尾に足す文字列 (shlex.split)")
    ap.add_argument("--thinking", choices=("off", "on", "default"), default="off",
                    help="reasoning_effort を揃えて thinking の on/off を比較可能にする。"
                         " mlx-serve は qwen4_exp で thinking を既定 off、mlxturbo は"
                         " テンプレート既定で on なので、揃えないと比較にならない。"
                         " off→reasoning_effort=none、on→medium、default→送らない"
                         " (各サーバーの既定のまま)")
    args = ap.parse_args()

    thinking_extra = {
        "off": {"reasoning_effort": "none"},
        "on": {"reasoning_effort": "medium"},
        "default": None,
    }[args.thinking]

    from transformers import AutoTokenizer
    from _bench_text import long_prompts

    t_start = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] (+{time.time() - t_start:6.1f}s) {msg}",
              flush=True)

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.model))
    ctxs = [int(c) for c in args.ctxs.split(",")]
    # **文脈・繰り返しの全組で、互いに重ならない窓を切る。**同じ本文を
    # 2 回送ると 2 回目が接頭辞キャッシュに当たり、「冷 TTFT」が冷えた 1 回と
    # キャッシュ当たり 1 回の平均になる。**2026-09-02 に実際にそれをやった**
    # (17k で 31.7s のはずが 16.89s = (31.7+2)/2 と出た)。文脈をまたいでも
    # 池の先頭から切ると接頭辞関係になるので、累積 offset を持って
    # 1 本切るごとに窓幅ぶん進める。
    prompts: dict[int, list[str]] = {}
    prompt_meta: dict[int, list[dict]] = {}
    offset = args.offset_tokens
    for i, c in enumerate(ctxs):
        if c == 0:
            # 短文脈は本文が固定なので、末尾に通し番号を足して別物にする
            prompts[c] = [f"{SHORT} (#{r})" for r in range(args.reps)]
            prompt_meta[c] = [dict(offset=None, tokens=len(tok.encode(prompts[c][r])))
                              for r in range(args.reps)]
        else:
            qs = [QUESTIONS[(args.question_index_base + i * args.reps + r) % len(QUESTIONS)]
                  for r in range(args.reps)]
            win = max(c - 200, 16)
            base = offset
            prompts[c] = long_prompts(tok, c, qs, offset_tokens=base)
            prompt_meta[c] = [dict(offset=base + r * win,
                                   tokens=len(tok.encode(prompts[c][r])))
                              for r in range(args.reps)]
            offset = base + win * args.reps

    # 温め 2 段目: 全測定窓の後ろ (未使用領域) から窓を切る
    warm_long_prompt = None
    if args.warm_long > 0:
        warm_long_prompt = long_prompts(
            tok, args.warm_long + 200, ["(warmup)"], offset_tokens=offset)[0]

    if args.serve_bin:
        if not args.serve_model:
            print("--serve-bin を使うときは --serve-model も要る")
            return 1
        label = "mlx-serve"
        argv = [os.path.expanduser(args.serve_bin), "--serve",
                "--model", os.path.expanduser(args.serve_model),
                "--host", "127.0.0.1", "--port", str(args.port), "--mtp",
                "--log-level", args.serve_log_level]
        if args.serve_extra:
            argv += shlex.split(args.serve_extra)
    else:
        label = "mlxturbo"
        argv = [sys.executable, "-m", "mlxturbo.server",
                "--model", os.path.expanduser(args.model),
                "--host", "127.0.0.1", "--port", str(args.port)]
        if args.ngram:
            argv += ["--ngram", os.path.expanduser(args.ngram)]
        if args.mtp:
            argv += ["--mtp", os.path.expanduser(args.mtp)]
        if args.turbo_extra:
            argv += shlex.split(args.turbo_extra)

    rows = []
    print(f"[{label}] thinking={args.thinking} extra_body={thinking_extra}")
    print(f"[{label}] 生成 {args.tokens} トークン、文脈 {ctxs}、各 {args.reps} 回の中央値\n")
    log(f"{label} 起動開始")
    with Server(label, argv, args.port, log_path=args.server_log):
        log(f"{label} 起動完了")
        mid = model_id(args.port)
        # 温め 1 段目: カーネルの初回コンパイルを済ませる。**測定と別のプロンプトで。**
        stream_once(args.port, [{"role": "user", "content": SHORT}], 8, mid,
                    extra_body=thinking_extra)
        log("温め 1 (短) 完了")
        # 温め 2 段目: 専門家重みと n-gram 行のページイン。**測定に使わない窓で。**
        if warm_long_prompt is not None:
            stream_once(args.port, [{"role": "user", "content": warm_long_prompt}], 8, mid,
                        extra_body=thinking_extra)
            log(f"温め 2 (長 {args.warm_long} tok) 完了")
        for c in ctxs:
            colds, warms, decs, ntok, nchunk, decs_ch = [], [], [], [], [], []
            for r in range(args.reps):
                msgs = [{"role": "user", "content": prompts[c][r]}]
                t_cold, dec, n_chunks, reply = stream_once(
                    args.port, msgs, args.tokens, mid, extra_body=thinking_extra)
                # MTPや別engineは複数tokenを1つのSSE deltaへまとめられる。
                # chunk数で割るとdecodeを過小評価するため、同じtokenizerで数え直す。
                n = _output_token_count(tok, reply, n_chunks)
                # 追記ターン: 実クライアントは履歴をまるごと送り直す
                msgs2 = msgs + [{"role": "assistant", "content": reply},
                                {"role": "user", "content": "続けて。"}]
                t_warm, _, _, _ = stream_once(args.port, msgs2, 8, mid,
                                               extra_body=thinking_extra)
                colds.append(t_cold)
                warms.append(t_warm)
                decs.append((n - 1) / dec if dec > 0 and n > 1 else 0.0)
                decs_ch.append((n_chunks - 1) / dec
                               if dec > 0 and n_chunks > 1 else 0.0)
                ntok.append(n)
                nchunk.append(n_chunks)
                # 両エンジンが同種のテキストを出しているか後で見るための痕跡
                prompt_meta[c][r]["reply_head"] = reply[:160]
                prompt_meta[c][r]["reply_chars"] = len(reply)
                log(f"文脈 {c} rep {r} 完了 (冷 {t_cold:.2f}s 温 {t_warm:.2f}s "
                    f"{n} tok / {n_chunks} chunk)")
            row = dict(ctx=c, cold_ttft=statistics.median(colds),
                       warm_ttft=statistics.median(warms),
                       decode_tps=statistics.median(decs),
                       n_tokens=statistics.median(ntok),
                       decode_tps_chunks=statistics.median(decs_ch),
                       n_chunks=statistics.median(nchunk),
                       colds=colds, warms=warms, decs=decs, ntoks=ntok,
                       nchunks=nchunk, decs_chunks=decs_ch,
                       prompts=prompt_meta[c])
            rows.append(row)
            pt = prompt_meta[c][0]["tokens"]
            print(f"  文脈 {c:>6} ({pt:>6} tok)  冷 TTFT {row['cold_ttft']:7.2f}s  "
                  f"温 TTFT {row['warm_ttft']:6.2f}s  "
                  f"decode {row['decode_tps']:6.1f} tok/s  "
                  f"prefill {pt / row['cold_ttft']:7.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(engine=label, rows=rows, tokens=args.tokens,
                       reps=args.reps, argv=argv, port=args.port,
                       thinking=args.thinking), f,
                  ensure_ascii=False, indent=2)
    print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
