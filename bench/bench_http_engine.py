#!/usr/bin/env python3
"""任意の OpenAI 互換サーバーを 1 つ立てて、`bench/self_snapshot.py` と
同じ定義で 冷 TTFT / 温 TTFT / decode tok/s を測る。

`bench/self_snapshot.py` は argv を直書きしている (mlxturbo と mlx-serve の
2 つだけ)。27B レーンでは同じ重みを 5 つのエンジン (mlxturbo / mlx-lm /
oMLX / MTPLX / mlx-serve) で回すので、**起動 argv を呼び手から渡せる**形が要る。
計測本体は `bench/vs_mlx_serve.py` の `stream_once` をそのまま使うので、
ここに計時のロジックは無い (定義がずれない)。

    tools/biglock.sh .venv/bin/python bench/bench_http_engine.py \\
        --engine mlx-lm --port 8151 \\
        --tokenizer ~/models/qwen38-27b-4bit \\
        --argv '.venv/bin/python -m mlx_lm.server --model ~/models/qwen38-27b-4bit
                --port 8151' \\
        --ctxs 0,4000 --tokens 64 --reps 1 \\
        --out bench/results/smoke-27b-mlx-lm-0904.json

出力 JSON の行は `self_snapshot.py` と同じ形 (`ctx` / `cold_ttft` /
`warm_ttft` / `decode_tps` / `n_tokens` / `colds` / `warms` / `decs` /
`ntoks` / `prompts`) にしてあるので、後で同じ表に並べられる。

**`n_tokens` は SSE のチャンク数ではなく、本文を測定側のトークナイザで
数え直した値。**`self_snapshot.py` は 1 チャンク = 1 トークンを前提に
チャンクを数えているが、oMLX は 1 チャンクに複数トークンを詰めて流す
(64 トークンの応答が 16 チャンク、2026-09-04 実測)。そのまま割ると decode が
1/4 に見える。5 エンジンとも vocab が同じなので数え直しが成り立つ。
チャンク基準の値は `decode_tps_chunks` / `n_chunks` に残してある
(1 チャンク = 1 トークンのエンジンでは両者が一致する — 突き合わせに使う)。

## thinking の切り方

エンジンごとに流儀が違うので `--thinking-how` で選び、**選んだものを JSON に
記録する** (後から「どの経路で off にしたか」が分かるように)。

- `reasoning_effort`: OpenAI 標準の `reasoning_effort: none` を本体に足す
  (mlxturbo / mlx-serve が読む)
- `chat_template_kwargs`: `{"chat_template_kwargs": {"enable_thinking": false}}`
  を本体に足す (mlx_lm.server 系)
- `prompt`: ユーザー本文の末尾に `/no_think` を付ける (テンプレートが
  `enable_thinking` を読まないエンジン向けの逃げ道)
- `none`: 何もしない (エンジンの既定のまま)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from vs_mlx_serve import (  # noqa: E402
    QUESTIONS, SHORT, Server, install_term_handler, model_id, stream_once,
)


class EngineServer(Server):
    """`Server` に「先に死んだら待たない」を足しただけのもの。

    素の `Server` は `/v1/models` が返るまで 900 秒待つので、起動に失敗した
    エンジン (引数違い、パックを読めない) で 15 分持っていかれる。5 エンジンを
    順に立ち上げる用途では致命的なので、プロセスの生存も見る。
    """

    def __init__(self, name, argv, port, log_path=None, timeout=900.0):
        super().__init__(name, argv, port, log_path=log_path)
        self.timeout = timeout

    def __enter__(self):
        import urllib.request

        if self.log_path:
            self._logf = open(self.log_path, "ab")
            out = err = self._logf
        else:
            out = err = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            self.argv, stdout=out, stderr=err, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            rc = self.proc.poll()
            if rc is not None:
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"{self.name} が起動前に終了した (rc={rc})。ログ: {self.log_path}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        return self
            except Exception:
                time.sleep(3)
        self.__exit__(None, None, None)
        raise RuntimeError(f"{self.name} が {self.timeout:.0f}s で起動しなかった")


def main() -> int:
    install_term_handler()
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="JSON に書くエンジン名")
    ap.add_argument("--argv", required=True,
                    help="サーバーの起動コマンド (1 つの文字列、shlex で割る)。"
                         " `~` は展開する")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="プロンプトを組むトークナイザ (重みのディレクトリ)")
    ap.add_argument("--ctxs", default="0,4000")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm-long", type=int, default=0,
                    help="温め 2 段目 (専門家重みのページイン) の窓。0 で無効")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--model-id", default=None,
                    help="/v1/models の名乗りを使わず、この id を送る")
    ap.add_argument("--thinking-how",
                    choices=("reasoning_effort", "chat_template_kwargs",
                             "prompt", "none"),
                    default="reasoning_effort",
                    help="thinking を off にする経路 (モジュール docstring 参照)")
    ap.add_argument("--extra-body", default=None,
                    help="リクエスト本体に足す JSON オブジェクト (文字列)。"
                         " --thinking-how の分に上書きで混ぜる。投機のスイッチが"
                         " リクエスト側にあるエンジン向け (mlx-serve の"
                         " `{\"enable_drafter\": true}` など。MoE の target では"
                         " drafter の既定が off なので、これを送らないと"
                         " ドラフタは動かない)")
    args = ap.parse_args()

    extra_body = None
    prompt_suffix = ""
    if args.thinking_how == "reasoning_effort":
        extra_body = {"reasoning_effort": "none"}
    elif args.thinking_how == "chat_template_kwargs":
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif args.thinking_how == "prompt":
        prompt_suffix = " /no_think"
    if args.extra_body:
        extra_body = dict(extra_body or {})
        extra_body.update(json.loads(args.extra_body))

    from transformers import AutoTokenizer
    from _bench_text import long_prompts

    t_start = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] (+{time.time() - t_start:6.1f}s) {msg}",
              flush=True)

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer))
    ctxs = [int(c) for c in args.ctxs.split(",")]

    # 窓は self_snapshot.py と同じ切り方 (互いに重ならない、累積 offset)。
    # 同じ本文を 2 回送ると 2 回目が接頭辞キャッシュに当たって「冷」が冷えない。
    prompts: dict[int, list[str]] = {}
    prompt_meta: dict[int, list[dict]] = {}
    offset = 0
    for i, c in enumerate(ctxs):
        if c == 0:
            prompts[c] = [f"{SHORT} (#{r}){prompt_suffix}" for r in range(args.reps)]
            prompt_meta[c] = [dict(offset=None, tokens=len(tok.encode(prompts[c][r])))
                              for r in range(args.reps)]
        else:
            qs = [QUESTIONS[(i * args.reps + r) % len(QUESTIONS)]
                  for r in range(args.reps)]
            win = max(c - 200, 16)
            base = offset
            body = long_prompts(tok, c, qs, offset_tokens=base)
            prompts[c] = [b + prompt_suffix for b in body]
            prompt_meta[c] = [dict(offset=base + r * win,
                                   tokens=len(tok.encode(prompts[c][r])))
                              for r in range(args.reps)]
            offset = base + win * args.reps

    warm_long_prompt = None
    if args.warm_long > 0:
        warm_long_prompt = long_prompts(
            tok, args.warm_long + 200, ["(warmup)"], offset_tokens=offset)[0]

    argv = [os.path.expanduser(a) for a in shlex.split(args.argv)]

    rows = []
    print(f"[{args.engine}] argv: {' '.join(argv)}")
    print(f"[{args.engine}] thinking-how={args.thinking_how} extra_body={extra_body}"
          f" prompt_suffix={prompt_suffix!r}")
    print(f"[{args.engine}] 生成 {args.tokens} トークン、文脈 {ctxs}、"
          f"各 {args.reps} 回の中央値\n")
    log(f"{args.engine} 起動開始")
    t_boot = time.time()
    with EngineServer(args.engine, argv, args.port, log_path=args.server_log,
                      timeout=args.startup_timeout):
        boot_s = time.time() - t_boot
        log(f"{args.engine} 起動完了 ({boot_s:.1f}s)")
        mid = args.model_id or model_id(args.port)
        log(f"model id: {mid}")
        # 温め 1 段目: カーネルの初回コンパイル。**測定と別のプロンプトで。**
        stream_once(args.port, [{"role": "user", "content": SHORT + prompt_suffix}],
                    8, mid, extra_body=extra_body)
        log("温め 1 (短) 完了")
        if warm_long_prompt is not None:
            stream_once(args.port,
                        [{"role": "user", "content": warm_long_prompt + prompt_suffix}],
                        8, mid, extra_body=extra_body)
            log(f"温め 2 (長 {args.warm_long} tok) 完了")
        for c in ctxs:
            colds, warms, decs, ntok, nchunk, decs_ch = [], [], [], [], [], []
            for r in range(args.reps):
                msgs = [{"role": "user", "content": prompts[c][r]}]
                t_cold, dec, n_chunks, reply = stream_once(
                    args.port, msgs, args.tokens, mid, extra_body=extra_body)
                # **チャンク数をトークン数として数えないこと。**oMLX は
                # 1 チャンクに複数トークンを詰めて流す (64 トークンの応答が
                # 16 チャンク、2026-09-04 に実測)。チャンクで割ると decode が
                # 1/4 に見える。5 エンジンとも同じ vocab なので、本文を
                # 測定側のトークナイザで数え直すのが engine 非依存で正しい。
                n = len(tok.encode(reply)) if reply else n_chunks
                # 追記ターン: 実クライアントは履歴をまるごと送り直す
                msgs2 = msgs + [{"role": "assistant", "content": reply},
                                {"role": "user", "content": "続けて。" + prompt_suffix}]
                t_warm, _, _, _ = stream_once(args.port, msgs2, 8, mid,
                                              extra_body=extra_body)
                colds.append(t_cold)
                warms.append(t_warm)
                decs.append((n - 1) / dec if dec > 0 and n > 1 else 0.0)
                decs_ch.append((n_chunks - 1) / dec
                               if dec > 0 and n_chunks > 1 else 0.0)
                ntok.append(n)
                nchunk.append(n_chunks)
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
            pref = pt / row["cold_ttft"] if row["cold_ttft"] > 0 else float("nan")
            print(f"  文脈 {c:>6} ({pt:>6} tok)  冷 TTFT {row['cold_ttft']:7.2f}s  "
                  f"温 TTFT {row['warm_ttft']:6.2f}s  "
                  f"decode {row['decode_tps']:6.1f} tok/s  "
                  f"prefill {pref:7.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(engine=args.engine, rows=rows, tokens=args.tokens,
                       reps=args.reps, argv=argv, port=args.port,
                       model_id=mid, boot_s=boot_s,
                       tokenizer=os.path.expanduser(args.tokenizer),
                       thinking="off" if args.thinking_how != "none" else "default",
                       thinking_how=args.thinking_how,
                       thinking_extra_body=extra_body,
                       extra_body_arg=args.extra_body,
                       thinking_prompt_suffix=prompt_suffix,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S")), f,
                  ensure_ascii=False, indent=2)
    print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
