"""mlx-serve と mlxturbo を、同じマシン・同じプロンプト・同じ HTTP 経路で比べる。

`docs/VS-MLX-SERVE.md` と KERNEL-BRIEF のスコアボードは手動手順で取っていて、
再現の手順がスクリプトになっていなかった。ここに固定する。

## 揃えるもの / 揃わないもの

揃える: プロンプト、生成トークン数、temperature=0、SSE のクライアント側計時
(TTFT = 最初の content delta まで、decode = 以降のチャンク数 / 経過秒)、
サーバーは 1 つずつ順に起動 (同時に載せない -- 128GB に 91GB を 2 つは載らない)。

**冷たい TTFT と温かい TTFT の両方を取る。**実クライアントは毎ターン
「前のやり取り + 新しい発言」をまるごと送るので、2 ターン目は前回の
プロンプトが接頭辞になる。そこを再利用できるかが温 TTFT で、うちは
checkpoint 復帰と BPE 末尾分割がここに乗っている (追記ターン 16k で
6.14s -> 1.1s の修正)。**冷えた TTFT だけ測ると、その仕事が 1 ミリも
見えない。**

揃わない: **重みが違う。**それぞれの推奨構成どうしの比較で、同一重みの比較では
ない (`docs/VS-MLX-SERVE.md` の注記と同じ)。

## 順序バイアス

文脈ごとに **A→B→B→A** で回す。サーバーの起動は重い (91GB のロード) ので、
「両方起動しっぱなしで交互」はメモリ的に無理。よって「A を起動して 2 本測る →
落とす → B を起動して 2 本測る → 落とす → …」ではなく、
**A(1本) B(1本) B(1本) A(1本)** の順で起動を 4 回行う。遅いが順序の効果は消える。
起動のたびに冷えるので、各セッションで最初の 1 本は捨てる。

    tools/biglock.sh .venv/bin/python bench/vs_mlx_serve.py \\
        --serve-bin ~/dev/mlx-serve/zig-out/bin/mlx-serve \\
        --serve-model ~/models/ddalcu-flashnext-serve-4bit \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep \\
        --ctxs 0,4000,17000,50000
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

SHORT = "分散システムにおける結果整合性について、具体例を挙げて説明してください。"
QUESTIONS = [
    "上の文書の要点を 5 つに整理してください。",
    "上の文書から、判断の根拠になっている数字だけを抜き出して並べてください。",
]


def wait_ready(port: int, timeout: float = 900.0) -> bool:
    """/v1/models が返るまで待つ (91GB のロードに数分かかる)。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def model_id(port: int) -> str:
    """/v1/models が名乗っている id を取る。

    **プレースホルダを送らないこと。**以前ここは `"model": "x"` を送っていて、
    404 で比較が丸ごと落ちた (2026-09-01)。

    **両者とも 404 を返す。**うちの `_check_model_openai` は既定で完全一致のみ
    (別名は `--model-alias` で明示したものだけ) で、mlx-serve も同じく見る。
    A→B→B→A の順で mlx-serve が先だったので先に鳴っただけで、うちに当てても
    同じく落ちていた。**「うちは model を見ないから気づかなかった」ではない**
    (最初そう誤診した)。

    それぞれが名乗っている id をそのまま送り返すのが正しい。
    """
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models", timeout=10
    ) as r:
        data = json.loads(r.read().decode())
    items = data.get("data") or []
    if not items:
        raise RuntimeError(f"port {port} の /v1/models が空")
    return items[0]["id"]


def stream_once(port: int, messages: list, n_tokens: int, model: str = "x",
                 extra_body: dict | None = None):
    """SSE で 1 本流し、(TTFT 秒, decode 秒, チャンク数, 本文) を返す。

    本文を返すのは**追記ターン (warm TTFT) を作るため**。実クライアントは
    「前のやり取り + 新しい発言」を毎回まるごと送るので、2 ターン目は
    前回のプロンプトが接頭辞になる。そこを再利用できるかが warm TTFT で、
    うちは checkpoint 復帰と BPE 末尾分割がここに乗っている
    (追記ターン 16k で 6.14s -> 1.1s の修正)。**冷えた TTFT だけ測ると、
    その仕事が 1 ミリも見えない。**

    `extra_body` を渡すとリクエスト本体にマージする (例: `reasoning_effort`
    で thinking の on/off を揃える)。両サーバーとも OpenAI 標準の
    `reasoning_effort` を読む ("none" で off、"medium" 等で on)。
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": n_tokens,
        "temperature": 0,
        "stream": True,
    }
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            # **思考 (reasoning_content) も数える。**以前は content だけ見て
            # いて、長い文脈だと 256 トークンが思考で尽きて本文が 1 つも
            # 来ず、うちの側が nan / 0 tok になっていた (2026-09-02)。
            # 速度の比較としては「どの経路であれトークンが出る速さ」が
            # 見たいものなので、両方数えるのが正しい。両サーバーに同じ
            # 扱いをする。
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    t_dec = time.perf_counter()
                n += 1
                parts.append(piece)
    if ttft is None:
        return float("nan"), float("nan"), 0, ""
    return ttft, time.perf_counter() - t_dec, n, "".join(parts)


def install_term_handler() -> None:
    """SIGTERM/SIGINT で `with` を巻き戻して、起動したサーバーを道連れにする。

    `Server` は `start_new_session=True` で別セッションに切り離す (ハーネスが
    Ctrl-C を受けてもサーバーが即死しないように)。その代わり、**親が外から
    殺されると `__exit__` が走らず、91GB を抱えたサーバーだけが残る。**
    2026-09-01 に実際に起きた: 比較を止めたら mlx-serve が 69.5GB を抱えたまま
    15 分居座り、biglock が「ロック無しのプロセスが走っている」と正しく検出して
    後続の測定が全部待たされた。

    SystemExit を投げれば `with` の `__exit__` が走るので、殺し方が SIGKILL で
    ない限り後始末が付く。
    """

    def _bye(signum, frame):
        raise SystemExit(f"signal {signum} を受けたので終了する")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _bye)


class Server:
    def __init__(self, name: str, argv: list[str], port: int,
                 log_path: str | None = None):
        self.name, self.argv, self.port = name, argv, port
        self.log_path = log_path
        self.proc = None
        self._logf = None

    def __enter__(self):
        if self.log_path:
            self._logf = open(self.log_path, "ab")
            out = err = self._logf
        else:
            out = err = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            self.argv, stdout=out, stderr=err,
            start_new_session=True)
        if not wait_ready(self.port):
            self.__exit__(None, None, None)
            raise RuntimeError(f"{self.name} が起動しなかった")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        if self._logf:
            self._logf.close()
        time.sleep(5)  # メモリが返るのを待つ
        return False


def main() -> int:
    install_term_handler()
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-bin", required=True)
    ap.add_argument("--serve-model", required=True)
    ap.add_argument("--serve-port", type=int, default=11234)
    ap.add_argument("--model", required=True, help="mlxturbo 側のモデル")
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--ctxs", default="0,4000,17000",
                    help="0 は短いプロンプト。それ以外は実文から窓を切る")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", default="bench/results/vs-mlx-serve.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.model))
    from _bench_text import long_prompts

    ctxs = [int(c) for c in args.ctxs.split(",")]
    prompts: dict[int, str] = {}
    for c in ctxs:
        prompts[c] = SHORT if c == 0 else long_prompts(tok, c, QUESTIONS[:1])[0]

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

    sides = {
        "mlx-serve": (Server, serve_argv, args.serve_port),
        "mlxturbo": (Server, turbo_argv, args.port),
    }

    print("判定基準はモジュール docstring のとおり (測る前に宣言済み)。")
    print(f"生成 {args.tokens} トークン、文脈 {ctxs}、順序は A B B A。\n")

    rows = []
    for name in ("mlx-serve", "mlxturbo", "mlxturbo", "mlx-serve"):
        cls, argv, port = sides[name]
        with cls(name, argv, port):
            mid = model_id(port)
            for c in ctxs:
                msgs = [{"role": "user", "content": prompts[c]}]
                # **温めに測定と同じプロンプトを使わないこと。**同じものを
                # 投げると接頭辞キャッシュに当たり、次の「冷 TTFT」が
                # 冷えなくなる。2026-09-02 に相手側で 17k の冷 TTFT が
                # 0.21s と出て気づいた (17k の prefill が 0.21s はあり得ない)。
                # 温めの目的はカーネルの初回コンパイルを済ませることなので、
                # 短い別プロンプトで足りる。
                stream_once(port, [{"role": "user", "content": SHORT}], 8, mid)
                ttft, dec, n, reply = stream_once(port, msgs, args.tokens, mid)
                tps = (n - 1) / dec if dec > 0 and n > 1 else 0.0
                # 追記ターン: 実クライアントと同じく履歴をまるごと送り直す。
                # 前ターンのプロンプトが接頭辞になるので、再利用が効けば
                # TTFT が落ちる
                msgs2 = msgs + [{"role": "assistant", "content": reply},
                                {"role": "user", "content": "続けてください。"}]
                w_ttft, _, _, _ = stream_once(port, msgs2, 8, mid)
                rows.append(dict(engine=name, ctx=c, ttft_s=ttft,
                                 warm_ttft_s=w_ttft, decode_tps=tps, tokens=n))
                print(f"  {name:10s} ctx={c:6d}  冷 TTFT {ttft:7.2f}s  "
                      f"温 TTFT {w_ttft:6.2f}s  decode {tps:6.1f} tok/s"
                      f" ({n} tok)", flush=True)

    print("\n=== まとめ (2 本の平均) ===")
    hdr = ("文脈", "冷TTFT serve", "冷TTFT turbo", "温TTFT serve",
           "温TTFT turbo", "tok/s serve", "tok/s turbo")
    print("".join(f"{h:>14s}" for h in hdr))
    for c in ctxs:
        def avg(engine, key):
            v = [r[key] for r in rows
                 if r["engine"] == engine and r["ctx"] == c
                 and r[key] == r[key]]  # NaN を除く
            return statistics.mean(v) if v else float("nan")
        vals = (avg("mlx-serve", "ttft_s"), avg("mlxturbo", "ttft_s"),
                avg("mlx-serve", "warm_ttft_s"), avg("mlxturbo", "warm_ttft_s"),
                avg("mlx-serve", "decode_tps"), avg("mlxturbo", "decode_tps"))
        print(f"{c:>14d}" + "".join(f"{v:14.2f}" for v in vals))
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
