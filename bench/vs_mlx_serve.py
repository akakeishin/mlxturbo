"""mlx-serve と mlxturbo を、同じマシン・同じプロンプト・同じ HTTP 経路で比べる。

`docs/VS-MLX-SERVE.md` と KERNEL-BRIEF のスコアボードは手動手順で取っていて、
再現の手順がスクリプトになっていなかった。ここに固定する。

## 揃えるもの / 揃わないもの

揃える: プロンプト、生成トークン数、temperature=0、SSE のクライアント側計時
(TTFT = 最初の content delta まで、decode = 以降のチャンク数 / 経過秒)、
サーバーは 1 つずつ順に起動 (同時に載せない -- 128GB に 91GB を 2 つは載らない)。

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


def stream_once(port: int, prompt: str, n_tokens: int) -> tuple[float, float, int]:
    """SSE で 1 本流し、(TTFT 秒, decode 秒, チャンク数) を返す。"""
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
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
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    t_dec = time.perf_counter()
                n += 1
    if ttft is None:
        return float("nan"), float("nan"), 0
    return ttft, time.perf_counter() - t_dec, n


class Server:
    def __init__(self, name: str, argv: list[str], port: int):
        self.name, self.argv, self.port = name, argv, port
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
        time.sleep(5)  # メモリが返るのを待つ
        return False


def main() -> int:
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

    from mlx_lm.tokenizer_utils import load_tokenizer  # noqa: F401
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
            for c in ctxs:
                stream_once(port, prompts[c], 8)  # 温め (捨てる)
                ttft, dec, n = stream_once(port, prompts[c], args.tokens)
                tps = (n - 1) / dec if dec > 0 and n > 1 else 0.0
                rows.append(dict(engine=name, ctx=c, ttft_s=ttft,
                                 decode_tps=tps, tokens=n))
                print(f"  {name:10s} ctx={c:6d}  TTFT {ttft:7.2f}s  "
                      f"decode {tps:6.1f} tok/s ({n} tok)", flush=True)

    print("\n=== まとめ (2 本の平均) ===")
    print(f"{'文脈':>7s} {'mlx-serve TTFT':>15s} {'mlxturbo TTFT':>14s} "
          f"{'mlx-serve tok/s':>16s} {'mlxturbo tok/s':>15s}")
    for c in ctxs:
        def avg(engine, key):
            v = [r[key] for r in rows if r["engine"] == engine and r["ctx"] == c]
            return statistics.mean(v) if v else float("nan")
        print(f"{c:7d} {avg('mlx-serve', 'ttft_s'):15.2f} "
              f"{avg('mlxturbo', 'ttft_s'):14.2f} "
              f"{avg('mlx-serve', 'decode_tps'):16.1f} "
              f"{avg('mlxturbo', 'decode_tps'):15.1f}")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
