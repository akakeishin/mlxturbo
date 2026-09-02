"""レーン5 ゲート: 走行中バッチへの途中参加が、単独x2に対して割に合っているか。

`docs/research/LANES-2026-09.md` の「## レーン 5」が宣言しているゲート:

    走行中バッチへの途中参加 (chunked prefill で入れる) を配線し、2 本同時
    到着 (間隔 200ms) の壁時計が単独 x2 の 0.8 倍以下。

このスクリプトは、**既に起動しているサーバー**に対して:

  1. 単独リクエストを N 回 (既定 3) 流して壁時計 (送信 〜 最終チャンク) を測り、
     中央値 T1 を取る。
  2. 2 本を `--gap-ms` (既定 200ms) の間隔で送り、両方が終わるまでの壁時計を
     N 回測って中央値 T2 を取る。
  3. 比 T2 / (2*T1) を出し、`--gate` (既定 0.8) 以下なら exit 0、超えたら exit 1。

サーバー自体の起動はこのスクリプトの外で行う。同じゲートを非投機の継続
バッチングと、バッチ x 投機の両方に当てる想定 (`docs/research/LANES-2026-09.md`
レーン5)。例:

    # 非投機の継続バッチング (mlxturbo/batch.py)
    uv run mlxturbo-serve --model <model> --max-batch 8 --port 8765

    # バッチ x 投機 (mlxturbo/batch_spec.py)
    uv run mlxturbo-serve --model <model> --max-batch-spec 8 --port 8765

使い方の例:

    # 既定 (3 回、間隔 200ms、ゲート 0.8)
    .venv/bin/python bench/parallel_join_gate.py

    # 回数やゲートを変えたい場合
    .venv/bin/python bench/parallel_join_gate.py -N 5 --gate 0.85

    # サーバー無しで動作確認だけ (ネットワークを叩かない)
    .venv/bin/python bench/parallel_join_gate.py --dry-run

## プロンプトの作り方

`tools/_bench_text.py` の `long_prompts` でリポジトリ内の実文から窓を切る
(繰り返し文字列は n-gram/MTP が当てすぎて受理率が嘘になるので使わない、
CLAUDE.md の計測の作法と同じ)。**混在長のバッチを踏ませる**のが目的なので、
長さの違う2本 (`--ctx-a` 既定1000、`--ctx-b` 既定300) を使う。単独計測
(T1) も同じ2本を交互に使い、T1 は「A の中央値」と「B の中央値」の平均に
する (どちらか一方だけに寄せない)。

接頭辞キャッシュに当たって壁時計が嘘になるのを避けるため、反復ごとに
`offset_tokens` をずらして、全ての draw (solo A/B, paired A/B x N 回ずつ)
が互いに重ならない窓を使う (`bench/self_snapshot.py` と同じ作法)。

thinking は off で揃える (`reasoning_effort: "none"` を `extra_body` で送る。
`bench/vs_mlx_serve.py` の `stream_once` がこれを受け取る)。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from vs_mlx_serve import QUESTIONS, model_id, stream_once  # noqa: E402


def _port_from_base_url(base_url: str) -> int:
    """`--base-url` からポートだけ取り出す。

    `stream_once` (bench/vs_mlx_serve.py) は `127.0.0.1` 決め打ちなので、
    ここでもホストは実質 127.0.0.1 前提。別ホストを渡された場合は警告だけ
    出してポートは使う (トンネル越し等、稀なケースの救済)。
    """
    u = urlparse(base_url)
    if u.hostname not in ("127.0.0.1", "localhost"):
        print(f"警告: stream_once は 127.0.0.1 決め打ちだが --base-url のホストは "
              f"{u.hostname!r}。127.0.0.1 として扱う。", file=sys.stderr)
    if u.port is None:
        raise SystemExit(f"--base-url にポートが無い: {base_url!r}")
    return u.port


def _tok_s(n_chunks: int, decode_s: float) -> float:
    return (n_chunks - 1) / decode_s if decode_s > 0 and n_chunks > 1 else 0.0


def _wall_request(port: int, prompt: str, n_tokens: int, model: str,
                   extra_body: dict, delay_s: float = 0.0) -> dict:
    """`delay_s` だけ待ってから1本流し、送信 (delay を含む) 〜 最終チャンクの
    壁時計を含めて結果を返す。

    2本を別スレッドで同時に立ち上げつつ片方だけ `delay_s` だけ遅らせる、が
    使い方 (paired 計測)。solo 計測では `delay_s=0`。
    """
    if delay_s > 0:
        time.sleep(delay_s)
    ttft, decode_s, n_chunks, _ = stream_once(
        port, [{"role": "user", "content": prompt}], n_tokens, model,
        extra_body=extra_body)
    wall_s = float("nan") if math.isnan(ttft) else delay_s + ttft + decode_s
    return dict(delay_s=delay_s, ttft_s=ttft, decode_s=decode_s,
                n_chunks=n_chunks, tok_s=_tok_s(n_chunks, decode_s), wall_s=wall_s)


def measure_solo(port: int, prompts: list[str], n_tokens: int, model: str,
                  extra_body: dict, label: str) -> list[dict]:
    rows = []
    for i, p in enumerate(prompts):
        r = _wall_request(port, p, n_tokens, model, extra_body)
        r.update(label=label, rep=i)
        rows.append(r)
        print(f"  solo {label} #{i}: ttft={r['ttft_s']:.2f}s "
              f"decode={r['tok_s']:.1f}tok/s ({r['n_chunks']}tok) "
              f"wall={r['wall_s']:.2f}s")
    return rows


def measure_paired(port: int, prompts_a: list[str], prompts_b: list[str],
                    n_tokens: int, model: str, extra_body: dict,
                    gap_s: float) -> list[dict]:
    """A を即座に、B を `gap_s` 後に別スレッドで送り、両方終わるまでの壁時計を
    実測する (perf_counter で外側から測る。個々の `wall_s` は近似、こちらが
    真値)。
    """
    rows = []
    for i, (pa, pb) in enumerate(zip(prompts_a, prompts_b)):
        result: dict = {}

        def _run(key: str, prompt: str, delay: float) -> None:
            result[key] = _wall_request(port, prompt, n_tokens, model,
                                         extra_body, delay_s=delay)

        ta = threading.Thread(target=_run, args=("a", pa, 0.0))
        tb = threading.Thread(target=_run, args=("b", pb, gap_s))
        t0 = time.perf_counter()
        ta.start()
        tb.start()
        ta.join()
        tb.join()
        wall_total = time.perf_counter() - t0
        row = dict(rep=i, a=result["a"], b=result["b"], wall_s=wall_total)
        rows.append(row)
        ra, rb = result["a"], result["b"]
        print(f"  paired #{i}: A ttft={ra['ttft_s']:.2f}s "
              f"decode={ra['tok_s']:.1f}tok/s ({ra['n_chunks']}tok) | "
              f"B ttft={rb['ttft_s']:.2f}s decode={rb['tok_s']:.1f}tok/s "
              f"({rb['n_chunks']}tok) | 合計壁時計={wall_total:.2f}s")
    return rows


def build_prompts(tok, ctx: int, n: int, offset: int, question: str) -> tuple[list[str], int]:
    """`ctx` の窓を `n` 本、`offset` から互いに重ならないように切る。

    次に安全に使える offset も一緒に返す (呼び出し側はこれを次の呼び出しに
    渡していけば、全 draw が池のどこにも重ならない)。
    """
    from _bench_text import long_prompts
    win = max(ctx - 200, 16)
    prompts = long_prompts(tok, ctx, [question] * n, offset_tokens=offset)
    return prompts, offset + win * n


def main() -> int:
    ap = argparse.ArgumentParser(
        description="レーン5ゲート: 2本同時到着(間隔200ms)の壁時計が単独x2の0.8倍以下か",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8765/v1",
                     help="サーバーのAPI base URL (既定: %(default)s)")
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm",
                     help="プロンプトを組むトークナイザのモデルパス (既定: %(default)s)。"
                          " mlx_lm.utils.load_tokenizer で読む")
    ap.add_argument("--served-model", default=None,
                     help="'model' フィールドに載せる値。省略時はサーバーの"
                          " /v1/models から自動取得する (--dry-run では未使用)")
    ap.add_argument("--ctx-a", type=int, default=1000, help="プロンプトAの文脈長 (既定: %(default)s)")
    ap.add_argument("--ctx-b", type=int, default=300, help="プロンプトBの文脈長 (既定: %(default)s)")
    ap.add_argument("--tokens", type=int, default=256, help="1リクエストあたりの生成上限 (既定: %(default)s)")
    ap.add_argument("--gap-ms", type=float, default=200.0,
                     help="paired計測でBをAから何ms遅らせて送るか (既定: %(default)s)")
    ap.add_argument("-N", "--runs", type=int, default=3,
                     help="solo/pairedそれぞれの反復回数 (既定: %(default)s)")
    ap.add_argument("--gate", type=float, default=0.8,
                     help="T2/(2*T1) がこの値以下ならPASS (既定: %(default)s)")
    ap.add_argument("--out", default="bench/results/parallel-join-gate.json",
                     help="結果JSONの書き出し先 (既定: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                     help="ネットワークを叩かず、切るはずのプロンプトと計画だけ表示する")
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs は1以上を指定してください")
    if args.ctx_a == args.ctx_b:
        ap.error("--ctx-a と --ctx-b は異なる長さにすること (混在長を踏ませるのが目的)")

    gap_s = args.gap_ms / 1000.0
    extra_body = {"reasoning_effort": "none"}

    print(f"base_url={args.base_url}  model(tok)={args.model}  "
          f"ctx_a={args.ctx_a} ctx_b={args.ctx_b} tokens={args.tokens}  "
          f"gap={args.gap_ms:.0f}ms  N={args.runs}  gate<={args.gate}")

    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(str(Path(args.model).expanduser()))

    # 全 draw (solo A, solo B, paired A, paired B) が互いに重ならないよう、
    # 単一の offset を順送りする (CLAUDE.md の計測の作法: 反復ごとに窓をずらす)。
    offset = 0
    solo_a_prompts, offset = build_prompts(tok, args.ctx_a, args.runs, offset, QUESTIONS[0])
    solo_b_prompts, offset = build_prompts(tok, args.ctx_b, args.runs, offset, QUESTIONS[1])
    paired_a_prompts, offset = build_prompts(tok, args.ctx_a, args.runs, offset, QUESTIONS[0])
    paired_b_prompts, offset = build_prompts(tok, args.ctx_b, args.runs, offset, QUESTIONS[1])

    if args.dry_run:
        print(f"[dry-run] 窓の合計トークン数(池からの消費) = {offset}")
        for label, prompts in (("solo A", solo_a_prompts), ("solo B", solo_b_prompts),
                                ("paired A", paired_a_prompts), ("paired B", paired_b_prompts)):
            for i, p in enumerate(prompts):
                n_tok = len(tok.encode(p))
                print(f"  {label} #{i}: {n_tok} tok, prefix={p[:40]!r}")
        print(f"[dry-run] --out={args.out} (書き出しは実走のみ)")
        return 0

    port = _port_from_base_url(args.base_url)
    try:
        served_model = args.served_model or model_id(port)
    except Exception as exc:  # noqa: BLE001 — サーバー未起動時に分かりやすく落とす
        print(f"サーバーに繋がらない ({args.base_url}): {exc}", file=sys.stderr)
        return 2
    print(f"served_model={served_model!r}\n")

    print("=== solo (単独) ===")
    solo_rows = (measure_solo(port, solo_a_prompts, args.tokens, served_model, extra_body, "A")
                 + measure_solo(port, solo_b_prompts, args.tokens, served_model, extra_body, "B"))

    print("\n=== paired (2本同時到着、間隔 %.0fms) ===" % args.gap_ms)
    paired_rows = measure_paired(port, paired_a_prompts, paired_b_prompts,
                                  args.tokens, served_model, extra_body, gap_s)

    def _median_wall(label: str) -> float:
        vals = [r["wall_s"] for r in solo_rows if r["label"] == label and not math.isnan(r["wall_s"])]
        if not vals:
            raise RuntimeError(f"solo {label} の有効なサンプルが1件も無い")
        return statistics.median(vals)

    t1_a = _median_wall("A")
    t1_b = _median_wall("B")
    t1 = (t1_a + t1_b) / 2.0

    t2_vals = [r["wall_s"] for r in paired_rows]
    t2 = statistics.median(t2_vals)

    ratio = t2 / (2.0 * t1) if t1 > 0 else float("inf")
    verdict = "PASS" if ratio <= args.gate else "FAIL"

    print("\n=== まとめ ===")
    print(f"T1 (単独, 中央値): A={t1_a:.2f}s  B={t1_b:.2f}s  平均={t1:.2f}s")
    print(f"T2 (2本同時到着, 中央値): {t2:.2f}s")
    print(f"比 T2/(2*T1) = {ratio:.3f}  (ゲート <= {args.gate})")
    print(f"GATE: {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(
        base_url=args.base_url, model=args.model, served_model=served_model,
        ctx_a=args.ctx_a, ctx_b=args.ctx_b, tokens=args.tokens,
        gap_ms=args.gap_ms, runs=args.runs, gate=args.gate,
        solo=solo_rows, paired=paired_rows,
        t1_a=t1_a, t1_b=t1_b, t1=t1, t2=t2, ratio=ratio, verdict=verdict,
    ), ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
