"""B=1 無劣化ゲート: バッチ機構を有効化した状態でも単独リクエストの短 decode
が劣化していないことを回帰として固定する。

判定プロトコル (docs/research/KERNEL-BRIEF-DECODE-BW.md 末尾「バッチ計測の
判定プロトコル」) の宣言:

    B=1 無劣化ゲート: バッチ機構を有効化しても単一リクエストの短 decode
    14.6ms/tok が動かないこと。これを bench の回帰に固定してから機構に
    着手する。

このスクリプトは「サーバーが --max-batch >1 (バッチコーディネータ有効) で
起動している状態」に対して、単独リクエストを N 回流し、短 decode の
ms/token を基準値 (--baseline、既定 14.6) と比較する。±5% を超えたら
exit code 1 で落ちる (CIやワンライナーで拾いやすくするため)。

サーバー自体の起動はこのスクリプトの外で行うこと。例:

    # 継続バッチング (非投機、mlxturbo/batch.py) を有効にした状態を見る場合
    uv run mlxturbo-serve --model <model> --max-batch 8 --port 8765

    # バッチ x 投機 (mlxturbo/batch_spec.py) を有効にした状態を見る場合。
    # こちらは待ち行列に 1 本しか無いとき単独経路をそのまま使う設計なので、
    # このゲートは「その設計が実際に守られているか」を測ることになる
    uv run mlxturbo-serve --model <model> --max-batch-spec 8 --port 8765

使い方の例:

    # 既定 (基準値14.6ms/tok、8回流して比較)
    .venv/bin/python bench/batch_b1_gate.py

    # 基準値を変えたい/回数を変えたい場合
    .venv/bin/python bench/batch_b1_gate.py --baseline 15.8 -N 16

    # サーバー無しで動作確認だけ (ネットワークを叩かない)
    .venv/bin/python bench/batch_b1_gate.py --dry-run

計時方式は bench/batch_throughput.py および scratchpad の xlong_probe.py
系と同じ「クライアント側 SSE 計時」(content delta チャンク数 ≈ トークン数)。
プロンプトは短 decode を測るための短い固定プロンプト1本のみを使う
(バッチ経路の判定プロトコルにある「1kと16kを同居させない」の対極、ここでは
単一の短いリクエストだけを繰り返し流して decode 速度のばらつきを見る)。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# 短 decode 計測用の固定プロンプト。短いプロンプト + ある程度の生成量で、
# prefill時間を無視できるくらいdecodeが支配的になるようにしてある。
_SHORT_PROMPT = "こんにちは。今日は天気がいいですね。今の気分を一言で教えてください。"


@dataclass
class GateSample:
    ok: bool
    ttft_s: float | None
    ms_per_token: float | None
    n_chunks: int
    error: str | None = None


def _stream_once(base_url: str, model: str | None, max_tokens: int, timeout: float) -> GateSample:
    body = {
        "messages": [{"role": "user", "content": _SHORT_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if model:
        body["model"] = model
    data = json.dumps(body).encode()
    req = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    tfirst = None
    n = 0
    try:
        with urlopen(req, timeout=timeout) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    now = time.time()
                    if tfirst is None:
                        tfirst = now
                    n += 1
        tend = time.time()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return GateSample(False, None, None, n, error=str(exc))

    if tfirst is None or n <= 1:
        return GateSample(False, tfirst - t0 if tfirst else None, None, n, error="not enough decode tokens")
    decode_time = tend - tfirst
    ms_per_token = (decode_time / (n - 1)) * 1000.0 if decode_time > 0 else None
    return GateSample(True, tfirst - t0, ms_per_token, n)


def run_gate(
    n_runs: int,
    max_tokens: int,
    base_url: str,
    model: str | None,
    baseline_ms: float,
    tolerance: float,
    timeout: float,
    dry_run: bool,
) -> int:
    """戻り値はプロセスの exit code (0=合格, 1=不合格/計測不能)。"""

    if dry_run:
        print(
            f"[dry-run] N={n_runs} max_tokens={max_tokens} base_url={base_url} "
            f"model={model!r} baseline={baseline_ms:.2f}ms/tok tolerance=±{tolerance * 100:.0f}%"
        )
        print(f"  prompt: {_SHORT_PROMPT!r}")
        return 0

    samples: list[GateSample] = []
    for i in range(n_runs):
        s = _stream_once(base_url, model, max_tokens, timeout)
        samples.append(s)
        if s.ok:
            print(f"run#{i}: ttft={s.ttft_s:.2f}s ms/tok={s.ms_per_token:.2f} ({s.n_chunks} tok)")
        else:
            print(f"run#{i}: ERROR ({s.error})")

    ok = [s for s in samples if s.ok and s.ms_per_token is not None]
    if not ok:
        print("GATE: FAIL — 有効なサンプルが1件も取れなかった (サーバー起動/接続を確認)")
        return 1

    ms_values = [s.ms_per_token for s in ok]
    median_ms = statistics.median(ms_values)
    ratio = median_ms / baseline_ms
    delta_pct = (ratio - 1.0) * 100.0
    verdict = "PASS" if abs(ratio - 1.0) <= tolerance else "FAIL"
    print(
        f"GATE: {verdict} — median={median_ms:.2f}ms/tok baseline={baseline_ms:.2f}ms/tok "
        f"比={ratio:.3f} ({delta_pct:+.1f}%) 許容=±{tolerance * 100:.0f}% "
        f"({len(ok)}/{n_runs} 件有効)"
    )
    return 0 if verdict == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B=1無劣化ゲート: バッチ機構有効時の単独リクエスト短decodeがベースラインから±5%以内かを見る",
    )
    parser.add_argument("-N", "--runs", type=int, default=8, help="単独リクエストを流す回数 (既定: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=64, help="1回あたりの生成トークン上限 (既定: %(default)s)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1", help="サーバーのAPI base URL (既定: %(default)s)")
    parser.add_argument("--model", default=None, help="'model' フィールドに載せる値。省略時はフィールド自体を送らない")
    parser.add_argument("--baseline", type=float, default=14.6, help="比較対象の基準 ms/token (既定: %(default)s、KERNEL-BRIEF-DECODE-BW.md の short decode 実測値)")
    parser.add_argument("--tolerance", type=float, default=0.05, help="許容比率 (既定: %(default)s = ±5%%)")
    parser.add_argument("--timeout", type=float, default=120.0, help="1リクエストあたりのタイムアウト秒 (既定: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="ネットワークを叩かず、実行内容だけ表示する")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs は1以上を指定してください")

    sys.exit(
        run_gate(
            n_runs=args.runs,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
            model=args.model,
            baseline_ms=args.baseline,
            tolerance=args.tolerance,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
