"""バッチ x 投機レーンのスループット計測 (計測足場のみ、機構本体は書かない)。

判定プロトコルは docs/research/KERNEL-BRIEF-DECODE-BW.md 末尾「バッチ計測の
判定プロトコル」を参照。ここではその宣言に沿って:

- 17k 級の長文プロンプトは含めない (QSA solo tier はバッチに入らない —
  スループットの土俵は短〜中尺の話だと明記しておく)。
- temperature=0 固定 (バッチ経路は QSA の causal 化を含み設計上ビット
  一致しないので、揺らぎの原因を増やさない)。
- プロンプトは固定シードで選ぶ内蔵 8 本のみ (bench/ 直下や scratchpad の
  prompts.json に依存しない自己完結スクリプト)。

計時は bench/ の外 (scratchpad) で使ってきた xlong_probe.py 系と同じ
「クライアント側 SSE 計時」方式: /v1/chat/completions に stream=true で
投げ、"content" delta のチャンクを数えて TTFT と decode tok/s を出す
(チャンク数がそのままトークン数の近似になる — サーバー側の on_tokens は
1 トークンごとに 1 chunk を流すのが通常経路なので、以前からこのリポジトリ
内の素の計測で使われてきた近似をそのまま踏襲する)。

使い方の例:

    # サーバーは別途起動しておく (docs/SERVER.md 参照)。例:
    #   uv run mlxturbo-serve --model <model> --max-batch 8 --port 8765

    # B=4 を同時投入、各 128 トークンまで
    .venv/bin/python bench/batch_throughput.py -B 4 --max-tokens 128

    # 1 本ずつ順番に投げて比較用のベースラインを取る
    .venv/bin/python bench/batch_throughput.py -B 4 --max-tokens 128 --mode sequential

    # サーバー無しで動作確認だけ (ネットワークを叩かない)
    .venv/bin/python bench/batch_throughput.py -B 8 --dry-run

    # 別ポート/別モデル名に向ける
    .venv/bin/python bench/batch_throughput.py -B 4 --base-url http://127.0.0.1:8765/v1 --model qwen38fn
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------- 内蔵プロンプト
#
# 8 種、短い一行質問から長めの要約依頼まで、長さ・種類を散らしてある。
# 実際のトークン数はサーバー側トークナイザ依存の近似 (日本語 SentencePiece
# 系で概ね 1 トークン/1〜2 文字) で、正確な値を保証するものではない —
# これは自己完結のクライアント側ベンチであり、狙いは「多様な長さの束を
# 毎回同じ構成で流せること」であって、トークン数の厳密さではない。

_ARTICLE_SNIPPET = (
    "分散システムにおいてノード間の通信は非同期であり、メッセージは遅延したり"
    "順序が入れ替わったりしうる。この前提の上で合意を取るための代表的な手法が"
    "Paxos や Raft であり、いずれも「過半数の合意」を単一の実行主体を経由せずに"
    "実現する点で共通している。Raft はリーダー選出とログ複製を分離して説明する"
    "ことで実装難易度を下げており、実運用のデータベースや分散ロックサービスで"
    "広く採用されている。一方で、ネットワーク分断が起きた際にどちらの合意手法も"
    "「安全性 (誤った合意をしない)」を優先し「可用性 (常に応答する)」を犠牲に"
    "する設計になっている点は変わらない。"
)

BUILTIN_PROMPTS: tuple[str, ...] = (
    # 1: 短い一行質問
    "富士山の標高は何メートルですか。一文で答えてください。",
    # 2: 短いコード依頼
    "Pythonで、リストから重複を除いた要素を順序を保ったまま返す関数を"
    "書いてください。",
    # 3: 中程度の説明依頼
    "TCPとUDPの違いを、それぞれが向いている用途の具体例を挙げながら"
    "300字程度で説明してください。",
    # 4: 中程度の翻訳依頼
    "次の英文を自然な日本語に翻訳してください。\n\n"
    "\"Continuous batching lets a serving system pack requests of different "
    "lengths into the same forward pass, admitting new requests and "
    "retiring finished ones every decoding step instead of waiting for the "
    "whole batch to finish together.\"",
    # 5: やや長い編集依頼 (コード)
    "次の関数に型ヒントとエラーハンドリング (ファイルが存在しない場合、"
    "JSONとして不正な場合の両方) を追加し、変更点を短く説明してください。\n\n"
    "```python\n"
    "def load_json(path):\n"
    "    import json\n"
    "    return json.loads(open(path).read())\n"
    "```",
    # 6: 長めの要約依頼 (埋め込み記事つき)
    "次の文章を3行で要約してください。\n\n" + _ARTICLE_SNIPPET,
    # 7: 長めの比較エッセイ依頼
    "強整合性と結果整合性のトレードオフについて、具体的な分散システムの"
    "例 (分散データベース、分散ロック、CDNのキャッシュなど) を3つ挙げ、"
    "それぞれでどちらが選ばれやすいか、その理由とともに詳しく説明して"
    "ください。",
    # 8: 長めの構造化依頼
    "オンライン書店のAPIを設計するとして、書籍・著者・注文の3つの"
    "リソースについて、それぞれのJSONスキーマ (フィールド名・型・必須/"
    "任意) を提案してください。フィールドの選定理由も添えてください。",
)


@dataclass
class ReqResult:
    idx: int
    label: str
    ok: bool
    ttft_s: float | None
    decode_tps: float | None
    n_chunks: int
    wall_s: float
    error: str | None = None


def _select_prompts(n: int, seed: int) -> list[str]:
    """固定シードで内蔵 8 本を並べ替え、n 本分を巡回的に取り出す (n>8 なら
    使い回す)。同じ seed なら常に同じ組み合わせ・同じ順序になる。"""

    order = list(range(len(BUILTIN_PROMPTS)))
    random.Random(seed).shuffle(order)
    return [BUILTIN_PROMPTS[order[i % len(order)]] for i in range(n)]


def _stream_one(
    idx: int, prompt: str, base_url: str, model: str | None, max_tokens: int, timeout: float
) -> ReqResult:
    """xlong_probe.py 系と同じ「クライアント側 SSE 計時」: content delta の
    到着だけを数え、最初のチャンクの到着 (TTFT) と、その後のチャンク間隔から
    decode tok/s を出す。1 chunk ≈ 1 トークンという近似は on_tokens が
    1トークンごとに1チャンクを流す通常経路 (FallbackRunner / バッチ経路とも)
    を前提にしている。"""

    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if model:
        body["model"] = model
    label = f"req#{idx} ({len(prompt)}chars)"
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
        return ReqResult(idx, label, False, None, None, n, time.time() - t0, error=str(exc))

    if tfirst is None:
        return ReqResult(idx, label, False, None, None, n, tend - t0, error="no content received")
    decode_tps = (n - 1) / (tend - tfirst) if n > 1 and tend > tfirst else 0.0
    return ReqResult(idx, label, True, tfirst - t0, decode_tps, n, tend - t0)


def _print_result(r: ReqResult) -> None:
    if not r.ok:
        print(f"{r.label}: ERROR ({r.error}) wall={r.wall_s:.2f}s")
        return
    print(
        f"{r.label}: ttft={r.ttft_s:.2f}s decode={r.decode_tps:.1f}tok/s "
        f"({r.n_chunks} tok) wall={r.wall_s:.2f}s"
    )


def run(
    batch_size: int,
    max_tokens: int,
    mode: str,
    base_url: str,
    model: str | None,
    seed: int,
    timeout: float,
    dry_run: bool,
) -> list[ReqResult]:
    prompts = _select_prompts(batch_size, seed)

    if dry_run:
        print(
            f"[dry-run] B={batch_size} max_tokens={max_tokens} mode={mode} "
            f"base_url={base_url} model={model!r} seed={seed}"
        )
        for i, p in enumerate(prompts):
            print(f"  req#{i}: {len(p)} chars, prefix={p[:40]!r}")
        return []

    results: list[ReqResult] = []
    wall_t0 = time.time()

    if mode == "concurrent":
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                pool.submit(_stream_one, i, p, base_url, model, max_tokens, timeout)
                for i, p in enumerate(prompts)
            ]
            for fut in futures:
                r = fut.result()
                results.append(r)
                _print_result(r)
    else:  # "sequential"
        for i, p in enumerate(prompts):
            r = _stream_one(i, p, base_url, model, max_tokens, timeout)
            results.append(r)
            _print_result(r)

    wall_total = time.time() - wall_t0

    ok = [r for r in results if r.ok]
    total_tokens = sum(r.n_chunks for r in ok)
    sum_decode_tps = sum(r.decode_tps or 0.0 for r in ok)
    wall_tps = total_tokens / wall_total if wall_total > 0 else 0.0
    mean_ttft = sum(r.ttft_s or 0.0 for r in ok) / len(ok) if ok else 0.0
    print(
        f"TOTAL: {len(ok)}/{len(results)} ok, tokens={total_tokens}, "
        f"wall={wall_total:.2f}s, mean_ttft={mean_ttft:.2f}s, "
        f"合計decode tok/s(単純和)={sum_decode_tps:.1f}, "
        f"実効tok/s(壁時計)={wall_tps:.1f}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="バッチ経路のスループット計測 (per-request decode tok/s・TTFT・合計)",
    )
    parser.add_argument("-B", "--batch-size", type=int, default=4, help="同時に流すリクエスト数 (既定: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=128, help="リクエストごとの生成上限 (既定: %(default)s)")
    parser.add_argument(
        "--mode",
        choices=("concurrent", "sequential"),
        default="concurrent",
        help="concurrent=B本同時投入 / sequential=1本ずつ順に投げる (既定: %(default)s)",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1", help="サーバーのAPI base URL (既定: %(default)s)")
    parser.add_argument("--model", default=None, help="'model' フィールドに載せる値。省略時はフィールド自体を送らない (server.py 側は省略を許可している)")
    parser.add_argument("--seed", type=int, default=0, help="内蔵プロンプトの選択/並び替えの固定シード (既定: %(default)s)")
    parser.add_argument("--timeout", type=float, default=300.0, help="1リクエストあたりのタイムアウト秒 (既定: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="ネットワークを叩かず、送るはずのリクエストだけ表示する")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size は1以上を指定してください")

    run(
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        mode=args.mode,
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
