"""QSA の端数ブロック因果性の修正 (5d1e1c5) が、速度と受理率をどう動かすか。

## 何を比べるか

A = 現行 (端数ブロックを因果性で切る)
B = 修正前 (端数ブロックを常に可視 = stock の `ones`)

B は `QSAIndexer.__call__` の返り値の端数列を True に戻すだけの薄い包みで
作る (写しは作らない)。他は 1 バイトも変えない。

## 判定基準 (測る前に宣言)

1. **対照 (短文脈、QSA 不活性)**: A と B で出力トークン列が完全一致すること。
   kv 長が indexer_budget を超えないので sparse は None のまま = 修正が
   触れない領域。ここが割れたら測定そのものが無効なので、以降の数字は
   読まない。
2. **本番 (17k、QSA 活性)**: tok/round と ms/token を比べる。これは正しさの
   修正なので合否は付けない (遅くなっても戻さない)。ただし tok/round の
   相対低下が 5% を超えたら、修正の代償として目立つ形で報告する。
3. **順序バイアス**: プロンプトごとに A→B→B→A で回し、A 2 本と B 2 本の
   平均を取る。1 プロセス内で交互 (CLAUDE.md の計測の作法)。
4. 生成長は全条件 512 トークンで揃える。

    tools/biglock.sh .venv/bin/python tools/qsa_causal_ab.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 長文脈の素材: リポジトリの文書を並べた池から、互いに重ならない窓を 3 つ
# 切る。繰り返し文字列で埋めると n-gram と MTP が当てすぎて受理率が嘘になる
# ので、実文で長さを作る
LONG_QUESTIONS = [
    "上の文書の要点を、初めて読む人向けに 5 つに整理してください。",
    "上の文書から、判断の根拠になっている数字だけを抜き出して並べてください。",
    "Summarize the document above, then list the open questions it leaves.",
]
SHORT_PROMPTS = [
    "分散システムにおける結果整合性について説明してください。",
    "Explain why speculative decoding helps when decoding is dispatch bound.",
    "Python でリストの重複を順序を保って除去する関数を書いてください。",
]


def install_stock_tail(Q):
    """B 条件: 端数ブロックを常に可視に戻す薄い包み。返り値だけを触る。"""
    import mlx.core as mx

    orig = Q.QSAIndexer.__call__

    def call(self, x, rope, cache, offset, positions=None):
        keep = orig(self, x, rope, cache, offset, positions)
        if keep is None:
            return None
        kv_len = keep.shape[-1]
        tail = kv_len % self.compress_ratio
        if not tail:
            return keep
        ones = mx.ones(keep.shape[:-1] + (tail,), dtype=keep.dtype)
        return mx.concatenate([keep[..., : kv_len - tail], ones], axis=-1)

    Q.QSAIndexer.__call__ = call
    return orig


def run_once(eng, ids, n_tokens, eos_ids):
    """1 本流して (トークン列, prefill 秒, decode 秒, accepted, rounds) を返す。"""
    import mlx.core as mx

    caches = eng.model.make_cache()
    mx.clear_cache()
    t0 = time.perf_counter()
    gen = eng.generate_stream(ids, n_tokens, caches=caches, eos_ids=eos_ids)
    out, t_prefill = [], None
    try:
        while True:
            toks = next(gen)
            if t_prefill is None:
                t_prefill = time.perf_counter() - t0
                t_dec0 = time.perf_counter()
            out.extend(toks)
    except StopIteration as e:
        val = e.value
        accepted, rounds = val[0], val[1]
    t_dec = time.perf_counter() - t_dec0
    return out, t_prefill, t_dec, accepted, rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None,
                    help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    from mlxturbo import fused, mtp_flash, spec_flash

    model_path = os.path.expanduser(args.model)
    model, tok = load(model_path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    fused.enable_hyper_connection_kernel()
    mtp_path = args.mtp or os.path.join(model_path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path),
                                   model.args.text, quantize=q)
    mx.eval(mtp.parameters())
    eng = spec_flash.FlashSpecEngine(model, mtp)

    import mlx_lm.models.qwen4_exp as Q

    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    eos_ids = tuple(eos) if eos else ()

    # ---- プロンプトを組む -------------------------------------------
    files = sorted(REPO_ROOT.glob("docs/**/*.md")) + [REPO_ROOT / "README.md"]
    pool = "\n\n".join(f.read_text() for f in files if f.exists())
    pool_ids = tok.encode(pool)
    win = args.ctx - 200  # 質問文とテンプレートのぶんを空ける
    longs = []
    for i, qtext in enumerate(LONG_QUESTIONS):
        lo = i * win
        if lo + win > len(pool_ids):
            print(f"素材が足りない (必要 {(i + 1) * win} tok, 手元 {len(pool_ids)})")
            return 1
        body = tok.decode(pool_ids[lo : lo + win])
        longs.append(f"{body}\n\n---\n\n{qtext}")

    def to_ids(text):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True))[None]

    cases = [("short", to_ids(p)) for p in SHORT_PROMPTS]
    cases += [("long", to_ids(p)) for p in longs]

    orig_call = Q.QSAIndexer.__call__

    def set_variant(v):
        if v == "A":
            Q.QSAIndexer.__call__ = orig_call
        else:
            Q.QSAIndexer.__call__ = orig_call  # 一度戻してから包み直す
            install_stock_tail(Q)

    print("判定基準はモジュール docstring のとおり (測る前に宣言済み)。")
    print(f"生成長 {args.tokens} トークンで全条件そろえる。\n")

    rows = []
    for kind, ids in cases:
        n = ids.shape[1]
        print(f"--- {kind} ctx={n} ---", flush=True)
        for v in ("A", "B", "B", "A"):
            set_variant(v)
            out, tp, td, acc, rounds = run_once(eng, ids, args.tokens, eos_ids)
            ms = td / max(len(out), 1) * 1000
            tpr = len(out) / max(rounds, 1)
            rows.append(dict(kind=kind, ctx=n, variant=v, n_out=len(out),
                             prefill_s=tp, decode_s=td, ms_per_tok=ms,
                             accepted=acc, rounds=rounds, tok_per_round=tpr,
                             head=out[:24]))
            print(f"  {v}: prefill {tp:6.2f}s  decode {td:6.2f}s  "
                  f"{ms:6.2f} ms/tok  tok/round {tpr:.3f}  "
                  f"({acc}/{rounds})", flush=True)
    Q.QSAIndexer.__call__ = orig_call

    # ---- まとめ -------------------------------------------------------
    print("\n=== まとめ ===")
    ok = True
    for kind in ("short", "long"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        for metric in ("ms_per_tok", "tok_per_round"):
            a = [r[metric] for r in sub if r["variant"] == "A"]
            b = [r[metric] for r in sub if r["variant"] == "B"]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            d = (ma - mb) / mb * 100
            print(f"  {kind:5s} {metric:14s} A={ma:8.3f}  B={mb:8.3f}  "
                  f"({d:+.2f}% vs 修正前)")
            if kind == "long" and metric == "tok_per_round" and d < -5:
                print("    ** tok/round が 5% 超落ちた。修正の代償として報告する **")

    # 対照: 短文脈は A と B で出力が完全一致するはず
    for ctx in sorted({r["ctx"] for r in rows if r["kind"] == "short"}):
        sub = [r for r in rows if r["ctx"] == ctx]
        heads = {tuple(r["head"]) for r in sub}
        if len(heads) != 1:
            ok = False
            print(f"  対照 NG: ctx={ctx} で A/B の出力が食い違う "
                  "(QSA 不活性なら一致するはず。測定は無効)")
    if ok:
        print("  対照 OK: 短文脈 (QSA 不活性) は A/B で出力が一致")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"\n書き出し: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
