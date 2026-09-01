"""decode 経路の A/B を 1 プロセス内で交互に測る (knob 差し替え式)。

A = 新しい側、B = 比較対象 (多くは修正前 / 既定 off)。knob ごとに
「どう切り替えるか」と「何をもって合格とするか」を `KNOBS` に書いてある。

## 共通の作法 (CLAUDE.md の計測の作法に従う)

- 1 プロセス内でプロンプトごとに A→B→B→A。線形の熱ドリフトを相殺する。
- **プロセスの最初の 1 本は捨てる。**実測で最初だけ 19.19 ms/tok
  (以降 16.45)、長文脈の TTFT は 73s (以降 35s) になる。混ぜると数 % の
  嘘の差が出る。温まってからの繰り返しは 0.2% 以内。
- 長時間回し続けると熱で 25% 落ちる (16.9k TTFT 35s -> 45s の実測)。
  絶対値を語るなら冷ましてから短く測る。ここで信じるのは A/B の差だけ。
- 生成長は全条件そろえる (既定 512)。長文脈は実文書から切った窓を使う
  (繰り返し文字列だと n-gram と MTP が当てすぎて受理率が嘘になる)。

## knob

`qsa-tail`   QSA の端数ブロック因果性 (5d1e1c5)。B は返り値の端数列を
             True に戻す薄い包みで再現する (写しは作らない)。
             合格条件: 短文脈 (QSA 不活性) で A/B の出力が完全一致すること。
             ここが割れたら測定は無効。長文脈は合否を付けない (正しさの
             修正なので遅くても戻さない) が、tok/round の相対低下が 5% を
             超えたら代償として目立つ形で報告する。
             結果 (2026-09-01): tok/round +2.1%、ms/token は符号が揃わず。

`moe-verify` 共有タイル gather v2 (MLXTURBO_MOE_VERIFY、既定 off)。
             verify 幅の MoE だけを差し替える。
             合格条件: **ms/token が短・長の両方で改善すること。**
             どちらかで悪化したら既定 off のまま据え置く。出力は
             累積順が変わるので一致を要求しない (tok/round の変化は
             テキスト運と区別できないので、判定は ms/token で行う)。

    tools/biglock.sh .venv/bin/python tools/decode_ab.py --knob moe-verify \\
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


def _knob_qsa_tail():
    """A = 現行 (因果性で切る) / B = 修正前 (端数を常に可視)。

    B は `QSAIndexer.__call__` の返り値の端数列を True に戻す薄い包みで作る。
    写しを作らないので、A 側の実装がこの先変わっても B の意味はずれない。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    orig = Q.QSAIndexer.__call__

    def stock_tail(self, x, rope, cache, offset, positions=None):
        keep = orig(self, x, rope, cache, offset, positions)
        if keep is None:
            return None
        kv_len = keep.shape[-1]
        tail = kv_len % self.compress_ratio
        if not tail:
            return keep
        ones = mx.ones(keep.shape[:-1] + (tail,), dtype=keep.dtype)
        return mx.concatenate([keep[..., : kv_len - tail], ones], axis=-1)

    def apply(variant):
        Q.QSAIndexer.__call__ = orig if variant == "A" else stock_tail

    return apply


def _knob_moe_verify():
    """A = 共有タイル gather v2 on / B = off (既定)。"""
    import os

    from mlxturbo import fused

    os.environ["MLXTURBO_MOE_VERIFY"] = "1"  # enable 側のゲートを開ける

    def apply(variant):
        if variant == "A":
            fused.enable_moe_verify_gather()
        else:
            fused.disable_moe_verify_gather()

    return apply


KNOBS = {
    # name: (setup -> apply(variant), 短文脈で A/B 出力の一致を要求するか)
    "qsa-tail": (_knob_qsa_tail, True),
    "moe-verify": (_knob_moe_verify, False),
}


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
    ap.add_argument("--knob", required=True, choices=sorted(KNOBS))
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

    setup, control_identical = KNOBS[args.knob]
    set_variant = setup()

    print(f"knob={args.knob}  判定基準はモジュール docstring のとおり"
          " (測る前に宣言済み)。")
    print(f"生成長 {args.tokens} トークンで全条件そろえる。"
          " 最初の 1 本は温めなので捨てる。\n")

    # 温め: 最初の 1 本は必ず遅いので、計測に混ぜず先に捨てる。短文脈だけ
    # 温めても長文脈の初回は重みのページインで 2 倍かかる (73s vs 35s の実測)
    # ので、長い方も 1 本捨てる
    set_variant("A")
    for kind, ids in cases:
        if kind == "short":
            run_once(eng, ids, 32, eos_ids)
            break
    for kind, ids in cases:
        if kind == "long":
            run_once(eng, ids, 32, eos_ids)
            break

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
    set_variant("A")

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
                print("    ** tok/round が 5% 超落ちた。代償として報告する **")

    if control_identical:
        # 対照: 短文脈は A と B で出力が完全一致するはず
        for ctx in sorted({r["ctx"] for r in rows if r["kind"] == "short"}):
            sub = [r for r in rows if r["ctx"] == ctx]
            if len({tuple(r["head"]) for r in sub}) != 1:
                ok = False
                print(f"  対照 NG: ctx={ctx} で A/B の出力が食い違う"
                      " (一致するはずの領域。測定は無効)")
        if ok:
            print("  対照 OK: 短文脈は A/B で出力が一致")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"\n書き出し: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
