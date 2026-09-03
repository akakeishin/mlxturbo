"""案 D1 の破断点だけを先に測る probe (Challenge 2026-09-03 08:20 の表)。

問い: **MTP 1 段を S 行の因果ブロックで通したとき、行 j の結果は S=1 の
逐次呼び出しと一致するか。**一致しなければ D1 の draft は現行と別物になり、
受理率の差がテキスト運に埋もれる (ms/round の A/B の意味が薄れる)。

台帳の破断点は「17k の MTP 層は QSA union gather で S=3 の因果ブロックが
ビット一致しない可能性」。ただし QSA (`QSAIndexer`) が発火するのは
**MTP キャッシュ自身の kv 長**が `indexer_budget` (2048) を越えたときで、
トランクの文脈長ではない。MTP キャッシュは `PRIME_WINDOW` (既定 512) から
始まってラウンドごとに (hit+1) 列ずつ伸びるので、512 トークン生成では
1000 前後にしかならない。そこでこの probe は 2 つの条件で測る:

  1. 実運用条件 (`--prime-window` 既定 = spec_flash.PRIME_WINDOW)。
     ベンチの A/B が実際に通る領域。
  2. QSA を踏ませる条件 (`--qsa-prime-window`、既定 4000)。
     MTP kv > 2048 にして union gather の経路を通す。長い生成
     (おおよそ 1500 トークン超) や PRIME_WINDOW を上げた構成で実際に来る。

やること: 本物の decode を数ラウンド流し、`_presync_step0` が呼ばれる
たびに (a) S 行ブロック (b) S=1 の逐次、を**同じキャッシュ状態の複製**から
それぞれ通し、mixer 出力・draft トークン・キャッシュに書かれた k/v を
突き合わせる。判定はブロックの行 j と逐次の j 回目の比較。

`--tiny` は合成モデル・CPU・数秒の一次検査 (モデルも GPU も要らない)。
受理数 hit を 0..S-1 と**強制**して、

    A: `_presync_step0` (S 行ブロック) + trim + `_draft_chain(first=...)`
    B: `_prime_accepted_gap` (hit 回の逐次) + `_draft_chain` (step 0 込み)

の後の MTP キャッシュ列数・k/v・出てくる draft を突き合わせる。実モデルでは
自然な受理を待たないと hit>=1 のラウンドが踏めないので、**帳尻 (どの列が
残るか) の検査はこちらの担当**。実モデル側 (既定) は逆に、本物の重みと
bf16・GPU カーネルでどれだけ値がずれるかを見る。

使い方 (GPU を使うので必ず biglock 経由。--tiny だけは素で走らせてよい):

    .venv/bin/python tools/draft_presync_check.py --tiny
    tools/biglock.sh .venv/bin/python tools/draft_presync_check.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctx 17000 --rounds 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mtp_out(eng, toks, hyper, cache):
    """MTP 1 段 (combine -> layers[0] -> mixer) を通す。

    `FlashMTPModule.__call__` はまさにこの 3 つなので、`_draft_chain` /
    `_presync_step0` の 1 段と同じ計算になる (写しを増やさない)。
    S=1 でも S=N でも同じ呼び出しで通る。
    """
    from mlxturbo import spec_flash

    Q = spec_flash._arch()
    emb = eng.model.model.embed_tokens(toks)
    mask = Q.create_attention_mask(emb, None)
    return eng.mtp(emb, hyper, eng.rope, mask, cache, cache.indexer)


def _cmp(mx, a, b):
    """(ビット一致か, max|diff|) を返す。"""
    same = bool(mx.all(a == b).item())
    d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
    return same, d


def tiny() -> int:
    """合成モデル (CPU、数秒) で「どの列が残るか」の帳尻だけを見る。

    実モデルでは hit>=1 のラウンドを自然に踏むまで待つ必要があるが、ここでは
    hit を 0..S-1 と強制して 3 通りとも通す。QSA を踏む構成 (budget=8) と
    踏まない構成 (budget=4096)、rerank あり/なしの 4 通り。
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    mx.set_default_device(mx.cpu)

    import mlxturbo  # noqa: F401
    from mlxturbo import mtp_flash, spec_flash

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from verify_batch_cache import TINY, build

    def make(budget, rerank):
        mx.random.seed(0)
        model = build(budget)
        if rerank:
            model.lm_head = nn.QuantizedLinear.from_linear(
                model.lm_head, group_size=64, bits=4)
            mx.eval(model.parameters())
        mx.random.seed(1)
        mtp = mtp_flash.FlashMTPModule(model.args.text, variant="lane")
        mtp.update(tree_map(
            lambda a: mx.random.normal(a.shape) * 0.05
            if a.dtype == mx.float32 else a, mtp.parameters()))
        mx.eval(mtp.parameters())
        mtp.eval()
        return spec_flash.FlashSpecEngine(model, mtp)

    ok = True
    for rerank in (False, True):
        for budget in (8, 4096):
            eng = make(budget, rerank)
            model = eng.model
            ids = mx.array([(i * 7 + 3) % TINY["vocab_size"]
                            for i in range(40)])[None]
            caches = model.make_cache()
            with spec_flash.capture(model) as cap:
                h = model.model(ids, cache=caches)
                logits = eng._head(h[:, -1:])
                mx.eval(logits)
            mtp_cache = eng._prime_draft_cache(ids, cap.hyper)
            cur = mx.argmax(logits[:, -1], axis=-1).reshape(1, 1)
            depth = 2
            drafts = eng._draft_chain(cur, cap.hyper[:, -1:], mtp_cache, depth)
            mx.eval(drafts)
            pair = mx.concatenate([cur] + drafts, axis=1)
            total = pair.shape[1]
            with spec_flash.capture(model) as cap2:
                lg = spec_flash._staged_forward(model, pair, caches)
            nxt_all = mx.argmax(lg, axis=-1)
            mx.eval(lg, nxt_all)
            base = spec_flash.snapshot_mtp_cache(mtp_cache)

            for hit in range(total):
                toks = [nxt_all[:, j:j + 1] for j in range(hit + 1)]
                hypers = [cap2.hyper[:, j:j + 1] for j in range(hit + 1)]

                cA = spec_flash.restore_mtp_cache(base)
                keep, ptoks, px, _ = eng._presync_step0(
                    nxt_all, cap2.hyper, cA, total)
                mx.eval(ptoks, px)
                spec_flash.trim_attn_cache(cA, keep + len(toks))
                m = len(toks) - 1
                dA = eng._draft_chain(
                    toks[-1], hypers[-1], cA, depth,
                    first=(ptoks[:, m:m + 1], px[:, m:m + 1], None))
                mx.eval(dA)

                cB = spec_flash.restore_mtp_cache(base)
                eng._prime_accepted_gap(toks, hypers, cB)
                dB = eng._draft_chain(toks[-1], hypers[-1], cB, depth)
                mx.eval(dB)

                n = min(cA.size(), cB.size())
                k_same, k_d = _cmp(mx, cA.keys[..., :n, :], cB.keys[..., :n, :])
                v_same, v_d = _cmp(mx, cA.values[..., :n, :],
                                   cB.values[..., :n, :])
                tA = [int(t.item()) for t in dA]
                tB = [int(t.item()) for t in dB]
                szok = cA.size() == cB.size()
                ok &= szok and tA == tB
                print(f"  rerank={rerank} budget={budget} hit={hit}: "
                      f"列数 A={cA.size()} B={cB.size()}"
                      f" {'OK' if szok else 'NG'}  draft A={tA} B={tB}"
                      f" {'一致' if tA == tB else '不一致'}  "
                      f"k bit={k_same}({k_d:.1e}) v bit={v_same}({v_d:.1e})")
    print("tiny:", "OK" if ok else "NG")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                    help="合成モデル (CPU、数秒) の帳尻検査だけを走らせる。"
                         "--model は要らない")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--rounds", type=int, default=5,
                    help="突き合わせるラウンド数 (それ以降は素通り)")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prime-window", type=int, default=None,
                    help="条件 1 の PRIME_WINDOW (既定 = spec_flash の既定値)")
    ap.add_argument("--qsa-prime-window", type=int, default=4000,
                    help="条件 2 の PRIME_WINDOW。MTP kv を indexer_budget "
                         "(2048) 超にして QSA union gather を踏ませる。"
                         "0 でこの条件を飛ばす")
    ap.add_argument("--depth", type=int, default=2,
                    help="投機深さを固定する (ブロック幅 S = depth + 1)。"
                         "台帳の破断点は S=3 なので既定 2。長文脈では静的規則も "
                         "depth 適応も depth 1 (S=2) に落とすので、固定しないと "
                         "S=3 の行が 1 度も踏めない。`decode_ab.py --depth` と "
                         "同じく `depth_ctx_limit` を上げ、さらに controller も "
                         "切って完全に固定する (0 で engine の既定に任せる)")
    args = ap.parse_args()

    if args.tiny:
        return tiny()
    if not args.model:
        ap.error("--model が要る (合成モデルだけで済ませるなら --tiny)")

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    mx.random.seed(0)

    import mlxturbo  # noqa: F401
    from mlxturbo import mtp_flash, spec_flash
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    model_path = os.path.expanduser(args.model)
    model, tok = load(model_path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[presync_check]")
    set_wired_limit_default(log_prefix="[presync_check]")
    mtp_path = args.mtp or os.path.join(model_path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path),
                                   model.args.text, quantize=q)
    mx.eval(mtp.parameters())
    eng = spec_flash.FlashSpecEngine(model, mtp)
    if args.depth:
        # 破断点は「S=3 の因果ブロック」なので、S を固定できないと probe が
        # 問いに答えられない。ctx_limit を外すだけでは足りない
        # (MLXTURBO_DEPTH_ADAPT が既定 on で、長文脈では controller が
        # 上書きする) ので、controller ごと切る。
        eng.depth = args.depth
        eng.depth_ctx_limit = 1 << 30
        eng._depth_adapt = False

    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    eos_ids = tuple(eos) if eos else ()

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts

    text = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True))[None]
    print(f"ctx={ids.shape[1]} tok  indexer_budget="
          f"{model.args.text.indexer_budget}", flush=True)

    orig_presync = spec_flash.FlashSpecEngine._presync_step0
    rows: list[dict] = []

    def probing(self, nxt_all, hyper_all, cache, S, want_margin=False):
        if len(rows) < args.rounds:
            snap = spec_flash.snapshot_mtp_cache(cache)
            kv0 = cache.size()

            c_blk = spec_flash.restore_mtp_cache(snap)
            out_blk = _mtp_out(self, nxt_all[:, :S], hyper_all[:, :S], c_blk)

            c_seq = spec_flash.restore_mtp_cache(snap)
            outs = [
                _mtp_out(self, nxt_all[:, j:j + 1], hyper_all[:, j:j + 1], c_seq)
                for j in range(S)
            ]
            out_seq = mx.concatenate(outs, axis=1)
            mx.eval(out_blk, out_seq, c_blk.keys, c_blk.values,
                    c_seq.keys, c_seq.values)

            rec = {"S": S, "kv_before": kv0, "kv_after": kv0 + S, "rows": []}
            for j in range(S):
                same, d = _cmp(mx, out_blk[:, j:j + 1], out_seq[:, j:j + 1])
                t_blk = self._draft_argmax(out_blk[:, j:j + 1])
                t_seq = self._draft_argmax(out_seq[:, j:j + 1])
                mx.eval(t_blk, t_seq)
                rec["rows"].append({
                    "j": j, "bitident": same, "max_abs": d,
                    "tok_blk": int(t_blk.item()), "tok_seq": int(t_seq.item()),
                })
            n = kv0 + S
            rec["kv_same"], rec["kv_max_abs"] = _cmp(
                mx, c_blk.keys[..., :n, :], c_seq.keys[..., :n, :])
            rec["v_same"], rec["v_max_abs"] = _cmp(
                mx, c_blk.values[..., :n, :], c_seq.values[..., :n, :])
            rows.append(rec)
        return orig_presync(self, nxt_all, hyper_all, cache, S, want_margin)

    def run(label: str, prime_window: int) -> None:
        rows.clear()
        spec_flash.PRIME_WINDOW = prime_window
        spec_flash._DRAFT_PRESYNC = True
        spec_flash.FlashSpecEngine._presync_step0 = probing
        caches = model.make_cache()
        mx.clear_cache()
        gen = eng.generate_stream(ids, args.tokens, caches=caches, eos_ids=eos_ids)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass
        spec_flash.FlashSpecEngine._presync_step0 = orig_presync

        print(f"\n=== {label} (PRIME_WINDOW={prime_window}) ===", flush=True)
        if not rows:
            print("  ラウンドが 1 回も回らなかった (--tokens を増やすこと)")
            return
        qsa = rows[0]["kv_after"] > model.args.text.indexer_budget
        print(f"  MTP kv = {rows[0]['kv_before']} -> {rows[-1]['kv_after']}"
              f"  QSA {'発火' if qsa else '不発 (kv <= budget)'}")
        n_row = n_ident = n_tok = 0
        worst = 0.0
        for r in rows:
            for e in r["rows"]:
                n_row += 1
                n_ident += e["bitident"]
                n_tok += e["tok_blk"] == e["tok_seq"]
                worst = max(worst, e["max_abs"])
        print(f"  行 {n_row} 個: ビット一致 {n_ident}/{n_row}、"
              f"draft トークン一致 {n_tok}/{n_row}、max|diff| {worst:.3e}")
        kv_ident = sum(r["kv_same"] and r["v_same"] for r in rows)
        kv_worst = max(max(r["kv_max_abs"], r["v_max_abs"]) for r in rows)
        print(f"  キャッシュ k/v: ビット一致 {kv_ident}/{len(rows)}、"
              f"max|diff| {kv_worst:.3e}")
        for r in rows:
            cells = " ".join(
                f"j={e['j']}:{'=' if e['bitident'] else 'x'}"
                f"{'T' if e['tok_blk'] == e['tok_seq'] else 'F'}"
                f"({e['max_abs']:.1e})" for e in r["rows"])
            print(f"    S={r['S']} kv={r['kv_before']}  {cells}")

    saved_pw = spec_flash.PRIME_WINDOW
    run("条件 1: 実運用の priming 窓",
        args.prime_window if args.prime_window is not None else saved_pw)
    if args.qsa_prime_window:
        run("条件 2: QSA を踏ませる (MTP kv > indexer_budget)",
            args.qsa_prime_window)
    spec_flash.PRIME_WINDOW = saved_pw

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
