"""段 P1 の融合カーネル (`mlxturbo/kernels/prefill_attn.py`) の正しさを見る。

`tools/verify_gather_attn.py` と同じ作りだが、あちらは CPU + 合成モデルで
gather 経路 (汎用 op 2 段) を見るのに対し、こちらは **GPU でカーネルを実際に
走らせる**。カーネルは `mx.fast.metal_kernel` なので CPU では動かない。

見るのは 2 段:

1. **配列レベル** (実モデルの形): head_dim 256 / n_heads 24 / kv head 2 /
   compress_ratio 4 で、`keep_block` を QSA の規約どおりに作った上で
   「カーネル」と「同じ集合を素の dense sdpa で計算したもの」を突き合わせる。
   fp32 で通す版 (丸めの水準まで一致するはず) と、本番の dtype である
   bf16 版の両方を踏む。端数ブロック (tail) が出る kv 長も別に踏む
2. **モデルレベル** (合成 Flash-Next、GPU): `tools/verify_gather_attn.py` と
   同じ呼び出し列を流し、カーネル on/off の logits を突き合わせる。
   QSA が不活性な域 (kv_len <= token_budget) はコードパスが完全一致するので
   ビット一致を要求する

**ビット一致は要求しない。**注意する集合は同じだが、加算順と (online softmax
の) スケーリング順が変わる。gather 経路自体が元々ビット一致しないのと同じ
前提 (`Attention._gather_forward` の docstring)。

使い方 (GPU を使うので biglock 経由で):

    tools/biglock.sh .venv/bin/python tools/verify_prefill_attn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import mlxturbo  # noqa: E402,F401
from mlxturbo import gather_attn  # noqa: E402
from mlxturbo.kernels import prefill_attn as PA  # noqa: E402
from verify_batch_cache import TINY, build  # noqa: E402

# fp32 で通したときの許容差。オンライン softmax は加算順が違うだけなので
# fp32 の丸め水準に収まるはず (`tools/vendor_fingerprint.py` の 5e-6 に合わせ、
# 列数が多いぶん少し緩める)。
TOL_F32 = 1e-4
# bf16 の 1 ulp は相対 2^-8 = 3.9e-3。列 2000 本ぶんの累積を見込んで
# **相対**で判定する (絶対値で切ると出力の大きさに依存して意味を失う)。
TOL_BF16_REL = 2e-2
# モデルレベル (合成モデル、fp32) の許容差。verify_gather_attn.py と同じ水準。
TOL_MODEL = 1e-4


def _keep_block(rng, B, S, n_blocks, cr, offset, block_topk):
    """QSA の規約どおりの ``keep_block`` を作る。

    `QSAIndexer._pooled_and_top` は ``block_end <= q_col`` を満たすブロック
    だけを top-k の候補にするので、可視ブロックの本数は ``(q_col+1)//cr``。
    そこから最大 ``block_topk`` 本を選ぶ。
    """
    keep = np.zeros((B, S, n_blocks), dtype=bool)
    for b in range(B):
        for s in range(S):
            q_col = offset + s
            n_vis = min(n_blocks, (q_col + 1) // cr)
            if n_vis <= 0:
                continue
            k = min(block_topk, n_vis)
            sel = rng.choice(n_vis, size=k, replace=False)
            keep[b, s, sel] = True
    return mx.array(keep)


def _dense_reference(q, k, v, keep_block, cr, kv_len, n_blocks, offset, scale):
    """同じ可視集合を素の dense sdpa で計算する (基準)。

    `QSAIndexer.__call__` と同じ手順でブロック bool をトークン幅へ展開し、
    tail を足す。tail の規則は `mlxturbo/qsa_tail.py` の ``MODE`` に従う
    (``global`` = 端数列を因果窓で / ``query`` = クエリごとに
    ``[cr*floor((q+1)/cr), q]``)。カーネル側も同じ knob を見るので、
    この基準は両方のモードで正しい相手になる。
    """
    from mlxturbo import qsa_tail as QT

    B, S, _ = keep_block.shape
    keep = mx.repeat(keep_block, cr, axis=-1)
    tail = kv_len - n_blocks * cr
    q_col = mx.arange(offset, offset + S)
    if QT.MODE == "query":
        cols = mx.arange(kv_len)
        own = ((q_col + 1) // cr) * cr
        keep = mx.concatenate(
            [keep, mx.zeros((B, S, tail), dtype=mx.bool_)], axis=-1
        ) if tail else keep
        own_keep = (cols[None, :] >= own[:, None]) & (cols[None, :] <= q_col[:, None])
        keep = keep | own_keep[None]
    elif tail:
        tail_col = n_blocks * cr + mx.arange(tail)
        keep = mx.concatenate(
            [
                keep,
                mx.broadcast_to(
                    tail_col[None, None, :] <= q_col[None, :, None], (B, S, tail)
                ),
            ],
            axis=-1,
        )
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=keep[:, None]
    )
    return out.transpose(0, 2, 1, 3)  # (B, S, H, D)


def check_array(dtype, n_heads, n_kv, head_dim, cr, block_topk, offset, S, tag):
    """配列レベル: カーネル vs 同じ集合の dense sdpa。"""
    from mlx_lm.models.cache import KVCache

    rng = np.random.default_rng(0)
    B = 1
    kv_len = offset + S
    n_blocks = kv_len // cr
    tail = kv_len - n_blocks * cr

    mx.random.seed(0)
    q = (mx.random.normal((B, n_heads, S, head_dim)) * 0.3).astype(dtype)
    k = (mx.random.normal((B, n_kv, kv_len, head_dim)) * 0.3).astype(dtype)
    v = (mx.random.normal((B, n_kv, kv_len, head_dim)) * 0.3).astype(dtype)
    keep_block = _keep_block(rng, B, S, n_blocks, cr, offset, block_topk)
    scale = head_dim ** -0.5

    # 本番と同じく、KV はキャッシュの途中切りビューとして渡す
    cache = KVCache()
    k_view, v_view = cache.update_and_fetch(k, v)

    ok_elig = PA.eligible(
        q, k_view, v_view, keep_block, cache, cr, kv_len, n_blocks, block_topk
    )
    if not ok_elig:
        print(f"  {tag}: eligible=False (カーネルが引き受けられない形)")
        return False

    got = PA.prefill_attn(
        q, k_view, v_view, keep_block, cache,
        cr=cr, kv_len=kv_len, n_blocks=n_blocks, block_topk=block_topk,
        offset=offset, scale=scale,
    )
    ref = _dense_reference(
        q, k_view, v_view, keep_block, cr, kv_len, n_blocks, offset, scale
    )
    mx.eval(got, ref)

    a = got.astype(mx.float32)
    b = ref.astype(mx.float32)
    diff = float(mx.max(mx.abs(a - b)))
    scale_ref = float(mx.max(mx.abs(b)))
    rel = diff / scale_ref if scale_ref > 0 else diff
    tol = TOL_F32 if dtype == mx.float32 else None
    if tol is not None:
        ok = diff <= tol
        verdict = f"max|diff|={diff:.3e} (許容 {tol:.0e})"
    else:
        ok = rel <= TOL_BF16_REL
        verdict = (
            f"max|diff|={diff:.3e} 相対={rel:.3e} (許容 {TOL_BF16_REL:.0e})"
        )
    print(
        f"  {tag}: kv_len={kv_len} n_blocks={n_blocks} tail={tail} "
        f"{verdict} -> {'合格' if ok else '不合格'}"
    )
    return ok


def _run_model(model, mode, calls):
    """mode: 'off' 通常経路 / 'gather' 汎用 op 2 段 / 'kernel' 融合カーネル。"""
    gather_attn.disable_gather_attn(model)
    if mode == "gather":
        gather_attn.enable_gather_attn(model)
    elif mode == "kernel":
        gather_attn.enable_prefill_attn(model)
    cache = model.make_cache()
    out = []
    for ids in calls:
        logits = model(mx.array(ids)[None], cache=cache)
        mx.eval(logits)
        out.append(logits)
    return out


def _build_calls(vocab):
    """verify_gather_attn.py と同じ割り方 (prefill 幅チャンクと decode 幅)。"""
    ids = [(i * 7 + 3) % vocab for i in range(102)]
    calls = []
    body = ids[:59]
    for lo in range(0, len(body), 8):
        calls.append(body[lo : lo + 8])
    calls.append(ids[59:91])          # prefill 幅チャンク (S=32)
    for i in range(91, 95):
        calls.append([ids[i]])
    calls.append(ids[95:97])          # verify 幅を模した S=2
    for i in range(97, 102):
        calls.append([ids[i]])
    return calls


def check_model(budget=8):
    """モデルレベル: 合成 Flash-Next を GPU で流して on/off を比べる。"""
    model = build(budget)
    calls = _build_calls(TINY["vocab_size"])

    base = _run_model(model, "off", calls)

    # カーネルが 1 度も走らないまま「一致」を出さないよう、実際の呼び出しを数える
    # (適格でなければ `_gather_forward` は黙って既存のタイル経路へ落ちる)
    fired = [0]
    orig = PA.prefill_attn

    def counted(*a, **kw):
        fired[0] += 1
        return orig(*a, **kw)

    PA.prefill_attn = counted
    try:
        kern = _run_model(model, "kernel", calls)
    finally:
        PA.prefill_attn = orig

    ok = True
    print(f"  カーネルが走った回数: {fired[0]}")
    if fired[0] == 0:
        print("  ★カーネルが一度も走っていない (適格判定で落ちている)★")
        ok = False
    offset = 0
    for i, (ids, a, b) in enumerate(zip(calls, base, kern)):
        S = len(ids)
        d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        tag = "一致(bit)" if d == 0.0 else f"max|diff|={d:.3e}"
        # QSA が不活性な域はコードパスが完全一致するのでビット一致を要求する
        inactive = offset + S <= budget
        if inactive and d != 0.0:
            ok = False
            tag += " ★QSA 不活性域でビット一致していない★"
        if d > TOL_MODEL:
            ok = False
        print(f"  call {i:2d}  offset={offset:3d} S={S:2d}  {tag}")
        offset += S
    print(f"  判定: {'合格' if ok else '不合格'} (許容 {TOL_MODEL:.0e})")
    return ok


def main() -> int:
    if not mx.metal.is_available():
        print("Metal が使えないのでこの検査は走らせられない")
        return 1
    mx.set_default_device(mx.gpu)

    print("=== 配列レベル (実モデルの形: n_heads 24 / kv 2 / head_dim 256 /"
          " compress_ratio 4) ===")
    ok = True
    # fp32: 丸めの水準まで一致するはず。端数ブロックの有無を両方踏む
    ok &= check_array(mx.float32, 24, 2, 256, 4, 128, 4096, 64, "fp32 tail=0")
    ok &= check_array(mx.float32, 24, 2, 256, 4, 128, 4095, 64, "fp32 tail>0")
    # bf16: 本番の dtype。判定は相対誤差
    ok &= check_array(mx.bfloat16, 24, 2, 256, 4, 128, 4096, 64, "bf16 tail=0")
    ok &= check_array(mx.bfloat16, 24, 2, 256, 4, 512, 8192, 64, "bf16 budget2048")

    print("\n=== モデルレベル (合成 Flash-Next、GPU、fp32) ===")
    ok &= check_model()

    print(f"\n=== 総合判定: {'合格' if ok else '不合格'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
