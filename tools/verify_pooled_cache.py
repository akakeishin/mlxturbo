"""pooled キーの増分キャッシュ (段 X1、`docs/research/KERNEL-PROGRAM.md`、
実体は `mlxturbo/_vendor/qwen4_exp.py` の `_IndexerCache.pooled` /
`QSAIndexer._pooled_and_top`、口は `mlxturbo/pooled_cache.py`) の正しさを、
合成した小さい Flash-Next を CPU で流して確かめる。

見るのは 2 つ:

    1. 増分キャッシュ (既定 on) と毎回全ブロック作り直し (旧経路、
       `pooled_cache.disable_pooled_cache`) が、同じ入力に対して
       **ビット一致**すること (段 X1 の反転条件どおり、ここで差が出るのは
       実装のバグ)。QSA が不活性な域 (kv_len<=budget) から活性域への遷移、
       端数ブロック (tail)、decode 幅 (S=1)、verify 幅 (S=2) を混ぜて踏む。
    2. 縮み (`mlxturbo.spec_flash.trim_attn_cache` と同じ経路、
       `_IndexerCache.keys` の setter を通る) のあとで pooled キャッシュが
       きちんと捨てられ、次のフォワードが古いブロックを静かに使い回さない
       こと。ロールバック後に続けた出力が「最初からその長さまでしか
       進めていない参照」と一致することを確認する。

使い方:

    .venv/bin/python tools/verify_pooled_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401
from mlxturbo import pooled_cache  # noqa: E402
from mlxturbo.spec_flash import trim_attn_cache  # noqa: E402
from verify_batch_cache import TINY, build  # noqa: E402


def _calls(vocab: int):
    """budget=8, compress_ratio=2 (TINY 既定) を前提にした呼び出し列。

    - call 0: kv_len=8 == budget。QSA 自体が不活性 (早期 return)
    - call 1..4 (各 S=8): kv_len が budget を超えて QSA 活性。tail=0
    - call 5 (S=3): kv_len=43 (奇数) になり tail=1 を作る
    - call 6..9 (S=1 x4): decode 幅
    - call 10 (S=2): verify 幅を模す
    - call 11..15 (S=1 x5): decode 幅
    """
    ids = [(i * 7 + 3) % vocab for i in range(54)]
    calls = [ids[0:8]]
    for lo in range(8, 40, 8):
        calls.append(ids[lo : lo + 8])
    calls.append(ids[40:43])
    for i in range(43, 47):
        calls.append([ids[i]])
    calls.append(ids[47:49])
    for i in range(49, 54):
        calls.append([ids[i]])
    return calls


def check_bit_identical(budget: int) -> bool:
    print(f"=== 増分 vs 毎回作り直し (budget={budget}) ===")
    model_a = build(budget)
    model_b = build(budget)
    pooled_cache.enable_pooled_cache(model_a)
    pooled_cache.disable_pooled_cache(model_b)

    calls = _calls(TINY["vocab_size"])
    cache_a = model_a.make_cache()
    cache_b = model_b.make_cache()

    offset = 0
    ok = True
    for i, ids in enumerate(calls):
        la = model_a(mx.array(ids)[None], cache=cache_a)
        lb = model_b(mx.array(ids)[None], cache=cache_b)
        mx.eval(la, lb)
        same = bool(mx.all(la == lb))
        d = float(mx.max(mx.abs(la.astype(mx.float32) - lb.astype(mx.float32))))
        tag = "一致(bit)" if same else f"max|diff|={d:.3e}"
        print(f"  call {i:2d}  offset={offset:3d} S={len(ids):2d}  {tag}")
        ok = ok and same
        offset += len(ids)

    print(f"  判定: {'合格' if ok else '不合格'}\n")
    return ok


def check_rollback_invalidates(budget: int) -> bool:
    """QSA を活性域まで進めて pooled キャッシュを積んだあと、
    `trim_attn_cache` (spec_flash が投機のロールバックに使う経路そのもの)
    で縮める。縮めた直後に pooled キャッシュが空になっていること、続けた
    出力が「最初からその長さまでしか進めていない参照」と一致することを見る。

    フルモデルではなく ``Attention`` 単体を直接呼ぶ (GDN の再帰状態は
    ``trim_attn_cache`` の対象外 -- ロールバックするのは attention 側の
    KV と indexer だけなので、GDN 込みでフルモデルを比較すると GDN 側の
    ずれが混ざって pooled キャッシュ自体の正しさを覆い隠してしまう)。
    """
    print(f"=== rollback 後の pooled キャッシュ無効化 (budget={budget}) ===")
    import mlx_lm.models.qwen4_exp as Q

    model = build(budget)
    layer = next(l for l in model.model.layers if l.layer_type != "linear_attention")
    attn = layer.self_attn
    rope = model.model.rope
    hidden_size = TINY["hidden_size"]

    # 固定シードで作った 1 本の隠れ状態列を、両方の実行が同じ位置では
    # 必ず同じ値になるように切り出して使う (チャンクの割り方が変わっても
    # 中身は不変)。
    mx.random.seed(1)
    x_all = mx.random.normal((1, 60, hidden_size)) * 0.05
    mx.eval(x_all)

    keep, grow_to, tail_len = 32, 40, 20

    def run(cache, x):
        mask = Q.create_attention_mask(x, [cache])
        return attn(x, rope, mask, cache, cache.indexer)

    # 本線: grow_to まで進めて pooled を積んでから keep へロールバック
    cache = Q._AttnCache()
    got = run(cache, x_all[:, :grow_to])
    mx.eval(got)
    assert cache.indexer._pooled is not None, "ここまでで pooled が積まれているはず"
    pooled_before = cache.indexer._pooled_n

    trim_attn_cache(cache, keep)  # 40 -> 32 (indexer.keys の setter を通る)
    pooled_reset = cache.indexer._pooled is None and cache.indexer._pooled_n == 0
    print(
        f"  rollback 前の pooled ブロック数: {pooled_before}、"
        f"rollback 直後に空になっている: {pooled_reset}"
    )

    for i in range(keep, keep + tail_len):
        got = run(cache, x_all[:, i : i + 1])
        mx.eval(got)

    # 参照: 最初から keep=32 までしか進めていない、同じ Attention インスタンス
    ref_cache = Q._AttnCache()
    run(ref_cache, x_all[:, :keep])
    ref_got = None
    for i in range(keep, keep + tail_len):
        ref_got = run(ref_cache, x_all[:, i : i + 1])
        mx.eval(ref_got)

    same = bool(mx.all(got == ref_got))
    d = float(mx.max(mx.abs(got.astype(mx.float32) - ref_got.astype(mx.float32))))
    print(
        "  rollback 後の続きが「最初から keep=32 の参照」と一致: "
        f"{'一致(bit)' if same else f'max|diff|={d:.3e}'}\n"
    )
    return pooled_reset and same


def main() -> int:
    ok1 = check_bit_identical(budget=8)
    ok2 = check_rollback_invalidates(budget=8)
    all_ok = ok1 and ok2
    print(f"=== 総合判定: {'合格' if all_ok else '不合格'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
