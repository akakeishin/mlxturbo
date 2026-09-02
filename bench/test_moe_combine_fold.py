"""SparseMoeBlock._combine_fold (mlxturbo/_vendor/qwen4_exp.py) の数値検査。

MLXTURBO_MOE_COMBINE_FOLD=1 で有効になる経路 (ルータ重み w を down_proj の
入力 (SwiGLU 出力、(rows, moe_intermediate_size)) に先掛けしてから down_proj
を通し、top_k 軸の和は down_proj の出力側 (rows, hidden_size) で取る) が、
素の経路 `(switch_mlp(x, idx) * w[..., None]).sum(-2)` (switch_mlp の出力を
先に実体化してから w を掛けて和を取る) と数式上は同じ値になることを確認
する。down_proj は bias 無しの線形写像なので、w を掛ける位置を前後に動かし
ても値は変わらないはず -- 実際にずれるとすれば量子化 4bit + bf16 活性の
積和順が変わることによる丸めだけ。

`tools/vendor_fingerprint.py` と同じ作り方 (合成した小さい Flash-Next 形状、
CPU、乱数固定、乱数分布は N(0, 0.05^2)) で SparseMoeBlock を単体で作る。
switch_mlp の 3 射影 (gate/up/down) だけ量子化して本番の量子化+bf16 の丸めを
再現し、router/shared_expert は密 bf16 のまま (本番の量子化構成と同じ切り分
け)。並べ替え閾値 `MLXTURBO_SORT_MIN` をまたぐ複数のトークン数
(未ソート経路・ソート経路の両方、decode 1 トークン相当・verify 幅相当・
prefill 相当) を全部検査する。

判定は RMS 相対誤差 <= 1e-2 (`_rel_err` 参照)。要素ごとの
|got-ref|/|ref| ではなく RMS で正規化するのは、top_k=2 の専門家出力が
router 重みでほぼ 0.5/0.5 に混ぜられて符号違いでほぼ相殺する要素がある
ため (分母がほぼ 0 になり、要素ごとの相対誤差だと発散する。実装のバグでは
なく `mlxturbo/kernels/moe_verify_gather.py` の `_max_rel_err` と同じ理由)。

実行: uv run python bench/test_moe_combine_fold.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401 -- sys.meta_path フックを入れて mlx_lm.models.qwen4_exp を
# mlxturbo/_vendor/qwen4_exp.py へリダイレクトする (相対 import `.base` 等は
# mlx_lm.models パッケージ配下として import しないと解決しないので、
# mlxturbo._vendor.qwen4_exp を直接 import してはいけない)。
from mlx_lm.models import qwen4_exp as Q  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

# 相対誤差の許容 (量子化 4bit + bf16 の丸みを見込んだ緩め、依頼の基準どおり)
TOL = 1e-2
GROUP_SIZE = 32
BITS = 4
HIDDEN = 64
MOE_HIDDEN = 32
NUM_EXPERTS = 8
TOP_K = 2


def _build_block(seed: int = 0) -> "Q.SparseMoeBlock":
    mx.random.seed(seed)
    args = Q.TextArgs(
        hidden_size=HIDDEN,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_HIDDEN,
        shared_expert_intermediate_size=MOE_HIDDEN,
    )
    mlp = Q.SparseMoeBlock(args)
    # 既定初期化は形ごとに違う一様分布 (nn.Linear 等) なので、
    # tools/vendor_fingerprint.py と同じ流儀 (verify_batch_cache.build) で
    # N(0, 0.05^2) に置き直してから bf16 に落とす (本番の重み dtype)。
    mlp.update(
        tree_map(
            lambda a: (mx.random.normal(a.shape) * 0.05).astype(mx.bfloat16)
            if a.dtype == mx.float32
            else a,
            mlp.parameters(),
        )
    )
    # switch_mlp (experts) だけ 4bit 量子化する。router (self.gate) と
    # shared_expert は本番同様に密 bf16 のまま (enable_moe_shared_fold は
    # 既定で呼ばれていないので、この検査でも r513 分岐は使わない)。
    sw = mlp.switch_mlp
    sw.gate_proj = sw.gate_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    sw.up_proj = sw.up_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    sw.down_proj = sw.down_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    mx.eval(mlp.parameters())
    mlp.eval()
    return mlp


def _rel_err(got: mx.array, ref: mx.array) -> float:
    """相対誤差 (RMS)。`ref` の RMS で割る (`tools/verify_gdn_metal.py` /
    `tools/verify_gdn_blocked.py` の `_rel_err` と同じ流儀)。

    単純な要素ごとの |got-ref|/|ref| は使わない -- top_k=2 個の専門家の
    出力が符号違いでほぼ相殺する要素 (router 重みがほぼ 0.5/0.5 のとき、
    top_k 軸の和がたまたま 0 付近に来る) では分母が潰れて発散する
    (`mlxturbo/kernels/moe_verify_gather.py` の `_max_rel_err` の docstring
    と同じ理由 -- 実装のバグではなく、bf16 蓄積誤差の絶対的な床が
    ほぼ 0 の出力要素にそのまま乗るだけ)。RMS で正規化すれば、少数の
    ほぼ 0 の要素に引きずられずに全体としての一致度を見られる。
    """
    got32 = got.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    d = got32 - ref32
    scale = math.sqrt(float(mx.mean(ref32 * ref32)))
    if scale == 0.0:
        scale = 1.0
    return math.sqrt(float(mx.mean(d * d))) / scale


def _check(seq_len: int, sort_min: int, batch: int = 1, seed: int = 0) -> bool:
    mlp = _build_block(seed=seed)
    x = (mx.random.normal((batch, seq_len, HIDDEN)) * 0.1).astype(mx.bfloat16)

    # _moe_combine_fold は実行時に一度だけ読んだモジュール定数
    # _MOE_COMBINE_SORT_MIN を見る (MLXTURBO_SORT_MIN の値そのもの)。
    # ここでは env var を経由せず、その定数を直接この検査の間だけ差し替えて
    # 未ソート/ソートの両方の分岐を確実に踏む。
    old_sort_min = Q._MOE_COMBINE_SORT_MIN
    Q._MOE_COMBINE_SORT_MIN = sort_min
    try:
        mlp._combine_fold = False
        ref = mlp(x)
        mlp._combine_fold = True
        got = mlp(x)
        mx.eval(ref, got)
    finally:
        Q._MOE_COMBINE_SORT_MIN = old_sort_min

    err = _rel_err(got, ref)
    do_sort = batch * seq_len * TOP_K >= sort_min
    ok = err <= TOL
    print(
        f"batch={batch} seq_len={seq_len:>3} top_k={TOP_K} sort_min={sort_min:>2} "
        f"do_sort={do_sort!s:<5} rel_err(rms)={err:.3e}  {'OK' if ok else 'FAILED'}"
    )
    return ok


def main() -> int:
    ok = True
    # decode 1 トークン (S=1): idx.size=2 < 16、既定 sort_min のまま未ソート
    ok &= _check(seq_len=1, sort_min=16)
    # decode の小さいバッチ: idx.size=8 < 16、未ソート
    ok &= _check(seq_len=4, sort_min=16)
    # 検証 (verify) 幅相当: idx.size=6、sort_min を下げてソート経路を強制
    # (本番の既定 MLXTURBO_SORT_MIN=16 が verify 幅 T=2..4 を拾うのと同じ形)
    ok &= _check(seq_len=3, sort_min=4)
    # prefill 相当の大きいバッチ: idx.size=40 >= 16、ソート経路
    ok &= _check(seq_len=20, sort_min=16)
    # 境界ちょうど (idx.size == sort_min): ソート経路に入る側
    ok &= _check(seq_len=8, sort_min=16)
    # バッチ検証形状 (B, T, top_k)。fused.py の moe_verify 実装コメントが
    # 警告する通り、`indices.shape[-2]` ではなく `indices.size` から
    # トークン数を出さないと壊れる形 -- combine-fold も同じ罠を踏みうる
    # ので、B>1 も別に検査する。idx.size=2*3*2=12 < 16 で未ソート。
    ok &= _check(seq_len=3, sort_min=16, batch=2)
    # 同じバッチ形状でソート経路も踏む。idx.size=2*20*2=80 >= 16。
    ok &= _check(seq_len=20, sort_min=16, batch=2)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
