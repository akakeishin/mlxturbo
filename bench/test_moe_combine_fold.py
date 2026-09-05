"""SparseMoeBlock の MoE combine-fold (mlxturbo/_vendor/qwen4_exp.py の
`_moe_combine_fold` / `_combine_fold_min_s`) の数値検査。

MLXTURBO_MOE_COMBINE_FOLD (既定 on、`=0` で無効化) で有効になる経路
(ルータ重み w を down_proj の入力 (SwiGLU 出力、(rows, moe_intermediate_size))
に先掛けしてから down_proj を通し、top_k 軸の和は down_proj の出力側
(rows, hidden_size) で取る) が、素の経路
`(switch_mlp(x, idx) * w[..., None]).sum(-2)` (switch_mlp の出力を先に
実体化してから w を掛けて和を取る) と数式上は同じ値になることを確認する。
down_proj は bias 無しの線形写像なので、w を掛ける位置を前後に動かしても
値は変わらないはず -- 実際にずれるとすれば量子化 4bit + bf16 活性の積和順
が変わることによる丸めだけ。

行数ゲート: 初回の in-model A/B (2026-09-03、行数ゲート無し) で prefill は
勝った (8k -2.2%/17k -2.5%) が decode は負けた (8k +1.3%/17k +0.6%/短文脈
ms/round +1.4%、tok/round -4.0%)。そこで行数 (B×S) が
`MLXTURBO_MOE_COMBINE_FOLD_MIN_S` (既定 64、`enable_moe_combine_fold` が
起動時に 1 回だけ読んで `_combine_fold_min_s` に積む) 未満のときは必ず
素の経路に落ちる (decode/verify 幅 S<=8 はここに入る)。この検査は
「閾値未満はビット一致 (素の経路そのもの)」と「閾値以上は RMS 相対誤差
<= 1e-2」の両方を、`_combine_fold_min_s` を直接差し替えて確認する
(env var は経由しない -- `enable_moe_combine_fold`/`disable_moe_combine_fold`
はモデル全体を組んでからでないと呼べないので、単体の SparseMoeBlock に
対してはそれらが属性へ積むのと同じ値を直接セットする)。

`tools/vendor_fingerprint.py` と同じ作り方 (合成した小さい Flash-Next 形状、
CPU、乱数固定、乱数分布は N(0, 0.05^2)) で SparseMoeBlock を単体で作る。
switch_mlp の 3 射影 (gate/up/down) だけ量子化して本番の量子化+bf16 の丸めを
再現し、router/shared_expert は密 bf16 のまま (本番の量子化構成と同じ切り分
け)。行数ゲートの境界 (閾値未満/以上) と、fold 内部の並べ替え閾値
`MLXTURBO_SORT_MIN` の境界 (未ソート経路・ソート経路の両方、decode 1
トークン相当・verify 幅相当・prefill 相当、batch>1 の (B,S,top_k) 形状)
を全部検査する。

閾値以上の判定は RMS 相対誤差 <= 1e-2 (`_rel_err` 参照)。要素ごとの
|got-ref|/|ref| ではなく RMS で正規化するのは、top_k=2 の専門家出力が
router 重みでほぼ 0.5/0.5 に混ぜられて符号違いでほぼ相殺する要素がある
ため (分母がほぼ 0 になり、要素ごとの相対誤差だと発散する。実装のバグでは
なく `mlxturbo/kernels/moe_verify_gather.py` の `_max_rel_err` と同じ理由)。
閾値未満の判定はビット一致 (同じコード経路を通るだけなので diff は厳密に
0 になるはず)。

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


def _check(seq_len: int, min_s: int, sort_min: int, batch: int = 1, seed: int = 0) -> bool:
    """min_s: `_combine_fold_min_s` 相当 (行数 B×S と比較する行数ゲート)。
    sort_min: `_MOE_COMBINE_SORT_MIN` 相当 (fold が発火したときだけ意味を
    持つ、fold 内部の並べ替え閾値)。"""
    mlp = _build_block(seed=seed)
    x = (mx.random.normal((batch, seq_len, HIDDEN)) * 0.1).astype(mx.bfloat16)
    rows = batch * seq_len
    do_fold = rows >= min_s

    # _moe_combine_fold は実行時に一度だけ読んだモジュール定数
    # _MOE_COMBINE_SORT_MIN を見る (MLXTURBO_SORT_MIN の値そのもの)。
    # ここでは env var を経由せず、その定数を直接この検査の間だけ差し替えて
    # 未ソート/ソートの両方の分岐を確実に踏む (fold が発火する場合のみ意味
    # を持つ、行数ゲートで素の経路に落ちるときは参照しない)。
    old_sort_min = Q._MOE_COMBINE_SORT_MIN
    Q._MOE_COMBINE_SORT_MIN = sort_min
    try:
        # 参照値は行数ゲートを完全に外した (素の経路しか使わない) 状態。
        mlp._combine_fold_min_s = None
        ref = mlp(x)
        # enable_moe_combine_fold が積むのと同じ形 (閾値そのものを属性に
        # 持つ) で行数ゲートを掛ける。
        mlp._combine_fold_min_s = min_s
        got = mlp(x)
        mx.eval(ref, got)
    finally:
        Q._MOE_COMBINE_SORT_MIN = old_sort_min

    if do_fold:
        err = _rel_err(got, ref)
        ok = err <= TOL
        print(
            f"batch={batch} seq_len={seq_len:>3} rows={rows:>3} min_s={min_s:>3} "
            f"sort_min={sort_min:>2} do_fold=True  rel_err(rms)={err:.3e}"
            f"  {'OK' if ok else 'FAILED'}"
        )
    else:
        # 閾値未満は SparseMoeBlock.__call__ が use_fold=False の分岐 (素の
        # 経路そのもの) を通るはずなので、参照値とビット一致する。
        diff = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))))
        ok = diff == 0.0
        print(
            f"batch={batch} seq_len={seq_len:>3} rows={rows:>3} min_s={min_s:>3} "
            f"sort_min={sort_min:>2} do_fold=False bit_exact_diff={diff:.3e}"
            f"  {'OK' if ok else 'FAILED'}"
        )
    return ok


def main() -> int:
    ok = True

    # --- 行数ゲート未満 (既定 MIN_S=64 相当): 必ず素の経路、ビット一致 ---
    # decode 1 トークン (rows=1)
    ok &= _check(seq_len=1, min_s=64, sort_min=16)
    # decode/verify 幅の上限相当 (rows=8、runner.py のコメントの S<=8)
    ok &= _check(seq_len=8, min_s=64, sort_min=16)
    # 境界のすぐ下 (rows=63)
    ok &= _check(seq_len=63, min_s=64, sort_min=16)
    # batch を含めた行数で判定していること (B×S=2*31=62 < 64)。
    # S だけで判定していたら誤って fold してしまう形。
    ok &= _check(seq_len=31, min_s=64, batch=2, sort_min=16)

    # --- 行数ゲート境界ちょうど・以上: fold 発火、RMS 相対誤差で判定 ---
    # 境界ちょうど (rows=64 == min_s)。既定 sort_min=16 で idx.size=128 -> ソート
    ok &= _check(seq_len=64, min_s=64, sort_min=16)
    # batch 込みで境界ちょうど (B×S=2*32=64)。S だけなら未満に見える形。
    ok &= _check(seq_len=32, min_s=64, batch=2, sort_min=16)
    # prefill 相当の大きいバッチ
    ok &= _check(seq_len=100, min_s=64, sort_min=16)
    # 閾値を下げて decode 幅でも fold を強制発火させ、fold 内部の未ソート
    # 経路 (idx.size=4*2=8 < sort_min=16) を検査する。
    ok &= _check(seq_len=4, min_s=4, sort_min=16)
    # 同じく閾値を下げて、fold 内部のソート経路 (verify 幅相当、
    # idx.size=8*2=16 >= sort_min=4) を検査する。
    ok &= _check(seq_len=8, min_s=4, sort_min=4)

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    raise SystemExit(main())
