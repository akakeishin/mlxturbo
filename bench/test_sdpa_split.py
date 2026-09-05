"""sdpa 幅分割 (`MLXTURBO_SDPA_SPLIT`、`mlxturbo/_vendor/qwen4_exp.py` の
`Attention.__call__` / `_gather_tile_attn`) の正しさを CPU だけで検査する。

対象は decode/verify 幅 (``1 < S <= 8``) で ``S * gqa_factor > 32`` の
とき、q と mask を幅 ``max(1, 32 // gqa_factor)`` で S 軸に割って
``mx.fast.scaled_dot_product_attention`` を複数回呼び、
``mx.concatenate(axis=2)`` で戻す分岐 (docs/research/SDPA-WIDTH-WALL.md、
KERNEL-BRIEF-DECODE-BW.md)。モデルは組み立てない --- Flash-Next の形
(Hq=24, Hk=2, gqa_factor=12) を模した合成 q/k/v/mask に対して
``mx.fast.scaled_dot_product_attention`` を直接呼び、分割ありと分割なしの
出力を突き合わせる。

分割ありは分割なしと**厳密なビット一致は要求しない** --- 割ったほうは
S が小さいので vector カーネル、割らないほうは materialize 経路に落ちる
可能性があり (Metal バックエンドの適格判定、実際の分岐は GPU 側でしか
起きない)、選ばれるカーネル自体が違いうるので丸めが ulp オーダーで
ずれうる。ここで確かめるのは「同じ入力に対して同じ答えを返すこと」
(各クエリ行が独立という数学的な前提が壊れていないこと) で、許容誤差は
CLAUDE.md の「品質を売って速度を買わない」に合わせて bf16 の丸み
(1e-2) 以内に収める。

実行: .venv/bin/python bench/test_sdpa_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

# Flash-Next の形 (mlxturbo/_vendor/qwen4_exp.py の TextArgs 既定)。
N_HEADS = 24
N_KV_HEADS = 2
HEAD_DIM = 128
GQA = N_HEADS // N_KV_HEADS  # 12
STEP = max(1, 32 // GQA)  # 2 (`Attention.__call__` と同じ式)
TOL = 1e-2  # bf16 丸めの許容誤差 (CLAUDE.md: KLD 受け入れ幅 +0.0005 と同じ精神)


def _split_sdpa(q, k, v, mask, scale, step):
    """`Attention.__call__` の split_mask 分岐そのものの再現。"""
    S = q.shape[2]
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + step], k, v, scale=scale,
                mask=mask[..., i : i + step, :],
            )
            for i in range(0, S, step)
        ],
        axis=2,
    )


def _make_qkv(rng, B, S, kv_len, dtype):
    q = mx.array(rng.standard_normal((B, N_HEADS, S, HEAD_DIM)).astype(np.float32)).astype(dtype)
    k = mx.array(rng.standard_normal((B, N_KV_HEADS, kv_len, HEAD_DIM)).astype(np.float32)).astype(dtype)
    v = mx.array(rng.standard_normal((B, N_KV_HEADS, kv_len, HEAD_DIM)).astype(np.float32)).astype(dtype)
    return q, k, v


def _causal_bool_mask(offset, S, kv_len):
    """`_final_mask`/`__call__` の causal 文字列 -> bool 変換式そのもの。"""
    m = (
        mx.arange(kv_len)[None, :]
        <= (offset + mx.arange(S))[:, None]
    )[None, None]
    return m


def _qsa_style_bool_mask(rng, B, S, kv_len, offset):
    """QSA の `sparse & causal` に近い形の疎な bool マスク。

    各クエリ行に最低 1 本 (自分自身の対角、常に causal 内) は可視列を残す
    ---全 False の行があると softmax が -inf ばかりになり NaN で誤判定に
    化けるため、テストの入力としてそれを避ける。
    """
    causal = np.array(_causal_bool_mask(offset, S, kv_len))
    sparse = rng.random(causal.shape) < 0.3
    mask_np = causal & sparse
    # 対角 (自分自身) だけは常に可視にする (causal 上で必ず存在する列)。
    for s in range(S):
        col = offset + s
        if col < kv_len:
            mask_np[0, 0, s, col] = True
    return mx.array(mask_np)


def check_bool_mask_matches(dtype, tag) -> bool:
    ok = True
    rng = np.random.default_rng(0)
    for B in (1, 2):
        for S in (2, 3, 4, 6, 8):
            for kv_len in (16, 64, 512, 4096):
                offset = max(0, kv_len - S)
                scale = HEAD_DIM ** -0.5
                q, k, v = _make_qkv(rng, B, S, kv_len, dtype)
                mask = _qsa_style_bool_mask(rng, B, S, kv_len, offset)
                if B > 1:
                    mask = mx.broadcast_to(mask, (B, 1, S, kv_len))

                ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
                split = _split_sdpa(q, k, v, mask, scale, STEP)
                mx.eval(ref, split)

                diff = float(mx.max(mx.abs(ref.astype(mx.float32) - split.astype(mx.float32))))
                good = diff <= TOL
                ok &= good
                status = "OK" if good else "NG"
                print(
                    f"  [{tag}] B={B} S={S} kv={kv_len}: max|diff|={diff:.6g}"
                    f" (tol={TOL}) -> {status}"
                )
    return ok


def check_causal_string_vs_split_bool(dtype, tag) -> bool:
    """`__call__` の elif 分岐: mask="causal" (分割なし基準) と、そこから
    組んだ bool マスクを分割ありで通した結果が一致すること
    (kv_len>=512 のときだけ elif が発火する、`Attention.__call__` の条件)。
    """
    ok = True
    rng = np.random.default_rng(1)
    for S in (2, 3, 4, 6, 8):
        for kv_len in (512, 1024, 4096):
            offset = kv_len - S
            scale = HEAD_DIM ** -0.5
            q, k, v = _make_qkv(rng, 1, S, kv_len, dtype)

            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
            bool_mask = _causal_bool_mask(offset, S, kv_len)
            split = _split_sdpa(q, k, v, bool_mask, scale, STEP)
            mx.eval(ref, split)

            diff = float(mx.max(mx.abs(ref.astype(mx.float32) - split.astype(mx.float32))))
            good = diff <= TOL
            ok &= good
            status = "OK" if good else "NG"
            print(
                f"  [{tag}] S={S} kv={kv_len}: causal文字列 vs 分割bool"
                f" max|diff|={diff:.6g} (tol={TOL}) -> {status}"
            )
    return ok


def check_ineligible_widths_are_noop() -> bool:
    """S<=1 または S>8 は `__call__` の分岐に入らない (`1 < S <= 8` の外)。
    このスクリプトはその境界そのものは検査対象ではないが (分岐の選択は
    Python 側の if 文で、GPU も乱数も要らない)、境界に近い S=1 と S=9 でも
    split_sdpa 自身は壊れず動く (呼び出し側が使わないだけ) ことだけ確認する。
    """
    ok = True
    rng = np.random.default_rng(2)
    scale = HEAD_DIM ** -0.5
    for S in (1, 9, 16):
        kv_len = max(64, S)
        offset = kv_len - S
        q, k, v = _make_qkv(rng, 1, S, kv_len, mx.float32)
        mask = _causal_bool_mask(offset, S, kv_len)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        split = _split_sdpa(q, k, v, mask, scale, STEP)
        mx.eval(ref, split)
        diff = float(mx.max(mx.abs(ref - split)))
        good = diff <= TOL
        ok &= good
        print(f"  [境界] S={S} (壁の外): max|diff|={diff:.6g} -> {'OK' if good else 'NG'}")
    return ok


def main() -> int:
    print("=== sdpa 幅分割 (MLXTURBO_SDPA_SPLIT) の CPU 正しさ検査 ===")
    print(f"Hq={N_HEADS} Hk={N_KV_HEADS} gqa_factor={GQA} step={STEP} head_dim={HEAD_DIM}\n")
    ok = True

    print("-- bool マスク (QSA 疎マスク相当)、float32 --")
    ok &= check_bool_mask_matches(mx.float32, "fp32")
    print("\n-- bool マスク (QSA 疎マスク相当)、bfloat16 --")
    ok &= check_bool_mask_matches(mx.bfloat16, "bf16")

    print("\n-- causal 文字列 (分割なし基準) vs 分割 bool マスク、float32 --")
    ok &= check_causal_string_vs_split_bool(mx.float32, "fp32")
    print("\n-- causal 文字列 (分割なし基準) vs 分割 bool マスク、bfloat16 --")
    ok &= check_causal_string_vs_split_bool(mx.bfloat16, "bf16")

    print("\n-- 壁の外 (S<=1 or S>8) でも split_sdpa 自体は壊れない --")
    ok &= check_ineligible_widths_are_noop()

    print(f"\n=== 総合判定: {'合格' if ok else '不合格'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    raise SystemExit(main())
