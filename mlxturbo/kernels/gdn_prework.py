"""GDN の前処理 (conv1d 以降・再帰カーネル手前) を 1 本の Metal カーネルに畳む。

`GatedDeltaNet.__call__` は再帰カーネル本体 (`gated_delta_kernel`) の手前に
別々の dispatch が並ぶ:

    conv1d -> silu -> q の rms_norm+スケール -> k の rms_norm+スケール
    -> 次段 conv 状態の書き出し -> g (compute_g) -> beta (sigmoid)

decode/verify 幅 (S が小さい) では 1 op あたりの要素数がごく少なく、
`hyper_connection.py` / `rms_norm_gated.py` と同じ「起動回数そのものが
コストの大半」というディスパッチ律速の構図になる。このモジュールはこの並びを
1 回の起動に畳む。

## 適用範囲

decode/verify 幅だけ (:data:`MAX_S` / :data:`MAX_M`)。prefill 幅は対象外で、
`GatedDeltaNet.__call__` 側の既存経路 (conv1d -> silu -> rms_norm -> ...) が
そのまま使われる。単一系列 (`mask is None`、右パディング無し) のみを扱う。
バッチの右パディング (`cache.lengths` あり) は `_tail_window` の分岐が要る
ため対象外にしてある -- ここは decode のホットパスを削る話で、右パディング
バッチ decode は別の頻度の低い経路。

## スレッド配置 (2026-09-03 に書き直した。旧版が負けた理由もここ)

**旧版は 1 threadgroup = 1 トークン位置 (`m = b*S + s`) だった。**S=1 の decode
では threadgroup がちょうど 1 個しか立たず、40 コアのうち 1 コアで 250 KB を
読むことになる。冷えた DRAM のレイテンシを隠す並列度が無く、in-model で
+5.3% 遅い側に出た (帰属は 2026-09-03 12:00、`custom_kernel_overhead_micro.py`
の「重みを巡回させると融合カーネルだけ +74%」と同じ形)。

**現行はチャネル方向にも割る。**1 threadgroup = (1 トークン位置, 連続する
:data:`BLOCK` チャネル)。`conv_dim` を BLOCK で割った本数だけ threadgroup が
立つので、S=1 でも 40〜80 個 (BLOCK=256 / 128) になり、全コアに散る。

- 1 スレッド = 1 チャネル。`conv_w` の {K} タップは 1 チャネルぶんが連続
  (8 B) で、隣のスレッドは隣のチャネルを読むので load は全部まとまる。
- BLOCK は `dk` (= `dv` = 128) の倍数に取る。q/k の 1 head は必ず 1 つの
  threadgroup の中に収まるので、rms_norm の縮約が threadgroup 内で閉じる
  (simdgroup ごとに `simd_sum` -> threadgroup メモリの部分和 -> バリア 1 回)。
  **旧版のような threadgroup メモリへの q/k 全体の退避 (tg_q/tg_k) は要らない。**
- g・beta は専用の最終ブロック (`blk == n_blocks`) が受け持つ。conv の
  ブロックに相乗りさせると、そのブロックだけ仕事が増えて末尾が伸びる。
- 次段の conv 状態 (K-1 列) は全 s で同じ値なので `s == S-1` の行だけが書く。
  チャネルの持ち主は 1 threadgroup だけなので競合しない。

## 精度

conv1d は fp32 で溜めてから T (bf16/fp16) に丸める (`mx.conv1d` と同じ丸め
回数であることを実測で確認: 手書きの 4 タップ総和は mx.conv1d と fp32 で
2.98e-8 以内)。q/k の rms_norm は「参照が bf16 実体化する箇所」を 2 回とも
再現する (`bf16(bf16(x*rsqrt) * scale)`、`rms_norm_gated.py` と同じ形)。

**silu / beta / g の丸めは MLX 本体の式をそのまま写す** (2026-09-03)。
旧版は sigmoid を `1/(1+exp(-x))` と書いていて bf16 で 1 ulp ずれ、softplus も
fp32 で通していたので実モデル (`A_log`/`dt_bias` は bf16) と別の値になっていた。
現行は

- sigmoid: `unary_ops.h` の `Sigmoid` (`y=1/(1+exp(|x|))`, `x<0 ? y : 1-y`) の写し。
  `hyper_connection.py` が bf16 全 65536 パターンで一致を確認したのと同じ式。
- silu: `nn.silu` = `x * sigmoid(x)` の 2 段丸め (`T(T(x) * T(sigmoid(x)))`)。
- softplus: `binary_ops.h` の `LogAddExp` (`max + log1p(exp(min-max))`) を
  **T のまま** 1 段ずつ丸めて写す。`metal::exp(bfloat)` は
  `bf16_math.h` の定義どおり float で計算して bf16 へ戻る = 各段で丸まる。
- g の外側 2 つの exp だけ `metal::precise::exp` (本体の `Exp` は
  `unary_ops.h:177` で precise を明示している)。softplus の中の exp は
  素の `metal::exp`。
- `A_log`/`dt_bias` は実モデルの dtype (bf16) のまま受ける。**fp32 の写しは
  もう要らない** (旧版は fp32 限定だったので `enable_gdn_prework_kernel` が
  `_A_log_f32` を作っていた。渡されると `a + dt_bias` が fp32 になって素と
  別の値になるので、いまは `eligible()` が dtype 不一致として断る)。

**丸め位置の総当たり (2026-09-03、196608 標本、`compute_g` との突き合わせ)**:
どこで丸めるかを 1 つ取り違えるだけで黙って食い違う。実測した外れ方 --

| 変種 | 素との不一致 |
|---|---|
| log1p の結果を float のまま足す | 2.99% |
| softplus 内の exp を float / precise / fast にする | 15.45% |
| 外側の exp を素の `metal::exp` にする | 32.68% |
| **上を全部 T で丸め、外側だけ precise** | **0.00%** |

この形で **S ∈ {1,2,3,6} の q/k/v/g/beta/conv 状態・出力・再帰状態 (fp32) が
素とビット一致する** (`tools/gdn_decode_micro.py --mode check`)。
それでも品質を疑うときの物差しは `bench/quant_eval.py` の KLD。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

_KERNELS: dict[tuple, Any] = {}
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    """同じ理由の見送りを 1 度だけ知らせる (黙って落ちないため)。

    `prefill_attn.py` と同じ形 (2026-09-02、B-8 の穴埋め)。ここが無言のまま
    だと、`eligible` が常に False を返し続けても発火カウンタ (`_fire`) が
    0 のまま推移するだけで、なぜ 0 なのかが分からない。
    """
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] GDN 前処理カーネル: {msg}")


# 1 threadgroup が持つ連続チャネル数 (= threadgroup のスレッド数、1 スレッド
# 1 チャネル)。`dk` (= `dv`) の倍数であること -- q/k の 1 head が threadgroup を
# またぐと rms_norm の縮約が閉じない。conv_dim=10240 なので 128 -> 80 個、
# 256 -> 40 個の threadgroup が立つ (M3 Max は 40 コア)。
BLOCK = 128

# threadgroup メモリに置く部分和 (simdgroup ごとに float 1 個) の上限。
# BLOCK/32 個しか要らないので実質のガードは BLOCK の上限そのもの。
MAX_TG_BYTES = 28 * 1024

# decode/verify 幅の適格しきい値。競合 (mlx-serve) の同種融合が使っている
# 「S<=9 かつ batch*seq<=16」に合わせてある。外れたら素の経路 (conv1d ->
# silu -> rms_norm -> ...) にそのまま落ちる。
MAX_S = 9
MAX_M = 16

# MLX 本体の bf16 elementwise の式をそのまま写したヘッダ。
# `metal::exp(bfloat)` は bf16_math.h の定義で「float で計算して bf16 に戻す」
# = 1 段ごとに丸まる。数学的に等価な形へ書き換えると 1 ulp ずれるので、
# **本体の写しとして触らないこと** (本体は unary_ops.h の `Sigmoid` と
# binary_ops.h の `LogAddExp`)。
_MLX_ELEM_HEADER = """
template <typename T>
inline T mlx_sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}

template <typename T>
inline T mlx_softplus(T x) {
    // 本体 (binary_ops.h の LogAddExp) は log1p を**修飾なし**で呼ぶ
    // (`metal::log1p` は無い。MSL の大域側に bfloat の多重定義がある)。
    // **log1p の結果は T に丸めてから足す。**float のまま足すと bf16 で
    // 3% ずれる (総当たりで確認、下の「丸め位置の総当たり」を参照)。
    T zero = static_cast<T>(0.0f);
    T maxval = metal::max(x, zero);
    T minval = metal::min(x, zero);
    T e = metal::exp(minval - maxval);
    T l = static_cast<T>(log1p(static_cast<float>(e)));
    return maxval + l;
}
"""


def _source(cfg: dict) -> str:
    n_v = cfg["n_v"]
    dk = cfg["dk"]
    key_dim, value_dim, conv_dim = cfg["key_dim"], cfg["value_dim"], cfg["conv_dim"]
    K = cfg["K"]
    km1 = K - 1
    eps = cfg["eps"]
    q_scale, k_scale = cfg["q_scale"], cfg["k_scale"]
    two_key_dim = 2 * key_dim
    cb = cfg["block"]
    n_blocks = (conv_dim + cb - 1) // cb
    n_simd = cb // 32
    sg_per_head = dk // 32          # 1 head (dk チャネル) を担当する simdgroup 数

    # conv のタップは K しか無いので展開して書く (`conv_w` の 1 チャネル分
    # {K} 個は連続なので、展開すると 1 回の広い load にまとまる)。
    taps = "\n".join(
        f"""        {{
            int idx = s + {j};
            float v = (idx < {km1})
                ? (float)conv_state_in[((size_t)b * {km1} + idx) * {conv_dim} + c]
                : (float)mixed_qkv[((size_t)(base + idx - {km1})) * {conv_dim} + c];
            acc += v * (float)w_c[{j}];
        }}"""
        for j in range(K)
    )

    return f"""
    uint tid      = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int  zidx = (int)threadgroup_position_in_grid.z;
    int  blk  = zidx % {n_blocks + 1};
    int  m    = zidx / {n_blocks + 1};
    int  s    = m % S;
    int  base = m - s;   // この b の s=0 に対応する行番号 (= b * S)
    int  b    = base / S;

    // threadgroup 変数は分岐の手前で宣言する (早期 return の後ろに置くと
    // アドレス空間の宣言が非一様な制御フローに入ってしまう)
    threadgroup float tg_part[{n_simd}];

    // ---- 最終ブロック: g・beta だけを持つ ({n_v} レーン) --------------------
    // conv のブロックに相乗りさせない (そのブロックだけ末尾が伸びるため)。
    if (blk == {n_blocks}) {{
        if ((int)tid < {n_v}) {{
            int hv = (int)tid;
            size_t off = (size_t)m * {n_v} + hv;
            // 参照 compute_g: exp(-exp(f32(A_log)) * softplus(a + dt_bias))。
            // a / dt_bias は実モデルでは bf16 なので、和も softplus も
            // T のまま (= 素の経路と同じ丸め位置)。
            T xv = a[off] + dt_bias[hv];
            float sp = (float)mlx_softplus<T>(xv);
            float alog = (float)A_log[hv];
            // 本体の `Exp` は `metal::precise::exp` (unary_ops.h:177)。
            // 素の `metal::exp` (fast math) だと 48 個中 13 個が食い違う (実測)。
            g_out[off] = metal::precise::exp(-metal::precise::exp(alog) * sp);
            beta_out[off] = mlx_sigmoid<T>(b_in[off]);
        }}
        return;
    }}

    // ---- conv1d ({K} タップ、深さ方向) + silu -----------------------------
    // 1 スレッド = 1 チャネル。conv_state_in / mixed_qkv の該当タップは
    // この b の行 (base 起点) からしか読まない -- 単一系列 (mask なし) の前提。
    int c = blk * {cb} + (int)tid;

    float acc = 0.0f;
    if (c < {conv_dim}) {{
        const device T* w_c = conv_w + (size_t)c * {K};
{taps}
    }}
    // mx.conv1d の出力は T に丸まってから silu に渡る (実測で確認済み)
    T conv_bf = (T)acc;
    // 参照 nn.silu = x * sigmoid(x) は T どうしの積 (2 段丸め)
    T act = (T)((float)conv_bf * (float)mlx_sigmoid<T>(conv_bf));
    float af = (float)act;

    // ---- 次段 conv 状態 (K-1 列) ------------------------------------------
    // 全 s で同じ値になるので、この b の最後の s だけが書く。チャネルの
    // 持ち主は 1 threadgroup だけなので競合しない。
    if (s == S - 1 && c < {conv_dim}) {{
        for (int i = 0; i < {km1}; i++) {{
            int idx2 = S + i;
            float v2 = (idx2 < {km1})
                ? (float)conv_state_in[((size_t)b * {km1} + idx2) * {conv_dim} + c]
                : (float)mixed_qkv[((size_t)(base + idx2 - {km1})) * {conv_dim} + c];
            conv_state_out[((size_t)b * {km1} + i) * {conv_dim} + c] = (T)v2;
        }}
    }}

    // ---- q/k の rms_norm + スケール ---------------------------------------
    // BLOCK は dk の倍数なので 1 head は threadgroup 内で閉じる。simdgroup で
    // 部分和 -> threadgroup メモリ -> バリア 1 回 -> head 内の {sg_per_head} 本を足す。
    float ss = simd_sum(af * af);
    if (simd_lid == 0) {{
        tg_part[simd_gid] = ss;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (c < {two_key_dim}) {{
        uint hbase = (simd_gid / {sg_per_head}) * {sg_per_head};
        float tot = 0.0f;
        for (uint i = 0; i < {sg_per_head}; i++) {{
            tot += tg_part[hbase + i];
        }}
        float r = metal::rsqrt(tot / {float(dk)}f + {eps!r}f);
        // 参照 (inv_scale**2) * rms_norm(q, None, eps) は
        // bf16(bf16(x*rsqrt) * scale) の 2 段丸め (実測で確認済み)
        float nv = (float)((T)(af * r));
        if (c < {key_dim}) {{
            q_out[(size_t)m * {key_dim} + c] = (T)(nv * {q_scale!r}f);
        }} else {{
            k_out[(size_t)m * {key_dim} + (c - {key_dim})] = (T)(nv * {k_scale!r}f);
        }}
    }} else if (c < {conv_dim}) {{
        v_out[(size_t)m * {value_dim} + (c - {two_key_dim})] = act;
    }}
"""


def _as_dtype_scalar(value: float, dtype) -> float:
    """スカラーを `dtype` に丸めてから float に戻す。

    カーネルに埋め込む定数を、参照実装が実際に掛けている値 (弱い昇格で
    テンソルの dtype に落ちたもの) に揃えるために使う。
    """
    return float(mx.array(value, dtype))


def _get_kernel(cfg: dict):
    key = tuple(sorted((k, v) for k, v in cfg.items() if k != "dtype")) + (
        ("dtype", cfg["dtype"]),
    )
    kern = _KERNELS.get(key)
    if kern is not None:
        return kern
    suffix = (
        f"{cfg['conv_dim']}_{cfg['n_k']}x{cfg['dk']}_{cfg['n_v']}"
        f"_b{cfg['block']}_{len(_KERNELS)}"
    )
    kern = mx.fast.metal_kernel(
        name=f"gdn_prework_{suffix}",
        input_names=[
            "mixed_qkv", "conv_state_in", "conv_w", "a", "b_in",
            "A_log", "dt_bias", "S",
        ],
        output_names=[
            "q_out", "k_out", "v_out", "g_out", "beta_out", "conv_state_out",
        ],
        header=_MLX_ELEM_HEADER,
        source=_source(cfg),
    )
    _KERNELS[key] = kern
    return kern


def eligible(
    mixed_qkv: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    n_k: int,
    n_v: int,
    dk: int,
    key_dim: int,
    value_dim: int,
) -> bool:
    """このカーネルで扱える形・幅・量子化かを判定する。外れたら呼び出し側は素の経路へ。"""

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if mixed_qkv.dtype not in (mx.float16, mx.bfloat16):
        _warn_once("dtype", f"mixed_qkv の dtype {mixed_qkv.dtype} は非対応 (fp16/bf16 のみ)")
        return False
    if conv_w.dtype != mixed_qkv.dtype or conv_state.dtype != mixed_qkv.dtype:
        _warn_once("dtype_conv", "conv_w/conv_state の dtype が mixed_qkv と揃っていない")
        return False
    if a.dtype != mixed_qkv.dtype or b.dtype != mixed_qkv.dtype:
        _warn_once("dtype_ab", "a/b の dtype が mixed_qkv と揃っていない")
        return False
    # A_log/dt_bias は**実モデルの dtype (bf16) のまま**受ける。素の経路の
    # compute_g は `a + dt_bias` を bf16 で足して bf16 で softplus するので、
    # ここで fp32 を要求すると素と別の値になる (旧版はそれで fp32 の写しを
    # 作らせていた)。fp32 で来た場合も式は同じなので通す。
    if A_log.dtype not in (mixed_qkv.dtype, mx.float32):
        _warn_once("dtype_alog", "A_log が mixed_qkv と同じ dtype でも fp32 でもない")
        return False
    if dt_bias.dtype != A_log.dtype:
        _warn_once("dtype_dtbias", "dt_bias の dtype が A_log と揃っていない")
        return False
    if dt_bias.dtype != mixed_qkv.dtype:
        # fp32 で来た場合は a (bf16) との和が fp32 に昇格する = 素の経路と
        # 丸め位置が違う。カーネルは T (=mixed_qkv.dtype) で足すので、
        # 揃っていないときは引き受けない (素の経路の方が参照そのもの)。
        _warn_once(
            "dtype_alog_mismatch",
            "A_log/dt_bias が mixed_qkv と別の dtype (fp32 の写し) なので使わない。"
            " 素の経路と丸め位置が変わる -- 写しを作らずに素の重みを渡すこと",
        )
        return False
    if mixed_qkv.ndim != 3:
        _warn_once("ndim", f"mixed_qkv.ndim={mixed_qkv.ndim} は 3 でない")
        return False
    B, S, conv_dim = mixed_qkv.shape
    if S > MAX_S or B * S > MAX_M:
        _warn_once(
            "width",
            f"S={S} B*S={B * S} は decode/verify 幅の上限 (MAX_S={MAX_S}, "
            f"MAX_M={MAX_M}) を超えるので使わない (prefill 幅は素の経路のまま)",
        )
        return False
    if conv_dim != 2 * key_dim + value_dim or key_dim != n_k * dk:
        _warn_once("shape_conv", "conv_dim/key_dim の関係が想定と違う")
        return False
    if conv_w.ndim != 3 or conv_w.shape[0] != conv_dim or conv_w.shape[2] != 1:
        _warn_once("shape_convw", "conv_w の形が (conv_dim, K, 1) でない")
        return False
    K = conv_w.shape[1]
    if conv_state.shape != (B, K - 1, conv_dim):
        _warn_once("shape_convstate", "conv_state の形が (B, K-1, conv_dim) でない")
        return False
    if a.shape != (B, S, n_v) or b.shape != (B, S, n_v):
        _warn_once("shape_ab", "a/b の形が (B, S, n_v) でない")
        return False
    if A_log.shape != (n_v,) or dt_bias.shape != (n_v,):
        _warn_once("shape_alog", "A_log/dt_bias の形が (n_v,) でない")
        return False
    if n_k <= 0:
        _warn_once("nk", f"n_k={n_k} が 0 以下")
        return False
    # 列ブロックの割り付けの前提: BLOCK は dk の倍数で、q/k の境目 (key_dim /
    # 2*key_dim) もブロック境界に乗る。乗らないと 1 head が threadgroup を
    # またぎ、rms_norm の縮約が閉じない。
    if BLOCK % dk != 0 or dk % 32 != 0:
        _warn_once("block_dk", f"BLOCK={BLOCK} が dk={dk} の倍数でない (または dk が 32 の倍数でない)")
        return False
    if key_dim % BLOCK != 0 or (2 * key_dim) % BLOCK != 0:
        _warn_once("block_qk", f"key_dim={key_dim} が BLOCK={BLOCK} の境界に乗らない")
        return False
    if n_v > BLOCK:
        _warn_once("nv_block", f"n_v={n_v} が BLOCK={BLOCK} を超える (g/beta が 1 ブロックに収まらない)")
        return False
    return True


def wants(module: Any, mask, cache) -> bool:
    """GDN 前処理融合を試す外側の条件を共有する。

    これはテンソルの形・dtype を見る ``eligible()`` の前段で、呼び出し側の
    共通条件 (有効化、mask、cache、training、cache.lengths) だけを判定する。
    ``cache.lengths`` があるときは拒否する。融合経路は ``_store_conv_state``
    を通らず ``cache[0]`` に直接書くため、実長に基づく窓取りを行うキャッシュを
    受けると状態を壊す (BACKLOG B9)。
    """

    return bool(
        getattr(module, "_gdn_prework", False)
        and mask is None
        and cache is not None
        and not module.training
        and getattr(cache, "lengths", None) is None
    )


def explain_gate_miss(mask, cache, training: bool) -> None:
    """呼び出し側 (``GatedDeltaNet.__call__`` / ``spec_flash.capture()`` の
    ``gdn``) にある外側のガード (``mask is None``・``cache is not None``・
    ``not training``・``cache.lengths is None``) で ``eligible()`` に届く前に
    弾かれた理由を 1 度だけ知らせる。

    `eligible()` 自身の約 20 条件は `_warn_once` で理由が出るようになった
    (2026-09-02、BACKLOG.md の B-8) が、**その手前にあるこの 4 条件は今まで
    無言のままだった。** `_gdn_prework=True` にしても発火カウンタが 0 のまま
    のとき、原因が `eligible()` の内側か手前かをここで区別する
    (2026-09-02、gdn_prework が投機の検証フォワードで一度も発火しない事例の
    調査で見つかった穴)。呼び出し側は「有効化されているのにここに来た」
    ときだけ呼ぶこと (無効時に毎回呼ぶと無意味な print になる)。
    """
    if mask is not None:
        _warn_once(
            "gate_mask",
            "外側のガード: mask が None でない"
            " (バッチの右パディングなど) ので eligible() まで届かない",
        )
        return
    if cache is None:
        _warn_once("gate_cache", "外側のガード: cache が None なので eligible() まで届かない")
        return
    if training:
        _warn_once(
            "gate_training",
            "外側のガード: self.training が True なので eligible() まで届かない"
            " (model.eval() 漏れの疑い)",
        )
        return
    if getattr(cache, "lengths", None) is not None:
        _warn_once(
            "gate_lengths",
            "外側のガード: cache.lengths が None でない"
            " (バッチの右パディング) ので eligible() まで届かない",
        )
        return
    # ここに来た = 4 条件は全部通っていた (= 呼び出し側の判定ミス)。


def fused_gdn_prework(
    mixed_qkv: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    n_k: int,
    n_v: int,
    dk: int,
    dv: int,
    key_dim: int,
    value_dim: int,
    eps: float = 1e-6,
    block: int | None = None,
):
    """`GatedDeltaNet.__call__` の conv1d 以降・再帰カーネル手前を 1 本のカーネルで計算する。

    戻り値は `(q, k, v, g, beta, conv_state_out)`。q/k/v は `(B, S, n_k|n_v, dk|dv)`、
    g は `(B, S, n_v)` (fp32、`compute_g` と同じ dtype)、beta は `(B, S, n_v)`
    (`mixed_qkv.dtype`、`mx.sigmoid(b)` と同じ dtype)、conv_state_out は
    `(B, K-1, conv_dim)` (次段の `cache[0]` にそのまま入る形)。
    """

    _fire.bump("gdn_prework")
    B, S, conv_dim = mixed_qkv.shape
    K = conv_w.shape[1]
    inv_scale = dk**-0.5
    cfg = {
        "n_k": n_k, "n_v": n_v, "dk": dk, "dv": dv,
        "key_dim": key_dim, "value_dim": value_dim, "conv_dim": conv_dim,
        "K": K, "eps": float(eps),
        "block": int(block or BLOCK),
        # 参照は `inv_scale * mx.fast.rms_norm(...)` と書いていて、MLX の
        # 弱い昇格で Python スカラーが**相手配列の dtype に落ちてから**掛かる。
        # つまり参照が実際に掛けているのは T に丸めた値。カーネルに fp32 の
        # 全精度を埋めると、そこだけ精度が高くなって最終丸めが割れる。
        # dk=128 では q 側 (2^-7) はちょうど表せるので差が出ないが、
        # k 側 (2^-3.5) は bf16 で仮数が丸まり、1 ulp の食い違いになる。
        "q_scale": _as_dtype_scalar(inv_scale**2, mixed_qkv.dtype),
        "k_scale": _as_dtype_scalar(inv_scale, mixed_qkv.dtype),
        "dtype": mixed_qkv.dtype,
    }
    kernel = _get_kernel(cfg)

    M = B * S
    flat_qkv = mx.contiguous(mixed_qkv.reshape((M, conv_dim)))
    flat_a = mx.contiguous(a.reshape((M, n_v)))
    flat_b = mx.contiguous(b.reshape((M, n_v)))

    cb = cfg["block"]
    # z = 行 m x (conv ブロック n_blocks 本 + g/beta の 1 本)。S=1 でも
    # n_blocks+1 個の threadgroup が立つ (conv_dim=10240, BLOCK=256 -> 41)。
    n_blocks = (conv_dim + cb - 1) // cb
    outs = kernel(
        inputs=[
            flat_qkv, conv_state, conv_w, flat_a, flat_b, A_log, dt_bias, S,
        ],
        template=[("T", mixed_qkv.dtype)],
        grid=(cb, 1, M * (n_blocks + 1)),
        threadgroup=(cb, 1, 1),
        output_shapes=[
            (M, key_dim), (M, key_dim), (M, value_dim),
            (M, n_v), (M, n_v),
            (B, K - 1, conv_dim),
        ],
        output_dtypes=[
            mixed_qkv.dtype, mixed_qkv.dtype, mixed_qkv.dtype,
            mx.float32, mixed_qkv.dtype,
            mixed_qkv.dtype,
        ],
    )
    q_out, k_out, v_out, g_out, beta_out, conv_state_out = outs
    q_out = q_out.reshape((B, S, n_k, dk))
    k_out = k_out.reshape((B, S, n_k, dk))
    v_out = v_out.reshape((B, S, n_v, dv))
    g_out = g_out.reshape((B, S, n_v))
    beta_out = beta_out.reshape((B, S, n_v))
    return q_out, k_out, v_out, g_out, beta_out, conv_state_out


__all__ = [
    "wants", "eligible", "fused_gdn_prework", "explain_gate_miss",
    "BLOCK", "MAX_S", "MAX_M", "MAX_TG_BYTES",
]
