"""decode / verify 幅 (行数 <= 8) の MoE の「糊」を 2 本の Metal カーネルに畳む。

**結論 (2026-09-03): 既定 off のまま置く。18 -> 4 dispatch/層 は機構としては
完成した (合成でビット一致〜1 ulp、冷 micro で S=1 0.57 倍) が、in-model は
短 ms/round +0.05% で判定線 (-3%) に届かない。**

理由は「decode は dispatch 数ではなく GPU 時間で律速している」こと。
`tools/decode_gpu_trace.py --cases short` (head4、depth 2 固定) の A/B:

| | dispatch/round | GPU 和 ms/round | 稼働率 | 隙間 | 壁 ms/round |
|---|---|---|---|---|---|
| 素 | 3722.8 | 32.41 | 96.1% | 0.4 us/dispatch | 33.71 |
| 自前 | 3078.1 (**-17.3%**) | 32.87 (**+1.4%**) | 95.9% | 0.5 us/dispatch | 34.26 |

**稼働率 96%、隙間 0.4〜0.5 us/dispatch。**645 本消しても取り返せる隙間は
0.26 ms しか無く、融合先が増やした GPU 時間 (+0.46 ms) に負ける。

いちばん強い証拠は shared expert のゲートだけを畳んだ変種 (下の (a')、
`shared_gate` だけを差す): **出力が素とビット一致**なので生成トークン
列もラウンド数も完全に同じ = 交絡の無い対照になる。-192 dispatch/round
(-5.2%) で **ms/round ±0.0%** (3 プロンプト x 2 反復すべて)。
2026-09-03 21:10 の「糊 1 本 = 1.9 us」(GDN の 468 本で -0.88 ms) は
**dispatch を消したこと自体の取り分ではなく、融合先のカーネルが素より
GPU 時間を使わなかったこと**の取り分だった、と読み直すべき。

route (a) と combine (b) はどちらも素の並びより**並列度が低い**:
(a) は 1 行 = 1 threadgroup (512 スレッド、20 バリア)、(b) は 2560 スレッド
(80 threadgroup x 32) しか立たない。素は `v_copy` が 25600 スレッド、
`carg_block_sort` が専用の block sort。冷の連鎖 micro (何も並走しない直列)
では自前が 0.57 倍だが、**in-model の 96% 稼働の中ではその差が出ない**。

以下は機構の説明 (残してある。反転条件は「decode の稼働率が下がる、
または融合先の GPU 時間を素より下げられる形を見つけたとき」)。

## 何を置き換えるのか

素の `SparseMoeBlock.__call__` (`mlxturbo/_vendor/qwen4_exp.py`) は decode 幅で
**18 dispatch/層**を出す (`bench/results/decode-anatomy-s1-short.json` の
region `V/MoE(SparseMoeBlock)` が 864 / 48 層)。内訳 (op -> カーネル名):

    x.astype(float32)          v_copy bf16->f32         (1)  ← 残す
    self.gate(...)             affine_qmv_fast          (1)  ← 行列積。触らない
      └ scales/biases の昇格   v_copy bf16->f32         (2)  ← 事前 f32 化で消える
    -logits                    v_Negative               (1)  ┐
    argpartition               carg_block_sort          (1)  │
    take_along_axis            gather_axis              (1)  │
    softmax(precise)           block_softmax            (1)  ├ (a) へ
    shared_expert_gate(x)      dot_product + all_reduce (2)  │
      └ f32->bf16              s_copy                   (1)  │
    sigmoid                    (unary)                  (1)  ┘
    switch_mlp(x,idx) * w      v_copy bf16->f32         (1)  ┐
                               g2_Multiply f32          (1)  │
    .sum(axis=-2)              col_reduce_small         (1)  ├ (b) へ
    .astype(x.dtype)           v_copy f32->bf16         (1)  │
    sigmoid(sg) * shared       (bf16 multiply)          (1)  │
    out + ...                  vv_Add bf16              (1)  ┘

**行列積 (`switch_mlp` の gather_qmm) は触らない。**decode の専門家 GEMV は既に
帯域のピーク (395 GB/s) に張り付いていることが 2026-09-03 20:08 の PoL で
決着している。ここで取るのは糊だけ。18 -> 4 (x の f32 化 / router の行列積 /
(a) / (b))。

## (a) `moe_route_decode`

`logits` (rows, E) f32 から top-k の添字と softmax 重み、ついでに shared expert の
ゲート `sigmoid(bf16(x . sgw))` を 1 本で出す。

**旧 `moe_route.py` が負けた形を避けてある。**あちらは threadgroup=(32,1,1) で、
行数 1 では 1 コアの 1 simdgroup が 512 要素を逐次に 10 周し、softmax も lane 0 の
逐次だった (`moe_route.py` の docstring)。ここは threadgroup を 512 スレッド
(= 16 simdgroup) にして 1 スレッド 1 専門家を持たせ、1 周を「simdgroup 内の
ラダー縮約 + 16 個の縮約」の 2 バリアで終える。10 周で 20 バリア。

行数が 1 なら threadgroup は 1 個しか立たない。GDN/HC の融合で負けた形と
同じに見えるが、**読むのは logits 2 KB + x 5 KB + sgw 5 KB しか無い**ので、
DRAM レイテンシを隠すための並列度は要らない (GDN の旧版が負けたのは 1 コアで
250 KB を読んでいたため)。効くかどうかは冷の連鎖 micro
(`tools/moe_route_decode_micro.py`) で判定する。

## (b) `moe_combine_decode`

`y` (rows, k, H) にルータ重みを掛けて k 軸を潰し、shared expert を合流する。
1 threadgroup = (1 行, 連続 TG*VEC 列)。H=2560 なので VEC=1 / TG=32 なら
**行数 1 でも 80 個の threadgroup が立つ** (反転条件への備え)。

## 丸め

素の丸め位置を写している (「品質を売って速度を買わない」):

    Σ_k w[k]*y[k]          f32 累算 -> bf16 に 1 回丸め  (素の col_reduce f32 + astype)
    sigmoid(sg) * shared   bf16 の積 (float で計算して bf16 に丸め)
    その和                  bf16 の和

素と違うのは **f32 の 10 項の足し込み順だけ** (こちらは値の降順、素は
argpartition の未規定順)。top-k の集合は一致する (同値は添字の小さい方を残す =
torch.topk / HF と同じ決め方)。softmax は `mx.softmax(..., precise=True)` と
同じく最大値を引いてから `metal::precise::exp`。

## 配線

**無い。**上の判定で `mlxturbo/fused.py` の enable/disable、`runner.py` の
呼び出し、`tools/decode_ab.py` の knob は取り除いた (効かない変種を二重に
持たない方針)。このファイルは道具として残してあるだけで、本番の経路からは
呼ばれない。

再開するときに要るもの (どれも当時の実装をそのまま起こせばよい):

- `SparseMoeBlock.__call__` の差し替え。行数 (`x.shape[0]*x.shape[1]`) が
  8 以下のときだけ通し、`_router513` / `_wide_shared` が立っていたら素へ。
- router の scales/biases を f32 で持たせると `mx.quantized_matmul` の昇格
  copy が 2 本消える (bf16 -> f32 は誤差ゼロ、実測 `max|diff|=0.0`)。
  **これも取り分ゼロだったので入れていない。**
- 検査は `tools/verify_moe_route_decode.py` (合成、S ∈ {1,2,3,6})、
  冷の連鎖は `tools/moe_route_decode_micro.py`。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

_KERNELS: dict[tuple, Any] = {}

# (a) の threadgroup 上限。専門家 1 個 = 1 スレッドで持つ
ROUTE_MAX_THREADS = 1024

# (b) の既定配置。行数 1 でも 2560/32 = 80 threadgroup 立つ
COMBINE_VEC = 1
COMBINE_TG = 32

# decode / verify 幅の既定の上限行数
MAX_ROWS = 8


_HEADER = """
#define MT_UNROLL _Pragma("clang loop unroll(full)")

// MLX 本体の Sigmoid と同じ形 (mlxturbo/kernels/hyper_connection.py の写し)
template <typename T>
inline T mlxturbo_sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
"""


# --------------------------------------------------------------------- (a)

_SGATE_BLOCK = """
  // ---- shared expert のゲート: sigmoid(bf16(x . sgw)) --------------------
  // 素は dot_product + all_reduce + s_copy + sigmoid の 4 dispatch。
  // 1 threadgroup で持てる大きさ (H=2560、bf16 で 5 KB) なので相乗りさせる。
  // 累算は f32、bf16 に落としてから sigmoid = 素と同じ丸め位置
  {
    threadgroup float tg_dot[NSIMD];
    float acc = 0.0f;
    const device T* xr = xin + (size_t)row * HID;
    for (uint j = tid; j < (uint)HID; j += TG) {
      acc += (float)xr[j] * (float)sgw[j];
    }
    MT_UNROLL
    for (int off = 16; off > 0; off >>= 1) {
      acc += simd_shuffle_down(acc, off);
    }
    if (lane == 0) { tg_dot[sgi] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float s = 0.0f;
      for (int i = 0; i < NSIMD; i++) { s += tg_dot[i]; }
      sgate[row] = mlxturbo_sigmoid<T>((T)s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

_ROUTE_HEAD = """
  // grid = (TG, rows, 1)、threadgroup = (TG, 1, 1) -> threadgroup は rows 個
  const uint tid  = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sgi  = simdgroup_index_in_threadgroup;
  const int  row  = (int)threadgroup_position_in_grid.y;

  threadgroup float tg_v[NSIMD];
  threadgroup int   tg_i[NSIMD];
  threadgroup float tg_sel[TOPK];
  threadgroup int   tg_sidx[TOPK];
  threadgroup int   tg_win;
"""

_ROUTE_BODY = """
  // ---- logits をレジスタへ (1 スレッド 1 専門家) -------------------------
  const device float* lr = logits + (size_t)row * NEXP;
  float v = lr[tid];
  int   e = (int)tid;

  // ---- 「最大を取って外す」を TOPK 周。1 周 = 2 バリア --------------------
  for (int r = 0; r < TOPK; r++) {
    float best = v;
    int   bi   = e;
    MT_UNROLL
    for (int off = 16; off > 0; off >>= 1) {
      float ov = simd_shuffle_down(best, off);
      int   oi = simd_shuffle_down(bi, off);
      // 同値は添字の小さい方を残す (torch.topk / HF と同じ決め方)
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) { tg_v[sgi] = best; tg_i[sgi] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
      float b2 = (tid < (uint)NSIMD) ? tg_v[tid] : -INFINITY;
      int   i2 = (tid < (uint)NSIMD) ? tg_i[tid] : 0x7fffffff;
      MT_UNROLL
      for (int off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(b2, off);
        int   oi = simd_shuffle_down(i2, off);
        if (ov > b2 || (ov == b2 && oi < i2)) { b2 = ov; i2 = oi; }
      }
      if (tid == 0) { tg_sel[r] = b2; tg_sidx[r] = i2; tg_win = i2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == tg_win) { v = -INFINITY; }
  }

  // ---- softmax (降順に取ったので tg_sel[0] が最大) -----------------------
  if (tid == 0) {
    const float m = tg_sel[0];
    float s = 0.0f;
    for (int p = 0; p < TOPK; p++) {
      float ex = metal::precise::exp(tg_sel[p] - m);
      tg_sel[p] = ex;
      s += ex;
    }
    const size_t base = (size_t)row * TOPK;
    for (int p = 0; p < TOPK; p++) {
      idx[base + p] = (uint)tg_sidx[p];
      wout[base + p] = tg_sel[p] / s;
    }
  }
"""


def _route_kernel(with_sgate: bool):
    key = ("route", bool(with_sgate))
    k = _KERNELS.get(key)
    if k is None:
        src = _ROUTE_HEAD + (_SGATE_BLOCK if with_sgate else "") + _ROUTE_BODY
        k = mx.fast.metal_kernel(
            name="mlxturbo_moe_route_decode" + ("_sg" if with_sgate else ""),
            input_names=["logits", "xin", "sgw"] if with_sgate else ["logits"],
            output_names=["idx", "wout", "sgate"] if with_sgate
            else ["idx", "wout"],
            source=src,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = k
    return k


def _rows_of(shape) -> int:
    rows = 1
    for d in shape:
        rows *= d
    return rows


def route_eligible(logits: mx.array, top_k: int, max_rows: int = MAX_ROWS) -> bool:
    """(a) が使える形か。**host 同期はしない** (形と dtype だけ見る)。"""

    if mx.default_device() != mx.gpu:
        return False
    if logits.dtype != mx.float32 or logits.ndim < 1:
        return False
    n = logits.shape[-1]
    # 1 スレッド 1 専門家。512 experts が本番の形
    if n % 32 != 0 or not (32 <= n <= ROUTE_MAX_THREADS):
        return False
    if not (1 <= top_k <= 32) or top_k > n:
        return False
    return 1 <= _rows_of(logits.shape[:-1]) <= max_rows


def route(logits: mx.array, top_k: int, x: mx.array | None = None,
          sgw: mx.array | None = None):
    """top-k の添字 (uint32) と softmax 重み (float32) を返す。

    `x` と `sgw` を渡すと shared expert のゲート `sigmoid(bf16(x . sgw))` も
    同じカーネルで作って 3 つ目の戻り値にする (渡さなければ None)。
    添字は**値の降順** (同値は添字の小さい方が先)。
    """

    _fire.bump("moe_route_decode")
    lead = logits.shape[:-1]
    n = logits.shape[-1]
    rows = _rows_of(lead)
    flat = logits.reshape((rows, n))

    tg = n                      # 1 スレッド 1 専門家
    nsimd = tg // 32
    with_sgate = x is not None and sgw is not None
    tmpl = [("TG", tg), ("NSIMD", nsimd), ("NEXP", n), ("TOPK", int(top_k))]
    if with_sgate:
        hid = x.shape[-1]
        tmpl += [("HID", int(hid)), ("T", x.dtype)]
        inputs = [flat, x.reshape((rows, hid)), sgw.reshape((hid,))]
        shapes = [(rows, top_k), (rows, top_k), (rows,)]
        dtypes = [mx.uint32, mx.float32, x.dtype]
    else:
        inputs = [flat]
        shapes = [(rows, top_k), (rows, top_k)]
        dtypes = [mx.uint32, mx.float32]

    outs = _route_kernel(with_sgate)(
        inputs=inputs,
        template=tmpl,
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    idx = outs[0].reshape((*lead, top_k))
    w = outs[1].reshape((*lead, top_k))
    sg = outs[2].reshape((*lead, 1)) if with_sgate else None
    return idx, w, sg


# ------------------------------------------------------------- (a') 単独版

# `_SGATE_BLOCK` と同じ計算を単独のカーネルにしたもの。素はこれだけで
# **4 dispatch/層** 使う (dot_product + all_reduce + s_copy + Sigmoid) --
# 2560 要素の内積 1 本を 2 段の縮約カーネルで取るため。行数 <= 8 では
# `route` と同じ TG=512 で読み切れるので、丸め位置も足し込み順も (a) と同じ。
_SGATE_ONLY_HEAD = """
  const uint tid  = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sgi  = simdgroup_index_in_threadgroup;
  const int  row  = (int)threadgroup_position_in_grid.y;
"""


def _sgate_kernel():
    k = _KERNELS.get("sgate")
    if k is None:
        k = mx.fast.metal_kernel(
            name="mlxturbo_moe_shared_gate_decode",
            input_names=["xin", "sgw"],
            output_names=["sgate"],
            source=_SGATE_ONLY_HEAD + _SGATE_BLOCK,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS["sgate"] = k
    return k


def shared_gate(x: mx.array, sgw: mx.array, threads: int = 512):
    """`sigmoid(bf16(x . sgw))` を 1 dispatch で返す ((..., 1) の形)。"""

    _fire.bump("moe_shared_gate_decode")
    lead = x.shape[:-1]
    hid = x.shape[-1]
    rows = _rows_of(lead)
    tg = min(threads, hid)
    while tg > 32 and hid % tg != 0:
        tg //= 2
    (sg,) = _sgate_kernel()(
        inputs=[x.reshape((rows, hid)), sgw.reshape((hid,))],
        template=[("T", x.dtype), ("TG", int(tg)), ("NSIMD", int(tg // 32)),
                  ("HID", int(hid))],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[x.dtype],
    )
    return sg.reshape((*lead, 1))


# --------------------------------------------------------------------- (b)

_COMBINE_SOURCE = """
  // grid = (H/VEC, 1, rows)、threadgroup = (TG, 1, 1)
  const uint r   = threadgroup_position_in_grid.z;
  const uint lid = thread_position_in_threadgroup.x;

  // 同じ threadgroup のスレッドは全員同じ行を見るので、ルータ重みは 1 回だけ読む
  threadgroup float tg_w[TOPK];
  for (uint i = lid; i < (uint)TOPK; i += TG) {
    tg_w[i] = w[(size_t)r * TOPK + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n = thread_position_in_grid.x * VEC;
  float acc[VEC];
  MT_UNROLL
  for (short c = 0; c < VEC; c++) { acc[c] = 0.0f; }

  const device T* yb = y + (size_t)r * TOPK * HID + (size_t)n;
  MT_UNROLL
  for (short k = 0; k < TOPK; k++) {
    const float wk = tg_w[k];
    const device T* p = yb + (size_t)k * HID;
    MT_UNROLL
    for (short c = 0; c < VEC; c++) { acc[c] += wk * (float)p[c]; }
  }

  // 素の丸め位置の写し: f32 の和 -> bf16、sigmoid(sg)*shared は bf16 の積、
  // 最後に bf16 の和
  const float sg = (float)sgate[r];
  const device T* sh = shared + (size_t)r * HID + (size_t)n;
  device T* o = out + (size_t)r * HID + (size_t)n;
  MT_UNROLL
  for (short c = 0; c < VEC; c++) {
    T s1 = (T)acc[c];
    T m1 = (T)(sg * (float)sh[c]);
    o[c] = (T)((float)s1 + (float)m1);
  }
"""


def _combine_kernel():
    k = _KERNELS.get("combine")
    if k is None:
        k = mx.fast.metal_kernel(
            name="mlxturbo_moe_combine_decode",
            input_names=["y", "w", "sgate", "shared"],
            output_names=["out"],
            source=_COMBINE_SOURCE,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS["combine"] = k
    return k


def combine_eligible(y: mx.array, w: mx.array, sgate: mx.array,
                     shared: mx.array, top_k: int, vec: int = COMBINE_VEC,
                     max_rows: int = MAX_ROWS) -> bool:
    """(b) が使える形か。**host 同期はしない**。"""

    if mx.default_device() != mx.gpu:
        return False
    if y.ndim < 3 or shared.ndim < 2 or w.ndim < 2:
        return False
    if y.shape[-2] != top_k or y.shape[-1] != shared.shape[-1]:
        return False
    if y.dtype != shared.dtype or y.dtype != sgate.dtype:
        return False
    if y.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if w.dtype != mx.float32 or w.shape[-1] != top_k:
        return False
    if y.shape[-1] % vec != 0:
        return False
    rows = _rows_of(y.shape[:-2])
    if rows != int(sgate.size) or rows != _rows_of(shared.shape[:-1]):
        return False
    if rows * top_k != _rows_of(w.shape):
        return False
    return 1 <= rows <= max_rows and 1 <= top_k <= 32


def combine(y: mx.array, w: mx.array, sgate: mx.array, shared: mx.array,
            top_k: int, vec: int = COMBINE_VEC, tg: int = COMBINE_TG):
    """``(Σ_k w[k]*y[k]) + sigmoid 済みゲート * shared`` を 1 本で返す。

    `sgate` は (a) が出した **sigmoid 適用済み**のゲート (rows 個)。
    """

    _fire.bump("moe_combine_decode")
    lead = shared.shape[:-1]
    hid = y.shape[-1]
    rows = _rows_of(lead)
    n_x = hid // vec
    while tg > 1 and n_x % tg != 0:
        tg //= 2
    (out,) = _combine_kernel()(
        inputs=[y.reshape((rows * top_k, hid)), w.reshape((rows, top_k)),
                sgate.reshape((rows,)), shared.reshape((rows, hid))],
        template=[("T", y.dtype), ("TOPK", int(top_k)), ("VEC", int(vec)),
                  ("TG", int(tg)), ("HID", int(hid))],
        grid=(n_x, 1, rows),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, hid)],
        output_dtypes=[y.dtype],
    )
    return out.reshape((*lead, hid))


__all__ = [
    "COMBINE_TG", "COMBINE_VEC", "MAX_ROWS", "ROUTE_MAX_THREADS",
    "combine", "combine_eligible", "route", "route_eligible", "shared_gate",
]
