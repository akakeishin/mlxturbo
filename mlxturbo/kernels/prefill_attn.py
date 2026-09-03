"""prefill の QSA 注意を 1 本の Metal カーネルに畳む (段 P1)。

`docs/research/KERNEL-PROGRAM.md` 段 P1 の決着そのもの。いまの gather 経路
(`Attention._gather_tile_attn`) は汎用 op を 2 段重ねている --- 選んだ列を
`take_along_axis` で **書き**、sdpa がそれを **読み直し**、さらに union 幅の
bool マスクを **作って** 渡す。帯域律速の prefill ではこの往復が丸ごと差に
なる (相手 mlx-serve の `gatherQsa256` は書きがゼロで、交差点が kv 10-12k、
うちは 18k)。

ここは段 P1 に書いた形をそのまま実装する:

- **1 threadgroup = 1 クエリトークン x 1 kv head**。GQA の 12 本の q head を
  12 simdgroup が分担するので、threadgroup メモリに載せた K/V タイルを
  12 head で共有できる (この共有が 1 threadgroup 1 クエリの取り分)
- **キー方向にタイル**。1 タイル = `BB` ブロック = `BB * compress_ratio` 列。
  head_dim 256 / bf16 で BB=4 (16 列)、K と V で 16KB
- **online softmax**。スコアを全部実体化しない (中間バッファ無し)
- **gather 後のマスクを持たない**。読む範囲そのものが可視集合
- bf16 / head_dim 256 を想定するが、形は全部テンプレートに逃がしてある
  (合成モデルでの検証を GPU 上で回すため。適格判定は :func:`eligible`)

## 可視集合をどう決めているか (相手との違い)

相手は選択ブロックの添字列をカーネルに渡す。こちらは indexer が既に作って
いる **ブロック単位の bool (`keep_block`, (B,S,n_blocks)) をそのまま読み**、
カーネル内で昇順のまま詰めて添字列にする。添字列を host 側で作るには
`keep_block` に対して argpartition なり sort なりをもう一度掛けることになり、
それは indexer の top-k と同じ規模の仕事をもう一度払うのと同じになる
(`QSAIndexer` が内部で持っている `top` をそのまま貰えれば要らないが、
それは indexer 側のシームを増やす話なので、ここでは踏み込まない)。

ブロック bool の読み出し量は 1 クエリ 1 kv head あたり n_blocks バイト
(17k なら 4.2KB) で、同じ threadgroup が読む K/V の 2MB に対して 0.2%。
**「後段の gather 後マスク」ではない** --- クエリと列の直積のマスクは
どこにも作らず、選択そのものを添字の素材として読むだけ。

詰める作業は simdgroup ごとに区間を割って 3 パスでやる (数える -> 排他的
前置和 -> 詰めて書く)。1 simdgroup に全部やらせると n_blocks/32 回の直列
ループが残り、他の simdgroup が待つ時間が計算時間と同じ桁になる。

## 可視性の根拠 (マスクを持たなくてよい理由)

`QSAIndexer._pooled_and_top` は `block_end <= q_col` を満たすブロックだけを
top-k の候補にしている。つまり **選ばれた完全ブロックの中身は追加の因果
チェック無しにそのまま可視**。残るのは端数列 (`n_blocks * cr` 以降) だけで、
そちらは `col <= q_col` なので **先頭からの連続区間**になる
(`ntail_vis = q_col - n_blocks*cr + 1` を [0, tail] に丸めるだけ)。
どちらも添字の算数で閉じるので、マスク配列は要らない。

## 精度

参照 (`take_along_axis` -> sdpa) とビット一致はしない。注意する集合は同じで、
加算順とスケーリングの順が変わる。gather 経路自体が元々ビット一致しない
(`Attention._gather_forward` の docstring) ので、前提は変わらない。
累算は fp32、出力で入力 dtype へ丸める。

## 既定 off

`MLXTURBO_PREFILL_ATTN=1` (`mlxturbo/gather_attn.py` の
`enable_prefill_attn`) のときだけ動く。採否は in-model の壁時計で決める
(判定は親、`tools/decode_ab.py --knob prefill-attn`)。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

# prefill と decode/verify を分ける S の下限。上の `eligible` の注記を参照。
MIN_S = 64

# 1 threadgroup が使う threadgroup メモリの上限 (Apple GPU は 32KB)。
# K/V タイル + 選択ブロックの添字列 + simdgroup ごとの本数がここに載る。
MAX_TG_BYTES = 30 * 1024

# K/V タイルに載せる列数の目安。head_dim 256 / bf16 なら BB=6 ブロック
# (compress_ratio 4 なら 24 列) = K と V で 24576 バイト (MAX_TG_BYTES 以内)。
#
# **2026-09-03、in-model 判定で本番へ配線 (合成マイクロベンチ候補 (i) と同じ内容)。**
# 元は 16 列 (BB=4)。`tools/qsa_gather_micro.py` の内訳計測
# (`docs/research/LANES-2026-09.md` レーン 3) で、K/V の threadgroup 段階 load が
# 全体の 65-70% を占め、タイルを広げる (16→24 列) と barrier 回数が減って
# load が 11% 縮むことを確認。合成 (kv=16896, S=2048, dense sdpa 基準比):
# 元 (16 列) 1.41 倍 → 24 列 1.50 倍、誤差は両方とも相対 7.3e-3 (許容 1.5e-2 以内)
# で変化なし。24 列を超える (32-128 列) は head_dim=256/bf16 では threadgroup
# メモリの壁 (32KB) を超えるため不可能 --- 24 列が物理上限に近い。
# 分割案 (score/softmax を狭い register で処理) は逆に遅くなったため不採用
# (LANES-2026-09.md 参照)。
_TARGET_COLS = 24

_KERNELS: dict[tuple, Any] = {}
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    """同じ理由の見送りを 1 度だけ知らせる (黙って落ちないため)。"""
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] prefill attention カーネル: {msg}")


def _tile_blocks(cr: int, head_dim: int, itemsize: int) -> int:
    """1 タイルに載せるブロック数 ``BB`` (列数は ``BB * cr``)。

    threadgroup メモリに K と V を両方置くので、収まるまで半分にする。
    1 ブロックすら載らない形は :func:`eligible` が弾く。
    """
    bb = max(1, _TARGET_COLS // cr)
    while bb > 1 and 2 * bb * cr * head_dim * itemsize > MAX_TG_BYTES:
        bb //= 2
    return bb


def _vec_elems(itemsize: int) -> int:
    """1 つの ``uint4`` (16 バイト) に入る要素数。bf16 なら 8。"""
    return 16 // itemsize


def vec_ok(head_dim: int, itemsize: int, cap: int) -> bool:
    """段階 load を ``uint4`` (16 バイト) 単位でやってよい形か。

    K/V の 1 行は ``head_dim * itemsize`` バイト。行頭が 16 バイト整列して
    いなければ `uint4` の読み書きは未定義なので、

    - 1 行が 16 の倍数 (行から行への刻みが整列を崩さない)
    - head ごとの刻み ``cap * head_dim * itemsize`` も 16 の倍数
    - threadgroup タイルの要素数が ``uint4`` 個数で割り切れる

    を全部満たすときだけ vec 版を使う。**外れたら scalar ループに戻る**
    (カーネル自体は使える。速いか遅いかだけの話なので `eligible` は落とさない)。
    """
    if itemsize <= 0 or 16 % itemsize != 0:
        return False
    ve = _vec_elems(itemsize)
    row_bytes = head_dim * itemsize
    if row_bytes % 16 != 0 or head_dim % ve != 0:
        return False
    if (cap * row_bytes) % 16 != 0:
        return False
    return True


def _source(
    head_dim: int, gqa: int, cr: int, bb: int, kmax: int, scale: float,
    tail_mode: str = "global", vec: bool = False, itemsize: int = 2,
) -> str:
    d = head_dim
    dpl = (d + 31) // 32       # 1 lane が持つ要素数 (head_dim を 32 lane で割る)
    bk = bb * cr               # 1 タイルの列数
    nsg = gqa                  # simdgroup 数 = この kv head に属する q head 数
    nth = nsg * 32
    ve = _vec_elems(itemsize) if vec else 0
    if vec and (bk * d) % ve != 0:
        vec = False

    if vec:
        # K/V タイルを `uint4` 配列として確保して 16 バイト整列を保証し、
        # スコア側は `T*` に読み替える (間に barrier が入るので順序は保たれる)。
        tg_kv_src = f"""    threadgroup uint4 tg_k4[{bk} * {d} / {ve}];
    threadgroup uint4 tg_v4[{bk} * {d} / {ve}];
    threadgroup T* tg_k = (threadgroup T*)tg_k4;
    threadgroup T* tg_v = (threadgroup T*)tg_v4;"""
    else:
        tg_kv_src = f"""    threadgroup T   tg_k[{bk} * {d}];
    threadgroup T   tg_v[{bk} * {d}];"""

    if vec:
        # 1 列 = 1 simdgroup、1 レーン = 1 `uint4` (16 バイト)。列の添字計算と
        # `tg_sel[]` 参照が **列あたり 1 回** になり、32 レーン x 16 バイト の
        # 連続読みになる (head_dim=256/bf16 なら 1 レーン x 32 個でちょうど 1 行)。
        vpc = d // ve   # 1 列あたりの uint4 個数
        if vpc == 32:
            inner = """            tg_k4[cc * 32 + (int)lane] = ksrc[lane];
            tg_v4[cc * 32 + (int)lane] = vsrc[lane];
"""
        else:
            inner = f"""            for (int dv = (int)lane; dv < {vpc}; dv += 32) {{
                tg_k4[cc * {vpc} + dv] = ksrc[dv];
                tg_v4[cc * {vpc} + dv] = vsrc[dv];
            }}
"""
        load_src = f"""        for (int cc = (int)sg; cc < ncol; cc += {nsg}) {{
            const int col = is_tail
                ? (tail_base + tb + cc)
                : (tg_sel[t0 + cc / {cr}] * {cr} + (cc % {cr}));
            const device uint4* ksrc =
                (const device uint4*)(kbase + (size_t)col * {d});
            const device uint4* vsrc =
                (const device uint4*)(vbase + (size_t)col * {d});
{inner}        }}"""
    else:
        load_src = f"""        for (int e = (int)tid; e < ncol * {d}; e += {nth}) {{
            const int cc = e / {d};
            const int dd = e - cc * {d};
            const int col = is_tail
                ? (tail_base + tb + cc)
                : (tg_sel[t0 + cc / {cr}] * {cr} + (cc % {cr}));
            tg_k[e] = kbase[(size_t)col * {d} + dd];
            tg_v[e] = vbase[(size_t)col * {d} + dd];
        }}"""

    if tail_mode == "query":
        # HF 参照の tail (`mlxturbo/qsa_tail.py`)。クエリごとに
        # 「自分の未完成ブロックの先頭から自分自身まで」= 列
        # [cr*floor((q_col+1)/cr), q_col] だけを足す (0〜cr-1 列)。
        # q_col % cr == cr-1 の行では 0 列 (その行のブロックは完全ブロック
        # として top-k の候補に入っている)。選択ブロックとは重ならないので、
        # global 側と同じく「読む範囲そのものが可視集合」のまま。
        tail_src = f"""    const int tail_base = ((q_col + 1) / {cr}) * {cr};
    const int ntail_vis = q_col - tail_base + 1;   // 0..{cr - 1}"""
    else:
        # global tail: ブロック格子の外 (端数列) を因果性の範囲で。
        # col <= q_col なので先頭からの連続区間になる。マスクは要らない。
        tail_src = f"""    const int tail_base = NB * {cr};
    const int ntail = KVLEN - tail_base;
    int ntail_vis = q_col - tail_base + 1;
    ntail_vis = max(0, min(ntail_vis, ntail));"""

    return f"""
    const int S      = params[0];   // このチャンクのクエリ行数
    const int NB     = params[1];   // 完全ブロック数 (kv_len / cr)
    const int KVLEN  = params[2];   // 可視な列の総数 (offset + S)
    const int CAP    = params[3];   // KV キャッシュの列方向の確保幅
    const int OFFSET = params[4];   // このチャンク先頭のキャッシュ列位置
    const int NKV    = params[5];   // kv head 数

    const uint tid   = thread_position_in_threadgroup.x;
    const uint sg    = simdgroup_index_in_threadgroup;
    const uint lane  = thread_index_in_simdgroup;
    const int  s     = (int)threadgroup_position_in_grid.y;
    const int  bh    = (int)threadgroup_position_in_grid.z;
    const int  b     = bh / NKV;
    const int  kvh   = bh - b * NKV;
    const int  H     = NKV * {nsg};
    const int  h     = kvh * {nsg} + (int)sg;
    const int  q_col = OFFSET + s;

    threadgroup int tg_sel[{kmax}];   // 昇順に詰めた選択ブロックの添字
    threadgroup int tg_cnt[{nsg}];    // simdgroup ごとの本数
{tg_kv_src}

    const device bool* keep_row = keep + ((size_t)b * S + s) * (size_t)NB;

    // --- 1) keep_block を昇順のまま詰める -------------------------------
    // simdgroup ごとに区間を割り、(a) 数える (b) 排他的前置和 (c) 詰めて書く。
    // 昇順を崩さないので、後段の K/V 読みはブロック順に前へ進む。
    int chunk = (NB + {nsg} - 1) / {nsg};
    chunk = ((chunk + 31) / 32) * 32;
    const int lo = (int)sg * chunk;
    const int hi = min(lo + chunk, NB);

    int cnt = 0;
    for (int base = lo; base < hi; base += 32) {{
        const int idx = base + (int)lane;
        const uint f = (idx < hi && keep_row[idx]) ? 1u : 0u;
        cnt += (int)simd_sum(f);
    }}
    if (lane == 0) {{ tg_cnt[sg] = cnt; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int base_off = 0;
    for (uint g = 0; g < sg; ++g) {{ base_off += tg_cnt[g]; }}
    int total = 0;
    for (int g = 0; g < {nsg}; ++g) {{ total += tg_cnt[g]; }}
    total = min(total, {kmax});

    int w = base_off;
    for (int base = lo; base < hi; base += 32) {{
        const int idx = base + (int)lane;
        const uint f = (idx < hi && keep_row[idx]) ? 1u : 0u;
        const uint pre = simd_prefix_exclusive_sum(f);
        if (f != 0u) {{
            const int slot = w + (int)pre;
            if (slot < {kmax}) {{ tg_sel[slot] = idx; }}
        }}
        w += (int)simd_sum(f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 2) tail の可視範囲 ---------------------------------------------
{tail_src}

    // --- 3) online softmax ---------------------------------------------
    const device T* qrow = q + (((size_t)b * S + s) * H + h) * {d};
    float qv[{dpl}];
    float acc[{dpl}];
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        qv[t] = (dd < {d}) ? (float)qrow[dd] : 0.0f;
        acc[t] = 0.0f;
    }}
    float m = -INFINITY;
    float l = 0.0f;

    const device T* kbase = k + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    const device T* vbase = v + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};

    const int ntiles = (total + {bb} - 1) / {bb};
    const int ttiles = (ntail_vis + {bk} - 1) / {bk};

    for (int ti = 0; ti < ntiles + ttiles; ++ti) {{
        const bool is_tail = ti >= ntiles;
        const int t0 = is_tail ? 0 : ti * {bb};
        const int tb = is_tail ? (ti - ntiles) * {bk} : 0;
        const int ncol = is_tail ? min({bk}, ntail_vis - tb)
                                 : min({bb}, total - t0) * {cr};

        // 前のタイルを読み終わるまで上書きしない
        threadgroup_barrier(mem_flags::mem_threadgroup);
{load_src}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // スコア。ループ長を定数にして sc をレジスタに残す
        // (ncol で回すと動的添字になり thread ローカルへ落ちる)
        float sc[{bk}];
        float mt = -INFINITY;
        for (int j = 0; j < {bk}; ++j) {{
            float p = 0.0f;
            if (j < ncol) {{
                for (int t = 0; t < {dpl}; ++t) {{
                    const int dd = (int)lane + 32 * t;
                    if (dd < {d}) {{ p += qv[t] * (float)tg_k[j * {d} + dd]; }}
                }}
            }}
            const float sj = simd_sum(p) * {scale!r}f;
            sc[j] = (j < ncol) ? sj : -INFINITY;
            mt = metal::max(mt, sc[j]);
        }}

        const float mnew = metal::max(m, mt);
        const float corr = metal::exp(m - mnew);   // 初回は exp(-inf) = 0
        for (int t = 0; t < {dpl}; ++t) {{ acc[t] *= corr; }}
        l *= corr;
        for (int j = 0; j < {bk}; ++j) {{
            if (j >= ncol) {{ continue; }}   // タイル外の tg_v は前タイルの残り
            const float p = metal::exp(sc[j] - mnew);
            l += p;
            for (int t = 0; t < {dpl}; ++t) {{
                const int dd = (int)lane + 32 * t;
                if (dd < {d}) {{ acc[t] += p * (float)tg_v[j * {d} + dd]; }}
            }}
        }}
        m = mnew;
    }}

    // 可視列が 1 つも無い行は 0 を書く (本番構成では踏まない --- 呼び出し側が
    // offset >= compress_ratio-1 かつ kv_len > token_budget を確かめている)
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device T* orow = out + (((size_t)b * S + s) * H + h) * {d};
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        if (dd < {d}) {{ orow[dd] = (T)(acc[t] * inv); }}
    }}
"""


def _get_kernel(
    head_dim, gqa, cr, bb, kmax, scale, tail_mode="global",
    vec=False, itemsize=2,
):
    key = (head_dim, gqa, cr, bb, kmax, scale, tail_mode, vec, itemsize)
    kern = _KERNELS.get(key)
    if kern is None:
        suffix = f"_u4{itemsize}" if vec else ""
        kern = mx.fast.metal_kernel(
            name=(
                f"prefill_attn_{head_dim}_{gqa}_{cr}_{bb}_{kmax}"
                f"_{tail_mode}{suffix}"
            ),
            input_names=["q", "k", "v", "keep", "params"],
            output_names=["out"],
            source=_source(
                head_dim, gqa, cr, bb, kmax, scale, tail_mode, vec, itemsize
            ),
        )
        _KERNELS[key] = kern
    return kern


def _kv_buffers(cache, k: mx.array, v: mx.array, kv_len: int):
    """KV キャッシュの **確保済みバッファそのもの** と、その列方向の幅を返す。

    `KVCache.update_and_fetch` が返すのは ``self.keys[..., :offset, :]`` という
    途中切りのビューで、確保幅と長さが一致しない限り行連続ではない。そのまま
    `metal_kernel` に渡すと `ensure_row_contiguous` が KV 全体を毎回コピーする
    (17k で 35MB/層) ので、**バッファ本体 + 確保幅**を渡して添字側で吸収する。

    条件に合わないキャッシュでは ``None`` を返す (呼び出し側は既存経路へ)。
    """
    keys = getattr(cache, "keys", None)
    values = getattr(cache, "values", None)
    if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
        return None
    if keys.ndim != 4 or values.ndim != 4:
        return None
    if getattr(cache, "offset", None) != kv_len:
        return None
    if keys.shape[:2] != k.shape[:2] or values.shape[:2] != v.shape[:2]:
        return None
    if keys.shape[3] != k.shape[3] or values.shape[3] != v.shape[3]:
        return None
    if keys.shape[2] < kv_len or values.shape[2] != keys.shape[2]:
        return None
    if keys.dtype != k.dtype or values.dtype != v.dtype:
        return None
    return keys, values, keys.shape[2]


def eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    keep_block: mx.array,
    cache,
    cr: int,
    kv_len: int,
    n_blocks: int,
    block_topk: int,
) -> bool:
    """このカーネルで扱える形か。外れたら呼び出し側は既存の gather 経路へ。"""

    # decode/verify 幅には入れない。このカーネルは online softmax で加算順が
    # 変わるためビット一致しない。prefill では出力が下流の一致で吸収されるが、
    # verify 幅で使うと argmax がまれに割れて受理率が動く。2026-09-01 の A/B が
    # まさにそれで、prefill -1.4% に対し tok/round -0.7%、合計 ms/tok は +0.9%
    # の悪化になった (bench/results/prefill-attn-ab.json)。**prefill だけに効かせて
    # 測り直すための下限。**MIN_S は本番の verify 幅 (depth+1、最大でも 16) より
    # 十分大きく、prefill のチャンク幅 (PREFILL_STEP_SIZE=2048、group 分割後も
    # 数百以上) より十分小さい値。
    if q.shape[2] < MIN_S:
        _warn_once("min_s", f"S={q.shape[2]} は decode/verify 幅なので使わない")
        return False
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        _warn_once("dtype", f"dtype {q.dtype} は非対応")
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        _warn_once("dtype_kv", "q/k/v の dtype が揃っていない")
        return False
    if keep_block.dtype != mx.bool_ or keep_block.ndim != 3:
        _warn_once("keep", "keep_block が (B,S,n_blocks) の bool でない")
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        _warn_once("ndim", "q/k/v の ndim が 4 でない ((B, H, S, D) 前提)")
        return False
    head_dim = q.shape[3]
    if k.shape[3] != head_dim or v.shape[3] != head_dim:
        _warn_once("headdim", "k/v の head_dim が q と違う (V の次元差は非対応)")
        return False
    n_kv = k.shape[1]
    if n_kv <= 0 or q.shape[1] % n_kv != 0:
        _warn_once("n_kv", f"n_kv={n_kv} が 0 以下、または q の head 数を割り切れない")
        return False
    gqa = q.shape[1] // n_kv
    if gqa < 1 or gqa > 32:
        # threadgroup は 32*gqa スレッド。32 simdgroup (1024) が上限
        _warn_once("gqa", f"GQA {gqa} は 1..32 の外")
        return False
    if cr < 1 or block_topk < 1:
        _warn_once("cr_topk", f"cr={cr} block_topk={block_topk} のどちらかが 1 未満")
        return False
    if n_blocks * cr > kv_len or kv_len - n_blocks * cr >= cr:
        _warn_once("blocks", "n_blocks と kv_len の関係が想定と違う")
        return False

    itemsize = q.dtype.size
    bb = _tile_blocks(cr, head_dim, itemsize)
    tg = 2 * bb * cr * head_dim * itemsize + 4 * block_topk + 4 * gqa
    if tg > MAX_TG_BYTES:
        _warn_once(
            "tg",
            f"threadgroup メモリ {tg} バイトが上限 {MAX_TG_BYTES} を超える",
        )
        return False
    if _kv_buffers(cache, k, v, kv_len) is None:
        _warn_once(
            "cache",
            "KV キャッシュの確保済みバッファを取れない (このキャッシュ型は非対応)",
        )
        return False
    # 16 バイト整列 (`vec_ok`) はここでは見ない。外れても scalar の段階 load で
    # 動くので、適格性ではなく `_get_kernel` の版の選択だけの話になる。
    return True


def prefill_attn(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    keep_block: mx.array,
    cache,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    block_topk: int,
    offset: int,
    scale: float,
) -> mx.array:
    """選択ブロックの gather と softmax を 1 本のカーネルで済ませる。

    ``q`` は (B, n_heads, S, head_dim)、``k``/``v`` は
    ``cache.update_and_fetch`` の戻り値 (B, n_kv_heads, kv_len, head_dim)、
    ``keep_block`` は `QSAIndexer.select_blocks` が返す (B, S, n_blocks) の
    bool。戻り値は **(B, S, n_heads, head_dim)** で、呼び出し側は転置なしで
    そのまま `reshape(B, S, -1)` できる (カーネルがこの並びで書く)。
    """

    _fire.bump("prefill_attn")
    B, n_heads, S, head_dim = q.shape
    n_kv = k.shape[1]
    gqa = n_heads // n_kv
    itemsize = q.dtype.size
    bb = _tile_blocks(cr, head_dim, itemsize)

    keys, values, cap = _kv_buffers(cache, k, v, kv_len)

    # (B, n_heads, S, D) は転置ビューなので、転置し直すと射影直後の行連続な
    # 並びに戻る。カーネルはこちらの並び (B, S, H, D) で読む
    q_bshd = q.transpose(0, 2, 1, 3)
    params = mx.array(
        [S, n_blocks, kv_len, cap, offset, n_kv], dtype=mx.int32
    )

    from mlxturbo import qsa_tail as _qsa_tail

    kernel = _get_kernel(
        head_dim, gqa, cr, bb, block_topk, float(scale), _qsa_tail.MODE,
        vec_ok(head_dim, itemsize, cap), itemsize,
    )
    (out,) = kernel(
        inputs=[q_bshd, keys, values, keep_block, params],
        template=[("T", q.dtype)],
        grid=(32 * gqa, S, B * n_kv),
        threadgroup=(32 * gqa, 1, 1),
        output_shapes=[(B, S, n_heads, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


__all__ = ["eligible", "prefill_attn"]
