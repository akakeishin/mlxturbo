"""QSA block-sparse attention prefill カーネル: 骨組み (レーン 3、着手段階)。

設計の根拠は `docs/research/QSA-PREFILL-KERNEL-DESIGN.md` を見ること。
このファイルを書いたセッションは **GPU 実行を禁止されている** (別の計測が
走行中)。したがって:

- :func:`build_union_blocks` はホスト側の union 添字列生成で、``mx`` の
  通常演算 (any / sort / take_along_axis) だけで書けるので **CPU で動く
  実コード**。`bench/test_qsa_prefill_attn_host.py` で検証済み。
- :func:`_source` 以下の Metal カーネル本体は **構造だけ**。
  `mx.fast.metal_kernel` でコンパイル・実行したことは一度も無い。次に
  この骨組みへ触るセッションは、まず `tools/gather_union_stats.py
  --tiles 4,8` (無改造で通る) で union 比の実測を取り、それから GPU 上で
  この Metal ソースをコンパイルして構文・境界条件を直すところから始める
  こと。

## 前回のカーネルとの違い (要点だけ、詳しくは設計文書)

`mlxturbo/kernels/prefill_attn.py` は 1 threadgroup = 1 クエリ x 1 kv
head で、K/V タイルの読み出しがクエリをまたいで共有されない。in-model
A/B (`bench/results/prefill-attn-ab*.json`) は dense と同速 (-1.1〜
-1.25%) だった。この骨組みは 1 threadgroup に T クエリ (既定 4) を載せ、
K/V タイルを T クエリで共有する。可視集合の和集合 (union) はホスト側で
1 回のバッチ演算として作り (`build_union_blocks`)、カーネルへは昇順・
パディング済みの添字列として渡す --- 旧カーネルが毎クエリ x 毎 kv head
でやっていた bool→添字の 3 パス圧縮 (`prefill_attn.py:142-176`) がカーネル
本体から消える。

union はクエリ自身の選択の和集合なので、各行は union 列の一部しか実際
には選んでいない。そのため旧カーネルには無かった行ごとの小さい bool
マスク (``row_keep``, T x U_pad ビット) が要る --- ただし kv_len に比例
しないので dense マスクへの逆戻りではない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from . import _fire

# prefill と decode/verify を分ける S の下限。`prefill_attn.py` の MIN_S と
# 同じ理由 (verify 幅で使うと online softmax の加算順が argmax の割れを
# 呼ぶ懸念、prefill だけに効かせて測る)。
MIN_S = 64

# threadgroup メモリの上限 (Apple GPU、prefill_attn.py と同じ値)。
MAX_TG_BYTES = 30 * 1024

# K/V 列タイルの目安。head_dim 256 / bf16 で BK=32 なら K と V で 32KB
# (旧カーネルの BK=16 の倍。相手 mlx-serve の既定 QSA_GATHER_BK=32 に
# 合わせた --- `~/dev/mlx-serve/src/transformer.zig` の
# QSA_GATHER_BK_DEFAULT)。ただし threadgroup メモリ上限との兼ね合いは
# `_tile_cols` が持つ。
_TARGET_COLS = 32

# union 添字列の丸め幅。カーネルのループはこの倍数単位で打ち切る。
PAD_MULTIPLE = 32

# 既定のタイル幅 (1 threadgroup が同時に処理するクエリ本数)。設計文書
# (b) の T。4 か 8 かは union 統計の実測後に決める --- ここでは 4 を仮の
# 既定値とし、呼び出し側で上書きできるようにする。
DEFAULT_TILE = 4

_KERNELS: dict[tuple, Any] = {}
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] qsa_prefill_attn カーネル: {msg}")


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# ホスト側: union 添字列の構築 (純 MLX、CPU で動く。GPU 不要)
# ---------------------------------------------------------------------------


@dataclass
class UnionBlocks:
    """:func:`build_union_blocks` の戻り値。

    ``union_idx`` は (B, n_tiles, u_pad) int32 で、各 (batch, tile) 行に
    そのタイルの T クエリが選んだブロックの和集合が **昇順・sentinel
    (=n_blocks) パディング**で入る。``row_keep`` は (B, s_pad, u_pad) bool
    で、s_pad 行それぞれについて union 列のうち自分が実際に選んだ列だけ
    True になる (パディング行・パディング列は常に False)。``s_pad`` は
    S を ``tile`` の倍数へ切り上げた値 --- 呼び出し側は出力の先頭 S 行
    だけを使う。
    """

    union_idx: Any  # mx.array (B, n_tiles, u_pad) int32
    row_keep: Any  # mx.array (B, s_pad, u_pad) bool
    u_pad: int
    n_tiles: int
    s_pad: int
    tile: int
    n_blocks: int


def build_union_blocks(
    keep_block: mx.array,
    tile: int,
    block_topk: int,
    *,
    pad_multiple: int = PAD_MULTIPLE,
) -> UnionBlocks:
    """``QSAIndexer.select_blocks`` の ``keep_block`` から union 添字列を作る。

    設計文書 (c) の op 列そのもの: T 行ごとに ``mx.any`` で和集合を取り、
    ``key = where(union, block_id, n_blocks)`` を ``mx.sort`` するだけで
    昇順・パディングが同時に手に入る (``argsort(-union)`` を使う
    `Attention._gather_tile_attn` と違い、True 集合内部の順序がソートの
    安定性に依存しない)。パディング幅は「タイルごとの動的な真の union
    サイズの最大値」を ``pad_multiple`` に丸めたもの --- host 同期は
    ``.item()`` の 1 回だけ (層・チャンクごとに 1 回。旧カーネルの
    「クエリごとに 2 バリア」よりずっと軽い)。

    ``keep_block``: (B, S, n_blocks) bool。``tile``: T (1 threadgroup が
    同時に処理するクエリ本数)。``block_topk``: モデルの
    ``token_budget // compress_ratio`` (安全上限の計算にだけ使う。実際の
    ループ幅は動的な ``u_pad`` で決まる)。
    """
    if keep_block.dtype != mx.bool_ or keep_block.ndim != 3:
        raise ValueError("keep_block は (B, S, n_blocks) の bool であること")
    if tile < 1:
        raise ValueError(f"tile は 1 以上であること (tile={tile})")
    if block_topk < 1:
        raise ValueError(f"block_topk は 1 以上であること (block_topk={block_topk})")

    B, S, n_blocks = keep_block.shape
    n_tiles = (S + tile - 1) // tile
    s_pad = n_tiles * tile

    if s_pad != S:
        # 端数タイルは全 False 行で埋める。パディング行は union にも
        # row_keep にも実害を与えない (union に何も足さない、自分の
        # row_keep は全 False のまま) --- 呼び出し側が先頭 S 行だけ使う
        # 前提で安全。
        pad_rows = mx.zeros((B, s_pad - S, n_blocks), dtype=mx.bool_)
        keep_padded = mx.concatenate([keep_block, pad_rows], axis=1)
    else:
        keep_padded = keep_block

    keep_tiled = keep_padded.reshape(B, n_tiles, tile, n_blocks)
    union = mx.any(keep_tiled, axis=2)  # (B, n_tiles, n_blocks)

    block_ids = mx.arange(n_blocks, dtype=mx.int32)
    sentinel = mx.array(n_blocks, dtype=mx.int32)
    key = mx.where(union, block_ids, sentinel)
    sorted_idx = mx.sort(key, axis=-1)  # True の実添字が昇順、False は末尾

    # 静的な安全上限 (host 同期なしで決まる)。実際の幅はこれより小さい
    # ことがほとんど --- 設計文書 (b) の見込みでは T=4 で kv の 2-5 割台。
    u_cap = min(n_blocks, tile * block_topk)
    sorted_idx = sorted_idx[..., :u_cap]

    true_counts = mx.sum(union.astype(mx.int32), axis=-1)  # (B, n_tiles)
    u_dyn = int(mx.max(true_counts).item())  # host 同期はここ 1 回だけ
    u_pad = min(u_cap, _round_up(max(1, u_dyn), pad_multiple))

    union_idx = sorted_idx[..., :u_pad]  # (B, n_tiles, u_pad)

    # row_keep: 各行が union 列のうちどれを実際に選んでいるか。sentinel
    # 添字 (=n_blocks) がダミー列を指すよう、末尾に常に False の列を足す。
    keep_ext = mx.concatenate(
        [keep_tiled, mx.zeros((B, n_tiles, tile, 1), dtype=mx.bool_)], axis=-1
    )
    idx_bcast = mx.broadcast_to(
        union_idx[:, :, None, :], (B, n_tiles, tile, u_pad)
    )
    row_keep_tiled = mx.take_along_axis(keep_ext, idx_bcast, axis=-1)
    row_keep = row_keep_tiled.reshape(B, s_pad, u_pad)

    return UnionBlocks(
        union_idx=union_idx,
        row_keep=row_keep,
        u_pad=u_pad,
        n_tiles=n_tiles,
        s_pad=s_pad,
        tile=tile,
        n_blocks=n_blocks,
    )


# ---------------------------------------------------------------------------
# デバイス側: Metal カーネルの構造 (未検証、次段の GPU セッションで固める)
# ---------------------------------------------------------------------------


def _tile_cols(cr: int, head_dim: int, itemsize: int) -> int:
    """1 回の K/V ステージングで読む列数 ``bk`` (`prefill_attn._tile_blocks`
    と同じ役割。ただし union はブロック単位ではなくトークン単位で歩くので、
    ``cr`` の倍数に丸めるだけで良い --- ブロック境界を跨いでタイルを切っても
    ``union_idx`` はブロック添字の列举なので、1 タイル = ``bk // cr`` 個の
    union エントリとして読める)。"""
    bk = max(cr, (_TARGET_COLS // cr) * cr)
    while bk > cr and 2 * bk * head_dim * itemsize > MAX_TG_BYTES:
        bk -= cr
    return bk


def _source(
    head_dim: int, gqa: int, cr: int, tile: int, bk: int, scale: float
) -> str:
    """Metal カーネルソースの **構造だけ**。境界条件・数値の細部は未検証。

    設計:

    - grid の threadgroup 数は (1, n_tiles, B*n_kv) --- x 方向 32*gqa
      スレッドで 1 threadgroup。旧カーネル (`prefill_attn.py`) と同じ
      threadgroup 形だが、y 軸がクエリ 1 本ではなく **T 本のタイル**
      になった。
    - simdgroup 数は ``gqa`` のまま (T に依存させない --- 「T x gqa
      simdgroup」は Apple GPU の 32 simdgroup (1024 thread) 上限を
      T>=3 で超える。設計文書 (b) の注記どおり、1 simdgroup = 1 q head
      が T 行を順番に処理する)。
    - threadgroup メモリは K/V ステージングタイル (bk x head_dim, T と
      無関係) だけ。union の添字列・行マスクは device メモリ (`union`,
      `row_keep` 入力) をそのまま読む --- 旧カーネルの `tg_sel` 相当の
      threadgroup 配列は不要になった (ホスト側で先に作ってあるため)。
    - 各 simdgroup は T 本ぶんの online softmax 状態 (``m[T]``,
      ``l[T]``, ``acc[T][dpl]``) をレジスタに保持する。K/V タイルは
      T 行で共有して 1 回だけ読み、行ごとのループでスコアを計算する
      ときに ``row_keep`` でマスクする。
    - 端数ブロック (tail, ブロック格子の外) は旧カーネルと同じ因果窓の
      算数 (``col <= q_col``) で処理する。行ごとに ``q_col`` が違うだけ
      で、追加のマスク配列は要らない。
    """
    d = head_dim
    dpl = (d + 31) // 32  # 1 lane が持つ head_dim 要素数
    nsg = gqa  # simdgroup 数 = 1 kv head に属する q head 数 (T に依らない)
    nth = nsg * 32

    return f"""
    const int T       = {tile};       // 1 threadgroup が同時に処理するクエリ本数
    const int NB      = params[0];    // 完全ブロック数 (kv_len / cr)
    const int KVLEN   = params[1];    // 可視な列の総数 (offset + S)
    const int CAP     = params[2];    // KV キャッシュの列方向の確保幅
    const int OFFSET  = params[3];    // このチャンク先頭のキャッシュ列位置
    const int NKV     = params[4];    // kv head 数
    const int UPAD    = params[5];    // union 添字列の実幅 (呼び出しごとに動的)
    const int S_PAD   = params[6];    // T の倍数へ切り上げたクエリ総行数
    const int NTILES  = params[7];    // S_PAD / T (grid.y と一致するはず)

    const uint tid   = thread_position_in_threadgroup.x;
    const uint sg    = simdgroup_index_in_threadgroup;
    const uint lane  = thread_index_in_simdgroup;
    const int  g     = (int)threadgroup_position_in_grid.y;   // タイル (union) 添字
    const int  bh    = (int)threadgroup_position_in_grid.z;
    const int  b     = bh / NKV;
    const int  kvh   = bh - b * NKV;
    const int  h     = kvh * {nsg} + (int)sg;
    const int  row0  = g * T;   // このタイルの先頭クエリ行 (S_PAD 空間)

    // union 添字列とタイルごとの行マスク。どちらも device メモリのまま
    // 読む --- threadgroup メモリへは持ち上げない (最適化は後)。
    const device int*  union_row = union + ((size_t)b * NTILES + g) * (size_t)UPAD;
    const device bool* keep_base = row_keep + (size_t)b * S_PAD * (size_t)UPAD;

    threadgroup T tg_k[{bk} * {d}];
    threadgroup T tg_v[{bk} * {d}];

    // T 本ぶんの online softmax 状態。T は既定 4-8 なので acc は
    // T*dpl 個 (T=4, dpl=8 なら 32 float) --- レジスタに収まる想定。
    float qv[T][{dpl}];
    float acc[T][{dpl}];
    float m[T];
    float l[T];
    int   q_col[T];
    for (int r = 0; r < T; ++r) {{
        const int srow = row0 + r;
        // パディング行 (srow >= 実際の S) は q_col が実データ範囲外になり
        // うるが、union 側は build_union_blocks が全 False で埋めている
        // ので l[r] が 0 のまま = 出力 0、host 側 (`out[:, :S]`) が
        // そもそも捨てる行なので実害は無い (prefill_attn.py:133 と同じ
        // 「OFFSET + 行番号」の素直な形)。
        q_col[r] = OFFSET + srow;
        const device T* qrow = q + (((size_t)b * S_PAD + srow) * NKV * {nsg} + h) * {d};
        for (int t = 0; t < {dpl}; ++t) {{
            const int dd = (int)lane + 32 * t;
            qv[r][t] = (dd < {d}) ? (float)qrow[dd] : 0.0f;
            acc[r][t] = 0.0f;
        }}
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }}

    const device T* kbase = k + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    const device T* vbase = v + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};

    // --- union ブロックぶんのタイル (bk 列ずつ、ブロック添字を union から
    // 引いて token 列へ展開する) -----------------------------------------
    const int ntiles_union = (UPAD * {cr} + {bk} - 1) / {bk};
    for (int ti = 0; ti < ntiles_union; ++ti) {{
        const int col0 = ti * {bk};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int)tid; e < {bk} * {d}; e += {nth}) {{
            const int cc = e / {d};
            const int dd = e - cc * {d};
            const int ucol = col0 + cc;
            const int ublk = ucol / {cr};
            const bool valid = ublk < UPAD;
            const int blk = valid ? union_row[ublk] : NB;  // NB = out-of-range sentinel
            const int col = blk * {cr} + (ucol - ublk * {cr});
            const bool ok = valid && (blk < NB);
            tg_k[e] = ok ? kbase[(size_t)col * {d} + dd] : (T)0;
            tg_v[e] = ok ? vbase[(size_t)col * {d} + dd] : (T)0;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int r = 0; r < T; ++r) {{
            const device bool* row_mask = keep_base
                + (size_t)(row0 + r) * (size_t)UPAD;
            float sc[{bk}];
            float mt = -INFINITY;
            for (int j = 0; j < {bk}; ++j) {{
                const int ucol = col0 + j;
                const int ublk = ucol / {cr};
                const bool visible = (ublk < UPAD) && row_mask[ublk];
                float p = 0.0f;
                if (visible) {{
                    for (int t = 0; t < {dpl}; ++t) {{
                        const int dd = (int)lane + 32 * t;
                        if (dd < {d}) {{ p += qv[r][t] * (float)tg_k[j * {d} + dd]; }}
                    }}
                }}
                const float sj = simd_sum(p) * {scale!r}f;
                sc[j] = visible ? sj : -INFINITY;
                mt = metal::max(mt, sc[j]);
            }}

            const float mnew = metal::max(m[r], mt);
            const float corr = metal::exp(m[r] - mnew);
            for (int t = 0; t < {dpl}; ++t) {{ acc[r][t] *= corr; }}
            l[r] *= corr;
            for (int j = 0; j < {bk}; ++j) {{
                if (sc[j] == -INFINITY) {{ continue; }}
                const float p = metal::exp(sc[j] - mnew);
                l[r] += p;
                for (int t = 0; t < {dpl}; ++t) {{
                    const int dd = (int)lane + 32 * t;
                    if (dd < {d}) {{ acc[r][t] += p * (float)tg_v[j * {d} + dd]; }}
                }}
            }}
            m[r] = mnew;
        }}
    }}

    // --- 端数列 (ブロック格子の外)。旧カーネルと同じ因果窓の算数 -------
    // TODO: 構造のプレースホルダ。tail_base / ntail_vis の導出と
    // KV バッファからの読み出しは prefill_attn.py:178-183 と
    // 200-254 を T 行ループへ展開する形で次段に書く。

    for (int r = 0; r < T; ++r) {{
        const int srow = row0 + r;
        const float inv = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
        device T* orow = out + (((size_t)b * S_PAD + srow) * NKV * {nsg} + h) * {d};
        for (int t = 0; t < {dpl}; ++t) {{
            const int dd = (int)lane + 32 * t;
            if (dd < {d}) {{ orow[dd] = (T)(acc[r][t] * inv); }}
        }}
    }}
"""


def _get_kernel(head_dim, gqa, cr, tile, bk, scale):
    key = (head_dim, gqa, cr, tile, bk, scale)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"qsa_prefill_attn_{head_dim}_{gqa}_{cr}_{tile}_{bk}",
            input_names=["q", "k", "v", "union", "row_keep", "params"],
            output_names=["out"],
            source=_source(head_dim, gqa, cr, tile, bk, scale),
        )
        _KERNELS[key] = kern
    return kern


def eligible(
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
    tile: int,
) -> bool:
    """このカーネルで扱える形か。`prefill_attn.eligible` と同じ判定群に
    ``tile`` の妥当性チェックを足したもの。**この関数自体は GPU 分岐
    (``mx.default_device() == mx.gpu``) を持つので、CPU (`tools/
    vendor_fingerprint.py`) では常に False になる** --- 一次検査の限界は
    設計文書 (d) のとおり。
    """
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
        _warn_once("headdim", "k/v の head_dim が q と違う")
        return False
    n_kv = k.shape[1]
    if n_kv <= 0 or q.shape[1] % n_kv != 0:
        _warn_once("n_kv", f"n_kv={n_kv} が 0 以下、または q の head 数を割り切れない")
        return False
    gqa = q.shape[1] // n_kv
    if gqa < 1 or gqa > 32:
        # threadgroup は 32*gqa スレッド。prefill_attn.eligible と同じ理由
        # (32 simdgroup / 1024 thread が Apple GPU の上限)。T はここに
        # 効かない --- simdgroup 数は gqa のまま (設計文書 (b))。
        _warn_once("gqa", f"GQA {gqa} は 1..32 の外")
        return False
    if cr < 1 or block_topk < 1:
        _warn_once("cr_topk", f"cr={cr} block_topk={block_topk} のどちらかが 1 未満")
        return False
    if tile < 1:
        _warn_once("tile", f"tile={tile} は 1 以上であること")
        return False
    if n_blocks * cr > kv_len or kv_len - n_blocks * cr >= cr:
        _warn_once("blocks", "n_blocks と kv_len の関係が想定と違う")
        return False

    itemsize = q.dtype.size
    bk = _tile_cols(cr, head_dim, itemsize)
    tg = 2 * bk * head_dim * itemsize
    if tg > MAX_TG_BYTES:
        _warn_once(
            "tg", f"threadgroup メモリ {tg} バイトが上限 {MAX_TG_BYTES} を超える"
        )
        return False
    return True


def qsa_prefill_attn(
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
    tile: int = DEFAULT_TILE,
) -> mx.array:
    """設計文書 (b)/(c) の呼び出し口。**未検証**: `build_union_blocks` は
    CPU で確認済みだが、Metal カーネル本体 (`_source`) は構文・境界条件
    ともに未検証。GPU が使えるセッションで `tools/verify_prefill_attn.py`
    の流儀 (設計文書 (d)) の検査を通してから使うこと。
    """
    _fire.bump("qsa_prefill_attn")
    B, n_heads, S, head_dim = q.shape
    n_kv = k.shape[1]
    gqa = n_heads // n_kv
    itemsize = q.dtype.size
    bk = _tile_cols(cr, head_dim, itemsize)

    union = build_union_blocks(keep_block, tile, block_topk)

    q_bshd = q.transpose(0, 2, 1, 3)
    params = mx.array(
        [n_blocks, kv_len, k.shape[2], offset, n_kv, union.u_pad, union.s_pad,
         union.n_tiles],
        dtype=mx.int32,
    )

    kernel = _get_kernel(head_dim, gqa, cr, tile, bk, float(scale))
    (out,) = kernel(
        inputs=[q_bshd, k, v, union.union_idx, union.row_keep, params],
        template=[("T", q.dtype)],
        grid=(32 * gqa, union.n_tiles, B * n_kv),
        threadgroup=(32 * gqa, 1, 1),
        output_shapes=[(B, union.s_pad, n_heads, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out[:, :S]


__all__ = ["build_union_blocks", "UnionBlocks", "eligible", "qsa_prefill_attn"]
