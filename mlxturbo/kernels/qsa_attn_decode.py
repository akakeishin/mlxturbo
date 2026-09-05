"""decode / verify 幅 (S <= 8) の QSA attention を MLX の 2-pass vector の
**写し**として書き直す (段 K2b)。

置き換える相手は `Attention.__call__` の

    mask = self._final_mask(mask, sparse, cache, S, q.dtype)   # (B,1,S,kv_len) bool
    ...  scaled_dot_product_attention(q[:, :, i:i+2], k, v, mask=mask[..., i:i+2, :])

という並び、つまり **bool マスクの実体化 + `ceil(S*gqa/32)` 回の sdpa 呼び**。
MLX 側 (`mlx/backend/metal/kernels/sdpa_vector.h` の `sdpa_vector_2pass_1` /
`_2pass_2`) は kv を ``blocks`` 本に等分し、1 simdgroup が
``i = block_idx, block_idx+blocks, ...`` を**全部**回って、毎回
``bmask[0]`` (device メモリ 1 バイト) を待ってから K/V を読むか決める。
17k・blocks=512 なら 1 (head, 行) あたり 33 回の**依存ロードの直列**で、
当たりは平均 4 個しかない。

ここでやることはその 1 点だけの置き換えである:

- ``bmask[0]`` の device 読み → K2a (`mlxturbo/kernels/qsa_select.py`) が
  出した **keep ビットマップ** (ブロック単位、1 行 1.6 KB まで) と、
  **クエリごとの端数規則** (列 ``cr*floor((q+1)/cr) .. q``) の算数。
- 候補 (最大 256 個ずつ) の可視判定を **threadgroup の 384 スレッドで一度に**
  やって当たりのビットマスクを作り、各 simdgroup はそこから当たりだけを
  昇順に拾う。12 simdgroup が同じ判定を 12 回やるのをやめる分と、
  K/V のアドレスが判定より前に決まる分が取り分。

**それ以外は写しである。**同じ ``i ≡ block_idx (mod blocks)`` の分割、同じ
逐次 online softmax (`fast::exp`、`o = o*factor + p*v`、q に scale を先に
掛ける)、同じ bf16 partials、同じ pass 2。狙いは
`mx.fast.scaled_dot_product_attention` とのビット一致で、判定は
`tools/verify_qsa_attn_decode.py` が `mx.array_equal` で取る。

## S の割り方 (本家の分割との関係)

MLX の vector 経路は ``S * gqa <= 32`` しか受けないので、本家 (`__call__`)
は S>=3 を 2 行ずつに割って ``ceil(S/2)`` 回呼ぶ。2pass_1 の計算は
**(head, 行) ごとに独立**で、``q_seq_len`` は添字にしか出てこないから、
行のまとめ方は数値に影響しない。影響するのは ``blocks`` だけ
(`scaled_dot_product_attention.cpp:494-520` の表が ``n_simds = gqa *
q_seq_len`` を見る)。よってこのカーネルは **1 threadgroup = 1 行**
((32, gqa, 1) = 384 スレッド) に統一し、``blocks`` だけを本家の分割どおりに
選ぶ (:func:`mirror_blocks`。割った断片で値が食い違う形は不適格にする)。
threadgroup 数は本家の S 倍になるので、並列度は落ちない。

## 端数 (tail) の規約

`mlxturbo/qsa_tail.py` の ``"query"`` (HF 参照と同じ) だけを実装する。
すなわち可視集合は

    「完全ブロック (block_end <= q) の top-k」 ∪ 「列 cr*floor((q+1)/cr) .. q」

で、**kv 全体末尾の端数 (global tail) は使わない**。``MODE == "global"``
のときは :func:`eligible` が False を返す (規約が 2 つあると参照が 2 本に
なり、ビット一致の判定が意味を失うため)。

## ``blocks`` の選び方 (2026-09-03 の実測)

``blocks`` は「本家と同じ値」でなければビット一致しないが、**両側を
``MLX_SDPA_BLOCKS`` で同じ値に釘付けすれば、どの値でも一致する**。冷たい
12 層連鎖 (`tools/verify_qsa_attn_decode.py`) で測ると、このカーネルの
時間は本家の表 (M3 Max = arch 末尾 ``'s'`` で kv 17k -> 256、50k -> 512)
より **32〜64 に釘付けした方がずっと速く、しかも kv にほぼ依らない**
(us/層、S=2): 表どおり 123 (17k) / 170 (50k) に対し、64 で 92 / 100、
32 で 90 / 96。理由は 2 つ --- (a) partials の往復が ``blocks`` に比例する、
(b) このカーネルが実際に読む列は budget 2048 + 端数で頭打ちなので、
``blocks`` を減らしても 1 threadgroup の仕事は kv ではなく budget で決まる。
本家は逆に全キーの mask バイトを走査するので kv に比例して伸びる。
製品経路は :func:`decode_blocks` で、QSAの疎なdecodeに限りKV長16,000〜18,000を
64へ縮める。17kの実モデルA/Bでround -1.2%、長文課題も非退行だった範囲だけを
使い、品質低下が出た50kは本家の表へ戻す。``MLXTURBO_QSA_BLOCKS64=0`` で
本家の表だけに戻せる。

## 配線 (段 K2c)

`Attention._decode_qsa_forward` (`mlxturbo/_vendor/qwen4_exp.py`) が
`_gather_forward` より前の第 3 分岐として呼ぶ。**既定 off**
(`mlxturbo/qsa_decode.py`、環境変数 ``MLXTURBO_QSA_DECODE_KERNEL=1``)。
適格判定はキャッシュを触る前にホスト側の値だけで済ませる規約なので、
:func:`eligible` の**構造条件は呼び出し側にも写してある** --- ここを変えたら
`_decode_qsa_forward` の前半も見ること。配線そのものの一次検査は
`tools/verify_qsa_attn_decode.py` の「配線」節 (knob on/off の `array_equal`)。
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from . import _fire

# 2-pass の partials を畳む単位 (`sdpa_vector_2pass_2` の BN)。``blocks`` は
# この倍数でなければ末尾が黙って落ちる (MLX 側も同じ丸めをしている)。
BN = 32

# 1 threadgroup が 1 回に判定する候補数。候補は
# ``i = block_idx + t * blocks`` で、17k/blocks=512 なら 33 個、
# 50k/blocks=128 なら 391 個。これを超える分は同じ手順を繰り返す
# (走る順は昇順のままなので加算順は変わらない)。
TMAX = 256

_KERNELS: dict[tuple, Any] = {}
_SCALES: dict[float, mx.array] = {}
_warned: set = set()

# GPU アーキテクチャ名は 1 プロセスで変わらないのに、`mx.device_info()` は
# 呼ぶたびに C++ の map を Python の dict へ作り直す。`mirror_blocks` は
# 層ごと (S>=3 なら分割ごと) に呼ばれるので、decode の 1 フォワードで
# 12〜24 回ぶん積み上がる。1 度だけ引いて覚える。
_ARCH_CHAR: str | None = None

# 小さい入力配列 (params / blocks) の 1 個メモ。1 フォワードの 12 層は
# offset も kv_len も同じなので、鍵が一致する限り 12 回の `mx.array` 構築が
# 1 回になる。**鍵が値を完全に決める**ので、当たれば必ず同じ中身
# (スレッドが混ざっても外れるだけで、間違った配列は返らない)。
_P1_MEMO: tuple[tuple, mx.array] | None = None
_P2_MEMO: dict[int, mx.array] = {}


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] QSA decode attention カーネル: {msg}")


# --------------------------------------------------------------------------
# ``blocks`` の表 (本家 `scaled_dot_product_attention.cpp:487-522` の写し)
# --------------------------------------------------------------------------
def arch_char() -> str:
    """GPU アーキテクチャ名の末尾 1 文字 (本家の ``devc``)。

    1 プロセスで変わらないので覚える (`_ARCH_CHAR`)。`mirror_blocks` が
    decode の 1 フォワードで 12〜24 回呼ばれる口なので、`mx.device_info()`
    の dict 変換をそのつど払うと無視できない。
    """
    global _ARCH_CHAR
    if _ARCH_CHAR is None:
        try:
            info = (
                mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info()
            )
            arch = str(info.get("architecture", ""))
        except Exception:
            arch = ""
        _ARCH_CHAR = arch[-1] if arch else ""
    return _ARCH_CHAR


def sdpa_blocks(n_kv: int, n_simds: int, devc: str | None = None) -> int:
    """MLX が `sdpa_vector_2pass` で使う ``blocks`` を返す。

    ``n_simds = gqa_factor * q_seq_len``。``MLX_SDPA_BLOCKS`` の上書きも
    本家と同じ丸め (32 の倍数へ切り上げ) で効かせる --- 環境変数が唯一の
    真実になるので、ビット一致の検証では両側をこれで釘付けにする。
    """
    if devc is None:
        devc = arch_char()
    if devc == "s":
        blocks = 64
        if n_kv > 1024 and n_simds > 4:
            if n_kv <= 8192:
                blocks = 128
            elif n_kv <= 32768:
                blocks = 256
            elif n_kv <= 65536:
                blocks = 512
            else:
                blocks = 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and n_kv > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_kv < 65536:
                blocks = 512
            elif n_kv >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    try:
        env = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0)
    except ValueError:
        env = 0
    if env > 0:
        blocks = ((env + 31) // 32) * 32
    return blocks


def split_widths(s_len: int, gqa: int, split_enabled: bool = True) -> list[int]:
    """本家 `Attention.__call__` の sdpa 幅分割が作る呼び出しごとの行数。"""
    if split_enabled and 1 < s_len <= 8 and s_len * gqa > 32:
        step = max(1, 32 // gqa)
        return [min(step, s_len - i) for i in range(0, s_len, step)]
    return [s_len]


def mirror_blocks(n_kv: int, gqa: int, s_len: int, devc: str | None = None):
    """本家の分割が全断片で同じ ``blocks`` を使うならその値、違えば ``None``。

    値が断片ごとに違うと「1 本のカーネルで写す」が成り立たない (行ごとに
    kv の分割が変わる) ので、そのときは不適格にして既存経路へ落とす。
    """
    vals = {
        sdpa_blocks(n_kv, gqa * w, devc) for w in split_widths(s_len, gqa)
    }
    if len(vals) != 1:
        return None
    blocks = vals.pop()
    return blocks if blocks % BN == 0 else None


def decode_blocks(n_kv: int, gqa: int, s_len: int, devc: str | None = None):
    """QSA decode専用の分割数。長文品質を確認した17k帯だけ64へ縮める。"""
    blocks = mirror_blocks(n_kv, gqa, s_len, devc)
    if blocks is None:
        return None
    try:
        pinned = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0) > 0
    except ValueError:
        pinned = False
    if (
        not pinned
        and os.environ.get("MLXTURBO_QSA_BLOCKS64", "1") != "0"
        and 16000 <= n_kv <= 18000
    ):
        return 64
    return blocks


# --------------------------------------------------------------------------
# pass 1: `sdpa_vector_2pass_1` の写し (mask だけ差し替え)
# --------------------------------------------------------------------------
def stage_cols(d: int, itemsize: int) -> int:
    """1 バッチで threadgroup メモリへ載せる列数 ``BC``。

    K と V の両方を置くので ``2 * BC * d * itemsize`` バイト。Apple GPU の
    threadgroup メモリ 32 KB のうち、当たり添字 (`TMAX` 個の int) と候補
    ビットマスクの分を除いた 24 KB を上限にする。
    """
    bc = 16
    while bc > 1 and 2 * bc * d * itemsize > 24 * 1024:
        bc //= 2
    return bc


def _p1_source(d: int, gqa: int, cr: int, blocks: int, bc: int) -> str:
    qpt = d // 32
    nth = 32 * gqa
    cw = TMAX // 32
    return f"""
    typedef float U;
    constexpr int QPT    = {qpt};       // 1 レーンが持つ q/k/v の要素数
    constexpr int NTH    = {nth};       // threadgroup のスレッド数 (32 * gqa)
    constexpr int CW     = {cw};        // 候補ビットマスクの word 数
    constexpr int BLOCKS = {blocks};
    constexpr int BC     = {bc};        // 1 バッチで threadgroup に載せる列数

    const int NKV = params[0];          // kv head 数
    const int N   = params[1];          // kv_len
    const int CAP = params[2];          // KV キャッシュの列方向の確保幅
    const int NB  = params[3];          // 完全ブロック数 (kv_len / cr)
    const int NW  = params[4];          // keep ビットマップの word 数
    const int OFF = params[5];          // offset (= kv_len - S)
    const int S   = params[6];          // このフォワードのクエリ行数

    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint tid  = thread_index_in_threadgroup;

    const int kvh = (int)threadgroup_position_in_grid.x;
    const int bs  = (int)threadgroup_position_in_grid.y;   // b * S + s
    const int blk = (int)threadgroup_position_in_grid.z;   // block_idx
    const int b   = bs / S;
    const int s   = bs - b * S;
    const int HQ  = NKV * {gqa};
    const int h   = kvh * {gqa} + (int)sg;                 // q head
    const int q_col = OFF + s;

    // HF 参照の tail: 列 [cr*floor((q+1)/cr), q] は完全ブロックの選択と
    // 無関係に常に可視 (自分自身を含む)。q % cr == cr-1 の行では空になる。
    const int tail_base = ((q_col + 1) / {cr}) * {cr};

    threadgroup atomic_uint cand[CW];
    threadgroup int  hit[{TMAX}];       // 当たりの kv 列 (昇順)
    threadgroup int  tg_nhit;
    threadgroup T    tg_k[BC * {d}];    // 12 simdgroup で共有する K/V タイル
    threadgroup T    tg_v[BC * {d}];

    auto qp    = q + ((size_t)bs * HQ + h) * {d} + lane * QPT;
    auto kbase = keys   + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    auto vbase = values + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    auto brow  = bits + (size_t)bs * NW;

    const size_t o_off = ((size_t)b * HQ + h) * S + s;

    // ---- q を読んで出力累算器を 0 に (本家 2pass_1 と同じ順) -------------
    U qv[QPT];
    U o[QPT];
    for (int i = 0; i < QPT; i++) {{ qv[i] = static_cast<U>(scale[0]) * qp[i]; }}
    for (int i = 0; i < QPT; i++) {{ o[i] = 0; }}

    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;

    // ---- 候補 i = blk, blk+BLOCKS, ... を昇順に -------------------------
    const int T_all = (blk < N) ? ((N - blk + BLOCKS - 1) / BLOCKS) : 0;

    for (int c0 = 0; c0 < T_all; c0 += {TMAX}) {{
        for (int w = (int)tid; w < CW; w += NTH) {{
            atomic_store_explicit(&cand[w], 0u, memory_order_relaxed);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 可視判定は threadgroup 全体で 1 回だけ (12 simdgroup で共有する)
        const int nt = metal::min({TMAX}, T_all - c0);
        for (int t = (int)tid; t < nt; t += NTH) {{
            const int i = blk + (c0 + t) * BLOCKS;
            // 参照 (`QSAIndexer.__call__`) は「ブロック bool を cr 列へ展開」
            // と「tail」の**和集合**なので、そのまま和で書く。選ばれた完全
            // ブロックの列は必ず tail_base 未満なので実際には重ならないが、
            // 前提にはしない (keep が不正でも参照と同じものを返す)。
            const int bb = i / {cr};
            bool vis = (bb < NB) &&
                       (((brow[bb >> 5] >> ((uint)bb & 31u)) & 1u) != 0u);
            vis = vis || ((i >= tail_base) && (i <= q_col));
            if (vis) {{
                atomic_fetch_or_explicit(
                    &cand[t >> 5], 1u << ((uint)t & 31u), memory_order_relaxed);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 当たりを昇順に詰める (simdgroup 0 だけ、word あたり 1 レーン) --
        if (sg == 0u) {{
            const uint m0 = (lane < (uint)CW)
                ? atomic_load_explicit(&cand[lane], memory_order_relaxed) : 0u;
            const uint pc = (uint)popcount(m0);
            const uint off = simd_prefix_exclusive_sum(pc);
            uint m = m0;
            uint w = off;
            while (m != 0u) {{
                const uint t = (uint)metal::ctz(m);
                m &= (m - 1u);
                hit[w] = blk + (c0 + (int)lane * 32 + (int)t) * BLOCKS;
                w++;
            }}
            if (lane == 31u) {{ tg_nhit = (int)(off + pc); }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nhit = tg_nhit;

        // ---- BC 列ずつ threadgroup に載せてから回す ----------------------
        // K/V を読むのは threadgroup あたり 1 回だけになる (これまでは
        // 12 simdgroup が同じ行をそれぞれ読んでいた)。**演算の順は不変**
        // なので、本家とのビット一致は崩れない。
        for (int h0 = 0; h0 < nhit; h0 += BC) {{
            const int ncol = metal::min(BC, nhit - h0);
            threadgroup_barrier(mem_flags::mem_threadgroup);   // 前のタイルを守る
            for (int e = (int)tid; e < ncol * {d}; e += NTH) {{
                const int cc = e / {d};
                const int dd = e - cc * {d};
                const size_t col = (size_t)hit[h0 + cc];
                tg_k[e] = kbase[col * {d} + dd];
                tg_v[e] = vbase[col * {d} + dd];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- ここから先は本家 2pass_1 の写し (逐次 online softmax) ---
            for (int cc = 0; cc < ncol; ++cc) {{
                const threadgroup T* kp = tg_k + cc * {d} + lane * QPT;
                const threadgroup T* vp = tg_v + cc * {d} + lane * QPT;

                U score = 0;
                for (int j = 0; j < QPT; j++) {{ score += qv[j] * kp[j]; }}
                score = simd_sum(score);

                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;

                for (int j = 0; j < QPT; j++) {{
                    o[j] = o[j] * factor + exp_score * vp[j];
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // ---- partials を書く (本家と同じ配置・同じ bf16 丸め) ---------------
    if (lane == 0) {{
        sums[o_off * BLOCKS + blk] = sum_exp_score;
        maxs[o_off * BLOCKS + blk] = max_score;
    }}
    device T* op = partials + (o_off * BLOCKS + (size_t)blk) * {d} + lane * QPT;
    for (int j = 0; j < QPT; j++) {{ op[j] = static_cast<T>(o[j]); }}
"""


# --------------------------------------------------------------------------
# pass 2: `sdpa_vector_2pass_2` の写し (差分なし)
# --------------------------------------------------------------------------
def _p2_source(d: int) -> str:
    ept = d // 32
    return f"""
    typedef float U;
    constexpr int BN  = 32;
    constexpr int BD  = 32;
    constexpr int EPT = {ept};

    const int blocks = params[0];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    thread U o[EPT] = {{0}};
    threadgroup U outputs[BN * BD];

    const int head_idx  = (int)threadgroup_position_in_grid.x;
    const int q_seq_idx = (int)threadgroup_position_in_grid.y;
    const int q_offset  = head_idx * (int)threadgroups_per_grid.y + q_seq_idx;

    auto pp = partials + (size_t)q_offset * blocks * {d}
              + simd_gid * {d} + simd_lid * EPT;
    auto sp = sums + (size_t)q_offset * blocks;
    auto mp = maxs + (size_t)q_offset * blocks;
    device T* op = out + (size_t)q_offset * {d} + simd_gid * EPT;

    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;

    for (int b = 0; b < blocks / BN; ++b) {{
        max_score = max(max_score, mp[simd_lid + BN * b]);
    }}
    max_score = simd_max(max_score);

    for (int b = 0; b < blocks / BN; ++b) {{
        U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sp[simd_lid + BN * b];
    }}
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < blocks / BN; ++b) {{
        U factor = fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < EPT; i++) {{
            o[i] += factor * static_cast<U>(pp[i]);
        }}
        mp += BN;
        sp += BN;
        pp += BN * {d};
    }}

    for (int i = 0; i < EPT; i++) {{
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (simd_lid == 0) {{
        for (int i = 0; i < EPT; i++) {{ op[i] = static_cast<T>(o[i]); }}
    }}
"""


def _get_kernels(d: int, gqa: int, cr: int, blocks: int, bc: int):
    """(pass 1, pass 2) を返す。

    ``ensure_row_contiguous`` は既定 (True) のまま。``keys`` / ``values`` は
    KV キャッシュの**確保済みバッファそのもの**を渡す規約なので行連続で、
    コピーは起きない (`prefill_attn._kv_buffers` と同じ作法)。途中切りの
    ビューを渡すと MLX が毎回 KV 全体を複製するので注意。
    """
    p1 = _KERNELS.get(("p1", d, gqa, cr, blocks, bc))
    if p1 is None:
        p1 = mx.fast.metal_kernel(
            name=f"mlxturbo_qsa_attn_p1_{d}_{gqa}_{cr}_{blocks}_{bc}",
            input_names=["q", "keys", "values", "bits", "scale", "params"],
            output_names=["partials", "sums", "maxs"],
            source=_p1_source(d, gqa, cr, blocks, bc),
        )
        _KERNELS[("p1", d, gqa, cr, blocks, bc)] = p1
    p2 = _KERNELS.get(("p2", d))
    if p2 is None:
        p2 = mx.fast.metal_kernel(
            name=f"mlxturbo_qsa_attn_p2_{d}",
            input_names=["partials", "sums", "maxs", "params"],
            output_names=["out"],
            source=_p2_source(d),
        )
        _KERNELS[("p2", d)] = p2
    return p1, p2


def _params1(*vals: int) -> mx.array:
    """pass 1 の params。1 フォワードぶん (12 層) を 1 個メモで畳む。"""
    global _P1_MEMO
    memo = _P1_MEMO
    if memo is not None and memo[0] == vals:
        return memo[1]
    arr = mx.array(list(vals), dtype=mx.int32)
    _P1_MEMO = (vals, arr)
    return arr


def _params2(blocks: int) -> mx.array:
    """pass 2 の params (``blocks`` だけ)。取る値は数種類しかない。"""
    arr = _P2_MEMO.get(blocks)
    if arr is None:
        arr = mx.array([blocks], dtype=mx.int32)
        _P2_MEMO[blocks] = arr
    return arr


def _scale_arr(scale: float) -> mx.array:
    arr = _SCALES.get(scale)
    if arr is None:
        arr = mx.array([scale], dtype=mx.float32)
        _SCALES[scale] = arr
    return arr


# --------------------------------------------------------------------------
def eligible(
    q_bshd: mx.array,
    keys: mx.array,
    values: mx.array,
    bits: mx.array,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    offset: int,
    blocks: int | None,
) -> bool:
    """このカーネルで扱える形か。外れたら呼び出し側は既存の sdpa 経路へ。"""

    from mlxturbo import qsa_tail as _qsa_tail

    if _qsa_tail.MODE != "query":
        _warn_once(
            "tail_mode",
            f"MLXTURBO_QSA_TAIL={_qsa_tail.MODE} は未対応 (query だけを写した)",
        )
        return False
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if blocks is None or blocks % BN != 0 or blocks < BN:
        _warn_once("blocks", f"blocks={blocks} が本家の分割と揃わない")
        return False
    if q_bshd.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        _warn_once("ndim", "q は (B,S,Hq,D)、keys/values は (B,Hk,cap,D) が要る")
        return False
    if keys.shape != values.shape or keys.dtype != values.dtype:
        _warn_once("kv", "keys と values の形か dtype が揃っていない")
        return False
    if q_bshd.dtype != keys.dtype:
        _warn_once("dtype", "q と KV の dtype が揃っていない")
        return False
    B, S, hq, d = q_bshd.shape
    if B != keys.shape[0] or d != keys.shape[3]:
        _warn_once("shape", "q と KV の B / head_dim が食い違う")
        return False
    if d % 32 != 0:
        _warn_once("head_dim", f"head_dim={d} が 32 の倍数でない")
        return False
    n_kv = keys.shape[1]
    if n_kv < 1 or hq % n_kv != 0:
        _warn_once("gqa", f"Hq={hq} が Hk={n_kv} で割り切れない")
        return False
    if 32 * (hq // n_kv) > 1024:
        _warn_once("tg", f"gqa={hq // n_kv} は threadgroup 1024 スレッドを超える")
        return False
    if keys.shape[2] < kv_len or kv_len < 1:
        _warn_once("cap", "KV の確保幅が kv_len に足りない")
        return False
    if bits.dtype != mx.uint32 or bits.ndim < 2:
        _warn_once("bits", "keep ビットマップが (…, NW) の uint32 でない")
        return False
    if bits.size // bits.shape[-1] != B * S:
        _warn_once("bits_rows", "keep ビットマップの行数が B*S と合わない")
        return False
    if bits.shape[-1] < (n_blocks + 31) // 32:
        _warn_once("bits_nw", "keep ビットマップの word 数が n_blocks に足りない")
        return False
    if n_blocks * cr > kv_len or kv_len - n_blocks * cr >= cr:
        _warn_once("nblocks", "n_blocks と kv_len の関係が想定と違う")
        return False
    if offset + S != kv_len:
        _warn_once("offset", "offset + S が kv_len と一致しない")
        return False
    if offset < cr - 1:
        # 可視ブロックが 1 つも無い行の救済 (`__call__` 末尾) は写していない
        _warn_once("offset_small", f"offset={offset} < cr-1 の救済は未対応")
        return False
    return True


def qsa_attn_decode(
    q_bshd: mx.array,
    keys: mx.array,
    values: mx.array,
    bits: mx.array,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    offset: int,
    scale: float,
    blocks: int,
) -> mx.array:
    """QSA の可視集合で decode / verify 幅の attention を計算する。

    ``q_bshd``: (B, S, Hq, D) --- 射影直後の行連続な並び。``Attention`` が
        持っている (B, Hq, S, D) は転置ビューなので、``q.transpose(0,2,1,3)``
        で元に戻して渡す (コピーは起きない)。
    ``keys`` / ``values``: (B, Hk, cap, D) の**確保済みバッファそのもの**
        (`KVCache.keys` / `.values`)。途中切りのビューを渡すと行連続でなく、
        `metal_kernel` が毎回全体をコピーする。
    ``bits``: K2a (`qsa_select.select(..., mode="bits")`) の
        (B, S, ceil(n_blocks/32)) uint32。**端数 (tail) は入っていない**
        --- こちらが ``q`` から直に判定する。
    ``blocks``: MLX の 2-pass と同じ kv 分割数 (:func:`mirror_blocks`)。

    戻り値は sdpa と同じ (B, Hq, S, D)。
    """

    _fire.bump("qsa_attn_decode")
    B, S, hq, d = q_bshd.shape
    n_kv = keys.shape[1]
    gqa = hq // n_kv
    cap = keys.shape[2]
    nw = bits.shape[-1]

    p1, p2 = _get_kernels(d, gqa, cr, blocks, stage_cols(d, q_bshd.dtype.size))
    params1 = _params1(n_kv, kv_len, cap, n_blocks, nw, offset, S)
    partials, sums, maxs = p1(
        inputs=[
            q_bshd,
            keys,
            values,
            bits.reshape(B * S, nw),
            _scale_arr(float(scale)),
            params1,
        ],
        template=[("T", q_bshd.dtype)],
        grid=(32 * n_kv, gqa * B * S, blocks),
        threadgroup=(32, gqa, 1),
        output_shapes=[(B, hq, S, blocks, d), (B, hq, S, blocks), (B, hq, S, blocks)],
        output_dtypes=[q_bshd.dtype, mx.float32, mx.float32],
    )
    (out,) = p2(
        inputs=[partials, sums, maxs, _params2(blocks)],
        template=[("T", q_bshd.dtype)],
        grid=(1024 * B * hq, S, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, hq, S, d)],
        output_dtypes=[q_bshd.dtype],
    )
    return out


__all__ = [
    "BN",
    "TMAX",
    "stage_cols",
    "arch_char",
    "decode_blocks",
    "eligible",
    "mirror_blocks",
    "qsa_attn_decode",
    "sdpa_blocks",
    "split_widths",
]
