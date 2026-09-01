"""decode (MTP verify) の MoE gather 向け、共有タイル gather カーネルの下ごしらえ。

## 依頼1: ディスパッチ特定 (Qwen3.8-Flash-Next 4bit, M3 Max / applegpu_g15s)

verify の 1 フォワードは S=3 (cur + draft 2 本)。MoE 入力 x=(1,3,2560)、
ルータ添字 idx=(1,3,10) (E=512, top_k=10, hidden=2560, moe_intermediate=640,
4bit affine gs=64)。`mlxturbo.fused.enable_gather_sort` (既定 min_size=16) が
`SwitchGLU.__call__` を差し替えていて、indices.size=30 >= 16 で常にソート
経路に入る。

形状の流れ (`mlx_lm.models.switch_layers._gather_sort` を実測で追った。
`.venv/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py` が実体で、
このリポジトリの `mlxturbo/_vendor/switch_layers.py` という単独ファイルは
存在しない — `_vendor/qwen4_exp.py` は `mlx_lm.models.switch_layers` を
そのまま import しているだけ):

    x            (1, 3, 2560)
    expand_dims(x, (-2,-3))  -> (1, 3, 1, 1, 2560)
    _gather_sort -> x (30, 1, 2560), idx (30,) [昇順ソート済み], inv_order (30,)

つまり **M の数え方は 30 (= S×top_k) だが、これは gather_qmm 側では
「バッチ次元 B」であって「行列積の行数 M」ではない**。x の最後から2軸が
(1, 2560) なので C++ 側の `M = x.shape(-2)` は **1**。この違いが分岐選択に
そのまま効く (下記)。gate_proj/up_proj/down_proj いずれも `SwitchLinear.__call__`
経由で `mx.gather_qmm(x, weight, scales, biases, rhs_indices=idx,
transpose=True, group_size=64, bits=4, mode="affine", sorted_indices=True)`
を呼ぶ。x は (30,1,2560) (gate/up) / (30,1,640) (down)。

`GatherQMM::eval_gpu` (quantized_host.cpp) の分岐、この形状での実値:

    K = x.shape(-1)          # gate/up: 2560, down: 640
    M = x.shape(-2)          # 1 (どの層も)
    N = out.shape(-1)        # gate/up: 640, down: 2560
    B = out.size()/M/N       # 30
    E = w.size()/w.shape(-2)/w.shape(-1)  # 512

    1) M==1 && B>=16 && right_sorted_ && B/E>=4
       -> B/E = 30/512 = 0 (整数除算)。**ソート済みでもこの分岐には入らない**。
          この分岐は「バッチ全体で 1 エキスパートあたり平均 4 行以上」を前提に
          した prefill 向けの重み再利用パスで、decode の 30 行 / 512 エキスパート
          には届かない (今回のカーネルが埋める空白そのもの)。
    2) M >= vector_limit(transpose_ ? get_qmv_batch_limit(K,N,d) : 4)
       -> M3 Max (g15s, arch_gen=15) では get_qmv_batch_limit が gate/up・down
          いずれも 15 を返す (D<=4096 && O<=4096 の枝)。M=1 < 15 で不成立。
    3) transpose_==true -> gather_qmv(...) が実際に呼ばれる経路。

    gather_qmv 内部 (bn=8, bk=32, threadgroup=(32,2,1)=64スレッド/2simdgroup):
        grid = (M=1, ceil(N/8), B=30)
        fast = N%8==0 && K%qmv_fast_k_alignment(bits=4)==0
             qmv_fast_k_alignment(4) = pack_factor(4,32)*2*32 = 512
        gate/up: K=2560 (%512==0) かつ N=640 (%8==0) -> fast=True
                 カーネル "{mode}_gather_qmv_fast_{type}_gs_64_b_4"
                 (mode 文字列は quantization_mode_to_string の実体が同梱
                 ヘッダに無く未確認。affine 想定)
        down   : K=640  (640%512=128 != 0) -> fast=False
                 カーネル "{mode}_gather_qmv_{type}_gs_64_b_4"
                 **down だけ qmv_fast の対象から外れる** (K=640 が 512 の
                 倍数でないため)。gate/up が抜けているのと対称的な見落とし
                 やすい非対称性。

    ディスパッチ回数: 1 verify ラウンドで 1 MoE 層あたり **gate/up/down で
    Metal カーネル起動 3 回**。3 トークン分 (30 行) は grid.z=B=30 の中で
    1 回の起動にまとめて処理されるので「1 トークンあたり」という数え方は
    実体と合わない — 3 トークンぶんまとめて 3 回、が正しい数え方。
    48 層 (num_hidden_layers) なので MoE 全体では 1 verify あたり 144 回。

## 依頼2: v1 の設計

上の 1) が空振りする理由は「バッチ全体で見た再利用率」を条件にしている
ため。decode の実態は「S=3 トークンがそれぞれ top_k=10 を引き、稀に別の
トークン同士が同じエキスパートを引く (1 トークン内では argpartition の
性質上 top_k は相異なるので、重複はトークン間でしか起きず高々 S=3 重ね)」
なので、**個々のセグメント単位では再利用が起きても、全体平均 (B/E) では
絶対に閾値 4 に届かない**。ここを "average" ではなく "per-segment" で
見るのが v1 の要点: ソート済み idx を連続ランで区切り (エキスパートID,
開始行, 行数) をホスト側 (python) で計算し、1 threadgroup = 1 セグメント
として重み (gate_w/gate_s/gate_b) を 1 回だけ読み、セグメント内の全行に
適用する。

カーネル内部構造は `kernels/moe_glu.py` の qmv_fast 系 (simdgroup 2 本 x
出力 4 行 = ROWS_PER_TG=8、K を 512 ずつ読む、bias は sum(x) 使い回し) を
そのまま踏襲し、「1 pair」だったループを「1 セグメント内の最大 MAX_SEG 行」
に広げた。重み読み (gw2/gsl/gbl の advance) は it (K/512 のブロック) と
j (出力4行のどれか) の組でちょうど 1 回だけ起こり、その内側で
`for r in range(MAX_SEG)` として各行の x ベクトルに同じレジスタ上の重みを
適用する — ここが「重みは1回、行は複数」の実体。

MAX_SEG はホスト側で観測した最大セグメント長からカーネルをキャッシュ生成
する (`qmv_wide_nocap.py` の「M ごとに1本コンパイル」と同じ考え方)。ただし
そちらが警告する「配列の動的インデックスでレジスタに乗らない」問題は
M=9 以上の話で、ここでの MAX_SEG は decode の実測上 3 (S=3) 前後にしか
ならないため、`moe_glu.py` が既に採用している `rg[4]` 型の小さい2次元配列
+ 通常の C++ for ループ (Python 側でのソース展開はしない) をそのまま使う。
HARD_MAX_SEG_CAP=8 を超えたら明示的に例外にして、それ以上は将来 (per-M
コード展開が要る領域) の課題として残す。

v1 は **gate 単体** のみ (依頼の指示通り、gate+up 融合や down 側は次段)。
down 側は上記の通り qmv_fast の対象にすら入っていない (K=640 が 512 の
倍数でない) ので、down 向けにこのカーネルを転用する場合は K のブロック
幅を 512 から作り直す必要がある — これは v1 のスコープ外。

## 実測 (M3 Max, 単一プロセス内で (a)(b) を交互計測。ウォームアップ20回)

2026-09-01、decode 実形状 (E=512, K=2560, H=640, gs=64/4bit, P=30,
nseg=12, max_seg=3, seg_len降順=[3,3,3,3,3,3,3,3,2,2,1,1]) での1回:

    max relative error  (a) mx.gather_qmm      = 4.86e-01
    max relative error  (b) moe_verify_gather  = 1.82e-01
    (a) mx.gather_qmm      : 234.6 us/call (300 回平均、交互計測)
    (b) moe_verify_gather  : 230.3 us/call (300 回平均、交互計測)
    ratio b/a = 0.982

乱数シードを変えた 8 通り (seed=0..7) でも (b) の誤差は 0.16-0.20 の帯
に収まって全て通過。max_seg=1 (全行が別エキスパート)・max_seg=5 (意図的
に5行を同一エキスパートに寄せた敵対的ケース)・P=1 でも同じ許容内で通過
した (このファイルの `__main__` 以外、単発の手動テストで確認。回帰用の
自動テストとしては同梱していない)。

**正しさ判定のしきい値について**: 単純な `|out-ref|/|ref|` は K=2560 の
内積が bf16 の丸めでキャンセルする要素 (真値がほぼ0) で分母が潰れて
発散するため使えない。`mx.gather_qmm` 自身を fp64 (numpy) 参照と比較した
実測で、絶対誤差が rtol に依らず 0.02-0.036 の帯に収まることを確認し、
atol=0.05 / rtol=0.02 の allclose 型 (atol + rtol*|ref|) を判定に採用した
(`_max_rel_err` 参照)。これは実装のバグではなく、x 自体が bf16 であることに
由来する量子化ノイズが K 本足し合わされた蓄積誤差の床であり、bf16 で
K=2560 の内積を取る限り (a) 側にも同じ床が乗る。

速度は「勝ってはいるが誤差の範囲内 (0.98x、単発計測)」というのが正直な
評価で、依頼の完了条件 (速度は負けていてもよい、まず正しさ) は満たして
いる。ここから先の最適化 (down 側対応、gate+up 融合、K=512アライメント
が壊れる down 用のブロック幅作り直し、真の帯域改善が出るかの in-model
複数プロンプト平均での検証) は本依頼のスコープ外として明示的に残す。
measurement-discipline のメモ通り、この micro の絶対値そのものは信じ
すぎないこと — ここでの数字は「正しさが通った」ことの記録であって、
「速い」ことの証明ではない。

## 依頼3: v2 (gate+up 融合 / down 対応 / 配線)

**gate+up 融合** (`_source_gate_up` / `gather_gate_up`): v1 の「1 セグメント
= 1 threadgroup、重みは it x j の組につき 1 回だけ読む」構造はそのまま。
`kernels/moe_glu.py` が 1 pair 単位でやっている「gate と up を同じ x レジスタ
(xt/xsum) で読み、silu(gate)*up まで 1 カーネルで済ませる」を、v1 のセグメント
ループの内側に足しただけ。gate_w/up_w 用に gw2/uw2 の 2 本のポインタを持ち、
it ループの中で両方を 1 回ずつ読んで rg[r][j]/ru[r][j] を別々に積む。最後に
simd_sum 後の g, u から `silu(g)*u` を書き出す。K=hidden_size=2560 は常に 512
の倍数なので、v1 と同じ `assert K % 512 == 0` のままでよい (down のような
端数処理は不要)。

**down 対応** (`_source_down` / `gather_down`): down は K=moe_intermediate=640
で、512 の倍数ではない (640 = 512 + 128)。ただし group_size=64 は quantize の
制約上 K を必ず割り切る (640/64=10 グループ)。v1/gate+up の「1 イテレーション
= 8 グループ (512 値) ぶん 32 レーン全稼働」という前提を、最後のイテレーション
だけ崩す形で一般化した: `groups_active = (it < full_iters) ? 8 : tail_groups`
とし、`lane < groups_active*4` のレーンだけがその回の x/重み読み出しと積和を
実行する (`if` で丸ごと囲む)。ポインタの前進 (`gw2 += 32` 等) はレーンによらず
毎回行う — 単なるアドレス計算で参照はしないため、最後のイテレーションで使われ
なくても安全。K=640 なら full_iters=1 (前半 512 値, 8 グループ全稼働) + tail
イテレーション 1 回 (残り 128 値 = 2 グループ、レーン 0-7 だけ有効) の計2回。
これは「gather_qmv (非 fast) 相当の一般整列版でよい」という依頼の指示通り、
速度でなく正しさを優先した設計 (無効レーンは何もせず遊ぶだけで、512 の倍数
ケースほど効率は出ない)。K % 512 == 0 のとき (tail_groups=0) は v1 と全く同じ
動作になる。

**配線** (`fused.enable_moe_verify_gather`): `SwitchGLU.__call__` を差し替え、
indices.size < 64 (mlx_lm 自身のソート閾値、`enable_moe_glu` と同じ基準) かつ
gate_proj/up_proj/down_proj が全て eligible なときだけ v2 経路に入る。中身は
常に `_gather_sort` でソートしてから `gather_gate_up` → `gather_down` →
`_scatter_unsort` の 3 段。down の出力はここでは (P, hidden) の 2 次元のまま
返る (gather_qmm 経由だと M=1 の中間次元が挟まるが、自作カーネルはそれを
最初から持たないので `squeeze(-2)` が要らない) ―― ここが素の実装からの構造的な
違いで、真似て `[:, None, :]` を挟むと形が壊れるので注意。既定は off で、
環境変数 `MLXTURBO_MOE_VERIFY=1` のときだけパッチが入る (呼ぶだけでは何も
起きない設計。依頼の「既定は off」を関数自身が守る形にした)。prefill 幅
(indices.size >= 64) は無条件で素の経路のまま。

**正しさ検証**: `_verify_v2()` (この模块の `main()` から呼ぶ) で以下を確認—
(1) `gather_gate_up` 単体を fp32 参照 (dequantize + silu*mul) と比較、
(2) `gather_down` 単体を fp32 参照と比較、(3) `SparseMoeBlock` を合成量子化
重みで組み立て (`mlx_lm.models.switch_layers.SwitchGLU` 経由、モデル全体の
ロードはしない)、`fused.enable_moe_verify_gather` の on/off で出力が allclose
になることを確認。しきい値は v1 と同じ atol=0.05/rtol=0.02 (`_max_rel_err`
を流用。理由は同じ — bf16 で K 本足し合わせる蓄積誤差の床で、実装のバグでは
ない)。

速度について: いま裏で大きいダウンロードが並走しており、このプロセス内で
計測しても帯域が汚れるため、勝敗の判断はしない。ここで出す us/call の数字は
「動いた」ことの記録であって「速い」ことの証明ではない (v1 の注記と同じ)。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_KERNELS: dict[tuple, Any] = {}

GROUP_SIZE = 64
BITS = 4
NSIMD = 2                  # moe_glu.py と同じ simdgroup 数
ROWS_PER_TG = NSIMD * 4     # 出力4行 x simdgroup2本 = 8

# qmv_wide_nocap.py の M_MAX と同種の安全弁。decode の実態 (S=3 なら重複は
# 高々3) をはるかに超えたら、per-MAX_SEG コード展開が要る領域なので止める。
HARD_MAX_SEG_CAP = 8


def _source(K: int, H: int, P: int, max_seg: int) -> str:
    """gate 単体の共有タイル gather。moe_glu._source の「1 pair」を
    「1 セグメント (同一エキスパートの最大 max_seg 行)」に広げたもの。

    重み (gw2/gsl/gbl) の読み出しは it x j の組につき 1 回だけで、
    その内側の `for (int r = 0; r < max_seg; r++)` が「同じ重みレジスタを
    セグメント内の全行に適用する」の本体。末尾セグメントで行数が
    max_seg に満たない場合は `rows[r]` を P-1 にクランプして読み出しの
    OOB だけ避け、書き込みは `r < segn` で捨てる (計算自体は無駄になるが
    安全性を優先する v1 の判断)。
    """
    assert K % 512 == 0, "block=512 の等分が前提 (moe_glu.py と同じ制約)"
    n_iters = K // 512
    return f"""
    constexpr int VPT = 16;
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint seg  = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * ({NSIMD} * 4) + sg * 4;
    if (row0 >= {H}) return;

    const uint e     = seg_expert[seg];
    const uint segr0 = seg_row0[seg];
    const uint segn  = seg_len[seg];
    const size_t wrow2 = (size_t)({K} / 16);
    const size_t grow  = (size_t)({K} / 64);
    const size_t ebase = (size_t)e * {H};

    const device uint2* gw2 = (const device uint2*)gate_w + (ebase + row0) * wrow2 + lane;
    const device T* gsl = gate_s + (ebase + row0) * grow;
    const device T* gbl = gate_b + (ebase + row0) * grow;

    const uint gofs = lane / 4;

    // セグメント内の行ごとに x ポインタだけ独立させる。重み側 (gw2/gsl/gbl)
    // は下の it ループで 1 本だけを全行に使い回す。
    uint rows[{max_seg}];
    const device vec<T, 4>* xv[{max_seg}];
    #pragma unroll
    for (int r = 0; r < {max_seg}; r++) {{
        rows[r] = min(segr0 + (uint)r, (uint)({P} - 1));
        xv[r] = (const device vec<T, 4>*)(x + (size_t)rows[r] * {K}) + lane * 4;
    }}

    float rg[{max_seg}][4];
    #pragma unroll
    for (int r = 0; r < {max_seg}; r++) {{
        #pragma unroll
        for (int j = 0; j < 4; j++) rg[r][j] = 0.0f;
    }}

    float xt[{max_seg}][VPT];

    for (int it = 0; it < {n_iters}; it++) {{
        float xsum[{max_seg}];
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) {{
            xsum[r] = 0.0f;
            #pragma unroll
            for (int v = 0; v < 4; v++) {{
                const vec<T, 4> xx = xv[r][v];
                #pragma unroll
                for (int i = 0; i < 4; i++) {{
                    xt[r][v * 4 + i] = (float)xx[i];
                    xsum[r] += xt[r][v * 4 + i];
                }}
            }}
        }}
        const uint gbase = it * 8 + gofs;
        #pragma unroll
        for (int j = 0; j < 4; j++) {{
            const uint2 wg = gw2[(size_t)j * wrow2];
            const float sj = (float)gsl[j * grow + gbase];
            const float bj = (float)gbl[j * grow + gbase];
            #pragma unroll
            for (int r = 0; r < {max_seg}; r++) {{
                float ag = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {{
                    ag += xt[r][i]     * (float)((wg.x >> (4 * i)) & 0xF)
                        + xt[r][8 + i] * (float)((wg.y >> (4 * i)) & 0xF);
                }}
                rg[r][j] += sj * ag + bj * xsum[r];
            }}
        }}
        gw2 += 32;
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) xv[r] += 128;
    }}

    #pragma unroll
    for (int j = 0; j < 4; j++) {{
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) {{
            float g = simd_sum(rg[r][j]);
            if (lane == 0 && row0 + j < {H} && (uint)r < segn) {{
                out[(size_t)rows[r] * {H} + row0 + j] = (T)g;
            }}
        }}
    }}
"""


def _get_kernel(K: int, H: int, P: int, max_seg: int):
    key = (K, H, P, max_seg)
    k = _KERNELS.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"moe_verify_gather_gate_{K}x{H}x{P}_ms{max_seg}",
            input_names=["x", "seg_expert", "seg_row0", "seg_len",
                         "gate_w", "gate_s", "gate_b"],
            output_names=["out"],
            source=_source(K, H, P, max_seg),
        )
        _KERNELS[key] = k
    return k


def eligible(gate_proj) -> bool:
    """量子化・形状がこのカーネルの前提に合うか。外れたら素の経路へ。"""
    if not hasattr(gate_proj, "scales"):
        return False
    if gate_proj.bits != BITS or gate_proj.group_size != GROUP_SIZE:
        return False
    if getattr(gate_proj, "mode", "affine") != "affine":
        return False
    K = gate_proj.weight.shape[-1] * 8
    if K % 512:
        return False
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def segments_from_sorted_idx(idx_sorted: list[int]) -> tuple[list[int], list[int], list[int]]:
    """昇順ソート済み添字を連続ランで区切る (エキスパートID, 開始行, 行数)。

    `_gather_sort` は argsort するだけなので、同じエキスパートを引く行は
    ソート後に必ず連続する。ここは python (ホスト側) で計算してよい
    ―― 依頼の設計通り、境界計算そのものはこのカーネルの対象外。
    """
    seg_expert: list[int] = []
    seg_row0: list[int] = []
    seg_len: list[int] = []
    i = 0
    n = len(idx_sorted)
    while i < n:
        j = i + 1
        while j < n and idx_sorted[j] == idx_sorted[i]:
            j += 1
        seg_expert.append(idx_sorted[i])
        seg_row0.append(i)
        seg_len.append(j - i)
        i = j
    return seg_expert, seg_row0, seg_len


def gather_gate(
    x_sorted: mx.array,
    idx_sorted: mx.array,
    gate_w: mx.array,
    gate_s: mx.array,
    gate_b: mx.array,
    K: int,
    H: int,
) -> mx.array:
    """x_sorted (P, K) bf16 とソート済み idx (P,) から gate 出力 (P, H) を返す。

    セグメント境界はここ (python) で計算してからカーネルへ渡す。MAX_SEG は
    実データの最大セグメント長からその都度決め、キャッシュキーに含める
    (qmv_wide_nocap.py の「M ごとに1本コンパイル」と同じ考え方)。
    """
    P = x_sorted.shape[0]
    idx_list = [int(v) for v in idx_sorted.tolist()]
    seg_expert, seg_row0, seg_len = segments_from_sorted_idx(idx_list)
    max_seg = max(seg_len) if seg_len else 1
    if max_seg > HARD_MAX_SEG_CAP:
        raise ValueError(
            f"segment length {max_seg} exceeds HARD_MAX_SEG_CAP="
            f"{HARD_MAX_SEG_CAP} (v1 はここまでしか検証していない)"
        )
    nseg = len(seg_expert)
    kern = _get_kernel(K, H, P, max_seg)
    (out,) = kern(
        inputs=[
            x_sorted,
            mx.array(seg_expert, dtype=mx.uint32),
            mx.array(seg_row0, dtype=mx.uint32),
            mx.array(seg_len, dtype=mx.uint32),
            gate_w.reshape(-1, gate_w.shape[-1]),
            gate_s.reshape(-1, gate_s.shape[-1]),
            gate_b.reshape(-1, gate_b.shape[-1]),
        ],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, H)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (H + ROWS_PER_TG - 1) // ROWS_PER_TG, nseg),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out


# ------------------------------------------------------------------ v2: gate+up 融合


_KERNELS_GATE_UP: dict[tuple, Any] = {}


def _source_gate_up(K: int, H: int, P: int, max_seg: int) -> str:
    """gate+up 融合版。`_source` のセグメントループはそのまま、
    `moe_glu._source` の「gate と up を同じ x レジスタで読む」を内側に足した。
    K=hidden_size は常に512の倍数 (down のような端数処理は不要)。"""
    assert K % 512 == 0, "block=512 の等分が前提 (gate/up は常にこの形)"
    n_iters = K // 512
    return f"""
    constexpr int VPT = 16;
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint seg  = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * ({NSIMD} * 4) + sg * 4;
    if (row0 >= {H}) return;

    const uint e     = seg_expert[seg];
    const uint segr0 = seg_row0[seg];
    const uint segn  = seg_len[seg];
    const size_t wrow2 = (size_t)({K} / 16);
    const size_t grow  = (size_t)({K} / 64);
    const size_t ebase = (size_t)e * {H};

    const device uint2* gw2 = (const device uint2*)gate_w + (ebase + row0) * wrow2 + lane;
    const device uint2* uw2 = (const device uint2*)up_w   + (ebase + row0) * wrow2 + lane;
    const device T* gsl = gate_s + (ebase + row0) * grow;
    const device T* gbl = gate_b + (ebase + row0) * grow;
    const device T* usl = up_s   + (ebase + row0) * grow;
    const device T* ubl = up_b   + (ebase + row0) * grow;

    const uint gofs = lane / 4;

    uint rows[{max_seg}];
    const device vec<T, 4>* xv[{max_seg}];
    #pragma unroll
    for (int r = 0; r < {max_seg}; r++) {{
        rows[r] = min(segr0 + (uint)r, (uint)({P} - 1));
        xv[r] = (const device vec<T, 4>*)(x + (size_t)rows[r] * {K}) + lane * 4;
    }}

    float rg[{max_seg}][4];
    float ru[{max_seg}][4];
    #pragma unroll
    for (int r = 0; r < {max_seg}; r++) {{
        #pragma unroll
        for (int j = 0; j < 4; j++) {{ rg[r][j] = 0.0f; ru[r][j] = 0.0f; }}
    }}

    float xt[{max_seg}][VPT];

    for (int it = 0; it < {n_iters}; it++) {{
        float xsum[{max_seg}];
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) {{
            xsum[r] = 0.0f;
            #pragma unroll
            for (int v = 0; v < 4; v++) {{
                const vec<T, 4> xx = xv[r][v];
                #pragma unroll
                for (int i = 0; i < 4; i++) {{
                    xt[r][v * 4 + i] = (float)xx[i];
                    xsum[r] += xt[r][v * 4 + i];
                }}
            }}
        }}
        const uint gbase = it * 8 + gofs;
        #pragma unroll
        for (int j = 0; j < 4; j++) {{
            const uint2 wg = gw2[(size_t)j * wrow2];
            const uint2 wu = uw2[(size_t)j * wrow2];
            const float gs_ = (float)gsl[j * grow + gbase];
            const float gb_ = (float)gbl[j * grow + gbase];
            const float us_ = (float)usl[j * grow + gbase];
            const float ub_ = (float)ubl[j * grow + gbase];
            #pragma unroll
            for (int r = 0; r < {max_seg}; r++) {{
                float ag = 0.0f;
                float au = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {{
                    ag += xt[r][i]     * (float)((wg.x >> (4 * i)) & 0xF)
                        + xt[r][8 + i] * (float)((wg.y >> (4 * i)) & 0xF);
                    au += xt[r][i]     * (float)((wu.x >> (4 * i)) & 0xF)
                        + xt[r][8 + i] * (float)((wu.y >> (4 * i)) & 0xF);
                }}
                rg[r][j] += gs_ * ag + gb_ * xsum[r];
                ru[r][j] += us_ * au + ub_ * xsum[r];
            }}
        }}
        gw2 += 32;
        uw2 += 32;
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) xv[r] += 128;
    }}

    #pragma unroll
    for (int j = 0; j < 4; j++) {{
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) {{
            float g = simd_sum(rg[r][j]);
            float u = simd_sum(ru[r][j]);
            if (lane == 0 && row0 + j < {H} && (uint)r < segn) {{
                const float sig = 1.0f / (1.0f + metal::exp(-g));
                out[(size_t)rows[r] * {H} + row0 + j] = (T)(g * sig * u);
            }}
        }}
    }}
"""


def _get_kernel_gate_up(K: int, H: int, P: int, max_seg: int):
    key = (K, H, P, max_seg)
    k = _KERNELS_GATE_UP.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"moe_verify_gather_gateup_{K}x{H}x{P}_ms{max_seg}",
            input_names=["x", "seg_expert", "seg_row0", "seg_len",
                         "gate_w", "gate_s", "gate_b",
                         "up_w", "up_s", "up_b"],
            output_names=["out"],
            source=_source_gate_up(K, H, P, max_seg),
        )
        _KERNELS_GATE_UP[key] = k
    return k


def eligible_gate_up(x, gate_proj, up_proj) -> bool:
    """gate+up 融合カーネルの前提 (K=hidden_size は常に512の倍数)。

    x はモデルの生の隠れ状態 (呼び出し元は SwitchGLU の入力)。カーネルは
    `template=[("T", mx.bfloat16)]` で T を bf16 に固定しているので、x が
    bf16 以外だとバッファを誤った幅で読む静かな誤りになる。ここで弾く
    (eligible_down 側の x は必ずこのカーネルの bf16 出力なのでチェック不要)。
    """
    if x.dtype != mx.bfloat16:
        return False
    for l in (gate_proj, up_proj):
        if not hasattr(l, "scales"):
            return False
        if l.bits != BITS or l.group_size != GROUP_SIZE:
            return False
        if getattr(l, "mode", "affine") != "affine":
            return False
    K = gate_proj.weight.shape[-1] * 8
    if K % 512:
        return False
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def gather_gate_up(
    x_sorted: mx.array,
    idx_sorted: mx.array,
    gate_w: mx.array,
    gate_s: mx.array,
    gate_b: mx.array,
    up_w: mx.array,
    up_s: mx.array,
    up_b: mx.array,
    K: int,
    H: int,
) -> mx.array:
    """x_sorted (P, K) bf16 とソート済み idx (P,) から silu(gate)*up (P, H) を返す。"""
    P = x_sorted.shape[0]
    idx_list = [int(v) for v in idx_sorted.tolist()]
    seg_expert, seg_row0, seg_len = segments_from_sorted_idx(idx_list)
    max_seg = max(seg_len) if seg_len else 1
    if max_seg > HARD_MAX_SEG_CAP:
        raise ValueError(
            f"segment length {max_seg} exceeds HARD_MAX_SEG_CAP="
            f"{HARD_MAX_SEG_CAP} (v1/v2 はここまでしか検証していない)"
        )
    nseg = len(seg_expert)
    kern = _get_kernel_gate_up(K, H, P, max_seg)
    (out,) = kern(
        inputs=[
            x_sorted,
            mx.array(seg_expert, dtype=mx.uint32),
            mx.array(seg_row0, dtype=mx.uint32),
            mx.array(seg_len, dtype=mx.uint32),
            gate_w.reshape(-1, gate_w.shape[-1]),
            gate_s.reshape(-1, gate_s.shape[-1]),
            gate_b.reshape(-1, gate_b.shape[-1]),
            up_w.reshape(-1, up_w.shape[-1]),
            up_s.reshape(-1, up_s.shape[-1]),
            up_b.reshape(-1, up_b.shape[-1]),
        ],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, H)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (H + ROWS_PER_TG - 1) // ROWS_PER_TG, nseg),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out


# ------------------------------------------------------------------ v2: down 対応


_KERNELS_DOWN: dict[tuple, Any] = {}


def _source_down(K: int, H: int, P: int, max_seg: int) -> str:
    """down 用の一般整列版。K=moe_intermediate_size (640) は512の倍数でない
    (640 = 512 + 128) が、group_size=64 は quantize の制約で K を必ず割り切る
    (640/64=10グループ)。最後のイテレーションだけ有効グループ数を8未満に落とし、
    該当しないレーンはその回の読み出し・積和を丸ごとスキップする
    (`lane < valid_lanes` ガード)。ポインタの前進 (gw2 += 32 等) はレーンに
    よらず毎回行う — アドレス計算だけで参照はしないので、末尾で使われなくても
    安全 (このイテレーションが最後なので次に使われることもない)。
    K % 512 == 0 のとき (tail_groups=0) は `_source` と同じ動作になる。"""
    assert K % GROUP_SIZE == 0, "group_size が K を割り切ることが前提 (quantize の制約)"
    full_iters = K // 512
    tail = K % 512
    tail_groups = tail // 64
    n_iters = full_iters + (1 if tail_groups else 0)
    return f"""
    constexpr int VPT = 16;
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint seg  = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * ({NSIMD} * 4) + sg * 4;
    if (row0 >= {H}) return;

    const uint e     = seg_expert[seg];
    const uint segr0 = seg_row0[seg];
    const uint segn  = seg_len[seg];
    const size_t wrow2 = (size_t)({K} / 16);
    const size_t grow  = (size_t)({K} / 64);
    const size_t ebase = (size_t)e * {H};

    const device uint2* gw2 = (const device uint2*)gate_w + (ebase + row0) * wrow2 + lane;
    const device T* gsl = gate_s + (ebase + row0) * grow;
    const device T* gbl = gate_b + (ebase + row0) * grow;

    const uint gofs = lane / 4;

    uint rows[{max_seg}];
    const device vec<T, 4>* xv[{max_seg}];
    #pragma unroll
    for (int r = 0; r < {max_seg}; r++) {{
        rows[r] = min(segr0 + (uint)r, (uint)({P} - 1));
        xv[r] = (const device vec<T, 4>*)(x + (size_t)rows[r] * {K}) + lane * 4;
    }}

    float rg[{max_seg}][4];
    #pragma unroll
    for (int r = 0; r < {max_seg}; r++) {{
        #pragma unroll
        for (int j = 0; j < 4; j++) rg[r][j] = 0.0f;
    }}

    float xt[{max_seg}][VPT];

    for (int it = 0; it < {n_iters}; it++) {{
        const uint groups_active = (it < {full_iters}) ? 8u : {tail_groups}u;
        const uint valid_lanes = groups_active * 4u;
        if (lane < valid_lanes) {{
            float xsum[{max_seg}];
            #pragma unroll
            for (int r = 0; r < {max_seg}; r++) {{
                xsum[r] = 0.0f;
                #pragma unroll
                for (int v = 0; v < 4; v++) {{
                    const vec<T, 4> xx = xv[r][v];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {{
                        xt[r][v * 4 + i] = (float)xx[i];
                        xsum[r] += xt[r][v * 4 + i];
                    }}
                }}
            }}
            const uint gbase = it * 8 + gofs;
            #pragma unroll
            for (int j = 0; j < 4; j++) {{
                const uint2 wg = gw2[(size_t)j * wrow2];
                const float sj = (float)gsl[j * grow + gbase];
                const float bj = (float)gbl[j * grow + gbase];
                #pragma unroll
                for (int r = 0; r < {max_seg}; r++) {{
                    float ag = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {{
                        ag += xt[r][i]     * (float)((wg.x >> (4 * i)) & 0xF)
                            + xt[r][8 + i] * (float)((wg.y >> (4 * i)) & 0xF);
                    }}
                    rg[r][j] += sj * ag + bj * xsum[r];
                }}
            }}
        }}
        gw2 += 32;
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) xv[r] += 128;
    }}

    #pragma unroll
    for (int j = 0; j < 4; j++) {{
        #pragma unroll
        for (int r = 0; r < {max_seg}; r++) {{
            float g = simd_sum(rg[r][j]);
            if (lane == 0 && row0 + j < {H} && (uint)r < segn) {{
                out[(size_t)rows[r] * {H} + row0 + j] = (T)g;
            }}
        }}
    }}
"""


def _get_kernel_down(K: int, H: int, P: int, max_seg: int):
    key = (K, H, P, max_seg)
    k = _KERNELS_DOWN.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"moe_verify_gather_down_{K}x{H}x{P}_ms{max_seg}",
            input_names=["x", "seg_expert", "seg_row0", "seg_len",
                         "gate_w", "gate_s", "gate_b"],
            output_names=["out"],
            source=_source_down(K, H, P, max_seg),
        )
        _KERNELS_DOWN[key] = k
    return k


def eligible_down(down_proj) -> bool:
    """down カーネルの前提。K は512の倍数でなくてよい (group_size を割り切れば足りる)。"""
    if not hasattr(down_proj, "scales"):
        return False
    if down_proj.bits != BITS or down_proj.group_size != GROUP_SIZE:
        return False
    if getattr(down_proj, "mode", "affine") != "affine":
        return False
    K = down_proj.weight.shape[-1] * 8
    if K % GROUP_SIZE:
        return False
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def gather_down(
    x_sorted: mx.array,
    idx_sorted: mx.array,
    down_w: mx.array,
    down_s: mx.array,
    down_b: mx.array,
    K: int,
    H: int,
) -> mx.array:
    """x_sorted (P, K) bf16 とソート済み idx (P,) から down 出力 (P, H) を返す。
    gate/gate_up と同じセグメント境界計算だが、こちらは K=640 の端数処理が要る
    (`_source_down` 参照)。"""
    P = x_sorted.shape[0]
    idx_list = [int(v) for v in idx_sorted.tolist()]
    seg_expert, seg_row0, seg_len = segments_from_sorted_idx(idx_list)
    max_seg = max(seg_len) if seg_len else 1
    if max_seg > HARD_MAX_SEG_CAP:
        raise ValueError(
            f"segment length {max_seg} exceeds HARD_MAX_SEG_CAP="
            f"{HARD_MAX_SEG_CAP} (v1/v2 はここまでしか検証していない)"
        )
    nseg = len(seg_expert)
    kern = _get_kernel_down(K, H, P, max_seg)
    (out,) = kern(
        inputs=[
            x_sorted,
            mx.array(seg_expert, dtype=mx.uint32),
            mx.array(seg_row0, dtype=mx.uint32),
            mx.array(seg_len, dtype=mx.uint32),
            down_w.reshape(-1, down_w.shape[-1]),
            down_s.reshape(-1, down_s.shape[-1]),
            down_b.reshape(-1, down_b.shape[-1]),
        ],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, H)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (H + ROWS_PER_TG - 1) // ROWS_PER_TG, nseg),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out


def _build_decode_case(E=512, K=2560, H=640, S=3, top_k=10, seed=0):
    """decode verify 実形状の合成データ。モデルのロードはしない。"""
    import random

    mx.random.seed(seed)
    rng = random.Random(seed)

    w_fp = mx.random.uniform(low=-0.05, high=0.05, shape=(E, H, K)).astype(mx.bfloat16)
    mx.eval(w_fp)
    gate_w, gate_s, gate_b = mx.quantize(w_fp, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(gate_w, gate_s, gate_b)

    # S トークン x top_k。1トークン内は相異なる (argpartition の性質)。
    # トークン間でわざと重複させ、セグメント長 >1 のケースを作る (実運用の
    # verify で起こり得る「近い分布のトークンが同じ上位エキスパートを引く」
    # 状況を模す)。
    shared_pool = rng.sample(range(E), top_k + 2)
    idx_rows = []
    for _t in range(S):
        row = list(shared_pool)
        rng.shuffle(row)
        row = row[:top_k]
        # 残り枠を無関係な添字で埋めて、完全一致トークンにはしない
        extra_pool = [e for e in range(E) if e not in shared_pool]
        rng.shuffle(extra_pool)
        n_extra = max(0, top_k - len(row))
        row = row[: top_k - n_extra] + extra_pool[:n_extra]
        idx_rows.append(row)
    idx_flat = [v for row in idx_rows for v in row]

    x = mx.random.normal(shape=(S * top_k, K)).astype(mx.bfloat16)
    mx.eval(x)

    order = sorted(range(len(idx_flat)), key=lambda i: idx_flat[i])
    idx_sorted = [idx_flat[i] for i in order]
    x_sorted = x[mx.array(order, dtype=mx.uint32)]
    mx.eval(x_sorted)

    return {
        "gate_w": gate_w, "gate_s": gate_s, "gate_b": gate_b,
        "x_sorted": x_sorted, "idx_sorted": idx_sorted,
        "E": E, "K": K, "H": H,
    }


def _reference_fp32(case: dict) -> mx.array:
    """該当エキスパートだけ dequantize した fp32 参照。"""
    idx_sorted = case["idx_sorted"]
    uniq = sorted(set(idx_sorted))
    pos = {e: i for i, e in enumerate(uniq)}
    uniq_arr = mx.array(uniq, dtype=mx.uint32)
    w_u = case["gate_w"][uniq_arr]
    s_u = case["gate_s"][uniq_arr]
    b_u = case["gate_b"][uniq_arr]
    deq_u = mx.dequantize(w_u, s_u, b_u, group_size=GROUP_SIZE, bits=BITS).astype(mx.float32)
    x_f32 = case["x_sorted"].astype(mx.float32)
    rows = []
    for p, e in enumerate(idx_sorted):
        rows.append(x_f32[p] @ deq_u[pos[e]].T)
    ref = mx.stack(rows)
    mx.eval(ref)
    return ref


def _max_rel_err(out: mx.array, ref: mx.array, rtol: float = 0.02, atol: float = 0.05) -> float:
    """allclose 型の相対誤差 (mx.allclose と同じ atol+rtol*|ref| の分母)。

    単純な |out-ref|/|ref| は ref がほぼ 0 の要素で分母が潰れて発散する
    (K=2560 の内積は bf16 の丸めでキャンセルが起き、値そのものが小さい
    出力要素が普通に出る)。atol=0.05 は実測 (mx.gather_qmm 自身を fp64
    参照と比較した際の絶対誤差が rtol に関係なく 0.02-0.036 の帯に収まる
    ことを確認して決めた床)。x 自体が bf16 でここに乗る量子化ノイズが
    K=2560 本足し合わされるぶんの絶対誤差で、rtol でスケールしない
    ―― bf16 蓄積の性質そのものであって実装のバグではない。
    """
    out32 = out.astype(mx.float32)
    err = mx.abs(out32 - ref)
    denom = atol + rtol * mx.abs(ref)
    return float(mx.max(err / denom))


def _verify_v2() -> None:
    """v2 (gate+up 融合 / down / SwitchGLU 配線) の正しさ検証。3段:
    (1) gather_gate_up 単体を fp32 参照 (dequantize + silu*mul) と比較、
    (2) gather_down 単体を fp32 参照と比較 (K=640 の端数処理が実際に通るかも
    ここで確認する)、(3) `mlx_lm.models.qwen4_exp.SparseMoeBlock` を合成量子化
    重みで組み立て、`fused.enable_moe_verify_gather` の on/off が allclose に
    なるかを確認する (モデル全体のロードはしない)。速度は測らない — 裏で
    ダウンロードが並走しており計測が汚れるため、正しさだけをここで見る。
    """
    import os

    case = _build_decode_case()  # gate 単体と同じ合成データ (K=2560, H=640)
    K, H = case["K"], case["H"]
    idx_sorted = case["idx_sorted"]
    idx_arr = mx.array(idx_sorted, dtype=mx.uint32)
    uniq = sorted(set(idx_sorted))
    pos = {e: i for i, e in enumerate(uniq)}
    uniq_arr = mx.array(uniq, dtype=mx.uint32)

    # (1) gate+up 融合。up は gate とは別の乱数重みを新規に量子化する
    w_up_fp = mx.random.uniform(low=-0.05, high=0.05, shape=(case["E"], H, K)).astype(mx.bfloat16)
    mx.eval(w_up_fp)
    up_w, up_s, up_b = mx.quantize(w_up_fp, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(up_w, up_s, up_b)

    out_gu = gather_gate_up(
        case["x_sorted"], idx_arr,
        case["gate_w"], case["gate_s"], case["gate_b"],
        up_w, up_s, up_b, K, H,
    )
    mx.eval(out_gu)

    gdeq = mx.dequantize(case["gate_w"][uniq_arr], case["gate_s"][uniq_arr], case["gate_b"][uniq_arr],
                          group_size=GROUP_SIZE, bits=BITS).astype(mx.float32)
    udeq = mx.dequantize(up_w[uniq_arr], up_s[uniq_arr], up_b[uniq_arr],
                          group_size=GROUP_SIZE, bits=BITS).astype(mx.float32)
    x_f32 = case["x_sorted"].astype(mx.float32)
    rows = []
    for p, e in enumerate(idx_sorted):
        g = x_f32[p] @ gdeq[pos[e]].T
        u = x_f32[p] @ udeq[pos[e]].T
        rows.append(g * mx.sigmoid(g) * u)
    ref_gu = mx.stack(rows)
    mx.eval(ref_gu)
    err_gu = _max_rel_err(out_gu, ref_gu)
    print(f"[v2] gather_gate_up  max relative error = {err_gu:.4e}")

    # (2) down。K=moe_intermediate_size(640) -> H=hidden_size(2560) に持ち替える。
    # 640 は 512 の倍数でない (640=512+128) ので、_source_down の端数処理が
    # ここで実際に通る。
    Kd, Hd = H, K  # 640, 2560
    w_down_fp = mx.random.uniform(low=-0.05, high=0.05, shape=(case["E"], Hd, Kd)).astype(mx.bfloat16)
    mx.eval(w_down_fp)
    down_w, down_s, down_b = mx.quantize(w_down_fp, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(down_w, down_s, down_b)

    x_down = out_gu  # 実運用と同じく gate+up の出力をそのまま down の入力にする
    out_down = gather_down(x_down, idx_arr, down_w, down_s, down_b, Kd, Hd)
    mx.eval(out_down)

    ddeq = mx.dequantize(down_w[uniq_arr], down_s[uniq_arr], down_b[uniq_arr],
                          group_size=GROUP_SIZE, bits=BITS).astype(mx.float32)
    x_down_f32 = x_down.astype(mx.float32)
    rows = []
    for p, e in enumerate(idx_sorted):
        rows.append(x_down_f32[p] @ ddeq[pos[e]].T)
    ref_down = mx.stack(rows)
    mx.eval(ref_down)
    err_down = _max_rel_err(out_down, ref_down)
    print(f"[v2] gather_down     max relative error = {err_down:.4e}")

    # (3) SparseMoeBlock を合成重みで組み立てて配線の on/off を比較する
    import mlxturbo  # noqa: F401  (mlx_lm.models.qwen4_exp -> _vendor への meta_path フック)
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo import fused

    mx.random.seed(7)
    args = Q.TextArgs()  # 既定値がそのまま E=512,K=2560,H=640,top_k=10 (実形状)
    block = Q.SparseMoeBlock(args)
    sw = block.switch_mlp
    # SwitchLinear の既定初期化は fp32 のままなので、量子化する前に bf16 に
    # 落とす (実チェックポイントは bf16 -> 量子化なので scales/biases も bf16
    # になる。fp32 のまま quantize するとカーネルの T=bfloat16 テンプレートと
    # 型が合わずコンパイルエラーになる)
    sw.gate_proj.weight = sw.gate_proj.weight.astype(mx.bfloat16)
    sw.up_proj.weight = sw.up_proj.weight.astype(mx.bfloat16)
    sw.down_proj.weight = sw.down_proj.weight.astype(mx.bfloat16)
    sw.gate_proj = sw.gate_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    sw.up_proj = sw.up_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    sw.down_proj = sw.down_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    mx.eval(block.parameters())

    x_in = mx.random.normal(shape=(1, 3, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(x_in)

    out_off = block(x_in)
    mx.eval(out_off)

    prev_env = os.environ.get("MLXTURBO_MOE_VERIFY")
    os.environ["MLXTURBO_MOE_VERIFY"] = "1"
    fused.enable_moe_verify_gather()
    out_on = block(x_in)
    mx.eval(out_on)
    fused.disable_moe_verify_gather()
    if prev_env is None:
        del os.environ["MLXTURBO_MOE_VERIFY"]
    else:
        os.environ["MLXTURBO_MOE_VERIFY"] = prev_env

    err_model = _max_rel_err(out_on, out_off.astype(mx.float32))
    print(f"[v2] SparseMoeBlock on/off max relative error = {err_model:.4e}")


def main() -> None:
    import time

    case = _build_decode_case()
    K, H = case["K"], case["H"]
    P = case["x_sorted"].shape[0]
    idx_sorted = case["idx_sorted"]
    seg_expert, seg_row0, seg_len = segments_from_sorted_idx(idx_sorted)
    print(f"P={P} nseg={len(seg_expert)} max_seg={max(seg_len)} seg_len_hist="
          f"{sorted(seg_len, reverse=True)}")

    ref = _reference_fp32(case)

    x3 = case["x_sorted"][:, None, :]
    idx_arr = mx.array(idx_sorted, dtype=mx.uint32)

    def run_a():
        out = mx.gather_qmm(
            x3, case["gate_w"], case["gate_s"], case["gate_b"],
            rhs_indices=idx_arr, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, mode="affine",
            sorted_indices=True,
        )
        return out.squeeze(-2)

    def run_b():
        return gather_gate(
            case["x_sorted"], idx_arr, case["gate_w"], case["gate_s"], case["gate_b"], K, H
        )

    out_a = run_a()
    mx.eval(out_a)
    out_b = run_b()
    mx.eval(out_b)

    err_a = _max_rel_err(out_a, ref)
    err_b = _max_rel_err(out_b, ref)
    print(f"max relative error  (a) mx.gather_qmm = {err_a:.4e}")
    print(f"max relative error  (b) moe_verify_gather = {err_b:.4e}")

    # ウォームアップ
    for _ in range(20):
        mx.eval(run_a())
        mx.eval(run_b())

    n_iters = 300
    t_a = 0.0
    t_b = 0.0
    for _ in range(n_iters):
        t0 = time.perf_counter()
        mx.eval(run_a())
        t_a += time.perf_counter() - t0

        t0 = time.perf_counter()
        mx.eval(run_b())
        t_b += time.perf_counter() - t0

    print(f"(a) mx.gather_qmm       : {t_a / n_iters * 1e6:.1f} us/call (avg of {n_iters}, interleaved)")
    print(f"(b) moe_verify_gather   : {t_b / n_iters * 1e6:.1f} us/call (avg of {n_iters}, interleaved)")
    print(f"ratio b/a = {t_b / t_a:.3f} (< 1 なら (b) が勝ち、暫定値。裏でダウンロードが"
          f"並走しているため勝敗の判断はしない)")

    print()
    _verify_v2()


if __name__ == "__main__":
    main()
