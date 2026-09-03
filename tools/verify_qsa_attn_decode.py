"""段 K2b の attention カーネル (`mlxturbo/kernels/qsa_attn_decode.py`) を検める。

見るのは 2 つ。

1. **ビット一致**。参照は本番と同じ並び --- `QSAIndexer.__call__` が作る
   bool マスク (per-query tail、`MLXTURBO_QSA_TAIL=query`) を
   `mx.fast.scaled_dot_product_attention` に渡し、S>=3 は `Attention.__call__`
   と同じく 2 行ずつに割って `concatenate` する --- で、判定は
   `mx.array_equal`。外れたら最初にずれる (行, head, 列) と値、bf16 の ulp 差、
   ずれた要素数を出す。
2. **時間**。カーネルと現行経路 (argpartition -> マスク -> sdpa 幅分割) を
   1 プロセス内 ABAB で測る。**12 層ぶんの別々の K/V を巡回する連鎖**で
   測ること --- 1 組の K/V を読み回す温かい連鎖では、並列度の低いカーネルが
   冷の DRAM レイテンシを隠せない負けが見えない (2026-09-03 12:00 の帰属)。
   17k なら層あたり K+V 34.8 MB x 12 層 = 418 MB で、L2 (32 MB) には載らない。

``blocks`` (MLX の 2-pass の kv 分割数) は `qsa_attn_decode.sdpa_blocks` が
本家の表を写して決める。``--pin-blocks N`` を付けると ``MLX_SDPA_BLOCKS`` で
両側を釘付けにする (表の読み違いと数値の問題を切り分けるため)。

使い方 (GPU を使うので biglock 経由で):

    tools/biglock.sh .venv/bin/python tools/verify_qsa_attn_decode.py
    tools/biglock.sh .venv/bin/python tools/verify_qsa_attn_decode.py --quick
    tools/biglock.sh .venv/bin/python tools/verify_qsa_attn_decode.py --pin-blocks 128
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 参照 (マスク) を HF と同じ per-query tail にする。`qsa_tail` は import 時に
# 環境変数を読むので、mlxturbo を触る前に決めておく。
os.environ.setdefault("MLXTURBO_QSA_TAIL", "query")

# `--pin-blocks` は MLX が dispatch のたびに getenv するので、mx を import した
# 後でも効く。ただし引数解析より前に import すると読み違えるので先に見ておく。
_PIN = None
for _i, _a in enumerate(sys.argv):
    if _a == "--pin-blocks" and _i + 1 < len(sys.argv):
        _PIN = int(sys.argv[_i + 1])
    elif _a.startswith("--pin-blocks="):
        _PIN = int(_a.split("=", 1)[1])
if _PIN:
    os.environ["MLX_SDPA_BLOCKS"] = str(_PIN)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlxturbo import qsa_tail as _qsa_tail  # noqa: E402
from mlxturbo.kernels import qsa_attn_decode as K2B  # noqa: E402
from mlxturbo.kernels import qsa_select as QS  # noqa: E402

# TextArgs (Qwen4-Exp) の実値
HEAD_DIM = 256        # attention の head_dim
IDX_HEAD_DIM = 128    # indexer_head_dim (スコアの /sqrt はこちら)
N_IHEADS = 4          # indexer_n_heads
N_HEADS = 24          # num_attention_heads
N_KV = 2              # num_key_value_heads
GQA = N_HEADS // N_KV
CR = 4                # indexer_compress_ratio
BUDGET = 2048         # indexer_budget
BLOCK_TOPK = BUDGET // CR   # 512
SCALE = HEAD_DIM ** -0.5

S_LIST = [1, 2, 3, 4, 6]
KV_LIST = [2049, 4096, 8500, 17000, 25000, 50000]
BENCH_KV = [4096, 17000, 25000, 50000]
CHAIN = 12            # 連鎖に並べる層の数 (K/V を冷やすため)


# --------------------------------------------------------------------------
# 参照 (本家の写し)
# --------------------------------------------------------------------------
def _device_info() -> dict:
    return mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()


def ref_keep_block(raw: mx.array, q_col: mx.array, n_blocks: int, k: int) -> mx.array:
    """`QSAIndexer._pooled_and_top` の選択部分 (argpartition 経路) の写し。"""
    B, S = raw.shape[0], raw.shape[1]
    scores = mx.maximum(raw, 0).sum(axis=-1) / math.sqrt(IDX_HEAD_DIM)
    block_end = mx.arange(n_blocks) * CR + (CR - 1)
    visible = block_end[None, None, :] <= q_col[None, :, None]
    scores = mx.where(visible, scores, -mx.inf)
    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    keep = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
    top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
    return mx.put_along_axis(keep, top, mx.array(True), axis=-1)[..., :n_blocks]


def ref_mask(keep_block: mx.array, n_blocks: int, kv_len: int, q_col: mx.array):
    """`QSAIndexer.__call__` の展開部 (``MODE == "query"``) の写し。

    `mlxturbo/_vendor/qwen4_exp.py` 528-556 行そのまま。**本家を変えたら
    ここも変える** (写しであることが判定の前提)。
    """
    B, S = keep_block.shape[0], keep_block.shape[1]
    keep = mx.repeat(keep_block, CR, axis=-1)
    tail = kv_len - n_blocks * CR
    keep = mx.concatenate(
        [keep, mx.zeros((B, S, tail + 1), dtype=mx.bool_)], axis=-1
    )
    own = ((q_col + 1) // CR) * CR
    cols = own[:, None] + mx.arange(CR - 1)[None, :]
    cols = mx.where(cols <= q_col[:, None], cols, kv_len)
    keep = mx.put_along_axis(
        keep,
        mx.broadcast_to(cols[None], (B, S, CR - 1)),
        mx.array(True),
        axis=-1,
    )[..., :kv_len]
    return keep[:, None]


def ref_attn(q_bhsd, k, v, mask):
    """`Attention.__call__` の sdpa 幅分割の写し (段 K2b の比較相手)。"""
    S = q_bhsd.shape[2]
    if 1 < S <= 8 and S * GQA > 32:
        step = max(1, 32 // GQA)
        return mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    q_bhsd[:, :, i : i + step], k, v,
                    scale=SCALE, mask=mask[..., i : i + step, :],
                )
                for i in range(0, S, step)
            ],
            axis=2,
        )
    return mx.fast.scaled_dot_product_attention(
        q_bhsd, k, v, scale=SCALE, mask=mask
    )


# --------------------------------------------------------------------------
# 入力
# --------------------------------------------------------------------------
def _raw_scores(rng, S: int, n_blocks: int, kind: str) -> np.ndarray:
    shape = (1, S, n_blocks, N_IHEADS)
    if kind == "rand":
        return rng.standard_normal(shape, dtype=np.float32)
    if kind == "relu_zero":
        # 本番のスコア分布に近い形 (relu で 4-5 割が 0 に潰れる)
        x = rng.standard_normal(shape, dtype=np.float32)
        x[rng.random(shape) < 0.45] = -1.0
        return x
    if kind == "ties":
        # 粗い格子で同点を大量に作る。閾値の同点処理 (添字の昇順) を直撃する
        g = rng.integers(0, 3, size=shape).astype(np.float32) * np.float32(0.25)
        return g - np.float32(0.25)
    if kind == "all_zero":
        # すべて同点。選択は「可視のうち添字の小さい方から k 個」になる
        return np.full(shape, -1.0, dtype=np.float32)
    raise ValueError(kind)


SCORE_KINDS = ("rand", "relu_zero", "ties", "all_zero")


class Layer:
    """1 層ぶんの入力 (q / KV バッファ / 選択)。"""

    def __init__(self, rng, S: int, kv_len: int, kind: str = "rand", cap_pad: int = 0):
        self.S = S
        self.kv_len = kv_len
        self.offset = kv_len - S
        self.n_blocks = kv_len // CR
        self.k = min(BLOCK_TOPK, self.n_blocks)
        self.cap = kv_len + cap_pad
        self.q_col = mx.arange(self.offset, self.offset + S, dtype=mx.int32)

        f32 = np.float32
        self.q = mx.array(
            rng.standard_normal((1, S, N_HEADS, HEAD_DIM)).astype(f32)
        ).astype(mx.bfloat16)
        self.keys = mx.array(
            rng.standard_normal((1, N_KV, self.cap, HEAD_DIM)).astype(f32)
        ).astype(mx.bfloat16)
        self.values = mx.array(
            rng.standard_normal((1, N_KV, self.cap, HEAD_DIM)).astype(f32)
        ).astype(mx.bfloat16)
        self.raw = mx.array(_raw_scores(rng, S, self.n_blocks, kind))

        # 参照側 (argpartition -> マスク) とカーネル側 (K2a の bits)
        self.keep_block = ref_keep_block(self.raw, self.q_col, self.n_blocks, self.k)
        self.mask = ref_mask(self.keep_block, self.n_blocks, kv_len, self.q_col)
        self.n_vis = QS.visible_counts_host(self.offset, S, CR, self.n_blocks)
        (self.bits, self.cnt) = QS.select(
            self.raw, self.n_vis, self.k, head_dim=IDX_HEAD_DIM, mode="bits"
        )
        # sdpa が受け取る形 (途中切りのビュー、本番の `update_and_fetch` と同じ)
        self.k_view = self.keys[:, :, :kv_len, :]
        self.v_view = self.values[:, :, :kv_len, :]
        self.q_bhsd = self.q.transpose(0, 2, 1, 3)
        self.blocks = K2B.mirror_blocks(kv_len, GQA, S)
        mx.eval(
            self.q, self.keys, self.values, self.raw,
            self.keep_block, self.mask, self.bits, self.cnt, self.n_vis,
        )
        # 適格判定そのものも通しておく (本番の分岐条件と同じ口)
        self.ok = K2B.eligible(
            self.q, self.keys, self.values, self.bits,
            cr=CR, kv_len=kv_len, n_blocks=self.n_blocks,
            offset=self.offset, blocks=self.blocks,
        )

    def kernel(self):
        return K2B.qsa_attn_decode(
            self.q, self.keys, self.values, self.bits,
            cr=CR, kv_len=self.kv_len, n_blocks=self.n_blocks,
            offset=self.offset, scale=SCALE, blocks=self.blocks,
        )

    def reference(self):
        return ref_attn(self.q_bhsd, self.k_view, self.v_view, self.mask)

    def cur_full(self):
        """現行経路まるごと (argpartition -> マスク -> sdpa 幅分割)。"""
        keep = ref_keep_block(self.raw, self.q_col, self.n_blocks, self.k)
        mask = ref_mask(keep, self.n_blocks, self.kv_len, self.q_col)
        return ref_attn(self.q_bhsd, self.k_view, self.v_view, mask)

    def k2_full(self):
        """K2 経路まるごと (K2a の選択 -> K2b の attention)。"""
        (bits, _cnt) = QS.select(
            self.raw, self.n_vis, self.k, head_dim=IDX_HEAD_DIM, mode="bits"
        )
        return K2B.qsa_attn_decode(
            self.q, self.keys, self.values, bits,
            cr=CR, kv_len=self.kv_len, n_blocks=self.n_blocks,
            offset=self.offset, scale=SCALE, blocks=self.blocks,
        )


# --------------------------------------------------------------------------
# 比較
# --------------------------------------------------------------------------
def _bf16_ulp(a: mx.array, b: mx.array) -> np.ndarray:
    """bf16 2 つの ulp 差 (単調な整数表現の差)。"""
    ua = np.array(a.view(mx.uint16)).astype(np.int32)
    ub = np.array(b.view(mx.uint16)).astype(np.int32)

    def mono(u):
        neg = (u & 0x8000) != 0
        return np.where(neg, -(u & 0x7FFF), u & 0x7FFF)

    return np.abs(mono(ua) - mono(ub))


def compare(layer: Layer, label: str, problems: list) -> dict:
    if not layer.ok:
        problems.append(f"{label}: eligible() が False (適格判定が形を弾いた)")
    got = layer.kernel()
    want = layer.reference()
    mx.eval(got, want)
    same = bool(mx.array_equal(got, want))
    ulp = _bf16_ulp(got, want)
    n_bad = int((ulp != 0).sum())
    max_ulp = int(ulp.max()) if ulp.size else 0
    if not same:
        idx = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
        g = np.array(got.astype(mx.float32))[idx]
        w = np.array(want.astype(mx.float32))[idx]
        # 最初にずれる位置も出す (最大 ulp の場所とは別)
        first = np.unravel_index(int(np.flatnonzero(ulp.reshape(-1))[0]), ulp.shape)
        problems.append(
            f"{label}: 不一致 {n_bad}/{ulp.size} 要素、最大 {max_ulp} ulp "
            f"(b={idx[0]} head={idx[1]} row={idx[2]} d={idx[3]}: "
            f"kernel {g:.9g} / 参照 {w:.9g})、最初のずれは "
            f"head={first[1]} row={first[2]} d={first[3]}"
        )
    return {"same": same, "n_bad": n_bad, "max_ulp": max_ulp, "n": int(ulp.size)}


def check_bits_vs_argpartition(layer: Layer, label: str, problems: list) -> bool:
    """K2a の bits が argpartition 経路の keep_block と同じ集合か。

    ここが崩れていると K2b の不一致が「選択の違い」なのか「数値の違い」なのか
    分からなくなるので、attention を比べる前に必ず見る。
    """
    bits = np.array(layer.bits).reshape(layer.S, -1)
    keep = np.array(layer.keep_block).reshape(layer.S, layer.n_blocks)
    ok = True
    for r in range(layer.S):
        ref = set(np.flatnonzero(keep[r]).tolist())
        got = set()
        for wi, word in enumerate(bits[r]):
            word = int(word)
            while word:
                low = word & -word
                got.add(wi * 32 + low.bit_length() - 1)
                word ^= low
        if ref != got:
            ok = False
            problems.append(
                f"{label} row={r}: K2a の bits が argpartition と違う "
                f"(ref {len(ref)} / bits {len(got)})"
            )
    return ok


# --------------------------------------------------------------------------
# 時間 (12 層の別々の K/V を巡回する冷たい連鎖)
# --------------------------------------------------------------------------
def bench(S: int, kv_len: int, rounds: int = 7) -> dict:
    rng = np.random.default_rng(97 * S + kv_len)
    layers = [Layer(rng, S, kv_len, "relu_zero") for _ in range(CHAIN)]
    mx.eval([l.mask for l in layers], [l.bits for l in layers])

    def chain(fn):
        outs = [fn(l) for l in layers]
        mx.eval(outs)

    variants = {
        "cur_total": lambda l: l.cur_full(),
        "cur_sdpa": lambda l: l.reference(),
        "k2_total": lambda l: l.k2_full(),
        "k2b": lambda l: l.kernel(),
    }
    for fn in variants.values():          # 温め (カーネルの JIT を落としておく)
        chain(fn)

    ts = {name: [] for name in variants}
    for _ in range(rounds):               # ABAB (1 プロセス内で交互に)
        for name, fn in variants.items():
            t = time.perf_counter()
            chain(fn)
            ts[name].append((time.perf_counter() - t) / CHAIN)

    res = {"S": S, "kv": kv_len, "blocks": layers[0].blocks}
    for name in variants:
        res[name] = statistics.median(ts[name]) * 1e6
    res["k2a"] = res["k2_total"] - res["k2b"]
    res["cur_sel_mask"] = res["cur_total"] - res["cur_sdpa"]
    del layers
    return res


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="形を絞って早く回す")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument(
        "--pin-blocks", type=int, default=0,
        help="MLX_SDPA_BLOCKS を釘付けにして両側を同じ kv 分割で走らせる",
    )
    args = ap.parse_args()

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        print("GPU が要る (metal_kernel は CPU では動かない)")
        return 2
    if _qsa_tail.MODE != "query":
        print(f"MLXTURBO_QSA_TAIL={_qsa_tail.MODE} では参照が別物になる")
        return 2

    devc = K2B.arch_char()
    pin = os.environ.get("MLX_SDPA_BLOCKS", "")
    print(
        f"== 前提 == arch={_device_info().get('architecture')} "
        f"(devc={devc!r})  MLX_SDPA_BLOCKS={pin or '(未設定 / 表どおり)'}  "
        f"tail={_qsa_tail.MODE}"
    )
    print("  blocks の表 (S -> kv):")
    for S in S_LIST:
        row = "  ".join(
            f"{kv}:{K2B.mirror_blocks(kv, GQA, S)}" for kv in KV_LIST
        )
        print(f"    S={S}: {row}")

    s_list = [1, 2] if args.quick else S_LIST
    kv_list = [4096, 17000] if args.quick else KV_LIST

    rc = 0
    if not args.no_check:
        print("\n== ビット一致 (参照 = bool マスク + 分割 sdpa) ==")
        print(f"  {'S':>2} {'kv':>6} {'blocks':>6}  " + "  ".join(
            f"{kind:>10}" for kind in SCORE_KINDS
        ))
        problems: list[str] = []
        summary: list[dict] = []
        for S in s_list:
            for kv_len in kv_list:
                cells = []
                for ki, kind in enumerate(SCORE_KINDS):
                    rng = np.random.default_rng(1000 * S + kv_len + 7919 * ki)
                    layer = Layer(rng, S, kv_len, kind, cap_pad=16)
                    label = f"S={S} kv={kv_len} {kind}"
                    if layer.blocks is None:
                        cells.append("skip")
                        continue
                    check_bits_vs_argpartition(layer, label, problems)
                    r = compare(layer, label, problems)
                    summary.append({"S": S, "kv": kv_len, "kind": kind, **r})
                    cells.append(
                        "ok" if r["same"] else f"NG {r['max_ulp']}ulp"
                    )
                    del layer
                print(
                    f"  {S:>2} {kv_len:>6} "
                    f"{K2B.mirror_blocks(kv_len, GQA, S) or 0:>6}  "
                    + "  ".join(f"{c:>10}" for c in cells)
                )

        # tail の位相 (kv % cr) を全部踏む。行 0 が自分自身を見ない回が出る
        print("\n  tail の位相 (kv % 4 = 0..3):")
        for kv_len in ([17000] if args.quick else [17000, 17001, 17002, 17003]):
            for S in ([2] if args.quick else [1, 2, 4, 6]):
                rng = np.random.default_rng(31 * kv_len + S)
                layer = Layer(rng, S, kv_len, "relu_zero", cap_pad=8)
                if layer.blocks is None:
                    continue
                label = f"S={S} kv={kv_len} (phase {kv_len % CR})"
                r = compare(layer, label, problems)
                verdict = "ok" if r["same"] else f"NG {r['max_ulp']} ulp"
                print(f"    kv={kv_len} S={S}: {verdict}")
                del layer

        n_all = len(summary)
        n_same = sum(1 for r in summary if r["same"])
        print(f"\n  形 x ケース {n_all} 通り、うち完全一致 {n_same}")
        if problems:
            rc = 1
            print(f"  不一致 {len(problems)} 件:")
            for p in problems[:30]:
                print("   -", p)
            worst = max((r["max_ulp"] for r in summary), default=0)
            frac = sum(r["n_bad"] for r in summary) / max(
                1, sum(r["n"] for r in summary)
            )
            print(
                f"  最大 ulp 差 {worst}、ずれた要素の割合 {frac:.3e} "
                f"-> {'close 層 (1 ulp 以内)' if worst <= 1 else '要調査'}"
            )
        else:
            print("  すべてビット一致 (array_equal)")

    if args.no_bench:
        return rc

    print("\n== 時間 (12 層の別々の K/V を巡回する冷たい連鎖、ABAB、中央値) ==")
    print("  us/層。cur_* は現行経路、k2_* が新経路。")
    print("  K2a と cur選+mask は**引き算で出した派生値**なので中央値の差が")
    print("  そのまま乗る (10-20 us の揺れ、まれに負)。実測は 4 列だけ。")
    print(
        f"  {'S':>2} {'kv':>6} {'blk':>4} | "
        f"{'cur計':>7} {'cur選+mask':>10} {'cur sdpa':>9} | "
        f"{'K2計':>7} {'K2a':>7} {'K2b':>7} | {'比(計)':>7} {'比(attn)':>8}"
    )
    bench_kv = [4096, 17000] if args.quick else BENCH_KV
    rows = []
    for S in s_list:
        for kv_len in bench_kv:
            r = bench(S, kv_len)
            rows.append(r)
            print(
                f"  {r['S']:>2} {r['kv']:>6} {r['blocks']:>4} | "
                f"{r['cur_total']:>7.1f} {r['cur_sel_mask']:>10.1f} "
                f"{r['cur_sdpa']:>9.1f} | "
                f"{r['k2_total']:>7.1f} {r['k2a']:>7.1f} {r['k2b']:>7.1f} | "
                f"{r['cur_total'] / r['k2_total']:>7.2f} "
                f"{r['cur_sdpa'] / r['k2b']:>8.2f}"
            )

    tgt = [r for r in rows if r["S"] == 2 and r["kv"] == 17000]
    if tgt:
        us = tgt[0]["k2b"]
        print(
            f"\n  判定線 (17k S=2 で K2b <= 75 us): {us:.1f} us -> "
            f"{'通る' if us <= 75 else '通らない'}"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
