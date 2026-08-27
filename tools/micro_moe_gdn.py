"""MoE と GDN の融合案を、本体 98GB を読まずに比べる。

デコードはディスパッチ律速なので、勝敗は「カーネルを何回起動したか」でほぼ
決まる (docs/STATUS.md)。であれば本物の重みは要らない。同じ形の乱数ブロックを
1 つ作り、48 回 (GDN は 44 回) 直列に通して 1 回だけ eval すれば、本番の
パイプラインと同じ条件で案を比べられる。

絶対値は本番より速く出る (重みが 1 層ぶんしかなく、キャッシュに乗る)。ここで
見るのは案どうしの差だけで、勝った案は実機の tools/ablate.py で確かめる。

    uv run python tools/micro_moe_gdn.py
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

D = 2560
MOE_INTER = 640
N_EXPERTS = 512
TOP_K = 10
N_LAYERS = 48

CONV_K = 4
N_K, N_V = 16, 48
DK, DV = 128, 128
GDN_LAYERS = 44


def qlin(inp: int, out: int, bits: int = 8, gs: int = 64):
    lin = nn.Linear(inp, out, bias=False)
    return lin.to_quantized(group_size=gs, bits=bits)


@mx.compile
def _router_head(logits):
    """gate の出力 -> (選ばれた添字, 正規化された重み)。

    行列積を含まないので、elementwise の連鎖が途切れない。hyper-connections で
    mx.compile が効かなかったのは間に行列積が挟まるからで、ここは事情が違う。
    """

    idx = mx.argpartition(logits, N_EXPERTS - TOP_K, axis=-1)[..., -TOP_K:]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
    return idx, w


def timeit(fn, x, n_layers: int, reps: int = 12) -> float:
    """n_layers 回直列に通して 1 回 eval。中央値を ms で返す。"""

    for _ in range(3):
        mx.eval(fn(x, n_layers))
    ts = []
    for _ in range(reps):
        mx.eval(mx.array(0))
        t = time.perf_counter()
        mx.eval(fn(x, n_layers))
        ts.append((time.perf_counter() - t) * 1000)
    return sorted(ts)[len(ts) // 2]


# --------------------------------------------------------------------- MoE


class MoE:
    """SparseMoeBlock と同じ構成。重みは乱数。"""

    def __init__(self, expert_bits: int = 4):
        from mlx_lm.models.switch_layers import SwitchGLU

        self.gate = qlin(D, N_EXPERTS, bits=8)
        self.switch = SwitchGLU(D, MOE_INTER, N_EXPERTS)
        self.switch = _quantize_switch(self.switch, expert_bits)
        self.shared_gate = qlin(D, 1, bits=8)
        self.shared_gp = qlin(D, MOE_INTER, bits=8)
        self.shared_up = qlin(D, MOE_INTER, bits=8)
        self.shared_dp = qlin(MOE_INTER, D, bits=8)
        # 共有を第 512 番として持つ版 (merged_shared 用)
        self.switch_plus = _quantize_switch(
            SwitchGLU(D, MOE_INTER, N_EXPERTS + 1), expert_bits
        )
        self.shared_idx = mx.array([[[N_EXPERTS]]])

    # --- そのまま -----------------------------------------------------
    def plain(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        out = (self.switch(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        sh = self.shared_dp(nn.silu(self.shared_gp(x)) * self.shared_up(x))
        return out + mx.sigmoid(self.shared_gate(x)) * sh

    # --- ルータ頭を除いた場合の上限 (idx/w を外から与える) -------------
    def no_router(self, x, idx, w):
        out = (self.switch(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        sh = self.shared_dp(nn.silu(self.shared_gp(x)) * self.shared_up(x))
        return out + mx.sigmoid(self.shared_gate(x)) * sh

    # --- 共有エキスパートを除いた場合の上限 ---------------------------
    def no_shared(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        return (self.switch(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)

    # --- 否定を省く (argpartition の下側を取る) -----------------------
    def no_negate(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(logits, N_EXPERTS - TOP_K, axis=-1)[..., -TOP_K:]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        out = (self.switch(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        sh = self.shared_dp(nn.silu(self.shared_gp(x)) * self.shared_up(x))
        return out + mx.sigmoid(self.shared_gate(x)) * sh

    # --- ルータ頭を mx.compile に通す ---------------------------------
    def compiled_router(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx, w = _router_head(logits)
        out = (self.switch(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        sh = self.shared_dp(nn.silu(self.shared_gp(x)) * self.shared_up(x))
        return out + mx.sigmoid(self.shared_gate(x)) * sh

    # --- 共有エキスパートを switch に第 512 番として畳む ---------------
    # 共有は hidden が routed と同じ 640 なので、同じテンソルに 1 本足せる。
    # 合成は (y_e * w_e).sum なので、w の末尾に sigmoid(gate) を継げば
    # 数式としては元と一致する
    def merged_shared(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(logits, N_EXPERTS - TOP_K, axis=-1)[..., -TOP_K:]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        idx = mx.concatenate([idx, self.shared_idx], axis=-1)
        w = mx.concatenate([w, mx.sigmoid(self.shared_gate(x))], axis=-1)
        return (self.switch_plus(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)

    # --- 合成 (mul+sum+astype) を除いた場合の上限 ----------------------
    def no_combine(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K]
        y = self.switch(x, idx)[..., 0, :]
        sh = self.shared_dp(nn.silu(self.shared_gp(x)) * self.shared_up(x))
        return y + mx.sigmoid(self.shared_gate(x)) * sh


def bench_residual(reps: int = 12) -> None:
    """残差の合成 `hyper + (x[...,None,:] * inject[...,None]).reshape(...)`。

    DecoderLayer で 1 層 2 回 x 48 層 = 96 回走る。行列積が挟まらない純粋な
    elementwise の連なりなので、hyper-connections と違って mx.compile が
    効くはず (あちらは間に量子化行列積が入って連なりが 1-3 op に分断された)。

    ablate.py が「残り」としてまとめている約 10ms の中身の候補。
    """

    print("\n残差の合成 x 96 回\n")

    def plain(hyper, x, inject):
        return hyper + (x[..., None, :] * inject[..., None]).reshape(
            *x.shape[:-1], -1
        )

    @mx.compile
    def compiled(hyper, x, inject):
        return hyper + (x[..., None, :] * inject[..., None]).reshape(
            *x.shape[:-1], -1
        )

    hyper = mx.random.normal((1, 1, D * 4)).astype(mx.bfloat16)
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    inject = mx.random.normal((1, 1, 4)).astype(mx.bfloat16)

    # 数値が一致するかを先に見る
    mx.eval(plain(hyper, x, inject), compiled(hyper, x, inject))
    a, b = plain(hyper, x, inject), compiled(hyper, x, inject)
    mx.eval(a, b)
    err = float(mx.abs(a - b).max() / (mx.abs(a).max() + 1e-9))
    print(f"  compile 版との相対誤差: {err:.2e} "
          f"({'一致' if err == 0 else '不一致'})")

    for name, fn in (("そのまま", plain), ("mx.compile", compiled)):
        def run(h0, n, f=fn):
            h = h0
            for _ in range(n):
                h = f(h, x, inject)
            return h
        ms = timeit(run, hyper, 96, reps)
        print(f"  {name:14s} {ms:6.2f} ms  ({ms / 96 * 1000:5.1f} us/回)")


def check_merged_shared_algebra() -> float:
    """「共有を第 N 番のエキスパートとして畳んでよい」を小さい形で確かめる。

    合成は sum_e y_e * w_e なので、共有の出力に sigmoid(gate) を掛けて足すのと、
    w の末尾に sigmoid(gate) を継いで一緒に足すのは同じ式になる。実装ではなく
    式が合っているかを見たいので、量子化なしの小さい形でやる。

    返すのは相対誤差。
    """

    from mlx_lm.models.switch_layers import SwitchGLU

    d, hid, ne, k = 16, 8, 6, 3
    mx.random.seed(1)
    sw = SwitchGLU(d, hid, ne + 1)
    mx.eval(sw.parameters())
    x = mx.random.normal((1, 1, d))
    gate_w = mx.random.normal((ne, d)) * 0.5
    sgate_w = mx.random.normal((1, d)) * 0.5

    logits = x @ gate_w.T
    idx = mx.argpartition(logits, ne - k, axis=-1)[..., -k:]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
    shared_w = mx.sigmoid(x @ sgate_w.T)

    # 元の形: routed の重み付き和 + sigmoid(gate) * 共有
    routed = (sw(x, idx) * w[..., None]).sum(axis=-2)
    shared = sw(x, mx.array([[[ne]]]))[..., 0, :]
    ref = routed + shared_w * shared

    # 畳んだ形: 添字と重みを継いで一度に和を取る
    idx2 = mx.concatenate([idx, mx.array([[[ne]]])], axis=-1)
    w2 = mx.concatenate([w, shared_w], axis=-1)
    got = (sw(x, idx2) * w2[..., None]).sum(axis=-2)

    mx.eval(ref, got)
    return float(mx.abs(ref - got).max() / (mx.abs(ref).max() + 1e-9))


def _quantize_switch(sw, bits: int, gs: int = 64):
    """SwitchGLU の 3 本を QuantizedSwitchLinear へ差し替える。"""

    for name in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(sw, name)
        setattr(sw, name, lin.to_quantized(group_size=gs, bits=bits))
    return sw


# --------------------------------------------------------------------- GDN


class GDN:
    """GatedDeltaNet の入口 (投影 4 本 + conv + 正規化) だけを写したもの。

    gated_delta_update 本体は MLX 側のカーネルなので触らない。ここで見たいのは
    その周りの op 数。
    """

    KEY_DIM = DK * N_K
    VALUE_DIM = DV * N_V
    CONV_DIM = KEY_DIM * 2 + VALUE_DIM

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qkv = qlin(D, self.CONV_DIM, bits=bits)
        self.z = qlin(D, self.VALUE_DIM, bits=bits)
        self.b = qlin(D, N_V, bits=bits)
        self.a = qlin(D, N_V, bits=bits)
        self.conv_w = mx.random.normal((self.CONV_DIM, CONV_K, 1)) * 0.02
        self.conv = nn.Conv1d(
            self.CONV_DIM, self.CONV_DIM, kernel_size=CONV_K, groups=self.CONV_DIM,
            bias=False, padding=0,
        )
        self.conv.weight = self.conv_w
        # 4 本を 1 本に畳んだ版 (出力側で連結)
        self.merged = qlin(D, self.CONV_DIM + self.VALUE_DIM + 2 * N_V, bits=8)
        self.state = mx.zeros((1, CONV_K - 1, self.CONV_DIM))
        self.inv_scale = DK**-0.5
        self.k_w = mx.full((DK,), self.inv_scale)
        self.q_w = mx.full((DK,), self.inv_scale**2)

    def _conv_plain(self, mixed):
        conv_in = mx.concatenate([self.state, mixed], axis=1)
        keep = mx.contiguous(conv_in[:, -(CONV_K - 1):, :])
        return nn.silu(self.conv(conv_in)), keep

    def _qk_plain(self, q, k):
        q = (self.inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k

    def _qk_folded(self, q, k):
        """定数倍を rms_norm の weight に畳む。op が 4 -> 2 になる。"""
        return (
            mx.fast.rms_norm(q, self.q_w, 1e-6),
            mx.fast.rms_norm(k, self.k_w, 1e-6),
        )

    def front(self, x, folded_qk: bool, merged_proj: bool):
        B, S = x.shape[0], x.shape[1]
        if merged_proj:
            all_p = self.merged(x)
            c = self.CONV_DIM
            mixed = all_p[..., :c]
            z = all_p[..., c : c + self.VALUE_DIM]
            b = all_p[..., c + self.VALUE_DIM : c + self.VALUE_DIM + N_V]
            a = all_p[..., c + self.VALUE_DIM + N_V :]
        else:
            mixed = self.qkv(x)
            z = self.z(x)
            b = self.b(x)
            a = self.a(x)
        conv_out, _ = self._conv_plain(mixed)
        q, k, v = mx.split(conv_out, [self.KEY_DIM, 2 * self.KEY_DIM], axis=-1)
        q = q.reshape(B, S, N_K, DK)
        k = k.reshape(B, S, N_K, DK)
        v = v.reshape(B, S, N_V, DV)
        q, k = self._qk_folded(q, k) if folded_qk else self._qk_plain(q, k)
        # 本体 (gated_delta_update) は測らない。ここでは前段の op 数だけ見る
        return q.sum() + k.sum() + v.sum() + z.sum() + a.sum() + b.sum()


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-bits", type=int, default=4)
    args = ap.parse_args()

    mx.random.seed(0)
    bench_residual()
    err = check_merged_shared_algebra()
    print(f"共有を switch に畳む式の相対誤差: {err:.2e} "
          f"({'一致' if err < 1e-5 else '不一致 — 式が違う'})\n")

    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)

    print(f"MoE (hidden={D}, inter={MOE_INTER}, experts={N_EXPERTS}, "
          f"top_k={TOP_K}, {args.expert_bits}bit) x {N_LAYERS} 層\n")
    moe = MoE(args.expert_bits)
    idx = mx.array([[list(range(TOP_K))]])
    w = mx.softmax(mx.random.normal((1, 1, TOP_K)), axis=-1)

    def chain(fn):
        def run(x0, n):
            h = x0
            for _ in range(n):
                h = fn(h)
            return h
        return run

    base = timeit(chain(moe.plain), x, N_LAYERS)
    rows = [("そのまま", base)]
    for name, fn in (
        ("ルータ頭なし (上限)", lambda h: moe.no_router(h, idx, w)),
        ("共有エキスパートなし (上限)", moe.no_shared),
        ("合成なし (上限)", moe.no_combine),
        ("否定を省く", moe.no_negate),
        ("ルータ頭を compile", moe.compiled_router),
        ("共有を switch に畳む", moe.merged_shared),
    ):
        rows.append((name, timeit(chain(fn), x, N_LAYERS)))

    for name, ms in rows:
        d = base - ms
        print(f"  {name:30s} {ms:7.2f} ms  ({ms / N_LAYERS * 1000:6.1f} us/層)"
              + (f"   -{d:5.2f} ms" if d > 0.005 else ""))

    # ディスパッチ律速か帯域律速かの切り分け。ビットを落として時間が比例して
    # 減るなら帯域律速で、融合ではなくビット配分が効く
    print("\n  -- experts のビットを振る (帯域律速かの判定) --")
    per_layer_params = TOP_K * 3 * MOE_INTER * D
    for bits in (2, 3, 4, 6, 8):
        m = MoE(bits)
        ms = timeit(chain(m.plain), x, N_LAYERS)
        gb = per_layer_params * N_LAYERS * bits / 8 / 1e9
        print(f"  {bits}bit {ms:7.2f} ms   読み出し {gb:5.2f} GB/token"
              f"   実効 {gb / (ms / 1000):6.1f} GB/s")
        del m

    print(f"\nGDN (conv_dim={GDN.CONV_DIM}, K={CONV_K}) x {GDN_LAYERS} 層\n")
    gdn = GDN()
    variants = [
        ("そのまま", False, False),
        ("q/k の定数を rms_norm に畳む", True, False),
        ("投影 4 本を 1 本に", False, True),
        ("両方", True, True),
    ]
    gbase = None
    for name, folded, merged in variants:
        fn = chain(lambda h, f=folded, m=merged: (
            gdn.front(h, f, m) * mx.zeros((1, 1, D)) + h
        ))
        ms = timeit(fn, x, GDN_LAYERS)
        gbase = gbase if gbase is not None else ms
        d = gbase - ms
        print(f"  {name:30s} {ms:7.2f} ms  ({ms / GDN_LAYERS * 1000:6.1f} us/層)"
              + (f"   -{d:5.2f} ms" if d > 0.005 else ""))

    print("\n  -- 投影のビットを振る --")
    gdn_params = D * (GDN.CONV_DIM + GDN.VALUE_DIM + 2 * N_V)
    for bits in (3, 4, 6, 8):
        g = GDN(bits)
        fn = chain(lambda h: g.front(h, False, False) * mx.zeros((1, 1, D)) + h)
        ms = timeit(fn, x, GDN_LAYERS)
        gb = gdn_params * GDN_LAYERS * bits / 8 / 1e9
        print(f"  {bits}bit {ms:7.2f} ms   読み出し {gb:5.2f} GB/token"
              f"   実効 {gb / (ms / 1000):6.1f} GB/s")
        del g


if __name__ == "__main__":
    main()
