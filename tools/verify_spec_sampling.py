"""投機ありのサンプリングが、逐次サンプリングと同じ分布かを実測で確かめる。

## なぜ要るか

`FlashSpecEngine._verify` は 1 ラウンドで検証フォワードの**全位置を先に**
サンプルし、draft と一致したプレフィックスだけ採用する。位置 j のサンプルは
`lg[:, j]` にしか依存せず、受理判定は samples[0..j-1] にしか依存しないので、
条件付けても位置 j の分布は歪まない -- というのが根拠で、これは top_p /
top_k / min_p のような**位置局所な変換**を挟んでも成り立つ (2026-09-01 に
投機経路がこれらを受けるようになった、mlxturbo/runner.py 参照)。

理屈は通っているが、実装が「どの位置の logits を使うか」を取り違えても
出力は一見それらしいままになる。そこを実測で押さえる。

## 何を測るか

`_verify` を**直接**叩く。合成モデルを丸ごと回す形も試したが、乱数初期化の
MTP が引くドラフトはまず当たらず、**多トークン受理の経路を一度も踏まない**
ので検査にならなかった (受理が起きなければ 1 位置ずつのサンプルにしかならず、
主張そのものを触らない)。固定 logits と固定ドラフトを与えれば受理率を選べる。

    (a) 逐次: 位置 0 をサンプル -> ドラフトと一致したら位置 1 をサンプル -> ...
    (b) 投機: _verify (全位置を先にサンプルして一致プレフィックスを採用)

から**採用トークン列そのもの**の分布を N 回ずつ取って比べる。長さも列も
分布に含まれる (受理数が変わることも検出できる)。

## 判定基準 (測る前に宣言)

2 標本のカイ二乗検定 (期待度数 5 未満のセルは「その他」にまとめる)。

    p 値 >= 0.001 なら「同じ分布と矛盾しない」= 合格
    p 値 <  0.001 なら不合格 (実装を疑う)

閾値を 0.001 と緩めに置くのは、これが**同一性の証明ではなく破損の検出**
だから。台 (support) は `--vocab` で小さく保つ (既定 6) -- セルが全部
「期待度数 5 未満」に落ちると検定が空回りする。

**検定の感度は `--selftest` で確かめる。**`_verify` をわざと壊して
(採用した全位置に位置 0 のサンプルを使う = 位置を取り違える実装ミスの模擬)、
落ちることを確認する。

    .venv/bin/python tools/verify_spec_sampling.py
    .venv/bin/python tools/verify_spec_sampling.py --selftest
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401
from mlxturbo.runner import _position_local_sampler  # noqa: E402
from verify_batch_cache import TINY, build  # noqa: E402


class _Cap:
    """`_verify` が触る cap は hyper だけ。形さえ合っていればよい。"""

    def __init__(self, k1: int):
        self.hyper = mx.zeros((1, k1, 1))


def sequential(logits, drafts, sampler):
    """投機なしの等価な手続き。これが基準。

    位置 0 をサンプルし、ドラフトと一致したら次の位置へ進む。外れたらそこで
    打ち切る。**投機が主張しているのはこれと同じ分布**。
    """
    out = []
    for j in range(logits.shape[1]):
        tok = int(sampler(logits[:, j]).item())
        out.append(tok)
        if j >= len(drafts) or tok != drafts[j]:
            break
    return tuple(out)


def speculative(eng, logits, drafts, sampler):
    dr = [mx.array([[d]]) for d in drafts]
    toks, _hypers, _hit = eng._verify(
        _Cap(logits.shape[1]), logits, dr, 1.0, sampler=sampler
    )
    return tuple(int(t.item()) for t in toks)


def chi2_two_sample(a: Counter, b: Counter, min_expected=5.0):
    """2 標本のカイ二乗。期待度数の小さいセルは「その他」に畳む。"""
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    cells, rest_a, rest_b = [], 0, 0
    for k in keys:
        ca, cb = a.get(k, 0), b.get(k, 0)
        exp_a = (ca + cb) * na / (na + nb)
        exp_b = (ca + cb) * nb / (na + nb)
        if min(exp_a, exp_b) < min_expected:
            rest_a += ca
            rest_b += cb
        else:
            cells.append((ca, cb))
    if rest_a + rest_b:
        cells.append((rest_a, rest_b))
    stat = 0.0
    for ca, cb in cells:
        tot = ca + cb
        ea, eb = tot * na / (na + nb), tot * nb / (na + nb)
        if ea > 0:
            stat += (ca - ea) ** 2 / ea
        if eb > 0:
            stat += (cb - eb) ** 2 / eb
    df = max(len(cells) - 1, 1)
    # 生存関数 (正則化不完全ガンマ)。scipy を持ち込まないための最小実装
    p = _chi2_sf(stat, df)
    return stat, df, p


def _chi2_sf(x: float, k: int) -> float:
    """カイ二乗分布の上側確率。k は自由度。級数展開 (k が小さいので十分)。"""
    if x <= 0:
        return 1.0
    # P(X > x) = Q(k/2, x/2) を、正則化不完全ガンマの級数から
    a, z = k / 2.0, x / 2.0
    if z < a + 1:
        # 下側の級数
        term = 1.0 / a
        total = term
        n = 1
        while n < 10000:
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
            n += 1
        lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return max(0.0, 1.0 - lower)
    # 連分数 (上側)
    tiny = 1e-300
    b, c, d = z + 1 - a, 1 / tiny, 1 / (z + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--vocab", type=int, default=6,
                    help="台の大きさ。小さくしないと検定が空回りする")
    ap.add_argument("--depth", type=int, default=2, help="1 ラウンドのドラフト数")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--selftest", action="store_true",
                    help="_verify をわざと壊して、検定が落ちることを見る")
    args = ap.parse_args()

    import mlxturbo.spec_flash as SF

    mx.random.seed(0)
    V, k = args.vocab, args.depth
    # 位置ごとに違う、そこそこ尖った分布にする (一様だと受理率が読めない)
    logits = mx.random.normal((1, k + 1, V)) * 1.5
    mx.eval(logits)
    drafts = [int(mx.argmax(logits[0, j]).item()) for j in range(k)]

    sampler = _position_local_sampler(1.0, args.top_p, 0, 0.0, None)
    assert sampler is not None

    eng = SF.FlashSpecEngine.__new__(SF.FlashSpecEngine)  # _verify だけ使う
    if args.selftest:
        _orig = SF.FlashSpecEngine._verify

        def _broken(self, cap, lg, dr, temp, precomputed=None, sampler=None):
            toks, hypers, hit = _orig(self, cap, lg, dr, temp, precomputed, sampler)
            # 採用した全位置に位置 0 のサンプルを使う
            return [toks[0]] * len(toks), hypers, hit

        SF.FlashSpecEngine._verify = _broken

    print("判定基準はモジュール docstring のとおり (測る前に宣言済み)。")
    print(f"vocab={V} depth={k} top_p={args.top_p} N={args.samples} x 2 経路")
    print(f"ドラフト (各位置の argmax) = {drafts}\n")

    seq, spec = Counter(), Counter()
    for _ in range(args.samples):
        seq[sequential(logits, drafts, sampler)] += 1
        spec[speculative(eng, logits, drafts, sampler)] += 1

    n_multi = sum(c for t, c in spec.items() if len(t) > 1)
    print(f"  受理が起きた割合 (投機側、2 トークン以上): "
          f"{n_multi / args.samples:.3f}")
    if n_multi == 0:
        print("=== 検査になっていない: 受理が一度も起きていない ===")
        return 1

    print("\n  採用トークン列の分布 (上位 6)")
    for t, _ in seq.most_common(6):
        print(f"    {t}: 逐次 {seq[t]:5d}  投機 {spec[t]:5d}")

    stat, df, p = chi2_two_sample(seq, spec)
    print(f"\n  chi2={stat:.2f} df={df} p={p:.4f}")
    if args.selftest:
        if p < 0.001:
            print("=== selftest 合格: 壊れを検出できた (検定に感度がある) ===")
            return 0
        print("=== selftest 不合格: 壊したのに検出できていない。"
              "N を増やすか台を小さくすること ===")
        return 1
    if p >= 0.001:
        print("=== 合格: 同じ分布と矛盾しない ===")
        return 0
    print("=== 不合格: 分布がずれている。実装を疑うこと ===")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
