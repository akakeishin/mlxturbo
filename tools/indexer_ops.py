"""decode 幅 (S<=8) の経路で QSAIndexer が発行する op を列挙する (CPU、合成モデル)。

`tools/forward_split.py` の `_profiled_build` と同じ名前空間差し替え
(`mlx.core` / `mlx.core.fast` の呼び出し可能属性を一時的にラップして
呼ばれた回数を数える) を、`QSAIndexer.__call__` 1 回の呼び出しだけに絞って
使う。GPU は使わない (合成モデル、CPU、数秒)。

QSAIndexer 自体のコードは呼ぶだけで書き換えない --- ここは観測用の道具。
`MLXTURBO_INDEXER_LEAN` (`mlxturbo/indexer_lean.py`) の on/off で
`_pooled_and_top` の op 数がどう変わるかを、decode 幅 S<=8 のいくつかの
kv (= n_blocks) で比べる。あわせて **lean on/off で `__call__` の返り値
(bool (B,1,S,kv)) とキャッシュの状態がビット一致すること**も検査する
(「数値は変えない」の一次確認。`tools/vendor_fingerprint.py` は decode を
常に S=1 でしか回さないので、S=2/3/4/8 はここでしか検査しない)。

    .venv/bin/python tools/indexer_ops.py
    .venv/bin/python tools/indexer_ops.py --widths 1,2,4,8 --kvs 300,2000,9000

`tools/forward_split.py` は `mlx.core`/`mlx.core.fast` の名前空間だけを
パッチするので、`mx.arange(n) * compress_ratio` の `*` や
`pooled.astype(mx.float32)` の `.astype` のような `mx.array` の束縛
メソッド (`__mul__`/`.astype` など) は見えない、と明記している。ここでは
`_pooled_and_top`/`__call__` が実際に使う束縛メソッドの一部
(`_ARRAY_METHODS`) も追加でパッチして数える (`mx.array` は nanobind 型
だが、メソッドの再代入自体はできることを確認済み) --- ただし全メソッドを
網羅してはいないので、これでも実際の削減 op 数の下限にしかならない。
`mx.einsum` の内部実装 (トップレベルの呼び出し 1 回としてしか数えられない)
は変わらず見えない。CPU バックエンドのグラフ構築 (これがここで数えるもの)
と GPU/Metal 側の実カーネル本数は別物であることにも注意 --- ここが見ている
のは Python レベルの mx.* / mx.array.* ディスパッチ回数で、
`docs/research/SESSION-2026-09-02-CATCHUP.md` が言う「小さいカーネルの
起動レイテンシ」の正確な代理指標ではなく、それに強く相関するはずの
近似でしかない (in-model の壁時計 A/B は親が行う)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401  (arch_registry の meta_path フック)
import mlx_lm.models.qwen4_exp as Q  # noqa: E402


# ------------------------------------------------------------- op counting


def _patch_namespace(namespace, ns_label, counts):
    """`forward_split._profiled_build` と同じ差し替え。呼び出し可能な
    公開属性 (関数・builtin) を、呼ばれた回数を数えるラッパへ差し替える。
    型オブジェクト (`mx.array` 自体など) は対象外 (isinstance に使われて
    いるので差し替えると壊れる)。"""
    patched = []
    for name in dir(namespace):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(namespace, name)
        except Exception:
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        key = f"{ns_label}.{name}"

        def _make(orig, key=key):
            def _wrapper(*a, **kw):
                counts[key] = counts.get(key, 0) + 1
                return orig(*a, **kw)

            return _wrapper

        wrapper = _make(obj)
        try:
            setattr(namespace, name, wrapper)
        except Exception:
            continue
        patched.append((name, obj))
    return patched


def _unpatch(namespace, patched):
    for name, orig in patched:
        setattr(namespace, name, orig)


# `mx.arange(n) * compress_ratio` の `*` や `pooled.astype(mx.float32)` の
# ``.astype`` は `mx.array` の束縛メソッド (`__mul__`/`.astype`) であって
# `mx` の名前空間には出てこない --- `_patch_namespace` (forward_split.py と
# 同じ手法) では原理的に見えない (docstring の「既知の限界」参照)。
# `_pooled_and_top`/`__call__` が実際に使う束縛メソッドだけを別途パッチして
# 数える (мx.array 自体は nanobind 型だが、メソッドの再代入はできることを
# 確認済み)。
_ARRAY_METHODS = (
    "astype", "reshape", "transpose", "sum", "mean",
    "__getitem__", "__mul__", "__add__", "__sub__", "__neg__",
    "__le__", "__matmul__",
)


def _patch_array_methods(counts):
    patched = []
    for name in _ARRAY_METHODS:
        try:
            orig = getattr(mx.array, name)
        except Exception:
            continue
        key = f"mx.array.{name}"

        def _make(orig, key=key):
            def _wrapper(self, *a, **kw):
                counts[key] = counts.get(key, 0) + 1
                return orig(self, *a, **kw)

            return _wrapper

        try:
            setattr(mx.array, name, _make(orig))
        except Exception:
            continue
        patched.append((name, orig))
    return patched


def _unpatch_array_methods(patched):
    for name, orig in patched:
        setattr(mx.array, name, orig)


def count_ops(fn, include_array_methods: bool = True):
    """`fn()` を 1 回呼び、その間の `mx`/`mx.fast` (+ 任意で `mx.array` の
    束縛メソッド) の呼び出し回数を名前ごとに集計して返す
    (`{"total": n, "by_name": {...}}`)。戻り値は eval してから捨てる
    (計測は呼び出し回数だけで、時間は測らない)。"""
    counts: dict[str, int] = {}
    patched_core = _patch_namespace(mx, "mx", counts)
    patched_fast = _patch_namespace(mx.fast, "mx.fast", counts)
    patched_arr = _patch_array_methods(counts) if include_array_methods else []
    try:
        out = fn()
        outs = list(out) if isinstance(out, (list, tuple)) else [out]
        mx.eval([o for o in outs if isinstance(o, mx.array)])
    finally:
        _unpatch(mx, patched_core)
        _unpatch(mx.fast, patched_fast)
        _unpatch_array_methods(patched_arr)
    return {"total": sum(counts.values()), "by_name": counts}


# ------------------------------------------------------------- synthetic model


def build_indexer(
    hidden_size: int = 64,
    n_heads: int = 2,
    kv_heads: int = 1,
    head_dim: int = 16,
    budget: int = 64,
    compress_ratio: int = 8,
    rotary_dim: int = 8,
    rope_theta: float = 10000.0,
):
    """`tools/vendor_fingerprint.py` / `tools/verify_batch_cache.py` と同じ
    作り方 (乱数初期化、CPU) の、単体の `QSAIndexer` + `RotaryEmbedding`。
    フル `Model` を組まないのは、`QSAIndexer.__call__` が必要とするのは
    重み (index_qk_proj/q_layernorm/k_layernorm) と `rope` オブジェクトだけ
    で、その他 47 種類のモジュールは要らないため (op 数の計測が他の層の
    ノイズで汚れるのも避けられる)。``rotary_dim < head_dim`` は本番と同じ
    partial rotary (`_rope_partial`) を通すための構成。"""
    args = Q.TextArgs(
        hidden_size=hidden_size,
        indexer_n_heads=n_heads,
        indexer_kv_heads=kv_heads,
        indexer_head_dim=head_dim,
        indexer_budget=budget,
        indexer_compress_ratio=compress_ratio,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(0)
    idx = Q.QSAIndexer(args)
    from mlx.utils import tree_map

    idx.update(
        tree_map(
            lambda a: mx.random.normal(a.shape) * 0.05
            if a.dtype == mx.float32
            else a,
            idx.parameters(),
        )
    )
    mx.eval(idx.parameters())
    idx.eval()
    rope = Q.RotaryEmbedding(rotary_dim, rope_theta)
    return args, idx, rope


def prime(idx, rope, cache, target_offset: int, hidden_size: int, chunk: int = 97):
    """`offset` が `target_offset` に届くまで、適当なチャンク幅で
    `idx(x, rope, cache, offset)` を回してキャッシュ (raw key バッファ・
    pooled キャッシュ) を進める。値はどうでもよい (op 数・分岐は shape と
    python int の offset だけで決まる) ので `mx.random.normal` で十分。
    `chunk` を compress_ratio と互いに素な値にして、様々な端数状態を通す。
    """
    offset = 0
    while offset < target_offset:
        step = min(chunk, target_offset - offset)
        x = mx.random.normal((1, step, hidden_size))
        idx(x, rope, cache, offset)
        mx.eval(cache._buf)
        if cache._pooled is not None:
            mx.eval(cache._pooled)
        offset += step
    return offset


def clone_cache(cache: "Q._IndexerCache") -> "Q._IndexerCache":
    """`cache` と同じ状態を指す、独立した `_IndexerCache`。MLX の配列は
    `arr[...] = v` のような代入構文でも古い参照を書き換えない (新しい値へ
    ローカル名を束ね直すだけ) ので、属性を浅くコピーするだけで以後の
    `update()`/`pooled()` の呼び出しが元の `cache` に影響しない
    (`tools/prefill_anatomy.py` の snapshot/restore と同じ前提)。
    """
    out = Q._IndexerCache()
    out._buf, out.offset = cache._buf, cache.offset
    out._pooled, out._pooled_n = cache._pooled, cache._pooled_n
    out._bs, out._be, out._bs_n = cache._bs, cache._be, cache._bs_n
    out._pooled_f32 = cache._pooled_f32
    out._pooled_f32_n = cache._pooled_f32_n
    return out


# ------------------------------------------------------------- main


def report_ops(idx, rope, cache, offset: int, S: int, hidden_size: int) -> dict:
    x = mx.random.normal((1, S, hidden_size))
    return count_ops(lambda: idx(x, rope, cache, offset))


def align_phase(idx, rope, cache, offset: int, hidden_size: int,
                 compress_ratio: int, target_phase: int) -> int:
    """``offset % compress_ratio == target_phase`` になるまで 1 回追加で
    進める (lean のオン・オフどちらでも値は同じなので False のまま進める)。
    「定常状態 (ブロックがまたがない decode ラウンド)」を測るための下ごしらえ
    --- 境界の直後から測ると常に伸びが混ざってしまうので、境界から離れた
    位置を作る。"""
    idx._indexer_lean = False
    phase = offset % compress_ratio
    need = (target_phase - phase) % compress_ratio
    if need:
        x = mx.random.normal((1, need, hidden_size))
        idx(x, rope, cache, offset)
        mx.eval(cache._buf)
        offset += need
    return offset


def check_bitidentical(idx, rope, cache, offset: int, S: int, hidden_size: int) -> dict:
    """lean off/on を、同じ出発点から同じ入力で 1 回呼んで比べる。返り値
    (bool マスク) とキャッシュの状態 (raw key バッファ・pooled・fp32 pooled)
    がどちらもビット一致することを確認する。"""
    mx.random.seed(1234 + offset + S)
    x = mx.random.normal((1, S, hidden_size))

    c_off = clone_cache(cache)
    idx._indexer_lean = False
    out_off = idx(x, rope, c_off, offset)

    c_on = clone_cache(cache)
    idx._indexer_lean = True
    out_on = idx(x, rope, c_on, offset)

    def _arrays_equal(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(mx.all(a == b))

    ok_out = _arrays_equal(out_off, out_on)
    ok_buf = _arrays_equal(c_off.keys, c_on.keys)
    ok_pooled = _arrays_equal(c_off._pooled, c_on._pooled)
    return {
        "offset": offset, "S": S,
        "out_match": ok_out, "buf_match": ok_buf, "pooled_match": ok_pooled,
        "all_match": ok_out and ok_buf and ok_pooled,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--compress-ratio", type=int, default=32,
                     help="大きめの既定値 (32): warm 計測 (下記) が幅 S<=8 の"
                          "呼び出し 2 回でブロック境界をまたがないための余白")
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--widths", default="1,2,4,8",
                     help="decode 幅 S のカンマ区切り (すべて S<=8 の想定)")
    ap.add_argument("--kvs", default="300,2000,9000",
                     help="op を数える直前の kv (= offset)。budget を跨いで"
                          "疎化が効いた状態で計測する")
    ap.add_argument("--top", type=int, default=20, help="op の内訳、上位何件を出すか")
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    kvs = [int(v) for v in args.kvs.split(",") if v.strip()]

    _, idx, rope = build_indexer(
        hidden_size=args.hidden_size, budget=args.budget,
        compress_ratio=args.compress_ratio,
    )

    print(f"budget={args.budget} compress_ratio={args.compress_ratio}"
          f" hidden_size={args.hidden_size} widths={widths} kvs={kvs}")
    print(f"MLXTURBO_INDEXER_LEAN の on/off を _indexer_lean 属性で直接切り替える"
          f" (env var は使わない --- このスクリプト単体の検査用)。\n")

    all_checks = []
    for kv in kvs:
        cache = Q._IndexerCache()
        offset = prime(idx, rope, cache, kv, args.hidden_size)
        # ブロック境界から離れた位置に揃える (下の warm 計測が幅 S<=8 の
        # 呼び出し 1 回 (温め) + S<=8 (計測) の計 16 トークン以内で境界を
        # またがないように、compress_ratio の半分あたりへ)
        offset = align_phase(idx, rope, cache, offset, args.hidden_size,
                              args.compress_ratio, args.compress_ratio // 2)
        n_blocks = offset // args.compress_ratio
        print(f"=== kv(offset)={offset} n_blocks={n_blocks} "
              f"(budget={args.budget} 超なので疎化が効いている) ===")
        for S in widths:
            # cold: このラウンドが初めて lean を通る場合 (キャッシュがまだ
            # 無い、`_bs`/`_pooled_f32` を今回作る)。base と同じだけ作業する
            # ので delta はほぼ 0 になるはず (キャッシュを作る分、わずかに
            # 増えることもありうる)。
            idx._indexer_lean = False
            base_cold = report_ops(idx, rope, clone_cache(cache), offset, S,
                                    args.hidden_size)
            idx._indexer_lean = True
            lean_cold = report_ops(idx, rope, clone_cache(cache), offset, S,
                                    args.hidden_size)

            # warm: 1 回 lean で小さく進めて `_bs`/`_pooled_f32` を温めてから
            # (n_blocks が変わらない範囲で) 測る --- decode の大多数のラウンド
            # が実際に踏む経路 (ブロックがまたがない定常状態)。
            warm_cache = clone_cache(cache)
            idx._indexer_lean = True
            x_warm = mx.random.normal((1, 1, args.hidden_size))
            idx(x_warm, rope, warm_cache, offset)
            mx.eval(warm_cache._buf)
            warm_offset = offset + 1
            crosses = (warm_offset + S) // args.compress_ratio != n_blocks
            base_warm = report_ops(idx, rope, clone_cache(warm_cache),
                                    warm_offset, S, args.hidden_size)
            idx._indexer_lean = True
            lean_warm = report_ops(idx, rope, clone_cache(warm_cache),
                                    warm_offset, S, args.hidden_size)
            idx._indexer_lean = False
            base_warm_off = report_ops(idx, rope, clone_cache(warm_cache),
                                        warm_offset, S, args.hidden_size)

            d_cold = lean_cold["total"] - base_cold["total"]
            d_warm = lean_warm["total"] - base_warm_off["total"]
            print(f"  S={S}: cold base={base_cold['total']} lean={lean_cold['total']}"
                  f" (delta {d_cold:+d})"
                  f"  |  warm{'*' if crosses else ''} base={base_warm_off['total']}"
                  f" lean={lean_warm['total']} (delta {d_warm:+d},"
                  f" {d_warm / base_warm_off['total'] * 100:+.1f}%)")

            chk = check_bitidentical(idx, rope, cache, offset, S, args.hidden_size)
            all_checks.append(chk)
            tag = "OK" if chk["all_match"] else "MISMATCH"
            print(f"    数値一致 (cold, offset={offset}): {tag}"
                  f" (out={chk['out_match']} buf={chk['buf_match']}"
                  f" pooled={chk['pooled_match']})")

        # 内訳 (最後に見た S、base と lean、warm 側) を上位だけ出す
        idx._indexer_lean = False
        base_detail = report_ops(idx, rope, clone_cache(warm_cache),
                                  warm_offset, widths[-1], args.hidden_size)
        idx._indexer_lean = True
        lean_detail = report_ops(idx, rope, clone_cache(warm_cache),
                                  warm_offset, widths[-1], args.hidden_size)
        print(f"  --- op 内訳 (warm, S={widths[-1]}, 上位{args.top}) ---")
        print(f"  base: {sorted(base_detail['by_name'].items(), key=lambda kv: -kv[1])[:args.top]}")
        print(f"  lean: {sorted(lean_detail['by_name'].items(), key=lambda kv: -kv[1])[:args.top]}")
        print()

    n_bad = sum(1 for c in all_checks if not c["all_match"])
    print(f"数値一致チェック: {len(all_checks) - n_bad}/{len(all_checks)} OK")
    if n_bad:
        print("MISMATCH:")
        for c in all_checks:
            if not c["all_match"]:
                print(f"  {c}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
