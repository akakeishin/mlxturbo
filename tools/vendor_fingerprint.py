"""vendor の単一系列フォワードの指紋を出す (写しのシーム化の回帰ゲート)。

`tools/verify_prefill_bitident.py` は 91GB の実モデルで 4 分かかる本番ゲート
だが、group=0/4 の比較なので「vendor 自体の挙動が変わった」ことは検出できない。
こちらは合成した小さい Flash-Next を CPU で流し、logits と全キャッシュ配列の
md5 を出す。シーム化のように**挙動を変えないはずの変更**の前後で走らせ、
出力が一行残らず一致することを確認する用途。

    .venv/bin/python tools/vendor_fingerprint.py > before.txt
    (変更を入れる)
    .venv/bin/python tools/vendor_fingerprint.py | diff before.txt -

QSA (indexer) の活性/不活性と prefill の割り方を変えた 4 通りの指紋に加えて、
`mlxturbo/spec_flash.py` の写し 2 つ (`_staged_forward` / `_group_prefill_forward`)
が本家と一致することも見る (写しを触ったときの一次検査。実モデルでの本番
ゲートは `tools/verify_prefill_bitident.py`)。
budget=8 の chunk=4 と chunk=19 は**互いに食い違う**構成。QSA のブロック
格子は kv 長で決まるので、prefill の割り方が変われば選ばれるブロックも
変わる (端数ブロックの因果性を直しても、これは残る)。この道具が見るのは
「変更の前後で同じ値が出るか」だけなので、食い違ったままでよい。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401
from verify_batch_cache import TINY, build  # noqa: E402

CASES = [
    # (indexer_budget, prefill chunk, prompt 長)
    (8, 4, 40),
    (8, 19, 40),
    (4096, 8, 40),
    (8, 40, 64),
]


def digest(budget: int, chunk: int, plen: int):
    model = build(budget)
    prompt = [(i * 7 + 3) % TINY["vocab_size"] for i in range(plen)]
    cache = model.make_cache()
    body = prompt[:-1]
    for lo in range(0, len(body), chunk):
        model(mx.array(body[lo : lo + chunk])[None], cache=cache)
    logits = model(mx.array(prompt[-1:])[None], cache=cache)
    out = []
    cur = int(mx.argmax(logits[0, -1]))
    for _ in range(8):
        out.append(cur)
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1]))

    h = hashlib.md5()
    mx.eval(logits)
    h.update(bytes(memoryview(mx.contiguous(logits.astype(mx.float32)))))
    for c in cache:
        for k in range(4):
            try:
                s = c[k]
            except Exception:
                s = None
            if isinstance(s, mx.array):
                mx.eval(s)
                h.update(bytes(memoryview(mx.contiguous(s.astype(mx.float32)))))
    return h.hexdigest()[:16], out


def _cache_arrays(cache):
    """キャッシュの生きている配列を全部並べる (比較用)。"""
    out = []
    for c in cache:
        if hasattr(c, "keys") and c.keys is not None:
            out.append(c.keys[..., : c.offset, :])
            out.append(c.values[..., : c.offset, :])
        else:
            for k in range(4):
                try:
                    v = c[k]
                except Exception:
                    continue
                if isinstance(v, mx.array):
                    out.append(v)
    return out


def _same(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(
        x.shape == y.shape and bool(mx.all(x == y)) for x, y in zip(a, b)
    )


def _max_diff(a, b) -> float:
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        return float("inf")
    return max(
        (
            float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))))
            for x, y in zip(a, b)
        ),
        default=0.0,
    )


def check_copies() -> bool:
    """spec_flash の写し 2 つが本家と一致することを合成モデルで見る。

    実モデルのゲート (`tools/verify_prefill_bitident.py`) は 91GB を読むので、
    写しを触ったときの一次検査はこちらで済ませる。
    """
    import mlxturbo.spec_flash as SF

    model = build(8)
    plen = 40
    ids_list = [(i * 7 + 3) % TINY["vocab_size"] for i in range(plen)]

    # 写し 1: _staged_forward (段階投入) == Model.__call__
    ids = mx.array(ids_list)[None]
    c1, c2 = model.make_cache(), model.make_cache()
    a, b = model(ids, cache=c1), SF._staged_forward(model, ids, c2)
    mx.eval(a, b)
    staged_ok = bool(mx.all(a == b)) and _same(_cache_arrays(c1), _cache_arrays(c2))
    print(f"staged_forward == Model.__call__: {staged_ok}")

    # 写し 2: _group_prefill_forward (レイヤー主導) == チャンクを順に流すの
    #
    # ここだけはビット一致を要求しない。MoE の行を concat して 1 回の GEMM に
    # まとめるのがこの経路の本体で、行独立ではあるが**累積順が変わる**。
    # 量子化 GPU 経路 (affine_gather_qmm_rhs、BM 行タイル) では実測でビット
    # 一致するが、この合成モデルは非量子化 float32 を CPU で流すので下位ビット
    # が動く (実測 8.3e-7、float32 の丸めの水準)。実モデルでのビット一致は
    # tools/verify_prefill_bitident.py が見る。
    chunks = [mx.array(ids_list[i : i + 10])[None] for i in range(0, plen, 10)]
    c1, c2 = model.make_cache(), model.make_cache()
    for ch in chunks:
        model(ch, cache=c1)
    SF._group_prefill_forward(model, chunks, c2)
    diff = _max_diff(_cache_arrays(c1), _cache_arrays(c2))
    group_ok = diff < 5e-6
    print(f"group_prefill_forward == chunk-major: {group_ok} (max|diff|={diff:.3e})")
    return staged_ok and group_ok


def main() -> int:
    for budget, chunk, plen in CASES:
        d, out = digest(budget, chunk, plen)
        print(f"budget={budget} chunk={chunk} plen={plen}  {d}  {out}")
    return 0 if check_copies() else 1


if __name__ == "__main__":
    raise SystemExit(main())
