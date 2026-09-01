"""vendor の単一系列フォワードの指紋を出す (写しのシーム化の回帰ゲート)。

`tools/verify_prefill_bitident.py` は 91GB の実モデルで 4 分かかる本番ゲート
だが、group=0/4 の比較なので「vendor 自体の挙動が変わった」ことは検出できない。
こちらは合成した小さい Flash-Next を CPU で流し、logits と全キャッシュ配列の
md5 を出す。シーム化のように**挙動を変えないはずの変更**の前後で走らせ、
出力が一行残らず一致することを確認する用途。

    .venv/bin/python tools/vendor_fingerprint.py > before.txt
    (変更を入れる)
    .venv/bin/python tools/vendor_fingerprint.py | diff before.txt -

QSA (indexer) の活性/不活性と prefill の割り方を変えた 4 通りを見る。
budget=8 の chunk=4 と chunk=19 は**わざと食い違う**構成で、QSA の端数
ブロックが未来を見せる件 (docs/BACKLOG.md「QSA tail の因果性」) が直ると
一致に変わる。挙動を変えない変更の検査に使うぶんには、食い違ったままで
よい (前後で同じ値が出ることだけを見る)。
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


def main() -> int:
    for budget, chunk, plen in CASES:
        d, out = digest(budget, chunk, plen)
        print(f"budget={budget} chunk={chunk} plen={plen}  {d}  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
