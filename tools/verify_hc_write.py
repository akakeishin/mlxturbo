"""hyper-connection の書き戻し (`DecoderLayer._combine`) の mx.compile 融合が
出力を変えないことを確かめる。

対象は `mlxturbo.fused.enable_hc_write()` / `disable_hc_write()`。読み側
(`enable_hyper_connection_kernel`) と違って量子化重みも GPU 専用カーネルも
絡まない、素の mx 演算 (multiply → add) をまとめるだけの融合なので、
`tools/vendor_fingerprint.py` と同じく合成した小さい Flash-Next を CPU で
流して確認できる (GPU も実モデルも不要)。

decode 幅 (S=1..4 相当) と prefill 幅 (S=128 相当、CLAUDE.md の作法に沿って
合成モデルでは縮小) の両方で、on/off の出力トークン列と logits がビット
一致することを見る。

    .venv/bin/python tools/verify_hc_write.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401
from mlxturbo import fused  # noqa: E402
from verify_batch_cache import TINY, build  # noqa: E402

# (prefill 長, prefill chunk, 続けて流す decode 幅の並び)。decode 幅には
# MTP 検証で実際に出る S=1..4 に加え、prefill 相当の大きめの塊 (128) も混ぜる。
CASES = [
    (40, 19, [1, 1, 2, 3, 4]),
    (64, 8, [4, 3, 2, 1]),
    (16, 16, [128]),
]


def run(budget: int, plen: int, chunk: int, decode_widths: list[int]):
    """1 回の prefill + 指定した幅ぶんの decode を流し、logits と生成列を返す。"""
    model = build(budget)
    prompt = [(i * 7 + 3) % TINY["vocab_size"] for i in range(plen)]
    cache = model.make_cache()
    body = prompt[:-1]
    for lo in range(0, len(body), chunk):
        model(mx.array(body[lo : lo + chunk])[None], cache=cache)
    logits = model(mx.array(prompt[-1:])[None], cache=cache)
    mx.eval(logits)
    outs = [logits.astype(mx.float32)]
    cur = int(mx.argmax(logits[0, -1]))
    toks = [cur]
    for w in decode_widths:
        # 検証系の幅 S をそのまま模す (中身は繰り返しでも、テストしたいのは
        # 「S が 1 でも 128 でも _combine が正しく畳めるか」なので十分)
        ids = mx.array([[(cur + j * 13 + 1) % TINY["vocab_size"] for j in range(w)]])
        logits = model(ids, cache=cache)
        mx.eval(logits)
        outs.append(logits.astype(mx.float32))
        cur = int(mx.argmax(logits[0, -1]))
        toks.append(cur)
    return outs, toks


def main() -> int:
    ok = True
    fused.disable_hc_write()  # 既定 off から始める (前段が汚していても揃える)

    for plen, chunk, widths in CASES:
        budget = 4096  # QSA 不活性 (この検証は _combine だけを見たいので固定)

        fused.disable_hc_write()
        base_outs, base_toks = run(budget, plen, chunk, widths)

        fused.enable_hc_write()
        fused_outs, fused_toks = run(budget, plen, chunk, widths)
        fused.disable_hc_write()

        max_diff = max(
            float(mx.max(mx.abs(a - b))) for a, b in zip(base_outs, fused_outs)
        )
        toks_match = base_toks == fused_toks
        case_ok = max_diff == 0.0 and toks_match
        ok &= case_ok
        status = "OK" if case_ok else "NG"
        print(
            f"{status} plen={plen} chunk={chunk} widths={widths}: "
            f"max|diff|={max_diff:.3e} tokens一致={toks_match}"
        )
        if not toks_match:
            print(f"     base ={base_toks}")
            print(f"     fused={fused_toks}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
