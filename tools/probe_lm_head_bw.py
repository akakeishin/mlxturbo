"""lm_head の実効帯域が、カーネルの問題かメモリ圧の問題かを切り分ける。

Flash-Next では 98GB のモデルを載せた状態で 109GB/s しか出なかった。同じ
マシンで hyper-connections のカーネルは 227GB/s 出ているので、半分しか
使えていない (`docs/research/KERNEL-BRIEF-MOE-GDN.md:126-172`)。

測る状態は 3 つ。**同じプロセスで 3 つ全部は取れない** (非常駐はモデルを
載せる前にしか作れない) ので、`--resident` の有無で 2 回流す。

1. **非常駐**: lm_head の重みだけを読み込んで測る (既定)。帯域が跳ねれば
   原因はメモリ圧、変わらなければカーネル側。
2. **常駐アイドル**: `--resident` でモデル一式を載せ、何も走っていない
   状態で測る。
3. **decode 直後**: `--resident` で短い生成を 1 本流した直後に測る
   (直前の重みトラフィックが効いているかを見る)。

    uv run python tools/probe_lm_head_bw.py --model ~/models/qwen38fn-mlx-v-fast6
    BIGLOCK_NO_WORKER=1 tools/biglock.sh .venv/bin/python \\
        tools/probe_lm_head_bw.py --model ~/models/qwen38-27b-4bit --resident

lm_head のテンソル名は族で違う (`lm_head.weight` / 27B は
`language_model.lm_head.weight`) ので、**末尾一致**で拾う。
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))


def bench(fn, n=50, reps=5) -> float:
    import mlx.core as mx

    out = []
    for _ in range(reps):
        for _ in range(5):
            mx.eval(fn())
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(fn())
        out.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(out)


def _quant_cfg(cfg: dict) -> tuple[int, int]:
    """lm_head の (bits, group_size)。族ごとの入れ子の揺れを吸収する。"""
    q = cfg["quantization"]
    for key in ("language_model.lm_head", "lm_head"):
        sub = q.get(key)
        if isinstance(sub, dict) and "bits" in sub:
            return sub["bits"], sub["group_size"]
    return q["bits"], q["group_size"]


def _load_lm_head(root: Path, bits: int):
    """safetensors から lm_head の 3 本だけを拾う (モデル全体は載せない)。"""
    import mlx.core as mx

    suffixes = ("lm_head.weight", "lm_head.scales", "lm_head.biases")
    found: dict[str, object] = {}
    for f in sorted(glob.glob(str(root / "*.safetensors"))):
        d = mx.load(f)
        for name in d:
            for suf in suffixes:
                if name.endswith(suf) and suf not in found:
                    found[suf] = d[name]
        if len(found) == len(suffixes):
            break
    missing = [s for s in suffixes if s not in found]
    if missing:
        raise SystemExit(f"lm_head のテンソルが見つからない: {missing}")
    return found["lm_head.weight"], found["lm_head.scales"], found["lm_head.biases"]


def _measure(w, s, b, bits: int, gs: int, label: str) -> None:
    import mlx.core as mx

    n_out, k_words = w.shape
    k = k_words * 32 // bits
    nbytes = k * n_out * bits / 8 + s.size * 2 + b.size * 2
    print(f"\n[{label}] lm_head {k} -> {n_out} {bits}bit gs={gs}"
          f"  重み {nbytes / 1e6:.0f}MB  常駐 (peak) {mx.get_peak_memory() / 1e9:.2f}GB")
    for m in (1, 2, 4):
        x = mx.random.normal((m, k)).astype(mx.bfloat16)
        mx.eval(x)
        us = bench(lambda: mx.quantized_matmul(
            x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits))
        print(f"  M={m}  {us:8.2f} us  実効 {nbytes / us / 1000:6.1f} GB/s")

    # 比較: 同じ量のデータを素直に流したときに何 GB/s 出るか (帯域の天井)
    n_el = int(nbytes // 2)
    big = mx.zeros((n_el,), dtype=mx.bfloat16)
    mx.eval(big)
    us = bench(lambda: mx.sum(big), n=20, reps=3)
    print(f"  参考: 同量 ({nbytes / 1e6:.0f}MB) の総和  {us:8.2f} us"
          f"  実効 {nbytes / us / 1000:6.1f} GB/s  <- この状態の実力")
    del big


def _resident(args) -> int:
    """モデル一式を載せた状態で「アイドル」と「decode 直後」を測る。

    読み込みは `tools/decode_ab_generic.load_model` (出荷経路と同じ順) を
    そのまま使う。lm_head の重みはモデルのモジュールから直に取る (別途
    safetensors から読み直すと 0.64GB を二重に常駐させてしまう)。
    """
    from types import SimpleNamespace

    import decode_ab_generic as G

    ns = SimpleNamespace(model=args.model, mtp=args.mtp, mtp_bits=4, no_mtp=False)
    model, tok, eng, eos, _guard = G.load_model(ns)

    root = Path(args.model).expanduser()
    cfg = json.loads((root / "config.json").read_text())
    bits, gs = _quant_cfg(cfg)
    head = model.language_model.lm_head
    w, s, b = head["weight"], head["scales"], head["biases"]

    _measure(w, s, b, bits, gs, "常駐アイドル")

    cases = G.build_cases(tok, args.ctx)
    label, ids = cases[0]
    row = G.run_once(eng, ids, args.tokens, eos, 3, 8)
    print(f"\n  ({label} {len(ids)} トークン -> {row['n_out']} 生成、"
          f"prefill {row['prefill_s']:.2f}s / decode {row['decode_s']:.2f}s)")
    _measure(w, s, b, bits, gs, "decode 直後")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--resident", action="store_true",
                    help="モデル一式を載せて「常駐アイドル」と「decode 直後」を測る")
    ap.add_argument("--mtp", default=None, help="--resident のときの MTP サイドカー")
    ap.add_argument("--ctx", type=int, default=0,
                    help="--resident の decode に使う文脈 (0 = 短文脈)")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    import mlx.core as mx  # noqa: F401  (mx を先に立ち上げる)

    if args.resident:
        return _resident(args)

    root = Path(args.model).expanduser()
    cfg = json.loads((root / "config.json").read_text())
    bits, gs = _quant_cfg(cfg)
    w, s, b = _load_lm_head(root, bits)
    mx.eval(w, s, b)
    _measure(w, s, b, bits, gs, "非常駐 (lm_head だけ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
