"""1 トークン生成でどのテンソルを何バイト読むかを、重みを読まずに数える。

ablate.py は「どの部品が時間を食うか」を測るが、その時間が起動回数由来なのか
読み出し量由来なのかは分けられない。ここは後者だけを、safetensors のヘッダから
静的に出す。両方を並べて初めて、融合で削るのかビットで削るのかが決まる。

エキスパートは 512 個のうち top_k 個しか読まないので、その割引を入れる。
n-gram はサイドカー (ディスク) なので本体の予算からは外し、別に表示する。

    uv run python tools/byte_budget.py --model ~/models/qwen38fn-mlx-v-stream
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastmlx.convert_flash import _DTYPE_BYTES, iter_tensor_headers  # noqa: E402

# M3 Max の実測ピーク。micro_moe_gdn.py のビット掃引で見た限界コストが
# MoE 385 GB/s / GDN 投影 243 GB/s だったので、上下を両方出す
BW_HI, BW_LO = 385.0, 243.0


def classify(name: str) -> tuple[str, float]:
    """テンソル名 -> (表示するグループ, 1 トークンあたり読む割合)。"""

    if "ngram" in name:
        return "n-gram (サイドカー)", 0.0
    if name.startswith("model.embed_tokens"):
        return "embed_tokens (1 行だけ)", 0.0
    if "lm_head" in name:
        return "lm_head", 1.0
    if "switch_mlp" in name:
        return "MoE experts (top_k/512)", 10 / 512
    if "shared_expert" in name:
        return "MoE 共有エキスパート", 1.0
    if name.endswith("mlp.gate.weight") or ".mlp.gate." in name:
        return "MoE ルータ", 1.0
    if "linear_attn" in name or "in_proj" in name or "conv1d" in name or "A_log" in name:
        return "GDN 投影/conv", 1.0
    if "hyper_connection" in name or "hc_norm" in name or "input_mix" in name:
        return "hyper-connections", 1.0
    if "indexer" in name:
        return "QSA indexer", 1.0
    if "self_attn" in name or "attn." in name:
        return "full attention (12 層)", 1.0
    if "ple" in name:
        return "PLE", 1.0
    return "その他 (norm 等)", 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens-per-sec", type=float, default=19.4,
                    help="現在の実測。帯域の使用率を出すのに使う")
    args = ap.parse_args()

    by_group: dict[str, float] = defaultdict(float)
    stored: dict[str, float] = defaultdict(float)
    for name, dtype, shape, _ in iter_tensor_headers(Path(args.model)):
        n = 1
        for d in shape:
            n *= d
        nbytes = n * _DTYPE_BYTES.get(dtype.upper(), 4)
        group, frac = classify(name)
        stored[group] += nbytes
        by_group[group] += nbytes * frac

    total_read = sum(by_group.values())
    total_stored = sum(stored.values())

    print(f"モデル: {args.model}")
    print(f"格納 {total_stored / 1e9:.1f} GB / 1 トークンで読む "
          f"{total_read / 1e9:.3f} GB\n")
    print(f"  {'グループ':26s} {'格納 GB':>8s} {'読む GB':>8s} {'割合':>6s} "
          f"{'@385':>6s} {'@243':>6s}")
    for group in sorted(by_group, key=lambda g: -by_group[g]):
        rd = by_group[group]
        if rd == 0 and stored[group] == 0:
            continue
        pct = 100 * rd / total_read if total_read else 0
        print(f"  {group:26s} {stored[group] / 1e9:8.1f} {rd / 1e9:8.3f} "
              f"{pct:5.1f}% {rd / 1e9 / BW_HI * 1000:5.1f}ms "
              f"{rd / 1e9 / BW_LO * 1000:5.1f}ms")

    ms_now = 1000 / args.tokens_per_sec
    print(f"\n  帯域だけの下限   {total_read / 1e9 / BW_HI * 1000:5.1f} ms/token "
          f"({BW_HI / 1000 * 1e3 / (total_read / 1e9) if total_read else 0:.0f} tok/s)")
    print(f"  実測             {ms_now:5.1f} ms/token ({args.tokens_per_sec:.1f} tok/s)")
    print(f"  帯域の使用率     {total_read / 1e9 / (ms_now / 1000) / BW_HI * 100:.0f}%"
          f"  (残りは起動回数などの固定費)")


if __name__ == "__main__":
    main()
