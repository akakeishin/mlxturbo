"""4 段 (v-xl / v-l / v-m / v-s) を焼く。

レシピは `mlxturbo/convert_flash.py` の RECIPES に**実行時に注入**する。
あのファイルは親の領域なので書き換えない。焼き方が確定したら親が
RECIPES へ取り込む (内容は docs/BAKE-PLAN.md の「確定レシピ案」)。

配分の原則: **高いビットは後ろへ。削るなら前から。**感度は深さに対して
単調に増え、出口は入口より 1 層あたり 4.9 倍敏感 (docs/BAKE-PLAN.md)。

    uv run python tools/bake.py list
    uv run python tools/bake.py mtp --src <bf16>
    uv run python tools/bake.py convert v-l --src <bf16> --out ~/models/qwen38fn-mlx-v-l
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_LAYERS = 48


def _last(n: int) -> list[int]:
    """末尾 n 層。感度地図により高いビットは後ろへ寄せる。"""
    return list(range(N_LAYERS - n, N_LAYERS))


_COMMON = {
    "ngram": False,
    "ngram_disk": True,
    "router": False,
    "gdn": {"bits": 6, "group_size": 64},
    "head": {"bits": 6, "group_size": 64},
    "attn": {"bits": 6, "group_size": 64},
    "shared": {"bits": 6, "group_size": 64},
    # 融合カーネルが 4/8 しか受けない。6bit にすると黙って素の実装に落ちる
    "hc": {"bits": 8, "group_size": 64},
    "default": {"bits": 8, "group_size": 64},
}


def _tier(experts: int, hi: int, layers: list[int]) -> dict:
    return {
        "experts": {"bits": experts, "group_size": 64},
        "experts_hi": {"bits": hi, "group_size": 64},
        "experts_hi_layers": layers,
        **_COMMON,
    }


# mlx-serve に読ませる段。**ビット幅を混ぜない。**混在パックは向こうで生成が
# 壊れ、速い経路 (Metal ソースの分岐は 4/5/8 のみ) からも外れる
# (docs/research/MLX-SERVE-PACK-FORMAT.md)。
#
# 既知で動くパック (ddalcu の一律 4bit) からの差をエキスパートだけに絞る。
# パックの 93.7% がエキスパートなので、品質の取り分はここでほぼ全部取れる。
# 動かなければ「このアーキでは 4bit しか読めない」と分かり、それはそれで決着。
_SERVE_5 = {
    "experts": {"bits": 5, "group_size": 64},
    "router": {"bits": 4, "group_size": 64},  # 向こうは router を量子化する
    "shared": {"bits": 4, "group_size": 64},
    "gdn": {"bits": 4, "group_size": 64},
    "attn": {"bits": 4, "group_size": 64},
    "hc": {"bits": 4, "group_size": 64},
    "head": {"bits": 8, "group_size": 64},  # 向こうと同じ
    "ngram": False,
    "ngram_disk": True,
    "default": {"bits": 4, "group_size": 64},
}

# 一律 4bit より速い側。MoE の融合カーネルは 2/4/8 を受けるので、2 と 4 は
# どちらも速い経路の上にある。感度は深さに対して単調に増えるので、前を 2bit に
# 落として後ろ 16 層を 4bit で残す。エキスパートは 63.3 -> 44.6 GiB (-30%)。
_SERVE_B = {
    **{k: v for k, v in _SERVE_5.items() if k != "experts"},
    "experts": {"bits": 2, "group_size": 64},
    "experts_hi": {"bits": 4, "group_size": 64},
    "experts_hi_layers": _last(16),
}

# 品質レーン。エキスパートの重み誤差は 48 層でほぼ一定 (最大/最小 1.02-1.07 倍)
# なので、層を選ぶ根拠は深さの感度しかない。末尾から N 層を 8bit にする形だけ試す。
# 4bit と 8bit の重み誤差の比は 16.9 倍。MoE の速いカーネルは 2/4/8 を受けるので
# 8bit は経路に残る (5/6 は落ちる)。
_SERVE_Q4 = {
    **{k: v for k, v in _SERVE_5.items() if k != "experts"},
    "experts": {"bits": 4, "group_size": 64},
}
_SERVE_Q48 = {
    **_SERVE_Q4,
    "experts_hi": {"bits": 8, "group_size": 64},
    "experts_hi_layers": _last(8),
}

# 128GB 機の上限側。Metal の上限が 107.5GB なので常駐 95GB あたりが実用の天井。
# 全層 8bit だとエキスパートだけで 119.6 GiB になり載らない。
_SERVE_Q24 = {
    **_SERVE_Q4,
    "experts_hi": {"bits": 8, "group_size": 64},
    "experts_hi_layers": _last(24),
}

TIERS = {
    "serve-5": _SERVE_5,
    "serve-q4": _SERVE_Q4,
    "serve-q24": _SERVE_Q24,
    "serve-q48": _SERVE_Q48,
    "serve-b": _SERVE_B,
    # 前 8 層だけ 4bit。v-fast6 (4bit が全域に散っている) の同容量改良版
    "v-xl": _tier(4, 6, list(range(8, N_LAYERS))),
    "v-l": _tier(4, 6, _last(8)),
    "v-m": _tier(3, 4, _last(24)),
    "v-s": _tier(2, 3, _last(12)),
}

# 予測 (tools/predict_recipe.py、rebit は悲観側)
PREDICTED = {
    "serve-5": (81.5, "未測定 (一律 4bit より良い側なのは確かだが、幅は未知)"),
    "serve-q4": (66.0, "未測定 (ddalcu と同じ配分。基準線)"),
    "serve-q24": (95.0, "未測定 (末尾 24 層。128GB 機の上限側)"),
    "serve-q48": (76.0, "未測定 (末尾 8 層のエキスパートを 8bit に)"),
    "serve-b": (48.0, "未測定 (2bit は素の affine だと荒い。imatrix が要るかもしれない)"),
    "v-xl": (97.5, "0.0033 前後 (推定。rebit では検証できない段)"),
    "v-l": (76.8, "0.00598"),
    "v-m": (64.8, "0.01257"),
    "v-s": (45.9, "0.05961"),
}


def _inject():
    from mlxturbo import convert_flash

    for name, recipe in TIERS.items():
        convert_flash.RECIPES[name] = recipe
    return convert_flash


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("mtp", help="mtp.* をサイドカーへ抽出 (元重みがある間に)")
    p.add_argument("--src", required=True)
    p.add_argument("--out", default=None)
    p = sub.add_parser("convert")
    p.add_argument("tier", choices=sorted(TIERS))
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = ap.parse_args()

    cf = _inject()

    if args.cmd == "list":
        for name in ("serve-5", "serve-q4", "serve-q48", "serve-q24", "serve-b", "v-xl", "v-l", "v-m", "v-s"):
            gb, kld = PREDICTED[name]
            r = TIERS[name]
            if "experts_hi" in r:
                mix = (f"hi {r['experts_hi']['bits']}bit x "
                       f"{len(r['experts_hi_layers'])} 層")
            else:
                mix = "混在なし (mlx-serve 用)"
            print(f"  {name:7s} experts {r['experts']['bits']}bit / {mix}"
                  f"  -> {gb:.1f}GB ({gb / 1.073741824:.1f}GiB)  予測 KLD {kld}")
        return

    if args.cmd == "mtp":
        out = args.out or str(Path(args.src).parent / "qwen38fn-mtp.safetensors")
        cf.cmd_extract_mtp(argparse.Namespace(src=args.src, out=out))
        return

    gb, kld = PREDICTED[args.tier]
    print(f"=== {args.tier} を焼く === 予測 {gb:.1f}GB / KLD {kld}")
    cf.cmd_convert(argparse.Namespace(
        recipe=args.tier, src=args.src, out=args.out, device=args.device))


if __name__ == "__main__":
    main()
