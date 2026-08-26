"""Qwen3.8-Flash-Next (qwen4_exp) のクラス別レシピ変換ドライバ。

Phase Q (docs/STATUS.md)。bf16 アーカイブ (外付け SSD) から、128GB Mac に
収まる混合精度 MLX チェックポイントを焼く。レシピの根拠はテンソル台帳の
バイト予算表 (STATUS の Phase Q 節) と、後続の感度スキャンで更新する。

サブコマンド:
  install-arch  vendored qwen4_exp.py を site-packages の mlx_lm へ配置
  estimate      重みを読まずにレシピ適用後のサイズを見積もる (ヘッダのみ)
  extract-mtp   mtp.* テンソルをサイドカー safetensors へ抽出 (bf16 のまま。
                量子化はエンジン読込時の --mtp-bits に任せる、27B と同じ規約)
  convert       mlx_lm.convert + クラス別 quant_predicate で本体を変換

使い方 (例):
  uv run python -m fastmlx.convert_flash install-arch
  uv run python -m fastmlx.convert_flash estimate --recipe v0-95
  uv run python -m fastmlx.convert_flash extract-mtp \
      --src "/Volumes/Mobile SSD/models/Qwen3.8-Flash-Next" \
      --out "/Volumes/Mobile SSD/models/qwen38fn-mtp.safetensors"
  uv run python -m fastmlx.convert_flash convert --recipe v0-95 \
      --src "/Volumes/Mobile SSD/models/Qwen3.8-Flash-Next" \
      --out ~/models/qwen38fn-mlx-v0-95
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ARCH = REPO_ROOT / "tools" / "vendor" / "qwen4_exp.py"

# ---------------------------------------------------------------------------
# レシピ
# ---------------------------------------------------------------------------
# クラス判定はモジュールパス (quantize 時) とテンソル名 (estimate 時) の両方に
# 同じ規則を使う。qwen4_exp のパス構造:
#   experts:  model.layers.N.mlp.switch_mlp.{gate_up_proj,down_proj}
#   router:   model.layers.N.mlp.gate  (量子化しない — モデル既定と同じ)
#   shared:   model.layers.N.mlp.shared_expert.* / shared_expert_gate
#   ngram:    model.layers.N.ple_embedding.ngram_embedding.shard_i
#   gdn:      model.layers.N.linear_attn.*
#   qsa:      model.layers.N.self_attn.* (indexer 含む)
#   embed:    model.embed_tokens / lm_head

RECIPES: dict[str, dict[str, dict | bool]] = {
    # 常用 (~95GB): 既定 GPU wired limit に KV 込みで収まる
    "v0-95": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": {"bits": 3, "group_size": 64},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # 全部盛り (~103GB): n-gram も 4bit。iogpu.wired_limit_mb 引き上げ運用向け
    "v0-105": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": {"bits": 4, "group_size": 64},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
}


def classify(path: str) -> str:
    """モジュールパス/テンソル名 → レシピクラス。"""

    if "ngram" in path:
        return "ngram"
    if path.endswith("mlp.gate") or ".mlp.gate." in path:
        return "router"
    if "switch_mlp" in path or ".experts." in path:
        return "experts"
    return "default"


def build_predicate(recipe_name: str):
    recipe = RECIPES[recipe_name]

    def predicate(path, module, *_config):
        del module
        return recipe[classify(path)]

    return predicate


# ---------------------------------------------------------------------------
# safetensors ヘッダ読み (重み本体を読まない)
# ---------------------------------------------------------------------------


def iter_tensor_headers(src: Path):
    """(tensor_name, dtype, shape, shard_path) を全シャードから列挙する。"""

    for shard in sorted(src.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            yield name, info["dtype"], info["shape"], shard


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}


def cmd_estimate(args):
    recipe = RECIPES[args.recipe]
    totals: dict[str, float] = {}
    for name, dtype, shape, _ in iter_tensor_headers(Path(args.src)):
        if name.startswith(("mtp.", "vision_tower.", "model.visual.")):
            continue
        n = 1
        for d in shape:
            n *= d
        c = classify(name)
        rule = recipe[c]
        if rule is False or len(shape) < 2:
            # 非量子化 (router / norm / 1 次元テンソル) は bf16 のまま
            b = n * _DTYPE_BYTES.get(dtype, 2)
        else:
            # 実効 bits/weight = bits + 16*2/group_size (scale+bias が bf16)
            eff = rule["bits"] + 32 / rule["group_size"]
            b = n * eff / 8
        totals[c] = totals.get(c, 0.0) + b
    for c, b in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{c:10s} {b / 1e9:8.1f} GB")
    print(f"{'TOTAL':10s} {sum(totals.values()) / 1e9:8.1f} GB")


def cmd_install_arch(args):
    import mlx_lm.models as models_pkg

    dest = Path(models_pkg.__file__).parent / "qwen4_exp.py"
    if dest.exists() and not args.force:
        print(f"already installed: {dest} (--force で上書き)")
        return
    shutil.copyfile(VENDOR_ARCH, dest)
    print(f"installed: {dest}")


def cmd_extract_mtp(args):
    import mlx.core as mx

    src = Path(args.src)
    picked: dict[str, object] = {}
    shard_names: dict[Path, list[str]] = {}
    for name, _, _, shard in iter_tensor_headers(src):
        if name.startswith("mtp."):
            shard_names.setdefault(shard, []).append(name)
    for shard, names in sorted(shard_names.items()):
        tensors = mx.load(str(shard))
        for n in names:
            picked[n] = tensors[n]
        del tensors
    if not picked:
        raise SystemExit("mtp.* テンソルが見つからない")
    mx.save_safetensors(args.out, picked)
    total = sum(v.nbytes for v in picked.values())
    print(f"extracted {len(picked)} tensors, {total / 1e9:.2f} GB -> {args.out}")


def cmd_convert(args):
    from mlx_lm.convert import convert

    cmd_install_arch(argparse.Namespace(force=False))
    convert(
        hf_path=args.src,
        mlx_path=args.out,
        quantize=True,
        q_group_size=64,
        q_bits=4,
        quant_predicate=build_predicate(args.recipe),
    )
    print(f"converted -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("install-arch")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_install_arch)

    p = sub.add_parser("estimate")
    p.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    p.add_argument("--src", required=True)
    p.set_defaults(fn=cmd_estimate)

    p = sub.add_parser("extract-mtp")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_extract_mtp)

    p = sub.add_parser("convert")
    p.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_convert)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
