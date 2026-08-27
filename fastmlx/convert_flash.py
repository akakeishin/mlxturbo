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
  build-ngram   n-gram 表を量子化してサイドカーへ出す (ディスク運用向け)

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

# 層別上書き: "experts_hi_layers" に載る層の experts は "experts_hi" の
# ビットで焼く (入口/出口が感度高いという folklore 起点。感度スキャンの
# 層別 KLD で入れ替える)。48 層 (0..47)。
_FIRST5_LAST5 = list(range(5)) + list(range(43, 48))
_FIRST6_LAST6 = list(range(6)) + list(range(42, 48))


def _spread(n: int) -> list[int]:
    """48 層から n 層を等間隔で選ぶ (入口/出口に寄せない)。

    v-exp6 までは「入口と出口が効く」という folklore に従って端に寄せていたが、
    層別の感度は未測定。10 層を超えると端寄せの根拠がさらに薄くなるので、
    多層を 6bit にする構成では偏りの無い等間隔にする。
    """

    return sorted({round(i * 48 / n) for i in range(n)})

RECIPES: dict[str, dict] = {
    # 常用 (~96GB): 既定 GPU wired limit に KV 込みで収まる
    "v0-95": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": {"bits": 3, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # n-gram へ +6.4GB (~102GB)。v-exp6 との等バイト A/B の片割れ
    "v0-105": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": {"bits": 4, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # 同じ +6.3GB を experts 10 層の 6bit に使う (~102GB)。v0-105 と KLD を
    # 等予算で比較し「n-gram と experts のどちらにビットを盛るか」を決める
    "v-exp6": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _FIRST5_LAST5,
        "ngram": {"bits": 3, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # n-gram を RAM から追い出す構成 (~98GB)。表はサイドカーに置き、
    # 浮いた 19.2GB をすべて experts に回して 6bit の層を 10 -> 40 に増やす。
    # 使うには build-ngram でサイドカーを作り、読込後に
    # fastmlx.ngram_stream.install(model, <サイドカー>) を呼ぶ
    "v-stream": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _spread(40),
        "ngram": False,
        "ngram_disk": True,
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # n-gram 2bit の検証用 (~99GB)。experts は v-exp6 と同一にして n-gram の
    # ビットだけ 3 -> 2 に落とす。KLD が v-exp6 (0.00181) から大きく劣化しな
    # ければ、n-gram は 2bit で足りることになり 6.4GB が experts へ回せる
    "v-ng2": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _FIRST5_LAST5,
        "ngram": {"bits": 2, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # v-ng2 が通ったときの本命 (~112GB): n-gram を 2bit に抑えて浮いた分を
    # すべて experts に回し、6bit の層を 10 -> 32 に増やす
    "v-exp-max": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _spread(32),
        "ngram": {"bits": 2, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # ギリギリ構成 (~112GB): n-gram 4bit + experts 12 層 6bit。
    # iogpu.wired_limit_mb 引き上げ + 専用機運用前提。115GB 帯は OS が
    # 苦しくなるのでこれを上限とする。6bit の層数を 16 -> 12 に詰めたのは、
    # n-gram を group_size=32 にせざるを得ず (head_dim=160 が 64 で割れない)
    # +3.2GB 増えた分を天井内で吸収するため
    "v-max-112": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _FIRST6_LAST6,
        "ngram": {"bits": 4, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
}

_LAYER_RE = None


def _layer_index(path: str) -> int | None:
    global _LAYER_RE
    if _LAYER_RE is None:
        import re

        _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    m = _LAYER_RE.search(path)
    return int(m.group(1)) if m else None


def resolve_rule(recipe: dict, path: str):
    """パス → そのテンソル/モジュールに適用する量子化規則。"""

    c = classify(path)
    if c == "experts":
        hi_layers = recipe.get("experts_hi_layers")
        if hi_layers is not None and _layer_index(path) in hi_layers:
            return recipe["experts_hi"]
    return recipe[c]


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
        return resolve_rule(recipe, path)

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
        if c == "ngram" and recipe.get("ngram_disk"):
            continue  # サイドカーへ出すので本体には入らない
        rule = resolve_rule(recipe, name)
        if rule is False or len(shape) < 2 or shape[-1] % rule["group_size"]:
            # 非量子化のまま残るもの: router / norm / 1 次元テンソル に加えて、
            # 入力次元が group_size で割り切れないもの。mlx の quantize は
            # これを黙って素通しする (n-gram の head_dim=160 で踏んだ)。
            # 台帳が実測とずれる原因になるのでここでも同じ規則を適用する
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

    # convert と同じ理由で CPU 側に置く (外付け SSD の mmap を GPU から
    # 読むとコマンドバッファが監視タイムアウトする)
    mx.set_default_device(mx.cpu)
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
    import os

    if RECIPES[args.recipe].get("ngram_disk"):
        # vendored arch は import 時にこの旗を読む。mlx_lm を触る前に立てる
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
        print("n-gram はディスク運用: 本体には入れない")

    import mlx.core as mx
    from mlx_lm.convert import convert

    if args.device == "cpu":
        # 既定は CPU。重みは外付け SSD 上の mmap なので、GPU で量子化すると
        # カーネル実行中の page-in が USB 越しになり、Metal のコマンドバッファが
        # 監視タイムアウトで落ちる (kIOGPUCommandBufferCallbackErrorTimeout、
        # シャード 2 の保存で再現)。CPU には同じ監視が無い
        mx.set_default_device(mx.cpu)

    # vendor 側を編集したまま古いコピーが site-packages に残ると、旗も
    # 修正も効かないまま焼き上がる (n-gram が落ちず 169GB まで膨らんだ)。
    # vendor が正本なので毎回上書きする
    cmd_install_arch(argparse.Namespace(force=True))
    convert(
        hf_path=args.src,
        mlx_path=args.out,
        quantize=True,
        # mlx_lm.quantize_model は predicate を呼ぶ前に
        #   module.weight.shape[-1] % <この group_size> != 0 -> 量子化しない
        # という足切りをする。ここを 64 にすると n-gram (head_dim=160) が
        # レシピの group_size=32 に届く前に落ちて bf16 のまま残る (51B params
        # がそのまま残り 178GB になって OOM した)。レシピ中の最小値を渡す。
        q_group_size=min(
            r["group_size"]
            for r in RECIPES[args.recipe].values()
            if isinstance(r, dict)
        ),
        q_bits=4,
        quant_predicate=build_predicate(args.recipe),
    )
    print(f"converted -> {args.out}")


def cmd_build_ngram(args):
    from .ngram_stream import build_sidecar

    build_sidecar(
        Path(args.src), Path(args.out), bits=args.bits, group_size=32,
        layout=args.layout,
    )


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

    p = sub.add_parser("build-ngram")
    p.add_argument("--src", required=True, help="bf16 アーカイブ")
    p.add_argument("--out", required=True, help="サイドカーの出力先")
    p.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8))
    p.add_argument(
        "--layout", default="interleaved", choices=("interleaved", "separate"),
        help="interleaved=ディスク常駐向け / separate=RAM 常駐向け",
    )
    p.set_defaults(fn=cmd_build_ngram)

    p = sub.add_parser("convert")
    p.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.set_defaults(fn=cmd_convert)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
