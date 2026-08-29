"""mtp.* を保持したまま group_size 可変で 4bit 量子化する mlx-lm 互換コンバータ。

mlx_lm.convert は読み込み時の sanitize (qwen3_5.py の TextModel.sanitize) で
mtp.* を丸ごと捨てる。ここでは本体は mlx_lm と同じ経路（同じ sanitize、同じ
量子化関数）で変換しつつ、mtp.* は mlxturbo.mtp.load_mtp に通して同じ +1 norm
シフトを適用したうえで量子化し、"mtp." キーのまま同じ safetensors シャード集合
に同梱保存する。group_size は本体・mtp 側とも CLI から可変。

mlx_lm.convert からの差分（詳細は README/報告参照）:
  - mtp.* を破棄せず、mlxturbo.mtp の shift 規約で量子化して同梱保存する
  - --group-size で本体・mtp 双方の量子化グループサイズを指定できる
  - config.json に "fastmlx_mtp": true と "mtp_quantization" を書く
  - --dry-run で本体の先頭 N 層 + mtp.* だけを変換し、ロード検証と逆量子化誤差
    確認ができる（フル 56GB 読みを避けるための開発用モード）
  - upload-repo / mixed quant recipe / dequantize など mlx_lm.convert の付随機能は
    持たない（このプロジェクトの用途はローカル 1 回変換のみのため）

使い方:
    uv run python -m mlxturbo.convert --group-size 128 --out /path/to/out
    uv run python -m mlxturbo.convert --group-size 128 --out /tmp/dry --dry-run
"""

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path

from ._mlx_compat import (
    QWEN35_SHIFTED_NORM_SUFFIXES,
    TextModelArgs,
    create_model_card,
    get_total_parameters,
    mlx_lm_load,
    make_shards,
    quantize_model,
    save_config,
    validate_affine_quantization,
)

from .mtp import MTPModule, find_snapshot, load_mtp

# mlx_lm.convert.MODEL_CONVERSION_DTYPES と同じ集合。float 型 config dtype だけ
# キャスト対象にする。
MODEL_CONVERSION_DTYPES = ("float16", "bfloat16", "float32")

DEFAULT_HF_PATH = "Qwen/Qwen3.8-27B"


def resolve_hf_path(hf_path: str) -> str:
    """repo id ならローカルキャッシュのスナップショットへ解決する。

    mlx_lm.utils.load はローカルに無いパスを渡すとネットワークからダウンロード
    しようとする。56GB のチェックポイントで意図せぬ取得が起きないよう、ここでは
    mlxturbo.mtp.find_snapshot (ローカルキャッシュ限定) で解決してから渡す。
    """
    p = Path(hf_path)
    if p.exists():
        return str(p)
    return find_snapshot(hf_path)


def cast_to_config_dtype(model: nn.Module, config: dict) -> None:
    """mlx_lm.convert.convert と同じ dtype キャストを行う。

    config の torch_dtype (無ければ text_config.dtype) が float 系なら、
    model.cast_predicate (A_log を除外する) を尊重してキャストする。
    """
    dtype = config.get("torch_dtype")
    if dtype is None and (text_config := config.get("text_config")):
        dtype = text_config.get("dtype")
    if dtype not in MODEL_CONVERSION_DTYPES:
        return

    mx_dtype = getattr(mx, dtype)
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def set_dtype(k, v):
        if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
            return v.astype(mx_dtype)
        return v

    model.update(tree_map_with_path(set_dtype, model.parameters()))


def truncate_layers(model: nn.Module, num_layers: int) -> None:
    """--dry-run 用: 本体の decoder layers を先頭 num_layers 本に切り詰める。

    mlx.nn.Module はリスト属性の再代入でパラメータツリーが更新されるため、
    tree_flatten(model.parameters()) はここで切り詰めた本数分しか含まなくなる。
    以降の量子化・保存はこのツリーだけを対象にするので、元の safetensors
    シャード自体は mx.load の遅延評価により読まれない。
    """
    layers = model.language_model.model.layers
    if num_layers >= len(layers):
        return
    model.language_model.model.layers = layers[:num_layers]


def validate_dry_run_layers(model: nn.Module, config: dict, num_layers: int) -> None:
    """Reject dry-run artifacts whose config and physical layer tree would differ."""

    total_layers = len(model.language_model.model.layers)
    min_layers = config["text_config"]["full_attention_interval"]
    if not min_layers <= num_layers <= total_layers:
        raise ValueError(
            "dry_run_layers must satisfy "
            f"{min_layers} <= dry_run_layers <= {total_layers}; got {num_layers}"
        )


def normalize_quantization(
    group_size: int, bits: int, mtp_bits: Optional[int]
) -> tuple[int, int, int]:
    """Resolve MLX affine defaults and validate values before loading weights."""

    effective_group_size = group_size or 64
    effective_bits = bits or 4
    effective_mtp_bits = (
        mtp_bits if mtp_bits is not None else effective_bits
    ) or 4
    validate_affine_quantization(effective_group_size, effective_bits)
    validate_affine_quantization(effective_group_size, effective_mtp_bits)
    return effective_group_size, effective_bits, effective_mtp_bits


def flatten_mtp_weights(mtp: nn.Module) -> dict:
    """MTPModule のパラメータを 'mtp.' プレフィックス付きでフラット化する。

    元チェックポイントの mtp.* キー命名 (mtp.fc.weight, mtp.layers.0...,
    mtp.norm.weight, mtp.pre_fc_norm_*.weight) と一致させることで、
    mlxturbo.mtp.load_mtp と同じキー規約のまま量子化済みとして保存できる。
    """
    return {f"mtp.{k}": v for k, v in tree_flatten(mtp.parameters())}


def base_weights_for_mtp_artifact(model: nn.Module) -> dict:
    """Return base weights in qwen3_5's raw, zero-centred norm convention.

    ``mlx_lm_load`` has already applied +1 to these norms because the source
    checkpoint contains mtp.*.  The output also contains mtp.*, so its reload
    sanitizer will apply +1 again.  Reverse the first shift only in the saved
    weight view; the live converted model remains unchanged.
    """

    weights = dict(tree_flatten(model.parameters()))
    shifted = 0
    for name, value in list(weights.items()):
        if value.ndim == 1 and any(
            name.endswith(suffix) for suffix in QWEN35_SHIFTED_NORM_SUFFIXES
        ):
            weights[name] = value - 1.0
            shifted += 1
    if shifted == 0:
        raise RuntimeError(
            "no qwen3_5 RMSNorm weights matched the raw-save contract"
        )
    return weights


def load_quantized_mtp(out_dir, text_args: TextModelArgs) -> MTPModule:
    """変換済みディレクトリから mtp.* を量子化済みとして読み込む。

    mlxturbo.mtp.load_mtp は生の bf16 チェックポイントから読んで自前で量子化する
    経路しか持たない。ここでは convert() が書いた config["mtp_quantization"] を
    見て同じ形の量子化スケルトンを組み、保存済みの mtp.weight/scales/biases を
    そのまま load_weights する — これが「mlx_lm.load(out) 相当」の mtp 版。
    """
    out_dir = Path(out_dir)
    config = json.loads((out_dir / "config.json").read_text())
    q = config["mtp_quantization"]

    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(
        {v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    )
    weights = {}
    for shard_name in shard_names:
        for k, v in mx.load(str(out_dir / shard_name)).items():
            if k.startswith("mtp."):
                weights[k.removeprefix("mtp.")] = v

    mtp = MTPModule(text_args)
    nn.quantize(
        mtp,
        group_size=q["group_size"],
        bits=q["bits"],
        mode=q.get("mode", "affine"),
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    mtp.load_weights(list(weights.items()), strict=True)
    mtp.eval()
    return mtp


def save_with_mtp(
    dst_path: Path,
    src_path: Path,
    model: nn.Module,
    mtp: nn.Module,
    tokenizer,
    config: dict,
) -> None:
    """mlx_lm.utils.save 相当。本体の重みに mtp.* を同じシャード集合へ混ぜて保存する。"""
    dst_path.mkdir(parents=True, exist_ok=True)

    weights = base_weights_for_mtp_artifact(model)
    weights.update(flatten_mtp_weights(mtp))

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    total_parameters = get_total_parameters(model) + get_total_parameters(mtp)
    index_data = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": total_parameters,
        },
        "weight_map": {},
    }

    weights.clear()

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = dst_path / shard_name
        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})
        for weight_name in shard:
            index_data["weight_map"][weight_name] = shard_name

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(dst_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    save_config(config, config_path=dst_path / "config.json")
    tokenizer.save_pretrained(dst_path)

    for pattern in ("*.py", "generation_config.json"):
        for file in src_path.glob(pattern):
            shutil.copy(file, dst_path)

    create_model_card(dst_path, None)


def convert(
    hf_path: str = DEFAULT_HF_PATH,
    out: str = "mlx_model",
    group_size: int = 128,
    bits: int = 4,
    mtp_bits: Optional[int] = None,
    dry_run: bool = False,
    dry_run_layers: int = 4,
) -> dict:
    """本体 + mtp.* を group_size 可変で 4bit 量子化して out に保存する。

    dry_run=True の場合、本体は先頭 dry_run_layers 本だけを対象にする。
    mtp.* は元々 1 ブロックしかないため常にフル量子化される。
    戻り値は診断用に最終 config と mtp モジュールを含む dict。

    dry_run_layers は full_attention_interval (Qwen3.8-27B では 4) 以上を
    推奨する。qwen3_5.Qwen3_5TextModel は fa_idx = full_attention_interval - 1
    を固定のフル注意キャッシュ位置として forward 時に参照するため、それより
    層数が少ないと (mtp.* とは無関係に) forward がインデックス範囲外で落ちる。
    """
    out_path = Path(out)
    if out_path.exists():
        raise ValueError(
            f"Cannot save to {out_path} as it already exists."
            " Delete it or pick a new path."
        )

    snapshot = resolve_hf_path(hf_path)
    # mlx affine treats zero as "use the default".  Resolve it before any
    # module is quantized so config metadata records the effective values.
    group_size, bits, mtp_bits = normalize_quantization(
        group_size, bits, mtp_bits
    )

    print(f"[mlxturbo.convert] loading base model from {snapshot}")
    model, tokenizer, config = mlx_lm_load(snapshot, return_config=True, lazy=True)

    if dry_run:
        validate_dry_run_layers(model, config, dry_run_layers)
        print(
            f"[mlxturbo.convert] dry-run: truncating base model to first "
            f"{dry_run_layers} layer(s)"
        )
        truncate_layers(model, dry_run_layers)
        config = copy.deepcopy(config)
        config["text_config"]["num_hidden_layers"] = dry_run_layers
        if layer_types := config["text_config"].get("layer_types"):
            # transformers の AutoConfig は num_hidden_layers と layer_types の
            # 長さ一致を検証するため、tokenizer 読み込み時に壊れないよう揃える。
            config["text_config"]["layer_types"] = layer_types[:dry_run_layers]

    cast_to_config_dtype(model, config)

    text_args = TextModelArgs.from_dict(config["text_config"])

    print(
        f"[mlxturbo.convert] loading + quantizing mtp.* "
        f"(group_size={group_size}, bits={mtp_bits})"
    )
    mtp = load_mtp(
        snapshot, text_args, quantize={"group_size": group_size, "bits": mtp_bits}
    )

    print(
        f"[mlxturbo.convert] quantizing base model "
        f"(group_size={group_size}, bits={bits})"
    )
    model, config = quantize_model(model, config, group_size, bits, mode="affine")

    config["fastmlx_mtp"] = True
    config["mtp_quantization"] = {
        "group_size": group_size,
        "bits": mtp_bits,
        "mode": "affine",
    }
    config["mtp_quantization_config"] = config["mtp_quantization"]

    print(f"[mlxturbo.convert] saving to {out_path}")
    save_with_mtp(out_path, Path(snapshot), model, mtp, tokenizer, config)

    return {"out": str(out_path), "config": config, "model": model, "mtp": mtp}


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3.8-27B を mtp.* 保持のまま group_size 可変で 4bit 量子化する"
        )
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        dest="hf_path",
        default=DEFAULT_HF_PATH,
        help="変換元のリポジトリ id またはローカルスナップショットパス"
        f" (デフォルト: {DEFAULT_HF_PATH})",
    )
    parser.add_argument(
        "--out", "--mlx-path", dest="out", required=True, help="保存先ディレクトリ"
    )
    parser.add_argument(
        "--group-size",
        dest="group_size",
        type=int,
        default=128,
        help="量子化グループサイズ (本体・mtp 共通、デフォルト 128)",
    )
    parser.add_argument("--bits", type=int, default=4, help="本体の量子化ビット数")
    parser.add_argument(
        "--mtp-bits",
        dest="mtp_bits",
        type=int,
        default=None,
        help="mtp.* の量子化ビット数 (省略時は --bits と同じ)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="本体の先頭 --dry-run-layers 層 + mtp.* だけを変換する検証モード",
    )
    parser.add_argument(
        "--dry-run-layers",
        dest="dry_run_layers",
        type=int,
        default=4,
        help=(
            "--dry-run 時に変換する本体の層数"
            " (デフォルト 4 = full_attention_interval。"
            " これ未満だと forward がフル注意層不在で落ちる)"
        ),
    )
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()
    convert(
        hf_path=args.hf_path,
        out=args.out,
        group_size=args.group_size,
        bits=args.bits,
        mtp_bits=args.mtp_bits,
        dry_run=args.dry_run,
        dry_run_layers=args.dry_run_layers,
    )


if __name__ == "__main__":
    main()
