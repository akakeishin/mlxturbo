"""量子化済み MTP サイドカーの読み戻しを合成モデルで確認する。

`~/models/qwen38-27b-mtp` のような**既に量子化された** MTP サイドカーは、
以前の `load_mtp_file` では「重みを読む → nn.quantize」の順だったため
`fc.scales` などが「model に無い」と弾かれていた (BACKLOG「27B レーンの下調べ」
の 04:45 の追記)。ここでは実モデルを使わず、小さな `MTPModule` を作って
quantize → 保存 → `load_mtp_file` で読み戻し、パラメータがビット一致することを
確かめる。GPU も 98GB のチェックポイントも要らない (数秒)。

プロジェクトの依存に pytest が無いので、bench/test_convert.py と同じく
plain assert で書いてある。

    uv run python bench/test_mtp_quantized_sidecar.py
    pytest bench/test_mtp_quantized_sidecar.py     # pytest がある環境なら
"""

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModelArgs

from mlxturbo.mtp import MTPModule, load_mtp_file

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="mlxturbo_mtp_sidecar_test_"))


def tiny_args() -> TextModelArgs:
    """本物の 27B (hidden 5120 / head_dim 256 / fai 4) と同じ形をした小さい args。

    full_attention_interval を 4 のままにしておくのが要点で、MTPModule は
    layer_idx = fai - 1 を渡してフル注意層を作る。全射影の K が group_size で
    割り切れる必要がある (fc は 2*hidden = 256、o_proj は heads*head_dim = 128)。
    """
    return TextModelArgs(
        model_type="qwen3_5",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=512,
        full_attention_interval=4,
    )


def tiny_moe_args() -> TextModelArgs:
    """Qwen3.6 と同じ MoE 経路を小さい形で再現する。"""
    args = tiny_args()
    args.num_experts = 4
    args.num_experts_per_tok = 2
    args.moe_intermediate_size = 64
    args.shared_expert_intermediate_size = 64
    return args


def quantized_reference(args: TextModelArgs, group_size: int, bits: int) -> MTPModule:
    mtp = MTPModule(args)
    nn.quantize(
        mtp,
        group_size=group_size,
        bits=bits,
        mode="affine",
        class_predicate=lambda _, m: hasattr(m, "to_quantized"),
    )
    mx.eval(mtp.parameters())
    return mtp


def save_sidecar(mtp: MTPModule, out: Path, quantization: dict | None) -> Path:
    """MTPModule のパラメータを本物のサイドカーと同じ並びで書き出す。

    キーは `mtp.` 接頭辞なしで MTPModule のパラメータ名そのもの
    (`~/models/qwen38-27b-mtp/model.safetensors.index.json` と同じ)。
    """
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(mtp.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    index = {
        "metadata": {"total_size": sum(v.nbytes for v in weights.values())},
        "weight_map": {k: "model.safetensors" for k in weights},
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=4))
    if quantization is not None:
        config = {
            "block_size": 3,
            "model_type": "qwen3_5_mtp",
            "quantization": quantization,
        }
        (out / "config.json").write_text(json.dumps(config, indent=4))
    return out / "model.safetensors"


def assert_same_params(a: MTPModule, b: MTPModule, label: str) -> None:
    pa = dict(tree_flatten(a.parameters()))
    pb = dict(tree_flatten(b.parameters()))
    assert set(pa) == set(pb), (
        f"{label}: パラメータ名が違う "
        f"(欠け {sorted(set(pa) - set(pb))}, 余り {sorted(set(pb) - set(pa))})"
    )
    for k in sorted(pa):
        assert pa[k].shape == pb[k].shape, f"{label}: {k} の形が違う"
        assert pa[k].dtype == pb[k].dtype, f"{label}: {k} の dtype が違う"
        assert bool(mx.all(pa[k] == pb[k])), f"{label}: {k} がビット一致しない"


def test_quantized_sidecar_roundtrip():
    """quantize → 保存 → 読み戻しが、ディレクトリ指定でもファイル指定でも一致する。"""
    args = tiny_args()
    ref = quantized_reference(args, group_size=64, bits=4)
    d = _TMP_ROOT / "q4g64"
    f = save_sidecar(ref, d, {"group_size": 64, "bits": 4, "mode": "affine"})

    # 直っていなければ、ここが「Received N parameters not in model: fc.scales, ...」で落ちる
    assert_same_params(ref, load_mtp_file(str(d), args), "ディレクトリ指定")
    assert_same_params(ref, load_mtp_file(str(f), args), "ファイル指定")

    # cli.py は --mtp-bits から quantize={"bits": 4, "group_size": 64} を渡してくる。
    # 量子化済みならサイドカー側の設定が勝ち、この引数は無視される
    loaded = load_mtp_file(str(d), args, quantize={"bits": 4, "group_size": 64})
    assert_same_params(ref, loaded, "quantize 引数つき")


def test_sidecar_quantization_wins_over_argument():
    """サイドカーが 8bit/128 なら、渡された 4bit/64 ではなくサイドカーが勝つ。"""
    args = tiny_args()
    ref = quantized_reference(args, group_size=128, bits=8)
    d = save_sidecar(
        ref, _TMP_ROOT / "q8g128", {"group_size": 128, "bits": 8, "mode": "affine"}
    ).parent
    loaded = load_mtp_file(str(d), args, quantize={"bits": 4, "group_size": 64})
    assert_same_params(ref, loaded, "8bit/128 サイドカー")
    assert loaded.fc.bits == 8 and loaded.fc.group_size == 128, (
        f"サイドカーの量子化設定が効いていない (bits={loaded.fc.bits}, "
        f"group_size={loaded.fc.group_size})"
    )


def test_quantized_moe_sidecar_roundtrip_and_forward():
    """MoE の SwitchLinear 3 本にも scales/biases の受け皿を作る。"""
    args = tiny_moe_args()
    ref = quantized_reference(args, group_size=64, bits=5)
    d = save_sidecar(
        ref, _TMP_ROOT / "moe_q5g64", {"group_size": 64, "bits": 5, "mode": "affine"}
    ).parent
    loaded = load_mtp_file(str(d), args)
    assert_same_params(ref, loaded, "MoE 5bit サイドカー")

    params = dict(tree_flatten(loaded.parameters()))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"layers.0.mlp.switch_mlp.{proj}"
        assert f"{prefix}.scales" in params, f"{prefix}.scales が無い"
        assert f"{prefix}.biases" in params, f"{prefix}.biases が無い"

    embeds = mx.random.normal((1, 5, args.hidden_size)).astype(mx.bfloat16)
    hiddens = mx.random.normal((1, 5, args.hidden_size)).astype(mx.bfloat16)
    out = loaded(embeds, hiddens)
    mx.eval(out)
    assert out.shape == (1, 5, args.hidden_size), f"出力の形が違う: {out.shape}"
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32)))), "出力に NaN/Inf がある"


def test_quantized_sidecar_without_config():
    """config.json が無い単一ファイルでも fc の形から group_size/bits を割り出す。"""
    args = tiny_args()
    ref = quantized_reference(args, group_size=64, bits=4)
    d = _TMP_ROOT / "q4g64_noconfig"
    f = save_sidecar(ref, d, None)
    (d / "model.safetensors.index.json").unlink()
    assert not (d / "config.json").exists()
    assert_same_params(ref, load_mtp_file(str(f), args), "config.json 無し")


def test_bf16_sidecar_path_unchanged():
    """bf16 サイドカー (train_mtp.py の成果物) は従来どおり「読む → quantize」。"""
    args = tiny_args()
    plain = MTPModule(args)
    mx.eval(plain.parameters())
    d = _TMP_ROOT / "bf16"
    f = save_sidecar(plain, d, None)

    # quantize 無し: 素のまま読める
    assert_same_params(plain, load_mtp_file(str(f), args), "bf16 そのまま")

    # quantize あり: 読んでから量子化した結果と一致する
    expect = MTPModule(args)
    expect.load_weights(list(tree_flatten(plain.parameters())))
    nn.quantize(
        expect,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=lambda _, m: hasattr(m, "to_quantized"),
    )
    mx.eval(expect.parameters())
    loaded = load_mtp_file(str(f), args, quantize={"bits": 4, "group_size": 64})
    assert_same_params(expect, loaded, "bf16 を読んでから量子化")


def test_loaded_sidecar_runs_forward():
    """読み戻した頭が実際に forward を通り、NaN/Inf を出さない。"""
    args = tiny_args()
    ref = quantized_reference(args, group_size=64, bits=4)
    d = save_sidecar(
        ref, _TMP_ROOT / "fwd", {"group_size": 64, "bits": 4, "mode": "affine"}
    ).parent
    mtp = load_mtp_file(str(d), args)
    embeds = mx.random.normal((1, 5, args.hidden_size)).astype(mx.bfloat16)
    hiddens = mx.random.normal((1, 5, args.hidden_size)).astype(mx.bfloat16)
    out = mtp(embeds, hiddens)
    mx.eval(out)
    assert out.shape == (1, 5, args.hidden_size), f"出力の形が違う: {out.shape}"
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32)))), "出力に NaN/Inf がある"


def main() -> int:
    tests = [
        test_quantized_sidecar_roundtrip,
        test_sidecar_quantization_wins_over_argument,
        test_quantized_moe_sidecar_roundtrip_and_forward,
        test_quantized_sidecar_without_config,
        test_bf16_sidecar_path_unchanged,
        test_loaded_sidecar_runs_forward,
    ]
    failed = 0
    try:
        for t in tests:
            try:
                t()
                print(f"ok    {t.__name__}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
