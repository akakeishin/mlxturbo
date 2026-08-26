"""fastmlx.convert の dry-run 経路を検証するテスト兼ベンチ。

プロジェクトの依存に pytest が無い（pyproject.toml は変更しない方針）ため、
plain assert + unittest.SkipTest だけで書いてある。pytest が入っている環境なら
`pytest bench/test_convert.py` で拾える（pytest は unittest.SkipTest を
スキップとして扱う）。入っていなければ

    uv run python bench/test_convert.py

で直接実行できる。どちらの経路でも、ローカルに Qwen/Qwen3.8-27B の
スナップショットが無い環境では全体をスキップする（フル 56GB のダウンロードは
発生しない — fastmlx.mtp.find_snapshot はローカルキャッシュしか見ない）。

検証内容:
  1. --dry-run で本体の先頭 DRY_RUN_LAYERS 層 + mtp.* だけを変換する
     (56GB 全体は読まない。fastmlx.convert.truncate_layers が mlx の遅延評価を
     利用して未使用層のシャード読み込みを避けている)
  2. 出力ディレクトリが mlx_lm.load() でそのまま読め、forward が NaN/Inf なしで通る
     (mtp.* は qwen3_5.py の sanitize が捨てるので本体ロードは壊れない)
  3. mtp.* が fastmlx.convert.load_quantized_mtp で量子化済みのまま読み戻せる
  4. 量子化した mtp.fc / layer0.mlp.down_proj を無量子化の参照重みと比較し、
     逆量子化誤差 (mean abs / relative L1 / max abs) を測る
  5. group_size=64 と 128 を比較し、128 の方が出力バイト数が小さいことを確認する
     (README の "scales/biases が量子化バイトの 11.1%" を裏付ける実測)
  6. 4 層分の実測サイズから 64 層フルモデルの出力サイズを外挿する
"""

import functools
import json
import shutil
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.qwen3_5 import TextModelArgs

from fastmlx._mlx_compat import QWEN35_SHIFTED_NORM_SUFFIXES
from fastmlx.convert import (
    convert,
    load_quantized_mtp,
    normalize_quantization,
    validate_dry_run_layers,
)
from fastmlx.mtp import find_snapshot, load_mtp
import fastmlx.cli as cli_module

HF_PATH = "Qwen/Qwen3.8-27B"
# full_attention_interval (Qwen3.8-27B では 4) 以上が必須。qwen3_5 の forward は
# fa_idx = full_attention_interval - 1 の位置にフル注意層があることを前提にして
# いるため、それ未満だと (mtp.* とは無関係に) forward がインデックス範囲外で落ちる。
DRY_RUN_LAYERS = 4
FULL_NUM_LAYERS = 64  # Qwen3.8-27B の実層数。フル変換サイズ推定に使う。

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="fastmlx_convert_test_"))


def _cleanup_tmp_root() -> None:
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


def _snapshot_or_skip() -> str:
    try:
        return find_snapshot(HF_PATH)
    except FileNotFoundError as e:
        raise unittest.SkipTest(
            f"{HF_PATH} のローカルスナップショットが無いためスキップ: {e}"
        ) from e


def _dequant_error(quantized_module: nn.Module, ref_weight: mx.array) -> dict:
    dq = mx.dequantize(
        quantized_module.weight,
        quantized_module.scales,
        quantized_module.biases,
        quantized_module.group_size,
        quantized_module.bits,
        quantized_module.mode,
    ).astype(mx.float32)
    ref = ref_weight.astype(mx.float32)
    diff = mx.abs(dq - ref)
    ref_abs_sum = mx.abs(ref).sum().item()
    return {
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "rel_l1": diff.sum().item() / ref_abs_sum if ref_abs_sum else float("nan"),
    }


def _tensor_sizes(out_dir: Path) -> dict:
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    sizes = {}
    for shard_name in shard_names:
        for k, v in mx.load(str(out_dir / shard_name)).items():
            sizes[k] = v.nbytes
    return sizes


def _partition_sizes(sizes: dict) -> dict:
    layer_bytes = sum(v for k, v in sizes.items() if ".model.layers." in k)
    mtp_bytes = sum(v for k, v in sizes.items() if k.startswith("mtp."))
    other_bytes = sum(sizes.values()) - layer_bytes - mtp_bytes
    return {
        "layer_bytes": layer_bytes,
        "mtp_bytes": mtp_bytes,
        "other_bytes": other_bytes,
    }


@functools.lru_cache(maxsize=None)
def _dry_run(group_size: int) -> dict:
    """group_size ごとに 1 回だけ --dry-run を実行し、検証結果と実測値を返す。"""
    snap = _snapshot_or_skip()
    out_dir = _TMP_ROOT / f"dryrun_g{group_size}"

    t0 = time.perf_counter()
    result = convert(
        hf_path=snap,
        out=str(out_dir),
        group_size=group_size,
        bits=4,
        dry_run=True,
        dry_run_layers=DRY_RUN_LAYERS,
    )
    elapsed = time.perf_counter() - t0
    config = result["config"]
    source_loaded_model = result["model"]

    assert config["fastmlx_mtp"] is True
    assert config["mtp_quantization"] == {
        "group_size": group_size,
        "bits": 4,
        "mode": "affine",
    }
    assert config["quantization"]["group_size"] == group_size
    assert config["quantization"]["bits"] == 4
    assert config["text_config"]["num_hidden_layers"] == DRY_RUN_LAYERS
    assert len(config["text_config"]["layer_types"]) == DRY_RUN_LAYERS

    # 1) mlx_lm.load がそのまま読めて forward できる（mtp.* は本体 sanitize が捨てる）
    output_reloaded_model, _tokenizer = mlx_lm_load(str(out_dir))
    assert (
        len(output_reloaded_model.language_model.model.layers) == DRY_RUN_LAYERS
    )
    toks = mx.array([[1, 2, 3, 4, 5, 6]])
    source_logits = source_loaded_model(toks)
    reloaded_logits = output_reloaded_model(toks)
    mx.eval(source_logits, reloaded_logits)
    assert source_logits.shape == (
        1,
        6,
        config["text_config"]["vocab_size"],
    )
    assert not bool(mx.any(mx.isnan(reloaded_logits)).item())
    assert not bool(mx.any(mx.isinf(reloaded_logits)).item())
    assert bool(mx.array_equal(source_logits, reloaded_logits)), (
        "source-loaded and output-reloaded logits must match exactly; "
        f"max_abs={mx.abs(source_logits.astype(mx.float32) - reloaded_logits.astype(mx.float32)).max().item()}"
    )

    source_norms = {
        k: v
        for k, v in tree_flatten(source_loaded_model.parameters())
        if v.ndim == 1
        and any(k.endswith(s) for s in QWEN35_SHIFTED_NORM_SUFFIXES)
    }
    reloaded_norms = {
        k: v
        for k, v in tree_flatten(output_reloaded_model.parameters())
        if k in source_norms
    }
    assert source_norms, "the qwen3_5 RMSNorm contract matched no source weights"
    assert source_norms.keys() == reloaded_norms.keys()
    mx.eval(*source_norms.values(), *reloaded_norms.values())
    assert all(
        bool(mx.array_equal(value, reloaded_norms[name]))
        for name, value in source_norms.items()
    ), "source-loaded and output-reloaded RMSNorm weights differ"

    # 2) mtp.* が量子化済みとしてそのまま読める（fastmlx.mtp.load_mtp 相当の量子化版）
    text_args = TextModelArgs.from_dict(config["text_config"])
    mtp_q = load_quantized_mtp(out_dir, text_args)
    mx.eval(mtp_q.parameters())
    assert mtp_q.fc.group_size == group_size
    assert mtp_q.fc.bits == 4

    # 3) 逆量子化誤差: mtp.fc と layer0.mlp.down_proj を無量子化の参照と比較
    mtp_ref = load_mtp(snap, text_args, quantize=None)
    mx.eval(mtp_ref.parameters())
    mtp_fc_err = _dequant_error(mtp_q.fc, mtp_ref.fc.weight)

    raw_model, _, _ = mlx_lm_load(snap, return_config=True, lazy=True)
    ref_down_proj = raw_model.language_model.model.layers[0].mlp.down_proj.weight
    mx.eval(ref_down_proj)
    quant_down_proj = output_reloaded_model.language_model.model.layers[0].mlp.down_proj
    down_proj_err = _dequant_error(quant_down_proj, ref_down_proj)

    sizes = _tensor_sizes(out_dir)
    partition = _partition_sizes(sizes)
    dir_size_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.safetensors"))

    per_layer_bytes = partition["layer_bytes"] / DRY_RUN_LAYERS
    est_full_bytes = (
        per_layer_bytes * FULL_NUM_LAYERS
        + partition["mtp_bytes"]
        + partition["other_bytes"]
    )

    return {
        "group_size": group_size,
        "elapsed_s": elapsed,
        "dir_size_bytes": dir_size_bytes,
        "partition": partition,
        "per_layer_bytes": per_layer_bytes,
        "est_full_bytes": est_full_bytes,
        "mtp_fc_err": mtp_fc_err,
        "down_proj_err": down_proj_err,
    }


def test_dry_run_group64():
    r = _dry_run(64)
    assert r["down_proj_err"]["rel_l1"] < 0.25, r
    assert r["mtp_fc_err"]["rel_l1"] < 0.25, r


def test_dry_run_group128():
    r = _dry_run(128)
    assert r["down_proj_err"]["rel_l1"] < 0.25, r
    assert r["mtp_fc_err"]["rel_l1"] < 0.25, r


def test_group128_smaller_than_group64():
    r64 = _dry_run(64)
    r128 = _dry_run(128)
    assert r128["dir_size_bytes"] < r64["dir_size_bytes"], (
        "group_size=128 は group_size=64 より scales/biases が少ない分、"
        f"出力が小さいはず: g64={r64['dir_size_bytes']} g128={r128['dir_size_bytes']}"
    )


def test_dry_run_layer_bounds():
    class Inner:
        layers = [object()] * FULL_NUM_LAYERS

    class LanguageModel:
        model = Inner()

    class Model:
        language_model = LanguageModel()

    config = {"text_config": {"full_attention_interval": 4}}
    validate_dry_run_layers(Model(), config, 4)
    validate_dry_run_layers(Model(), config, FULL_NUM_LAYERS)
    for invalid in (3, FULL_NUM_LAYERS + 1):
        try:
            validate_dry_run_layers(Model(), config, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"dry_run_layers={invalid} should be rejected")


def test_quantization_preflight_and_effective_defaults():
    assert normalize_quantization(0, 0, 0) == (64, 4, 4)
    assert normalize_quantization(128, 4, None) == (128, 4, 4)
    for args in ((16, 4, None), (64, 7, None), (64, 4, 7)):
        try:
            normalize_quantization(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported quantization should fail: {args}")


def test_cli_prefers_bundled_mtp_artifact():
    bundled = object()
    with (
        mock.patch.object(
            cli_module, "resolve_local_model_path", return_value=Path("/artifact")
        ) as resolve,
        mock.patch.object(
            cli_module, "load_quantized_mtp", return_value=bundled
        ) as load_bundled,
        mock.patch.object(cli_module, "find_snapshot") as find_original,
    ):
        actual = cli_module.load_cli_mtp(
            "artifact-repo",
            {"fastmlx_mtp": True},
            object(),
            "raw-repo",
            4,
        )
    assert actual is bundled
    resolve.assert_called_once_with("artifact-repo")
    load_bundled.assert_called_once()
    find_original.assert_not_called()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def _print_report(results: dict) -> None:
    print("\n=== fastmlx.convert dry-run report ===")
    for gs in sorted(results):
        r = results[gs]
        print(f"\n-- group_size={gs} --")
        print(f"  dry-run elapsed:              {r['elapsed_s']:.2f}s")
        print(
            f"  dry-run artifact size:        {_fmt_bytes(r['dir_size_bytes'])}"
            f" ({DRY_RUN_LAYERS} layers + mtp + embed/lm_head/norm)"
        )
        print(f"  bytes/layer (quantized):      {_fmt_bytes(r['per_layer_bytes'])}")
        print(
            f"  estimated FULL ({FULL_NUM_LAYERS} layers) artifact size: "
            f"{_fmt_bytes(r['est_full_bytes'])}"
        )
        e = r["mtp_fc_err"]
        print(
            f"  mtp.fc dequant error:         mean_abs={e['mean_abs']:.5f} "
            f"rel_l1={e['rel_l1']:.4f} max_abs={e['max_abs']:.4f}"
        )
        e = r["down_proj_err"]
        print(
            f"  layer0.down_proj dequant err: mean_abs={e['mean_abs']:.5f} "
            f"rel_l1={e['rel_l1']:.4f} max_abs={e['max_abs']:.4f}"
        )

    if 64 in results and 128 in results:
        ratio = results[128]["est_full_bytes"] / results[64]["est_full_bytes"]
        print(f"\nestimated full-model size ratio group128/group64: {ratio:.3f}")


def main() -> None:
    """pytest 無しでの直接実行エントリポイント。

        uv run python bench/test_convert.py
    """
    quick_tests = [
        ("test_dry_run_layer_bounds", test_dry_run_layer_bounds),
        (
            "test_quantization_preflight_and_effective_defaults",
            test_quantization_preflight_and_effective_defaults,
        ),
        (
            "test_cli_prefers_bundled_mtp_artifact",
            test_cli_prefers_bundled_mtp_artifact,
        ),
    ]
    snapshot_tests = [
        ("test_dry_run_group64", test_dry_run_group64),
        ("test_dry_run_group128", test_dry_run_group128),
        ("test_group128_smaller_than_group64", test_group128_smaller_than_group64),
    ]
    failures = []
    snapshot_skips = 0
    try:
        for name, fn in quick_tests + snapshot_tests:
            print(f"[test_convert] running {name} ...")
            try:
                fn()
            except unittest.SkipTest as e:
                print(f"[test_convert] SKIP {name}: {e}")
                if (name, fn) in snapshot_tests:
                    snapshot_skips += 1
            except AssertionError as e:
                print(f"[test_convert] FAIL {name}: {e}")
                failures.append(name)
            else:
                print(f"[test_convert] PASS {name}")

        if snapshot_skips == 0:
            results = {gs: _dry_run(gs) for gs in (64, 128)}
            _print_report(results)

        if failures:
            raise SystemExit(f"{len(failures)} test(s) failed: {failures}")
        print(
            f"\n[test_convert] quick tests passed; "
            f"snapshot tests skipped={snapshot_skips}"
        )
    finally:
        _cleanup_tmp_root()


if __name__ == "__main__":
    main()
