"""本番パックの lm_head だけを、真の bf16 から再量子化して差し替える。

本番パック (既定 ``~/models/ddalcu-mlxlm``) は 4-bit affine g64 だが lm_head だけ
8-bit で焼いてある。これを bf16 の元 (``lm_head.weight`` は Linear 層の重みそのもの
で量子化済みではない) から `mx.quantize` で任意の bits/group_size に焼き直し、
新しいパックとして出力する。

やること:
  1. bf16 の元から lm_head.weight だけを読む (該当 shard は lm_head.weight 単独
     テンソルなので、これしか読まれない)。
  2. CPU 上で `mx.quantize` して (weight, scales, biases) を作る。
  3. 出力先に本番パックの全ファイルをハードリンクで並べ、lm_head を含む shard
     だけ新しく書き直す (同じ shard 内の他テンソルはそのまま転記)。
  4. config.json の `quantization` / `quantization_config` 双方にある `lm_head`
     エントリの bits/group_size を書き換える。
  5. model.safetensors.index.json の `metadata.total_size` を差分ぶん更新する。

本番パックは読むだけで一切書き換えない。GPU は使わない
(`mx.set_default_device(mx.cpu)` を最初に呼ぶ)。

Usage:
    .venv/bin/python tools/splice_head.py \\
        --bf16-src "/Volumes/Mobile SSD/models/qwen38fn-bf16" \\
        --prod-pack ~/models/ddalcu-mlxlm \\
        --out ~/models/ddalcu-mlxlm-head4 \\
        --bits 4 --group-size 64
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

# 最初に固定する。量子化は CPU の mx.quantize で十分で、他プロセスが同じ
# GPU (Metal) 上でモデルを読んでいるかもしれない環境で GPU を使わない。
mx.set_default_device(mx.cpu)

from safetensors import safe_open  # noqa: E402


LM_HEAD_TENSORS = ("lm_head.weight", "lm_head.scales", "lm_head.biases")


def link_or_copy(src: Path, dst: Path) -> str:
    """src を dst にハードリンク。失敗したらシンボリックリンク、それも失敗したらコピー。"""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def load_index(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_lm_head_shard(index: dict, label: str) -> str:
    wm = index["weight_map"]
    shards = {wm[k] for k in LM_HEAD_TENSORS if k in wm}
    missing = [k for k in LM_HEAD_TENSORS if k not in wm]
    if missing:
        raise SystemExit(f"{label}: index.json に {missing} が無い")
    if len(shards) != 1:
        raise SystemExit(f"{label}: lm_head テンソルが複数 shard に分散している: {shards}")
    return next(iter(shards))


def update_lm_head_cfg_block(block, bits: int, group_size: int) -> dict | None:
    """quantization / quantization_config の lm_head エントリを in-place で書き換える。

    書き換え前の値を返す (無ければ None)。
    """
    if not isinstance(block, dict) or "lm_head" not in block:
        return None
    old = dict(block["lm_head"])
    new = dict(old)
    new["bits"] = bits
    new["group_size"] = group_size
    block["lm_head"] = new
    return old


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bf16-src",
        default="/Volumes/Mobile SSD/models/qwen38fn-bf16",
        help="真の bf16 の元パック (model.safetensors.index.json を持つディレクトリ)",
    )
    ap.add_argument(
        "--prod-pack",
        default=str(Path.home() / "models" / "ddalcu-mlxlm"),
        help="本番パック。読むだけで一切書き換えない",
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "models" / "ddalcu-mlxlm-head4"),
        help="出力先の新パック",
    )
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    bf16_src = Path(args.bf16_src)
    prod = Path(args.prod_pack).expanduser()
    out = Path(args.out).expanduser()

    if out.resolve() == prod.resolve():
        raise SystemExit("--out が --prod-pack と同じ場所を指している。本番パックは書き換えない")
    if str(out.resolve()).startswith(str(prod.resolve()) + os.sep):
        raise SystemExit("--out が --prod-pack の内側を指している")

    out.mkdir(parents=True, exist_ok=True)

    # --- 1. shard の所在を index.json から調べる -----------------------------
    bf16_index = load_index(bf16_src / "model.safetensors.index.json")
    bf16_wm = bf16_index["weight_map"]
    if "lm_head.weight" not in bf16_wm:
        raise SystemExit("bf16-src の index.json に lm_head.weight が無い")
    bf16_shard_name = bf16_wm["lm_head.weight"]
    bf16_shard_path = bf16_src / bf16_shard_name

    prod_index = load_index(prod / "model.safetensors.index.json")
    prod_lm_head_shard = find_lm_head_shard(prod_index, "prod-pack")
    prod_shard_path = prod / prod_lm_head_shard

    print(f"[shard] bf16 lm_head: {bf16_shard_path}")
    print(f"[shard] prod lm_head: {prod_shard_path}")

    # --- 2. bf16 の lm_head.weight だけを読む ---------------------------------
    # safe_open で shard 内のテンソル一覧とメタ (shape/dtype) をまず確認する。
    # 実データの読み出しは safetensors の numpy バックエンドが bf16 を
    # (ml_dtypes 無しには) 復元できないため mx.load を使うが、この shard は
    # lm_head.weight 単独テンソルしか持たないと safe_open で確認済みなので、
    # mx.load で読んでも「shard 全体」= 「lm_head だけ」であり、他の 130 shard
    # は一切開かない。
    with safe_open(str(bf16_shard_path), framework="numpy") as f:
        keys = list(f.keys())
        if keys != ["lm_head.weight"]:
            raise SystemExit(f"bf16 lm_head shard に想定外のテンソル: {keys}")
        sl = f.get_slice("lm_head.weight")
        bf16_shape = tuple(sl.get_shape())
        bf16_dtype = sl.get_dtype()
    print(f"[bf16] lm_head.weight shape={bf16_shape} dtype={bf16_dtype}")

    lm_head_bf16 = mx.load(str(bf16_shard_path))["lm_head.weight"]
    mx.eval(lm_head_bf16)
    if tuple(lm_head_bf16.shape) != bf16_shape:
        raise SystemExit("mx.load の shape が safe_open のメタと不一致")

    # --- 3. CPU で量子化 -------------------------------------------------------
    w_q, scales, biases = mx.quantize(lm_head_bf16, group_size=args.group_size, bits=args.bits)
    mx.eval(w_q, scales, biases)
    print(
        f"[quantize] weight={tuple(w_q.shape)}/{w_q.dtype} "
        f"scales={tuple(scales.shape)}/{scales.dtype} "
        f"biases={tuple(biases.shape)}/{biases.dtype}"
    )

    # --- 4. 検証: dequantize した値と bf16 の元との最大絶対誤差 -----------------
    # CPU の mx.dequantize は一括で呼ぶと (248320, 2560) 全体に対して内部で
    # 数 GB 級の一時バッファを作る (実測: 1.66GB -> 8GB 超) ため、行方向に
    # チャンクして呼ぶ。誤差の値はチャンクしても一括でも同一 (実測確認済み)。
    max_abs_err = 0.0
    n_rows = lm_head_bf16.shape[0]
    chunk = 8192
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        deq_c = mx.dequantize(
            w_q[start:end],
            scales=scales[start:end],
            biases=biases[start:end],
            group_size=args.group_size,
            bits=args.bits,
        )
        diff_c = mx.abs(deq_c.astype(mx.float32) - lm_head_bf16[start:end].astype(mx.float32))
        mx.eval(diff_c)
        max_abs_err = max(max_abs_err, mx.max(diff_c).item())
        del deq_c, diff_c
    print(f"[verify] max |dequant(w_q) - bf16_original| = {max_abs_err:.6g}")
    del lm_head_bf16

    # --- 5. 出力先に本番パックの全ファイルをハードリンクで並べる -----------------
    link_methods = {}
    for entry in sorted(prod.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in (prod_lm_head_shard, "config.json", "model.safetensors.index.json"):
            continue  # これらは後で新しく書く
        method = link_or_copy(entry, out / entry.name)
        link_methods.setdefault(method, 0)
        link_methods[method] += 1
    print(f"[link] other files: {link_methods}")

    # --- 6. lm_head を含む shard を書き直す (同じ shard の他テンソルは転記) -----
    prod_shard_tensors = mx.load(str(prod_shard_path))
    old_lm_head_nbytes = sum(
        prod_shard_tensors[k].nbytes for k in LM_HEAD_TENSORS if k in prod_shard_tensors
    )
    new_shard_tensors = dict(prod_shard_tensors)
    new_shard_tensors["lm_head.weight"] = w_q
    new_shard_tensors["lm_head.scales"] = scales
    new_shard_tensors["lm_head.biases"] = biases
    new_lm_head_nbytes = w_q.nbytes + scales.nbytes + biases.nbytes

    out_shard_path = out / prod_lm_head_shard
    mx.save_safetensors(str(out_shard_path), new_shard_tensors)
    print(
        f"[write] {out_shard_path} "
        f"({len(new_shard_tensors)} tensors, "
        f"{out_shard_path.stat().st_size / 1e6:.1f} MB, "
        f"prod shard was {prod_shard_path.stat().st_size / 1e6:.1f} MB)"
    )

    # --- 7. config.json の lm_head bits を書き換える ---------------------------
    with open(prod / "config.json") as f:
        cfg = json.load(f)
    cfg_diff = {}
    for key in ("quantization", "quantization_config"):
        old = update_lm_head_cfg_block(cfg.get(key), args.bits, args.group_size)
        if old is not None:
            cfg_diff[key] = {"old": old, "new": cfg[key]["lm_head"]}
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("[config] lm_head diff:")
    for key, diff in cfg_diff.items():
        print(f"  {key}.lm_head: {diff['old']} -> {diff['new']}")
    if not cfg_diff:
        print("  (config.json に lm_head の量子化エントリが見つからなかった)")

    # --- 8. index.json の total_size をテンソルバイト数の差分で更新 ------------
    new_index = json.loads(json.dumps(prod_index))
    meta = new_index.setdefault("metadata", {})
    old_total = meta.get("total_size")
    if isinstance(old_total, int):
        meta["total_size"] = old_total - old_lm_head_nbytes + new_lm_head_nbytes
        print(f"[index] metadata.total_size: {old_total} -> {meta['total_size']}")
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=1, ensure_ascii=False)
        f.write("\n")

    # --- 9. 焼き直した shard を disk から読み直して shape/dtype を再確認 --------
    reloaded = mx.load(str(out_shard_path))
    print("[reload] on-disk shard after write:")
    for k in LM_HEAD_TENSORS:
        v = reloaded[k]
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    print(f"\ndone: {out}")


if __name__ == "__main__":
    main()
