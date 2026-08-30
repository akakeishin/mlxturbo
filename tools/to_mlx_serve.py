"""うちのパックを mlx-serve がそのまま読める並びに直す。

焼き直しではない。**重みの中身は 1 ビットも触らない。**違うのは 3 つだけ:

  1. テンソル名の頭。うちは `model.*` / `lm_head.*` / (別ファイルの) `mtp.*`、
     向こうは全部 `language_model.` が付く。vision だけは `model.visual.*` で
     prefix が付かない
  2. MTP。うちは `mtp.safetensors` に分けている。向こうは trunk のシャードに
     `language_model.mtp.*` として同居させる
  3. n-gram 表。うちはレコードごとに weight/scales/biases を並べた
     インタリーブ、向こうは 3 つのブロックに分けたプレーナを safetensors
     ヘッダ付きの 1 ファイル (`ngram_table.bin`) に入れる

シャードは safetensors のヘッダだけ書き直して本体はバイト列のままコピーする。
data_offsets は本体先頭からの相対値なので、ヘッダが伸びても中身は動かない。

ビット配分は config に書かない。向こうはテンソルの形からビット数を割り出す
ので、compact な `quantization` (向こうのパックと同じ 3 キー) だけ置く。
混在ビットが本当に読めるかは、これで焼かずに確かめられる。

    uv run python tools/to_mlx_serve.py shards \
        --src ~/models/qwen38fn-mlx-v-l --out ~/models/qwen38fn-v-l-serve
    uv run python tools/to_mlx_serve.py ngram \
        --src ~/models/qwen38fn-ngram-4bit --out ~/models/qwen38fn-v-l-serve
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
)


def rename(key: str) -> str:
    """うちの名前を向こうの名前にする。"""
    if key.startswith(("model.", "lm_head.", "mtp.")):
        return "language_model." + key
    raise ValueError(f"想定していない prefix: {key}")


def _read_header(fh) -> tuple[dict, int]:
    n = struct.unpack("<Q", fh.read(8))[0]
    return json.loads(fh.read(n)), 8 + n


def _write_shard(src: Path, dst: Path) -> dict[str, list[int]]:
    """ヘッダを書き直して本体をそのままコピーする。名前 -> shape を返す。"""
    with open(src, "rb") as fh:
        hdr, _ = _read_header(fh)
        meta = hdr.pop("__metadata__", None)
        new = {rename(k): v for k, v in hdr.items()}
        if meta is not None:
            new["__metadata__"] = meta
        blob = json.dumps(new, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            shutil.copyfileobj(fh, out, 1 << 24)
    return {k: v["data_offsets"] for k, v in new.items() if k != "__metadata__"}


def cmd_shards(args) -> None:
    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("model-0*.safetensors"))
    if (src / "mtp.safetensors").exists():
        shards.append(src / "mtp.safetensors")
    if not shards:
        sys.exit(f"シャードが無い: {src}")

    weight_map: dict[str, str] = {}
    total = 0
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}.safetensors"
        t0 = time.time()
        offsets = _write_shard(sh, out / name)
        for k, (a, b) in offsets.items():
            weight_map[k] = name
            total += b - a
        gb = (out / name).stat().st_size / 2**30
        print(f"  {sh.name} -> {name}  {gb:6.2f} GiB  {time.time() - t0:5.1f}s", flush=True)

    if args.vision:
        vis = Path(args.vision).expanduser()
        print(f"  vision <- {vis}", flush=True)
        shutil.copy2(vis, out / "model-vision.safetensors")
        with open(vis, "rb") as fh:
            vhdr, _ = _read_header(fh)
        for k, v in vhdr.items():
            if k == "__metadata__":
                continue
            weight_map[k] = "model-vision.safetensors"
            total += v["data_offsets"][1] - v["data_offsets"][0]

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )

    cfg = json.loads((src / "config.json").read_text())
    q = cfg["quantization"]
    cfg["quantization"] = {
        "group_size": q["group_size"], "bits": q["bits"], "mode": q["mode"],
    }
    cfg["quantization_config"] = dict(cfg["quantization"])
    cfg["ngram_table"] = {"file": "ngram_table.bin", "bits": 4, "group_size": 32}
    cfg.pop("eos_token_id", None)  # 向こうは text_config 側だけを見る
    if args.vision:
        ref = json.loads(Path(args.vision_config).expanduser().read_text())
        cfg["vision_config"] = ref["vision_config"]
        cfg["language_model_only"] = False
    else:
        cfg.pop("vision_config", None)
        cfg["language_model_only"] = True
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)

    print(f"\n{len(weight_map)} テンソル / {total / 2**30:.2f} GiB -> {out}")
    print("n-gram 表がまだ無い。`ngram` サブコマンドを続けて走らせること")


def cmd_ngram(args) -> None:
    """インタリーブのサイドカーをプレーナ 1 ファイルに組み直す。

    レコード内は weight(80B) / scales(10B) / biases(10B) の順。向こうは
    weight を全行ぶん、次に scales を全行ぶん、と続ける。3 周読むのは、
    一時ファイルを置かずに済ませるため。
    """
    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    man = json.loads((src / "manifest.json").read_text())
    if man.get("layout") != "interleaved":
        sys.exit(f"インタリーブのサイドカーが要る (これは {man.get('layout')})")

    rows, rec = man["rows"], man["record_bytes"]
    wb, sb = man["weight_bytes"], man["scale_bytes"]
    blocks = (("weight", 0, wb, "U32", man["packed_per_row"]),
              ("scales", wb, sb, "BF16", man["groups_per_row"]),
              ("biases", wb + sb, sb, "BF16", man["groups_per_row"]))

    hdr, off = {}, 0
    for name, _, width, dtype, cols in blocks:
        hdr[name] = {"dtype": dtype, "shape": [rows, cols],
                     "data_offsets": [off, off + rows * width]}
        off += rows * width
    hdr = {"__metadata__": {"format": "mlx-serve-ngram",
                            "bits": str(man["bits"]),
                            "group_size": str(man["group_size"])}, **hdr}
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    rows_bin = src / "rows.bin"
    if rows_bin.stat().st_size != rows * rec:
        sys.exit(f"rows.bin の大きさが manifest と合わない: {rows_bin.stat().st_size}")

    chunk = 1 << 20  # 行数。100B/行 なので 100MB ずつ
    dst = out / "ngram_table.bin"
    with open(dst, "wb") as o:
        o.write(struct.pack("<Q", len(blob)))
        o.write(blob)
        for name, start, width, _, _ in blocks:
            t0, done = time.time(), 0
            with open(rows_bin, "rb") as f:
                while done < rows:
                    n = min(chunk, rows - done)
                    buf = np.frombuffer(f.read(n * rec), dtype=np.uint8).reshape(n, rec)
                    o.write(buf[:, start:start + width].tobytes())
                    done += n
            print(f"  {name}: {rows} 行 / {rows * width / 2**30:.2f} GiB "
                  f"/ {time.time() - t0:.1f}s", flush=True)
    print(f"-> {dst}  {dst.stat().st_size / 2**30:.2f} GiB")


def cmd_mtp(args) -> None:
    """MTP ヘッドを量子化して 1 シャードに書き出す。

    うちは `mtp.safetensors` を bf16 のまま持っていて、読み込み時に
    `--mtp-bits` で量子化する。向こうは量子化済みが checkpoint に入っている
    前提で、scales が無いと MissingWeight で落ちる。

    どのテンソルを量子化するかは参照パックから引く。1 次元の norm や、
    行数の少ない block_inject_weight は向こうも生のままなので、当てずっぽうに
    規則を書くより、対応する名前が量子化されているかを見た方が確か。
    """
    import mlx.core as mx

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    ref = Path(args.reference).expanduser()
    ref_idx = json.loads((ref / "model.safetensors.index.json").read_text())
    quantized = {
        k[: -len(".scales")]
        for k in ref_idx["weight_map"]
        if ".mtp." in k and k.endswith(".scales")
    }

    w = mx.load(str(src))
    # 融合エキスパートをほどく。checkpoint は (E, 2*moe_inter, H) で、
    # 前半が gate・後半が up (mlx_lm の qwen4_exp.sanitize と同じ約束)。
    # 向こうは MLX の SwitchGLU と同じ 3 本立てを期待する。
    for k in [k for k in w if k.endswith("mlp.experts.gate_up_proj")]:
        base = k[: -len("experts.gate_up_proj")] + "switch_mlp."
        gate, up = mx.split(w.pop(k), 2, axis=1)
        w[base + "gate_proj.weight"] = gate
        w[base + "up_proj.weight"] = up
    for k in [k for k in w if k.endswith("mlp.experts.down_proj")]:
        base = k[: -len("experts.down_proj")] + "switch_mlp."
        w[base + "down_proj.weight"] = w.pop(k)

    packed: dict[str, mx.array] = {}
    n_q = 0
    for k in sorted(w):
        name = rename(k)
        if name.removesuffix(".weight") in quantized:
            wq, sc, bi = mx.quantize(w[k], group_size=args.group_size, bits=args.bits)
            stem = name.removesuffix(".weight")
            packed[f"{stem}.weight"] = wq
            packed[f"{stem}.scales"] = sc
            packed[f"{stem}.biases"] = bi
            n_q += 1
        else:
            packed[name] = w[k]
    mx.eval(list(packed.values()))
    mx.save_safetensors(str(out), packed)
    print(f"{n_q} テンソルを {args.bits}bit/gs{args.group_size} にした "
          f"({len(packed)} エントリ) -> {out}")


def _ref_quant_stems(ref: Path) -> set[str]:
    idx = json.loads((ref / "model.safetensors.index.json").read_text())
    return {k[: -len(".scales")] for k in idx["weight_map"] if k.endswith(".scales")}


def cmd_align(args) -> None:
    """量子化する範囲を参照パックに合わせ直す。

    どのモジュールを量子化するかはエンジンごとに決め打ちで、config には
    書かれていない。うちは `block_inject_weight` と `shared_expert_gate` を
    量子化していて、向こうは生の bf16 で読む。逆に router の `mlp.gate` は
    向こうが量子化していて、うちは生。**名前は合っているので読み込みは通り、
    出力だけが壊れる。**気づけるのは生成を見たときだけなので、ここで揃える。

    ビット数までは合わせない (混在ビットを向こうが読めるかは別の問題)。

    weight と scales が別シャードに入っていることがあるので、先に対象だけを
    集めてから直す。対象は router と gate 類だけで、量としては小さい。
    """
    import mlx.core as mx

    pack = Path(args.pack).expanduser()
    quant = _ref_quant_stems(Path(args.reference).expanduser())
    # index ではなくシャードそのものを見る。index は途中の工程で古くなる
    wm: dict[str, str] = {}
    for sh in sorted(pack.glob("model-*.safetensors")):
        with open(sh, "rb") as fh:
            hdr, _ = _read_header(fh)
        for k in hdr:
            if k != "__metadata__":
                wm[k] = sh.name

    groups: dict[str, set[str]] = {}
    for k in wm:
        stem, _, suf = k.rpartition(".")
        if suf in ("weight", "scales", "biases"):
            groups.setdefault(stem, set()).add(suf)

    todo = {
        stem: ("quantize" if stem in quant else "dequantize")
        for stem, sufs in groups.items()
        if (stem in quant) != ("scales" in sufs)
    }
    if not todo:
        print("合わせるものは無い")
        return
    print(f"対象 {len(todo)} 群 "
          f"(量子化 {sum(v == 'quantize' for v in todo.values())} / "
          f"復元 {sum(v == 'dequantize' for v in todo.values())})", flush=True)

    shards = sorted(pack.glob("model-0*.safetensors"))
    have: dict[str, mx.array] = {}
    for sh in shards:
        w = mx.load(str(sh))
        for stem in todo:
            for suf in ("weight", "scales", "biases"):
                k = f"{stem}.{suf}"
                if k in w:
                    have[k] = w[k]
        mx.eval(list(have.values()))

    fixed: dict[str, mx.array] = {}
    for stem, how in todo.items():
        if how == "dequantize":
            wq, sc, bi = (have[f"{stem}.{s}"] for s in ("weight", "scales", "biases"))
            fixed[f"{stem}.weight"] = mx.dequantize(
                wq, sc, bi, group_size=args.group_size,
                bits=_bits_of(wq, sc, args.group_size),
            ).astype(mx.bfloat16)
        else:
            wq, sc, bi = mx.quantize(have[f"{stem}.weight"],
                                     group_size=args.group_size, bits=args.bits)
            fixed[f"{stem}.weight"] = wq
            fixed[f"{stem}.scales"] = sc
            fixed[f"{stem}.biases"] = bi
    mx.eval(list(fixed.values()))

    # 直したものは .weight が元々あったシャードにまとめて置く
    home = {stem: wm[f"{stem}.weight"] for stem in todo}
    for sh in shards:
        w = mx.load(str(sh))
        out = {k: v for k, v in w.items()
               if k.rpartition(".")[0] not in todo}
        for stem, where in home.items():
            if where == sh.name:
                for suf in ("weight", "scales", "biases"):
                    if f"{stem}.{suf}" in fixed:
                        out[f"{stem}.{suf}"] = fixed[f"{stem}.{suf}"]
        mx.eval(list(out.values()))
        mx.save_safetensors(str(sh), out)
        print(f"  {sh.name}: {len(out)} テンソル", flush=True)

    wm2 = {}
    tot = 0
    for sh in sorted(pack.glob("model-*.safetensors")):
        with open(sh, "rb") as fh:
            hdr, _ = _read_header(fh)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            wm2[k] = sh.name
            tot += v["data_offsets"][1] - v["data_offsets"][0]
    (pack / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": tot}, "weight_map": wm2}))
    print(f"{len(wm2)} テンソル / {tot / 2**30:.2f} GiB")


def _bits_of(wq, scales, group_size: int) -> int:
    """パック済みの形からビット数を割り出す。"""
    values = scales.shape[-1] * group_size
    return round(wq.shape[-1] * 32 / values)


# このアーキの Qwen4ExpTextRMSNorm は x * (1 + w) を計算する。向こうのパックは
# **+1 を畳み込んだ状態**で重みを持ち、エンジン側は素直に w を掛ける。
# うちは HF の生の重み (ほぼ 0) をそのまま入れていたので、向こうで読むと
# ほぼ 0 倍になり、生成が同じトークンの反復になる。形も値も一致して見えるのに
# 出力だけが壊れるので、突き止めるのに時間がかかった。
#
# 一覧は向こうの tests/convert_qwen38_flash_next.py (MIT) の NORM_FOLD_SUFFIXES。
# linear_attn.norm はゲート付きで素の重みなので触らない。
NORM_FOLD_SUFFIXES = (
    "hc_norm.weight", "q_norm.weight", "k_norm.weight",
    "q_layernorm.weight", "k_layernorm.weight",
    "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight",
    "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
)

FOLD_MARK = ".norm_folded"


def cmd_fold(args) -> None:
    """RMSNorm の重みに +1 を畳み込む。**2 回かけてはいけない。**"""
    import mlx.core as mx

    pack = Path(args.pack).expanduser()
    mark = pack / FOLD_MARK
    if mark.exists() and not args.force:
        sys.exit(f"すでに畳み込み済み ({mark})。やり直すならパックを作り直すこと")

    n = 0
    for sh in sorted(pack.glob("model-*.safetensors")):
        if sh.name == "model-vision.safetensors":
            continue
        w = mx.load(str(sh))
        hit = [k for k in w if k.endswith(NORM_FOLD_SUFFIXES)]
        if not hit:
            continue
        for k in hit:
            w[k] = (w[k].astype(mx.float32) + 1.0).astype(w[k].dtype)
        mx.eval(list(w.values()))
        mx.save_safetensors(str(sh), w)
        n += len(hit)
        print(f"  {sh.name}: {len(hit)} 本", flush=True)
    mark.write_text("norm weights carry the +1\n")
    print(f"{n} 本に +1 を畳み込んだ")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shards", help="シャード・index・config を書き直す")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vision", default=None,
                   help="持ってくる model-vision.safetensors")
    p.add_argument("--vision-config", default=None,
                   help="vision_config を借りる config.json (--vision と対で)")
    p = sub.add_parser("ngram", help="n-gram 表をプレーナに組み直す")
    p.add_argument("--src", required=True, help="インタリーブのサイドカー dir")
    p.add_argument("--out", required=True, help="出力パック dir")
    p = sub.add_parser("align", help="量子化の範囲を参照パックに合わせる")
    p.add_argument("--pack", required=True, help="変換済みのパック dir (その場で書き換える)")
    p.add_argument("--reference", required=True, help="mlx-serve ネイティブのパック")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p = sub.add_parser("fold", help="RMSNorm の重みに +1 を畳み込む")
    p.add_argument("--pack", required=True)
    p.add_argument("--force", action="store_true", help="畳み込み済みの印を無視する")
    p = sub.add_parser("mtp", help="bf16 の MTP ヘッドを量子化して書き出す")
    p.add_argument("--src", required=True, help="mtp.safetensors")
    p.add_argument("--out", required=True, help="出力 safetensors")
    p.add_argument("--reference", required=True,
                   help="どれを量子化するかを引く mlx-serve ネイティブのパック")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    if args.cmd == "shards":
        if bool(args.vision) != bool(args.vision_config):
            sys.exit("--vision と --vision-config は対で渡すこと")
        cmd_shards(args)
    elif args.cmd == "ngram":
        cmd_ngram(args)
    elif args.cmd == "align":
        cmd_align(args)
    elif args.cmd == "fold":
        cmd_fold(args)
    else:
        cmd_mtp(args)


if __name__ == "__main__":
    main()
