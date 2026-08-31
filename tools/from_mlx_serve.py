"""mlx-serve ネイティブのパックを、うちのエンジンが読める形に戻す。

`to_mlx_serve.py` の逆。**重みの数値は 1 ビットも変えない** (RMSNorm の +1 を
引くところだけは値を触るが、それは向こうが足したものを戻しているだけ)。

戻すもの:

  1. 名前の頭。`language_model.model.*` -> `model.*`、
     `language_model.lm_head.*` -> `lm_head.*`
  2. MTP。trunk に同居している `language_model.mtp.*` を `mtp.safetensors` へ
     抜き出す (`mtp.*` の名前で)
  3. RMSNorm の +1。向こうは畳み込んだ重みを持つが、うちのアーキは
     `x * (1 + w)` を自分で計算するので **1 を引く**
  4. config。per-tensor の量子化エントリを**パックの実体から作り直す**。
     mlx_lm はここを見てどのモジュールを量子化するか決めるので、実体と
     ずれると読み込みが形の不一致で落ちる

エキスパートは向こうも `switch_mlp.{gate,up,down}_proj` の 3 本立てなので
そのまま。融合 (`experts.gate_up_proj`) に戻す必要はない。

n-gram 表は `ngram` サブコマンドでプレーナからインタリーブへ組み直す。

    uv run python tools/from_mlx_serve.py shards \
        --src ~/models/ddalcu-flashnext-serve-4bit --out ~/models/ddalcu-mlxlm
    uv run python tools/from_mlx_serve.py ngram \
        --src ~/models/ddalcu-flashnext-serve-4bit --out ~/models/ddalcu-ngram
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COPY_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "generation_config.json", "vocab.json", "merges.txt", "LICENSE",
)

# to_mlx_serve.py と同じ一覧。向こうが +1 を畳み込んでいるので、ここで引く。
NORM_FOLD_SUFFIXES = (
    "hc_norm.weight", "q_norm.weight", "k_norm.weight",
    "q_layernorm.weight", "k_layernorm.weight",
    "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight",
    "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
)


def unrename(key: str) -> str:
    if key.startswith("language_model."):
        return key[len("language_model."):]
    return key          # model.visual.* はそのまま (落とす側で判定する)


def _read_header(fh) -> dict:
    n = struct.unpack("<Q", fh.read(8))[0]
    return json.loads(fh.read(n))


def _bits_of(w_shape, s_shape, group_size: int) -> int:
    """パック済みの形からビット数を割り出す。"""
    return round(w_shape[-1] * 32 / (s_shape[-1] * group_size))


def cmd_shards(args) -> None:
    import mlx.core as mx

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    gs = args.group_size

    shards = sorted(p for p in src.glob("model-*.safetensors")
                    if p.name != "model-vision.safetensors")
    if not shards:
        sys.exit(f"シャードが無い: {src}")

    mtp: dict = {}
    quant_cfg: dict = {"group_size": gs, "bits": args.bits, "mode": "affine"}
    n_out = 0
    n_fold = 0

    for sh in shards:
        t0 = time.time()
        w = mx.load(str(sh))
        keep, drop_vision = {}, 0
        for k, v in w.items():
            if k.startswith("model.visual."):
                drop_vision += 1
                continue
            nk = unrename(k)
            if nk.endswith(NORM_FOLD_SUFFIXES):
                v = (v.astype(mx.float32) - 1.0).astype(v.dtype)
                n_fold += 1
            (mtp if nk.startswith("mtp.") else keep)[nk] = v
        if not keep:
            continue
        n_out += 1
        name = f"model-{n_out:05d}.safetensors"
        mx.eval(list(keep.values()))
        mx.save_safetensors(str(out / name), keep)
        print(f"  {sh.name} -> {name}  {len(keep)} テンソル"
              f"{f' (vision {drop_vision} 本を落とした)' if drop_vision else ''}"
              f"  {time.time() - t0:.1f}s", flush=True)
        del w, keep
        mx.clear_cache()

    if mtp:
        # 向こうは MTP を量子化済みで trunk に持つが、うちのエンジンは bf16 を
        # 受け取って --mtp-bits で自分で量子化する。復元して渡す。
        deq, n_deq = {}, 0
        stems = {k.rsplit(".", 1)[0] for k in mtp}
        for stem in sorted(stems):
            if f"{stem}.scales" in mtp:
                w, sc, bi = (mtp[f"{stem}.{x}"] for x in ("weight", "scales", "biases"))
                bits = _bits_of(w.shape, sc.shape, gs)
                deq[f"{stem}.weight"] = mx.dequantize(
                    w, sc, bi, group_size=gs, bits=bits).astype(mx.bfloat16)
                n_deq += 1
            elif f"{stem}.weight" in mtp:
                deq[f"{stem}.weight"] = mtp[f"{stem}.weight"]
            else:
                deq[stem] = mtp[stem]
        mx.eval(list(deq.values()))
        mx.save_safetensors(str(out / "mtp.safetensors"), deq)
        print(f"  mtp.safetensors  {len(deq)} テンソル ({n_deq} 本を bf16 に復元)")

    # index と、実体から起こした per-tensor の量子化エントリ
    wm, tot = {}, 0
    for p in sorted(out.glob("model-*.safetensors")):
        with open(p, "rb") as fh:
            hdr = _read_header(fh)
        shapes = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            wm[k] = p.name
            tot += v["data_offsets"][1] - v["data_offsets"][0]
            shapes[k] = v["shape"]
        for k in shapes:
            if not k.endswith(".scales"):
                continue
            stem = k[: -len(".scales")]
            quant_cfg[stem] = {
                "bits": _bits_of(shapes[stem + ".weight"], shapes[k], gs),
                "group_size": gs,
            }
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": tot}, "weight_map": wm}))

    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("vision_config", None)
    cfg.pop("ngram_table", None)
    cfg["language_model_only"] = True
    cfg["quantization"] = quant_cfg
    cfg["quantization_config"] = dict(quant_cfg)
    tc = cfg.get("text_config", {})
    if "eos_token_id" in tc:
        cfg["eos_token_id"] = tc["eos_token_id"]
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)

    n_q = sum(1 for v in quant_cfg.values() if isinstance(v, dict))
    print(f"\n{len(wm)} テンソル / {tot / 2**30:.2f} GiB -> {out}")
    print(f"RMSNorm から 1 を引いた: {n_fold} 本 / 量子化エントリ {n_q} 件")


def cmd_ngram(args) -> None:
    """プレーナの `ngram_table.bin` をインタリーブのサイドカーに組み直す。"""
    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    table = src / "ngram_table.bin"
    if not table.exists():
        sys.exit(f"{table} が無い")

    with open(table, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    meta = hdr.get("__metadata__", {})
    rows, cols_w = hdr["weight"]["shape"]
    cols_s = hdr["scales"]["shape"][1]
    wb, sb = cols_w * 4, cols_s * 2
    rec = wb + 2 * sb

    man = {
        "rows": rows, "rows_per_shard": rows, "n_shards": 1,
        "dim": cols_s * int(meta.get("group_size", 32)),
        "bits": int(meta.get("bits", 4)),
        "group_size": int(meta.get("group_size", 32)),
        "packed_per_row": cols_w, "groups_per_row": cols_s,
        "layout": "interleaved", "record_bytes": rec,
        "weight_bytes": wb, "scale_bytes": sb,
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))

    chunk = 1 << 20
    offs = {k: base + hdr[k]["data_offsets"][0] for k in ("weight", "scales", "biases")}
    t0 = time.time()
    with open(table, "rb") as fw, open(table, "rb") as fs, open(table, "rb") as fb, \
            open(out / "rows.bin", "wb") as o:
        done = 0
        while done < rows:
            m = min(chunk, rows - done)
            fw.seek(offs["weight"] + done * wb)
            fs.seek(offs["scales"] + done * sb)
            fb.seek(offs["biases"] + done * sb)
            W = np.frombuffer(fw.read(m * wb), dtype=np.uint8).reshape(m, wb)
            S = np.frombuffer(fs.read(m * sb), dtype=np.uint8).reshape(m, sb)
            B = np.frombuffer(fb.read(m * sb), dtype=np.uint8).reshape(m, sb)
            o.write(np.concatenate([W, S, B], axis=1).tobytes())
            done += m
            if done % (64 * chunk) == 0:
                print(f"  {done}/{rows} 行 ({time.time() - t0:.0f}s)", flush=True)
    p = out / "rows.bin"
    print(f"-> {p}  {p.stat().st_size / 2**30:.2f} GiB  {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shards", help="シャード・index・config を戻す")
    p.add_argument("--src", required=True, help="mlx-serve ネイティブのパック")
    p.add_argument("--out", required=True)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p = sub.add_parser("ngram", help="n-gram 表をインタリーブに組み直す")
    p.add_argument("--src", required=True, help="mlx-serve ネイティブのパック")
    p.add_argument("--out", required=True, help="サイドカーの出力先 dir")
    args = ap.parse_args()
    (cmd_shards if args.cmd == "shards" else cmd_ngram)(args)


if __name__ == "__main__":
    main()
