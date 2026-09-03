"""~/models/qwen38-27b-mtp/model.safetensors のキーに `mtp.` を前置した写しを作る。

テンソルは 1 バイトも触らない: ヘッダの JSON だけキー名を書き換え、
`data_offsets` はそのまま、データ領域は元ファイルからバイト列を丸ごと写す。
これで「キー名以外は同一」がバイト単位で保証される (検証も下でやる)。
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

SRC = Path("/Users/ht/models/qwen38-27b-mtp/model.safetensors")
DST = Path("/Users/ht/models/qwen38-27b-mtp-prefixed/mtp.safetensors")
PREFIX = "mtp."


def read_header(p: Path):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return n, hdr


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    n_src, hdr_src = read_header(SRC)
    meta = hdr_src.get("__metadata__")
    tensors = {k: v for k, v in hdr_src.items() if k != "__metadata__"}

    new_hdr = {}
    if meta is not None:
        new_hdr["__metadata__"] = meta          # metadata も写す
    for k, v in tensors.items():
        assert not k.startswith(PREFIX), f"既に接頭辞がある: {k}"
        new_hdr[PREFIX + k] = v                  # data_offsets はそのまま

    blob = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8                       # 8 バイト境界にそろえる (空白詰め)
    blob += b" " * pad

    DST.parent.mkdir(parents=True, exist_ok=True)
    data_start = 8 + n_src
    with open(SRC, "rb") as fin, open(DST, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        fin.seek(data_start)
        while True:
            chunk = fin.read(1 << 22)
            if not chunk:
                break
            fout.write(chunk)

    # ---- 検証: キー名以外が同一か -------------------------------------
    n_dst, hdr_dst = read_header(DST)
    t_dst = {k: v for k, v in hdr_dst.items() if k != "__metadata__"}
    ok = True
    if hdr_dst.get("__metadata__") != meta:
        print("NG: metadata が一致しない"); ok = False
    if set(t_dst) != {PREFIX + k for k in tensors}:
        print("NG: キー集合が想定と違う"); ok = False
    for k, v in tensors.items():
        w = t_dst.get(PREFIX + k)
        if w is None or w["dtype"] != v["dtype"] or w["shape"] != v["shape"] \
                or w["data_offsets"] != v["data_offsets"]:
            print(f"NG: {k} の dtype/shape/offsets が違う: {v} vs {w}"); ok = False

    # データ領域のバイト列を丸ごと突き合わせる
    hs, hd = hashlib.sha256(), hashlib.sha256()
    with open(SRC, "rb") as fs, open(DST, "rb") as fd:
        fs.seek(8 + n_src)
        fd.seek(8 + n_dst)
        while True:
            a, b = fs.read(1 << 22), fd.read(1 << 22)
            if not a and not b:
                break
            if a != b:
                print("NG: データ領域のバイト列が違う"); ok = False
                break
            hs.update(a); hd.update(b)
    print(f"データ領域 sha256 (元)   = {hs.hexdigest()}")
    print(f"データ領域 sha256 (写し) = {hd.hexdigest()}")
    print(f"ファイル sha256 (元)   = {sha256(SRC)}  {SRC.stat().st_size} bytes")
    print(f"ファイル sha256 (写し) = {sha256(DST)}  {DST.stat().st_size} bytes")
    print(f"テンソル数 {len(tensors)} -> {len(t_dst)}、ヘッダ長 {n_src} -> {n_dst}")
    print("キー名以外は同一:", "OK" if ok else "NG")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
