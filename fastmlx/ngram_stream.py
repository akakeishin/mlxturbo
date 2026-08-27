"""n-gram ハッシュ表を RAM から追い出し、必要な行だけディスクから引く。

Flash-Next の n-gram 表は 51.2B params あって、bf16 なら 102GB、2bit に潰しても
19.2GB を占める。モデル全体の 1/4 を食う最大の塊なのに、**1 トークンで触るのは
16 行だけ** (ngram_heads = (ngram_size-1) * heads_per_ngram = 16)。行列積ではなく
引きなので、常駐させる意味がほとんど無い。

実測 (docs/STATUS.md Phase Q):
  - n-gram を 2bit (相対誤差 0.36) まで潰しても top1 一致は変わらなかった
  - 誤差源として突出しているのは experts の方
つまり n-gram に払っていた容量は experts に回した方が良い。ディスクに置けば
RAM はゼロになり、しかも精度は上げられる (4bit で相対誤差 0.081)。

サイドカーの形式は memmap しやすい生バイナリにする。safetensors だと 25GB の
テンソルを 1 本抱えることになり、行単位の取り出しに余計な層が挟まる。

    <dir>/manifest.json   行数・次元・bits・group_size
    <dir>/rows.bin        (rows, record_bytes) uint8

**1 行を 1 レコードに連続配置する。**weight / scales / biases を別ファイルに
すると、行を 1 つ引くのに 3 箇所へ触ることになり、毎トークン 16 行 x 3 =
48 回のランダムアクセスが起きる。実測でこれが 1 トークンあたり 23.9ms
(生成 21.3 -> 14.1 tok/s) を食っていた。連続配置ならフォルトは 1/3 になり、
4KB ページに約 40 行が載るので局所性も出る。

128 枚のシャードは論理的に行ブロックを並べたものなので、連結して 1 枚の平坦な
表として持つ。こうすると引くときのシャード計算が消える。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_SHARD_RE = "ngram_embedding.shard_"


# --------------------------------------------------------------- サイドカー作成


def _iter_shard_tensors(src: Path):
    """bf16 アーカイブから n-gram のシャードを shard_0, shard_1, ... の順に返す。"""

    found: dict[int, tuple[Path, int, int, list[int]]] = {}
    for shard in sorted(src.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, info in header.items():
            if _SHARD_RE not in name:
                continue
            i = int(name.split(_SHARD_RE)[1].split(".")[0])
            a, b = info["data_offsets"]
            found[i] = (shard, base + a, base + b, info["shape"])
    for i in sorted(found):
        yield i, found[i]


def build_sidecar(src: Path, out: Path, bits: int = 4, group_size: int = 32) -> dict:
    """bf16 アーカイブから量子化済みサイドカーを作る。

    行ブロック単位で読み書きするのでメモリは数百 MB しか使わない。
    """

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    out.mkdir(parents=True, exist_ok=True)
    shards = list(_iter_shard_tensors(src))
    if not shards:
        raise SystemExit("n-gram のシャードが見つからない")

    dim = shards[0][1][3][-1]
    if dim % group_size:
        raise SystemExit(f"dim {dim} が group_size {group_size} で割り切れない")
    total_rows = sum(s[1][3][0] for s in shards)
    n_groups = dim // group_size
    n_packed = dim * bits // 32

    print(f"{len(shards)} シャード / {total_rows:,} 行 x {dim} 次元 -> {bits}bit")
    wb, sb = n_packed * 4, n_groups * 2
    rec_bytes = wb + sb * 2
    frows = open(out / "rows.bin", "wb")
    BLOCK = 1 << 16
    for i, (path, a, b, shape) in shards:
        rows, _ = shape
        mm = np.memmap(path, dtype=np.uint16, mode="r", offset=a, shape=(rows, dim))
        for lo in range(0, rows, BLOCK):
            blk = np.ascontiguousarray(mm[lo : lo + BLOCK])
            w = mx.array(blk).view(mx.bfloat16)
            q, sc, bi = mx.quantize(w, group_size=group_size, bits=bits)
            mx.eval(q, sc, bi)
            n = q.shape[0]
            buf = np.empty((n, rec_bytes), dtype=np.uint8)
            buf[:, :wb] = np.array(q, copy=False).view(np.uint8).reshape(n, wb)
            buf[:, wb : wb + sb] = (
                np.array(sc.view(mx.uint16), copy=False).view(np.uint8).reshape(n, sb)
            )
            buf[:, wb + sb :] = (
                np.array(bi.view(mx.uint16), copy=False).view(np.uint8).reshape(n, sb)
            )
            frows.write(buf.tobytes())
        del mm
        if (i + 1) % 16 == 0:
            print(f"  shard {i + 1}/{len(shards)}", flush=True)
    frows.close()

    manifest = {
        "rows": total_rows,
        "rows_per_shard": shards[0][1][3][0],
        "n_shards": len(shards),
        "dim": dim,
        "bits": bits,
        "group_size": group_size,
        "packed_per_row": n_packed,
        "groups_per_row": n_groups,
        "layout": "interleaved",
        "record_bytes": rec_bytes,
        "weight_bytes": wb,
        "scale_bytes": sb,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = (out / "rows.bin").stat().st_size
    print(f"wrote {out}  {size / 1e9:.1f} GB")
    return manifest


# --------------------------------------------------------------- 実行時の引き


class StreamNGram:
    """サイドカーから行だけ引く。`_ShardedEmbedding` と同じ呼び出し規約。

    nn.Module ではない。パラメータを 1 つも持たないので、モデルの
    `parameters()` に現れてはいけない (現れると保存や dtype 変換の対象になる)。
    """

    def __init__(self, sidecar: Path):
        self.dir = Path(sidecar)
        m = json.loads((self.dir / "manifest.json").read_text())
        self.rows = m["rows"]
        self.dim = m["dim"]
        self.bits = m["bits"]
        self.group_size = m["group_size"]
        self.npack, self.ngrp = m["packed_per_row"], m["groups_per_row"]
        self.rec = m.get("record_bytes", self.npack * 4 + self.ngrp * 4)
        self.wb = m.get("weight_bytes", self.npack * 4)
        self.sb = m.get("scale_bytes", self.ngrp * 2)
        self.mm = np.memmap(
            self.dir / "rows.bin", dtype=np.uint8, mode="r", shape=(self.rows, self.rec)
        )

    def __call__(self, gid):
        import mlx.core as mx

        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64)
        # numpy の fancy index は C 側で一度に集める。行ごとの Python ループに
        # すると生成時に効いてくるので、ここは必ず 1 回のギャザーで済ませる。
        # 1 行が連続レコードなので、触るページも 1 行あたり 1 枚で済む
        rec = self.mm[flat]
        n = rec.shape[0]
        w = mx.array(rec[:, : self.wb].copy().view(np.uint32).reshape(n, self.npack))
        s = mx.array(
            rec[:, self.wb : self.wb + self.sb].copy().view(np.uint16).reshape(n, self.ngrp)
        ).view(mx.bfloat16)
        b = mx.array(
            rec[:, self.wb + self.sb :].copy().view(np.uint16).reshape(n, self.ngrp)
        ).view(mx.bfloat16)
        out = mx.dequantize(w, s, b, group_size=self.group_size, bits=self.bits)
        return out.reshape(*gid.shape, self.dim)


def install(model, sidecar: str | Path) -> None:
    """読み込み済みモデルの n-gram 表をサイドカー参照に差し替える。

    量子化変換のときに `_ShardedEmbedding` をパラメータ無しの空実装にしてある
    ので、チェックポイント側に n-gram のテンソルは入っていない。ここで実体を
    与える。
    """

    stream = StreamNGram(Path(sidecar))
    n = 0
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = ple.ple_embedding
        if emb.ngram_embedding.dim != stream.dim:
            raise ValueError(
                f"次元が合わない: モデル {emb.ngram_embedding.dim} / "
                f"サイドカー {stream.dim}"
            )
        emb.ngram_embedding = stream
        n += 1
    if n == 0:
        raise ValueError("PLE 層が見つからない")
    print(f"n-gram をサイドカー参照に差し替えた ({n} 層, {stream.bits}bit, RAM 0)")


__all__ = ["StreamNGram", "build_sidecar", "install"]
