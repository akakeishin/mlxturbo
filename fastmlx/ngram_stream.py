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
import os
import struct
from concurrent.futures import ThreadPoolExecutor
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


def build_sidecar(
    src: Path, out: Path, bits: int = 4, group_size: int = 32, layout: str = "interleaved"
) -> dict:
    """bf16 アーカイブから量子化済みサイドカーを作る。

    行ブロック単位で読み書きするのでメモリは数百 MB しか使わない。

    layout="interleaved" は 1 行を 1 レコードに連続配置する。ディスクに置いた
    まま引く用で、触るページが 1 行あたり 1 枚で済む。
    layout="separate" は weight/scales/biases を別ファイルにする。RAM に載せて
    `mx.take` で引く用で、こちらは 3 本の配列がそのまま要る。
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
    sep = layout == "separate"
    if sep:
        fw = open(out / "weight.bin", "wb")
        fs = open(out / "scales.bin", "wb")
        fb = open(out / "biases.bin", "wb")
    else:
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
            if sep:
                fw.write(np.array(q, copy=False).tobytes())
                fs.write(np.array(sc.view(mx.uint16), copy=False).tobytes())
                fb.write(np.array(bi.view(mx.uint16), copy=False).tobytes())
                continue
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
    if sep:
        fw.close(), fs.close(), fb.close()
    else:
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
        "layout": layout,
        "record_bytes": rec_bytes,
        "weight_bytes": wb,
        "scale_bytes": sb,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(
        (out / f).stat().st_size
        for f in (("weight.bin", "scales.bin", "biases.bin") if sep else ("rows.bin",))
    )
    print(f"wrote {out}  {size / 1e9:.1f} GB")
    return manifest


# --------------------------------------------------------------- 実行時の引き


class StreamNGram:
    """サイドカーから行だけ引く。`_ShardedEmbedding` と同じ呼び出し規約。

    nn.Module ではない。パラメータを 1 つも持たないので、モデルの
    `parameters()` に現れてはいけない (現れると保存や dtype 変換の対象になる)。

    mmap の fancy index は 1 回のギャザーに見えるが、実体は行ごとにページ
    フォールトを起こし、それがカーネル内で直列に処理される。実測で 16 行
    (1 トークン分) の引きに 1.5-2ms かかっており、これがデコード全体の
    7-9% を占めていた。ddalcu/mlx-serve (Zig 実装) が同じ設計で同じ問題を
    踏んでいて、"serial mmap faults were ~5ms of every decode step" とある。

    `os.pread` は GIL を解放するので、行ごとに別スレッドで並列に投げれば
    フォールトの直列化が消える。マイクロベンチ (tools/ngram_pread_bench.py)
    では 16 行の引きが mmap 比 5-7x、128 行 (バッチ forward 相当) で
    10-12x 速い。スレッド数は 8-24 の間でほぼ横ばいで、性能コア数
    (このマシンでは 12) に合わせるのが無難と見て既定値にした。

    退行したときに戻せるよう mmap 経路は残す。`FASTMLX_NGRAM_BACKEND=mmap`
    か `backend="mmap"` で切り替えられる。
    """

    def __init__(
        self,
        sidecar: Path,
        backend: str | None = None,
        n_threads: int | None = None,
    ):
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
        rows_bin = self.dir / "rows.bin"
        self.mm = np.memmap(rows_bin, dtype=np.uint8, mode="r", shape=(self.rows, self.rec))

        self.backend = backend or os.environ.get("FASTMLX_NGRAM_BACKEND", "pread")
        if self.backend not in ("mmap", "pread"):
            raise ValueError(f"backend は mmap/pread のどちらか ({self.backend})")
        if self.backend == "pread":
            self.n_threads = n_threads or int(
                os.environ.get("FASTMLX_NGRAM_THREADS", "12")
            )
            # 呼び出しごとに作らない。プール生成はスレッド起動込みで数ms
            # かかり、毎トークン作っていたら並列化した意味が消える
            self._pool = ThreadPoolExecutor(max_workers=self.n_threads)
            self._fd = os.open(str(rows_bin), os.O_RDONLY)

    def _gather_pread(self, flat: np.ndarray) -> np.ndarray:
        """行 id 配列を受けて、対応するレコードを並列 pread で埋める。"""

        n = flat.shape[0]
        buf = np.empty((n, self.rec), dtype=np.uint8)
        rec_bytes = self.rec

        def read_one(i: int, row_id: int) -> None:
            # os.pread は GIL を解放するので、ここで実際にディスク I/O が
            # 並列に走る。書き込み先 buf[i] は行ごとに素なので競合しない
            buf[i] = np.frombuffer(
                os.pread(self._fd, rec_bytes, int(row_id) * rec_bytes), dtype=np.uint8
            )

        futures = [self._pool.submit(read_one, i, row_id) for i, row_id in enumerate(flat)]
        for f in futures:
            f.result()
        return buf

    def __call__(self, gid):
        import mlx.core as mx

        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64)
        if self.backend == "pread":
            rec = self._gather_pread(flat)
        else:
            # numpy の fancy index は C 側で一度に集める。行ごとの Python
            # ループにすると生成時に効いてくるので、ここは必ず 1 回の
            # ギャザーで済ませる。1 行が連続レコードなので、触るページも
            # 1 行あたり 1 枚で済む
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


__all__ = ["RamNGram", "StreamNGram", "build_sidecar", "install", "install_ram"]


class RamNGram:
    """連結済みの n-gram 表を RAM に持ち、`mx.take` 一発で引く。

    素の `_ShardedEmbedding` は 128 枚を別々に持つため、引くたびに
    `np.unique` でホストへ降りて (= 毎トークン GPU 同期)、触れたシャードごとに
    numpy と MLX を往復する。実測で 1 トークン 11-30ms を食っていた
    (デコード全体の 38.5%、docs/STATUS.md)。

    128 枚は論理的に行ブロックを並べたものなので、連結してしまえば
    シャード計算も分岐も要らない。gather 3 回 + dequantize の 4 op で済み、
    同期も Python ループも消える。メモリ量は連結前と同じ。
    """

    def __init__(self, sidecar: Path):
        import mlx.core as mx

        self.dir = Path(sidecar)
        m = json.loads((self.dir / "manifest.json").read_text())
        if m.get("layout") != "separate":
            raise ValueError(
                f"RAM 常駐には layout=separate のサイドカーが要る "
                f"(このサイドカーは {m.get('layout')})"
            )
        self.dim, self.bits = m["dim"], m["bits"]
        self.group_size = m["group_size"]
        rows, npack, ngrp = m["rows"], m["packed_per_row"], m["groups_per_row"]

        def load(name, dtype, cols):
            a = np.fromfile(self.dir / name, dtype=dtype).reshape(rows, cols)
            return mx.array(a)

        self.w = load("weight.bin", np.uint32, npack)
        self.s = load("scales.bin", np.uint16, ngrp).view(mx.bfloat16)
        self.b = load("biases.bin", np.uint16, ngrp).view(mx.bfloat16)
        mx.eval(self.w, self.s, self.b)

    @property
    def nbytes(self) -> int:
        return self.w.nbytes + self.s.nbytes + self.b.nbytes

    def __call__(self, gid):
        import mlx.core as mx

        flat = gid.reshape(-1)
        out = mx.dequantize(
            mx.take(self.w, flat, axis=0),
            mx.take(self.s, flat, axis=0),
            mx.take(self.b, flat, axis=0),
            group_size=self.group_size,
            bits=self.bits,
        )
        return out.reshape(*gid.shape, self.dim)


def install_ram(model, sidecar: str | Path) -> None:
    """n-gram 表を RAM 常駐の連結テーブルに差し替える (速度重視の経路)。"""

    table = RamNGram(Path(sidecar))
    n = 0
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        if ple.ple_embedding.ngram_embedding.dim != table.dim:
            raise ValueError("次元が合わない")
        ple.ple_embedding.ngram_embedding = table
        n += 1
    if n == 0:
        raise ValueError("PLE 層が見つからない")
    print(
        f"n-gram を連結テーブルに差し替えた "
        f"({n} 層, {table.bits}bit, RAM {table.nbytes / 1e9:.1f}GB, gather 1 回)"
    )
