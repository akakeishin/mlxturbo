"""Evict the n-gram hash tables from RAM and fetch only the needed rows from disk.

Flash-Next's n-gram tables are 51.2B params: 102GB in bf16, and still 19.2GB
even crushed down to 2bit. They are the single largest block, eating a quarter
of the whole model, and yet **only 16 rows are touched per token**
(ngram_heads = (ngram_size-1) * heads_per_ngram = 16). It is a lookup, not a
matmul, so there is almost no point in keeping it resident.

Measurements (docs/STATUS.md Phase Q):
  - Crushing the n-gram tables all the way to 2bit (relative error 0.36) left
    top1 agreement unchanged
  - The error source that stands out is the experts instead
So the capacity that was being spent on the n-gram tables is better handed to
the experts. Putting them on disk brings the RAM cost to zero, and on top of
that the precision can be raised (relative error 0.081 at 4bit).

The sidecar format is raw binary, so it is easy to memmap. With safetensors you
would end up holding a single 25GB tensor, with an extra layer in the way of
row-granular extraction.

    <dir>/manifest.json   row count, dimension, bits, group_size
    <dir>/rows.bin        (rows, record_bytes) uint8

**One row is laid out contiguously as one record.** Putting weight / scales /
biases in separate files would mean touching 3 places to fetch one row, i.e.
16 rows x 3 = 48 random accesses per token. Measured, that was costing 23.9ms
per token (generation 21.3 -> 14.1 tok/s). With contiguous layout the faults
drop to a third, and since about 40 rows fit in a 4KB page you get locality too.

The 128 shards are logically just row blocks laid end to end, so they are
concatenated and held as a single flat table. That way the shard arithmetic
disappears from the lookup path.
"""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_SHARD_RE = "ngram_embedding.shard_"


# --------------------------------------------------------- building the sidecar


def _iter_shard_tensors(src: Path):
    """Yield the n-gram shards from the bf16 archive in shard_0, shard_1, ...
    order."""

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
    """Build a quantized sidecar from the bf16 archive.

    Reading and writing happens in row blocks, so this only uses a few hundred
    MB of memory.

    layout="interleaved" lays out one row contiguously as one record. This is
    for fetching while the data stays on disk, and touches only one page per
    row.
    layout="separate" puts weight/scales/biases in separate files. This is for
    holding the table in RAM and fetching with `mx.take`, which needs the 3
    arrays as-is.
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


# ------------------------------------------------------------ runtime lookups


class StreamNGram:
    """Fetch only the needed rows from the sidecar. Same calling convention as
    `_ShardedEmbedding`.

    This is not an nn.Module. It holds no parameters at all, so it must not show
    up in the model's `parameters()` (if it did, it would become a target for
    saving and dtype conversion).

    A fancy index into an mmap looks like a single gather, but what actually
    happens is a page fault per row, processed serially inside the kernel.
    Measured, fetching 16 rows (one token's worth) took 1.5-2ms, which was
    7-9% of the entire decode. ddalcu/mlx-serve (a Zig implementation) hit the
    same problem with the same design; it says "serial mmap faults were ~5ms of
    every decode step".

    `os.pread` releases the GIL, so dispatching each row on a separate thread in
    parallel makes the serialization of the faults go away. In the microbenchmark
    (tools/ngram_pread_bench.py), fetching 16 rows is 5-7x faster than mmap, and
    128 rows (equivalent to a batched forward) is 10-12x faster. The thread count
    is essentially flat between 8 and 24, and matching the performance core count
    (12 on this machine) looked like the safe choice, so that is the default.

    The mmap path is kept so it can be reverted to if this regresses. Switch with
    `FASTMLX_NGRAM_BACKEND=mmap` or `backend="mmap"`.
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
            # Do not create this per call. Creating the pool takes a few ms
            # including thread startup, and creating one per token would undo
            # the point of parallelizing
            self._pool = ThreadPoolExecutor(max_workers=self.n_threads)
            self._fd = os.open(str(rows_bin), os.O_RDONLY)

    def _gather_pread(self, flat: np.ndarray) -> np.ndarray:
        """Take an array of row ids and fill in the corresponding records with
        parallel preads."""

        n = flat.shape[0]
        buf = np.empty((n, self.rec), dtype=np.uint8)
        rec_bytes = self.rec

        def read_one(i: int, row_id: int) -> None:
            # os.pread releases the GIL, so the disk I/O really does run in
            # parallel here. The destination buf[i] is disjoint per row, so
            # there is no contention
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
            # numpy's fancy index collects everything at once on the C side. A
            # per-row Python loop would show up during generation, so always
            # settle this with a single gather. Since one row is one contiguous
            # record, this also touches only one page per row
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
    """Swap the n-gram tables of an already-loaded model for sidecar lookups.

    At quantization/conversion time `_ShardedEmbedding` was made an empty
    parameter-free implementation, so the checkpoint contains no n-gram tensors.
    This is where the actual data gets supplied.
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
    # The backend can also be decided by an environment variable, so always
    # print which one was taken. If FASTMLX_NGRAM_BACKEND=mmap is left in the
    # environment, fetching 16 rows becomes 5-7x slower. The output is identical,
    # so if it runs silently the only thing that shifts is the numbers we publish
    if stream.backend == "pread":
        how = f"backend=pread threads={stream.n_threads}"
    else:
        how = "backend=mmap (既定は pread。FASTMLX_NGRAM_BACKEND を確認すること)"
    print(
        f"[mlxturbo] n-gram をサイドカー参照に差し替えた "
        f"({n} 層, {stream.bits}bit, RAM 0, {how}) <- {stream.dir}"
    )


def warn_if_not_installed(model) -> bool:
    """Raise the alarm at startup when the n-gram tables have not been swapped
    for the sidecar.

    Forgetting to pass `--ngram` means `FASTMLX_NGRAM_DISK` is never set, and
    qwen4_exp's `_ShardedEmbedding` ends up allocating the tables itself. A
    checkpoint whose n-gram data was split out into a sidecar contains no n-gram
    tensors, so those tables stay at their initial values. Generation itself
    runs to completion, so if nothing says anything this slips into both
    conversations and measurements.

    Does nothing for models with no PLE layers (Llama and the like). Returns
    True if the swap has been done.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return False
    stale = total = 0
    for layer in layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        total += 1
        if not isinstance(ple.ple_embedding.ngram_embedding, (StreamNGram, RamNGram)):
            stale += 1
    if total == 0:
        return False
    if stale:
        print(
            f"[mlxturbo] 警告: n-gram 表がサイドカーに差し替わっていない "
            f"({stale}/{total} 層)。--ngram <サイドカー> を渡していないなら、"
            "出力は初期値の表で生成されたものになる"
        )
        return False
    return True


__all__ = [
    "RamNGram",
    "StreamNGram",
    "build_sidecar",
    "install",
    "install_ram",
    "warn_if_not_installed",
]


class RamNGram:
    """Hold the concatenated n-gram table in RAM and fetch with a single
    `mx.take`.

    The stock `_ShardedEmbedding` holds the 128 shards separately, so every
    lookup descends to the host via `np.unique` (= a GPU sync every token) and
    round-trips between numpy and MLX once per shard touched. Measured, that was
    costing 11-30ms per token (38.5% of the entire decode, docs/STATUS.md).

    The 128 shards are logically just row blocks laid end to end, so once
    concatenated neither the shard arithmetic nor the branching is needed. It
    comes down to 4 ops — 3 gathers plus a dequantize — and both the sync and
    the Python loop disappear. Memory usage is the same as before concatenation.
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
    """Swap the n-gram tables for a RAM-resident concatenated table (the
    speed-first path)."""

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
