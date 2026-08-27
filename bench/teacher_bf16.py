"""bf16 の参照 logits を、重みをメモリに載せずに 1 パスで取る。

Flash-Next の bf16 は 360GB あって 128GB には載らない。しかし teacher forcing で
必要なのは **決まったトークン列を 1 回流すこと**だけで、生成のようにトークン毎の
繰り返しが要らない。だから層ごとに

    その層の重みだけ読む -> 全プロンプトの活性に適用する -> 捨てる

と進めれば、アーカイブを 1 回舐めるだけで済む。活性の総量は 31 プロンプト x
約 250 位置 x 10240 次元 x 4 バイト = 約 320MB で、何の問題もない。

読み出し量の内訳 (実測は --dry-run で出る):
  dense (attention / GDN / norm / hyper-connection)  約 30GB
  experts                                            約 240GB
  n-gram                                             位置あたり 16 行だけ引くので
                                                     102GB のうち実際に触るのは数 MB
  embed / lm_head                                    約 2.5GB

これで「参照は走る中で最大の変種」という相対レーンの制約が外れ、
量子化変種の絶対的な劣化 (bf16 からの KLD) が出せる。

使い方:
  uv run python bench/teacher_bf16.py --src "/Volumes/Mobile SSD/models/Qwen3.8-Flash-Next" \
      --continuations bench/results/qe-cont.json --out bench/results/qe-ref-bf16.npz
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------- シャード索引


class ShardIndex:
    """safetensors のヘッダだけを読み、テンソル名 -> (ファイル, 位置) を持つ。

    mx.load はファイル全体の辞書を返すので、層ごとに必要な分だけ取り出したい
    ここでは使えない。numpy の memmap で必要な範囲だけ読む。
    """

    _DTYPE = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I64": np.int64}

    def __init__(self, src: Path):
        self.entries: dict[str, tuple[Path, int, int, str, list[int]]] = {}
        self._mm: dict[Path, np.memmap] = {}
        for shard in sorted(src.glob("model-*.safetensors")):
            with open(shard, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
            base = 8 + hlen
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                a, b = info["data_offsets"]
                self.entries[name] = (
                    shard,
                    base + a,
                    base + b,
                    info["dtype"],
                    info["shape"],
                )

    def _map(self, path: Path) -> np.memmap:
        if path not in self._mm:
            self._mm[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._mm[path]

    def has(self, name: str) -> bool:
        return name in self.entries

    def raw(self, name: str) -> np.ndarray:
        shard, a, b, dtype, shape = self.entries[name]
        buf = self._map(shard)[a:b]
        arr = np.frombuffer(buf, dtype=self._DTYPE[dtype])
        return arr.reshape(shape)

    def rows(self, name: str, row_ids: np.ndarray) -> np.ndarray:
        """2 次元テンソルの指定行だけ読む。n-gram 表のための入口。"""
        shard, a, _, dtype, shape = self.entries[name]
        width = shape[-1]
        itemsize = np.dtype(self._DTYPE[dtype]).itemsize
        buf = self._map(shard)
        out = np.empty((len(row_ids), width), dtype=self._DTYPE[dtype])
        stride = width * itemsize
        for i, r in enumerate(row_ids):
            off = a + int(r) * stride
            out[i] = np.frombuffer(buf[off : off + stride], dtype=self._DTYPE[dtype])
        return out

    def nbytes(self, name: str) -> int:
        _, a, b, _, _ = self.entries[name]
        return b - a


def to_mx(arr: np.ndarray):
    """BF16 は numpy に型が無いので uint16 で運んで mlx 側で解釈する。"""
    import mlx.core as mx

    if arr.dtype == np.uint16:
        return mx.array(arr).view(mx.bfloat16)
    return mx.array(arr)


# --------------------------------------------------------------- 進捗の見積り


def cmd_plan(args):
    src = Path(args.src)
    idx = ShardIndex(src)
    groups: dict[str, int] = {}
    for name in idx.entries:
        if name.startswith("mtp.") or ".visual." in name:
            kind = "除外 (mtp/vision)"
        elif "ngram_embedding" in name:
            kind = "n-gram (行だけ引く)"
        elif "switch_mlp" in name or ".experts." in name:
            kind = "experts"
        elif "embed_tokens" in name or name.startswith("lm_head"):
            kind = "embed / lm_head"
        else:
            kind = "dense"
        groups[kind] = groups.get(kind, 0) + idx.nbytes(name)
    total_read = 0
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v / 1e9:8.1f} GB")
        if k not in ("除外 (mtp/vision)", "n-gram (行だけ引く)"):
            total_read += v
    print(f"  {'1 パスで実際に読む量':24s} {total_read / 1e9:8.1f} GB")
    for bw in (0.5, 1.0, 2.0):
        print(f"    {bw} GB/s なら {total_read / 1e9 / bw / 60:.0f} 分")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--continuations")
    ap.add_argument("--out")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--plan", action="store_true", help="読み出し量の見積りだけ出す")
    args = ap.parse_args()

    if args.plan:
        cmd_plan(args)
        return

    raise SystemExit(
        "層ごとの適用はこれから。まず --plan で読み出し量を確かめること"
    )


if __name__ == "__main__":
    main()
