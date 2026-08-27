"""bf16 の参照 logits を、重みをメモリに載せずに 1 パスで取る。

Flash-Next の bf16 は 360GB あって 128GB には載らない。しかし teacher forcing で
必要なのは **決まったトークン列を 1 回流すこと**だけで、生成のようにトークン毎の
繰り返しが要らない。だから層ごとに

    その層の重みだけ読む -> 全プロンプトの活性に適用する -> 捨てる

と進めれば、アーカイブを 1 回舐めるだけで済む。活性の総量は 31 プロンプト x
約 250 位置 x 10240 次元 x 4 バイト = 約 320MB で、何の問題もない。

読み出し量 (--plan で出る): 251.5GB。experts が 241.6GB で 96%、dense は 7.4GB。
n-gram の 102GB は位置あたり 16 行しか引かないので実質数 MB しか触らない。

これで「参照は走る中で最大の変種」という相対レーンの制約が外れ、量子化変種の
絶対的な劣化 (bf16 からの KLD) が出せる。基準を追い越す構成も順位が付く。

使い方:
  uv run python bench/teacher_bf16.py --src <bf16 dir> --plan
  uv run python bench/teacher_bf16.py --src <bf16 dir> \
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
                self.entries[name] = (shard, base + a, base + b, info["dtype"], info["shape"])

    def _map(self, path: Path) -> np.memmap:
        if path not in self._mm:
            self._mm[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._mm[path]

    def raw(self, name: str) -> np.ndarray:
        shard, a, b, dtype, shape = self.entries[name]
        arr = np.frombuffer(self._map(shard)[a:b], dtype=self._DTYPE[dtype])
        return arr.reshape(shape)

    def rows(self, name: str, row_ids: np.ndarray) -> np.ndarray:
        """2 次元テンソルの指定行だけ読む。n-gram 表のための入口。"""
        shard, a, _, dtype, shape = self.entries[name]
        np_dt = self._DTYPE[dtype]
        width = shape[-1]
        stride = width * np.dtype(np_dt).itemsize
        buf = self._map(shard)
        out = np.empty((len(row_ids), width), dtype=np_dt)
        for i, r in enumerate(row_ids):
            off = a + int(r) * stride
            out[i] = np.frombuffer(buf[off : off + stride], dtype=np_dt)
        return out

    def nbytes(self, name: str) -> int:
        _, a, b, _, _ = self.entries[name]
        return b - a


def to_mx(arr: np.ndarray):
    """BF16 は numpy に型が無いので uint16 で運んで mlx 側で解釈する。"""
    import mlx.core as mx

    if arr.dtype == np.uint16:
        return mx.array(np.ascontiguousarray(arr)).view(mx.bfloat16)
    return mx.array(np.ascontiguousarray(arr))


# ------------------------------------------------- n-gram をディスクから引く

_NGRAM_CTX: dict = {}


def _install_disk_ngram(Q):
    """`_ShardedEmbedding` を、行だけディスクから引く実装に差し替える。

    素の実装は 128 枚の nn.Embedding (合計 102GB) を確保するので、層を組んだ
    瞬間にメモリが尽きる。位置あたり 16 行しか触らないので、実体を持つ必要が
    そもそも無い。
    """

    import mlx.core as mx
    import mlx.nn as nn

    class DiskShardedEmbedding(nn.Module):
        def __init__(self, n_shards: int, rows: int, dim: int):
            super().__init__()
            self.n_shards, self.rows, self.dim = n_shards, rows, dim

        def __call__(self, gid: mx.array) -> mx.array:
            idx = _NGRAM_CTX["index"]
            prefix = _NGRAM_CTX["prefix"]
            flat = np.array(gid.reshape(-1), dtype=np.int64)
            out = np.zeros((flat.size, self.dim), dtype=np.uint16)
            shard_of = flat // self.rows
            row_of = flat % self.rows
            for s in np.unique(shard_of):
                sel = np.nonzero(shard_of == s)[0]
                out[sel] = idx.rows(f"{prefix}shard_{int(s)}.weight", row_of[sel])
            arr = mx.array(out).view(mx.bfloat16).astype(mx.float32)
            return arr.reshape(*gid.shape, self.dim)

    Q._ShardedEmbedding = DiskShardedEmbedding


# --------------------------------------------------------------- 層の組み立て

_LAYER_SKIP = ("ngram_embedding.shard_",)


def _sanitize_layer(w: dict):
    """port の sanitize のうち、層に効く部分だけを再現する。"""
    import mlx.core as mx

    out = {}
    for k, v in w.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("experts.gate_up_proj")] + "switch_mlp."
            gate, up = mx.split(v, 2, axis=1)
            out[base + "gate_proj.weight"] = gate
            out[base + "up_proj.weight"] = up
            continue
        if k.endswith("mlp.experts.down_proj"):
            out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
            continue
        if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1 and v.shape[1] == 1:
            v = v.transpose(0, 2, 1)
        out[k] = v
    return out


def _build_layer(Q, args, idx: int, index: ShardIndex):
    prefix = f"model.language_model.layers.{idx}."
    layer = Q.DecoderLayer(args.text, idx)
    w = {}
    for name in index.entries:
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix) :]
        if any(s in rel for s in _LAYER_SKIP):
            continue
        w[rel] = to_mx(index.raw(name))
    layer.load_weights(list(_sanitize_layer(w).items()), strict=False)
    return layer


# --------------------------------------------------------------- 見積り


def cmd_plan(args):
    idx = ShardIndex(Path(args.src))
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
    total = 0
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v / 1e9:8.1f} GB")
        if k not in ("除外 (mtp/vision)", "n-gram (行だけ引く)"):
            total += v
    print(f"  {'1 パスで実際に読む量':24s} {total / 1e9:8.1f} GB")
    for bw in (0.5, 1.0, 2.0):
        print(f"    {bw} GB/s なら {total / 1e9 / bw / 60:.0f} 分")


# --------------------------------------------------------------- 本体


def cmd_run(args):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    import mlx_lm.models.qwen4_exp as Q

    src = Path(args.src)
    index = ShardIndex(src)
    cfg = json.loads((src / "config.json").read_text())
    margs = Q.ModelArgs.from_dict(cfg)
    text = margs.text

    _install_disk_ngram(Q)
    _NGRAM_CTX["index"] = index
    ple_layer = next(
        i for i in range(text.num_hidden_layers) if (i + 1) in text.ple_layer_ids
    )
    _NGRAM_CTX["prefix"] = (
        f"model.language_model.layers.{ple_layer}.ple.ple_embedding.ngram_embedding."
    )

    cont = json.loads(Path(args.continuations).read_text())
    keys = list(cont["prompts"])
    seqs = [
        cont["prompts"][k]["prompt_ids"] + cont["prompts"][k]["continuation_ids"]
        for k in keys
    ]
    starts = [len(cont["prompts"][k]["prompt_ids"]) - 1 for k in keys]
    print(f"{len(keys)} プロンプト / 合計 {sum(len(s) for s in seqs)} トークン")

    t0 = time.time()
    embed_w = to_mx(index.raw("model.language_model.embed_tokens.weight"))
    hs, ids_list = [], []
    for s in seqs:
        ids = mx.array(s)[None]
        ids_list.append(ids)
        hs.append(mx.tile(embed_w[mx.array(s)][None], (1, 1, text.hc_count)))
    del embed_w
    mx.eval(hs)
    mx.clear_cache()
    print(f"  embed 済み {time.time() - t0:.0f}s")

    eos = text.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    prev_ctx = [
        mx.full((1, text.ngram_size - 1), eos, ids.dtype) for ids in ids_list
    ]
    masks = [create_attention_mask(h, None) for h in hs]
    rope = Q.RotaryEmbedding(
        int(text.head_dim * text.partial_rotary_factor), text.rope_theta
    )

    for li in range(text.num_hidden_layers):
        t1 = time.time()
        layer = _build_layer(Q, margs, li, index)
        for i in range(len(hs)):
            hs[i] = layer(hs[i], rope, masks[i], None, None, None, ids_list[i], prev_ctx[i])
        mx.eval(hs)
        del layer
        mx.clear_cache()
        print(f"  層 {li:2d}/{text.num_hidden_layers} {time.time() - t1:5.1f}s", flush=True)

    mixer = Q.GatedResidual(text, use_combine=False)
    mw = {
        k[len("model.language_model.hyper_connection_mixer.") :]: to_mx(index.raw(k))
        for k in index.entries
        if k.startswith("model.language_model.hyper_connection_mixer.")
    }
    mixer.load_weights(list(mw.items()))
    hs = [mixer(h) for h in hs]
    mx.eval(hs)
    del mixer, mw
    mx.clear_cache()

    lm_head = to_mx(index.raw("lm_head.weight"))
    arrays, meta = {}, {"model": str(src), "topk": args.topk, "prompts": {}}
    for key, h, s, st in zip(keys, hs, seqs, starts):
        logits = (h[0, st:-1] @ lm_head.T).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idxs = mx.argpartition(-logp, args.topk - 1, axis=-1)[:, : args.topk]
        top = mx.take_along_axis(logp, idxs, axis=-1)
        order = mx.argsort(-top, axis=-1)
        idxs = mx.take_along_axis(idxs, order, axis=-1)
        top = mx.take_along_axis(top, order, axis=-1)
        tgt = mx.array(s[st + 1 :])[:, None]
        tgt_logp = mx.take_along_axis(logp, tgt, axis=-1)[:, 0]
        mx.eval(idxs, top, tgt_logp)
        top_np = np.array(top, dtype=np.float32)
        arrays[f"{key}.idx"] = np.array(idxs, dtype=np.int32)
        arrays[f"{key}.logp"] = top_np
        arrays[f"{key}.tgt_logp"] = np.array(tgt_logp, dtype=np.float32)
        arrays[f"{key}.tail"] = np.log1p(
            -np.minimum(np.exp(top_np).sum(axis=-1), 1 - 1e-9)
        ).astype(np.float32)
        meta["prompts"][key] = {"positions": int(top_np.shape[0])}
        del logits, logp
        mx.clear_cache()

    np.savez_compressed(args.out, **arrays)
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta))
    print(f"wrote {args.out}  合計 {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--continuations")
    ap.add_argument("--out")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    if args.plan:
        cmd_plan(args)
        return
    if not (args.continuations and args.out):
        raise SystemExit("--continuations と --out が要る")
    cmd_run(args)


if __name__ == "__main__":
    main()
