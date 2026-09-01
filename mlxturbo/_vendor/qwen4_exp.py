# vendored from https://github.com/eauchs/mlx-lm branch add-qwen4-exp
# (ml-explore/mlx-lm PR #1788, MIT license). Imported 2026-08-27.
# Changes made on the mlxturbo side:
#   1. Drop mtp.* (extracted separately into a sidecar)
#   2. Fold model.language_model.* -> model.* (the public ckpt has VLM shape)
#   3. Split and rename mlp.experts.{gate_up,down}_proj -> switch_mlp.*
#   4. NGramEmbedding: take the hash multipliers from the ckpt's actual values
#      rather than from a recomputation
#   5. Fix RMSNorm to the (1 + weight) convention (the Gemma-style one, same as
#      the reference implementation)
#   6. With FASTMLX_NGRAM_DISK=1, do not hold the n-gram table (read rows from
#      disk instead)
# (mlx-lm proper has no MTP module, so a strict load fails. MTP is extracted
#  into a sidecar with mlxturbo/convert_flash.py extract-mtp and used from there.)
# Resolution: importing `mlxturbo` makes `mlxturbo/_arch_registry.py` install a
# `sys.meta_path` hook that redirects the import of `mlx_lm.models.qwen4_exp`
# straight to this file. Nothing is written into the user's site-packages or into
# mlx_lm proper (the old install-arch physically copied the file into
# site-packages, and was retired because of that side effect of polluting the
# user's mlx_lm).
# MLX port of Qwen3.8-Flash-Next (HF model_type: qwen4_exp)
# New compared to qwen3_next: QSA sparse attention, gated residual
# (hyper-connections), sharded n-gram / PLE embedding, split deltanet projections.

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, KVCache, _BaseCache
from .gated_delta import gated_delta_update
from .switch_layers import SwitchGLU

# mlxturbo: switch to keeping the n-gram table (51.2B params) out of RAM and
# reading only the needed rows from disk. This flag has to be set both at
# conversion time and at load time.
NGRAM_ON_DISK = os.environ.get("FASTMLX_NGRAM_DISK") == "1"


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # n-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 0
    eos_token_id: Any = 248044
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.text.partial_rotary_factor)
        )
        if not self.text.layer_types:
            n, k = self.text.num_hidden_layers, self.text.full_attention_interval
            self.text.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(n)
            ]


# --------------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """RMSNorm, normalized per group when group_size is given.

    Hyper-connections normalize each of the hc_count streams separately, hence the
    reshape: one weight of size hc_count*hidden, but one statistic per stream.
    """

    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        # The reference implementation (Qwen4ExpTextRMSNorm) initializes to zero
        # and applies the scale as (1 + weight) — the Gemma-style convention
        self.weight = mx.zeros(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} non divisible par group_size {group_size}")

    def __call__(self, x: mx.array) -> mx.array:
        # mlxturbo: the original port used x * weight and was missing the +1.
        # The trained weight is a delta near 0, so multiplying by it shrinks the
        # signal down to just its direction; the magnitude of the activations
        # still looks plausible while the information is destroyed (generation
        # degenerates into meaningless repetition).
        if self.group_size is None:
            return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * (1.0 + self.weight)


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        g = act(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


# ------------------------------------------------------------------- rope / helpers


def _rope_partial(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rope to the first `rotary_dim` dimensions only."""
    d = cos.shape[-1]
    # cos/sin are computed in float32: without this cast they promote x and the
    # whole attention falls back to float32.
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    xr, xp = x[..., :d], x[..., d:]
    half = d // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


class RotaryEmbedding:
    def __init__(self, dim: int, base: float):
        self.dim = dim
        self.inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)

    def __call__(self, positions: mx.array):
        # positions: (B, T) -> cos/sin (B, T, dim)
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


# ------------------------------------------------------------------------ QSA


# 集めた列が kv 長のこの割合を超えるなら gather しない (使う場所の注記を参照)。
#
# **モデルの形で変わる。**KV は (B, n_kv_heads, kv, head_dim) なので、kv 軸で
# 集めるときの連続長は `head_dim * 2` バイト。連続長が短いほど飛び飛びの読みは
# 効率が落ち、gather が割に合う境界は前に動く。QSA を持つモデルが増えるなら、
# ここをモデルごとに分けられる形にしておく必要がある。
#
# **外挿はしない。**機構モデル (Δ% = share*(u*k-1)) を 2 点に当てると
# k が 4.28 と 8.56 で揃わず、外挿に耐えないと分かった。よって
# **実測した形にだけ実測値を置き、それ以外は保守側の値を使ってログに出す。**
# 黙って未検証の値を使わない。
#
# ------------------------------------------------------------------
# 新しい QSA モデルを足すときの手順 (将来の自分あて)
# ------------------------------------------------------------------
#
# **0. まず前提を確かめる。**この判定は
#
#     union (集めるブロック数) <= 行数 * block_topk
#
# が成り立つことに全面的に乗っている。Flash-Next は「クエリごとに上位
# block_topk 個のブロックを選ぶ」ので成り立つ。**選ぶ数が可変のモデル**
# (閾値で切る、確率的に選ぶ、局所窓 + グローバルの二本立て 等) では成り立たず、
# 上限が上限にならない。その場合はこの表に足すのではなく、判定の式ごと
# 見直すこと。**ここを確かめずに数字だけ足すと、静かに間違える。**
#
# 1. 掃引する (文脈長を振って、集める割合と速度の関係を見る):
#
#      tools/biglock.sh .venv/bin/python tools/decode_ab.py \
#          --knob gather-attn --only long --ctx <17000|25000|32000|50000> \
#          --prefill-once --model <パック> --ngram <サイドカー>
#
# 2. **ms/round で読む。ms/token で読まない。**gather は加算順が変わるので
#    受理率がテキスト運で動く。Flash-Next の 17k では ms/token +5.5% のうち
#    実際の費用増は +1.1% だけだった (残りは受理率の揺れ)。
#
# 3. 集める割合 (= 行数 * token_budget / kv_len) を横軸に、ms/round の変化を
#    縦軸にして**ゼロ交差**を読む。交差点を挟む 2 点があれば内挿でよい。
#
# 4. 交差点から 1-2 割**安全側 (小さい方)** に倒して、この表に足す。
#    集めに行かずに外した損は 1-2% で頭打ちだが、攻めて外すと数の多い中尺で
#    損をする、という非対称があるため。
#
# 5. **外挿しない。**「head_dim が半分だから比も半分」のような式は書かない
#    (機構モデルを 2 点に当てたら係数が 4.28 と 8.56 で揃わず、外挿に耐えないと
#    実測で分かっている)。測っていない形は保守側の既定に落として警告を出す、
#    が現行の方針。
#
# 6. 機種を変えたときも測り直す (連続読みと飛び飛び読みのスケールの仕方が
#    機種で違う)。詳しくは docs/research/KERNEL-PROGRAM.md の段 C。
#
_GATHER_RATIO_MEASURED = {
    # head_dim -> 比。Flash-Next (qwen4_exp, head_dim 256) を M3 Max で実測:
    # 集める割合 24%/16%/8% で ms/round +1.1%/-6.7%/-15.4%、ゼロ交差 23%。
    # 安全側に倒して 0.20。
    256: 0.20,
}
# 実測の無い形に使う値。連続長が短いほど gather は不利なので、**小さめ**に
# 置く (= あまり集めに行かない = 従来経路に落ちる)。外すと損は 1-2% で頭打ち
# だが、攻めて外すと数の多い中尺で損をする。
_GATHER_RATIO_UNKNOWN = 0.10
_GATHER_RATIO_ENV = os.environ.get("MLXTURBO_GATHER_MAX_RATIO")
_gather_ratio_warned = set()


def _gather_max_ratio(head_dim: int) -> float:
    """この形のモデルで「集める価値がある」割合の上限。

    env > 実測表 > 保守側の既定、の順。実測の無い形では**一度だけ**
    警告を出す (較正すれば正しい値が出せる、と伝えるため)。
    """
    if _GATHER_RATIO_ENV:
        return float(_GATHER_RATIO_ENV)
    if head_dim in _GATHER_RATIO_MEASURED:
        return _GATHER_RATIO_MEASURED[head_dim]
    if head_dim not in _gather_ratio_warned:
        _gather_ratio_warned.add(head_dim)
        print(
            f"[mlxturbo] gather attention: head_dim={head_dim} は実測が無いので"
            f" 保守側の比 {_GATHER_RATIO_UNKNOWN} を使う (実測があるのは"
            f" {sorted(_GATHER_RATIO_MEASURED)})。"
            " docs/research/KERNEL-PROGRAM.md の段 C の手順で測り直せば"
            " MLXTURBO_GATHER_MAX_RATIO で指定できる。"
        )
    return _GATHER_RATIO_UNKNOWN


@dataclass
class QSABlockSelection:
    """段 3(b) の gather 経路 (``mlxturbo/gather_attn.py``) 向け: ブロック選択を
    トークン幅へ展開する前の、ブロック添字のままの形で持つ。

    ``QSAIndexer.select_blocks`` の戻り値。``__call__`` が返す ``keep``
    ((B,1,S,kv_len) の bool、trueの位置がトークン単位) とは違い、こちらは
    ``keep_block`` が (B,S,n_blocks) で、1 要素が 1 ブロック
    (``compress_ratio`` トークン分) を表す。Attention 側はこれの S 行の
    和集合を取ってから、対応する KV 列だけを集める。
    """

    keep_block: Any  # mx.array (B, S, n_blocks) bool
    n_blocks: int
    kv_len: int
    tail: int
    q_col: Any  # mx.array (S,) int


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks.

    The reference PyTorch implementation loops over (batch, query); here everything
    is vectorized: pooled keys do not depend on the query, so they are computed once
    and followed by a per-row top-k.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        # 段 X1 (docs/research/KERNEL-PROGRAM.md): pooled キーの増分キャッシュ。
        # 既定 on (ビット不変なので)。A/B 用の口は mlxturbo/pooled_cache.py。
        self._pooled_cache = True

    def _pooled_and_top(self, x, rope, cache, offset: int, positions=None):
        """pooled key の作成からブロック top-k 選択まで。``__call__`` と
        ``select_blocks`` の共通部 (段 3(b) で切り出した)。

        ``kv_len <= self.token_budget`` (疎化が要らない) のときは ``None``。
        それ以外は ``(keep_block, n_blocks, kv_len, q_col)`` を返す。
        ``keep_block`` は (B, S, n_blocks) の bool で、まだトークン幅へは
        展開していない。
        """
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = raw_k.shape[1]

        # No sparsification possible: every visible token fits in the budget, so the
        # top-k would keep them all. The usual causal mask is enough.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        block_starts = mx.arange(n_blocks) * self.compress_ratio

        # 段 X1: ブロックは compress_ratio トークン分が揃った時点で内容が
        # 確定し (mean・k_layernorm はブロック内で閉じている)、rope の角度も
        # block_starts だけで決まって以後変わらない。よって新しく完成した
        # ブロックぶんだけ計算して cache に積み増せば、毎回全ブロックを
        # 作り直すのとビット一致する (`docs/research/KERNEL-PROGRAM.md` 段 X1、
        # `tools/micro_indexer.py` の実測で pooled 関連が indexer の 45.5%)。
        if cache is not None and getattr(self, "_pooled_cache", True):

            def _new_pooled_blocks(start: int, end: int):
                seg = raw_k[
                    :, start * self.compress_ratio : end * self.compress_ratio
                ].reshape(B, end - start, self.compress_ratio, self.head_dim)
                seg = self.k_layernorm(
                    seg.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
                )
                starts = block_starts[start:end]
                cos_seg, sin_seg = rope(starts[None, :])
                return _rope_partial(seg, cos_seg, sin_seg)

            pooled = cache.pooled(n_blocks, _new_pooled_blocks)
        else:
            pooled = raw_k[:, : n_blocks * self.compress_ratio].reshape(
                B, n_blocks, self.compress_ratio, self.head_dim
            )
            pooled = self.k_layernorm(
                pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
            )
            cos_k, sin_k = rope(block_starts[None, :])
            pooled = _rope_partial(pooled, cos_k, sin_k)

        q_col = mx.arange(offset, offset + S)
        cos_q, sin_q = rope(q_col[None, :] if positions is None else positions)
        q = self.q_layernorm(q)
        q = _rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])

        # scores: sum over heads of relu(q.k), per block
        scores = mx.einsum(
            "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        # a block is only a candidate if it lies entirely in the query's past
        block_end = block_starts + self.compress_ratio - 1
        visible = block_end[None, None, :] <= q_col[None, :, None]
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]  # (B, S, k)

        keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
        top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
        keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[
            ..., :n_blocks
        ]
        return keep_block, n_blocks, kv_len, q_col

    def __call__(
        self, x, rope, cache, offset: int, positions=None
    ) -> Optional[mx.array]:
        """``offset`` はキャッシュの列位置 (ブロック格子と可視判定に使う)、
        ``positions`` はその系列の先頭から数えた真の位置 (rope の角度に使う)。
        パディングが無ければ両者は一致するので ``positions`` は省略できる。"""
        B, S, _ = x.shape
        res = self._pooled_and_top(x, rope, cache, offset, positions)
        if res is None:
            return None
        keep_block, n_blocks, kv_len, q_col = res

        # remap block -> tokens; the incomplete tail is visible as far as
        # causality allows.
        #
        # The port this file came from marks the tail `ones` ("always
        # visible"). But when `sparse` is not None, Attention throws the causal
        # mask away and goes by `sparse` alone, so those up-to
        # compress_ratio-1 trailing columns become visible to *every* query in
        # the call, including the ones that sit before them. Complete blocks
        # are screened by `visible` above, so the tail was the only place the
        # future leaked. It shows wherever S > 1 with QSA active: prefill
        # chunks whose kv length is not a multiple of compress_ratio, and
        # speculative verify rounds (there a draft token sees the drafts that
        # follow it). Synthetic probe: logits move by ~9e-2 with the stock
        # `ones`, exactly 0 with this.
        keep = mx.repeat(keep_block, self.compress_ratio, axis=-1)
        tail = kv_len - n_blocks * self.compress_ratio
        if tail:
            tail_col = n_blocks * self.compress_ratio + mx.arange(tail)
            keep = mx.concatenate(
                [
                    keep,
                    mx.broadcast_to(
                        tail_col[None, None, :] <= q_col[None, :, None],
                        (B, S, tail),
                    ),
                ],
                axis=-1,
            )
        if offset < self.compress_ratio - 1:
            # 可視ブロックが 1 つも無い行の救済。``visible`` は
            # block_end <= q_col を要求するので、q_col < compress_ratio-1 の
            # クエリはどのブロックも見えない。sparse が非 None のとき
            # Attention は causal を捨てる規約なので、その行は mask が全面
            # -inf になり、softmax が全列一様 = **未来まで見る**。因果窓
            # (自分の位置まで) を開けて塞ぐ。
            #
            # 本番構成では踏まない (budget 2048 / prefill チャンク 2048 なので、
            # QSA が効き始める時点で offset >= 2048)。budget をチャンク幅より
            # 小さくしたパックでだけ起きる。判定は offset (python int) だけで
            # 決まるので、踏まない構成ではこのブロック自体が走らない
            # (mask を読む形にすると毎層 GPU 同期が要る)。
            need = (q_col < self.compress_ratio - 1)[None, :, None]
            causal = mx.arange(kv_len)[None, None, :] <= q_col[None, :, None]
            keep = keep | (need & causal)
        return keep[:, None]  # (B, 1, S, kv_len)

    def select_blocks(
        self, x, rope, cache, offset: int, positions=None
    ) -> Optional["QSABlockSelection"]:
        """段 3(b) の gather 経路向け: ブロック選択をブロック添字のまま返す。

        ``__call__`` と選択ロジックそのものは ``_pooled_and_top`` を共有する
        (計算内容は同一)。違うのはトークン幅への展開をしないことだけ。
        呼び出し側 (``Attention._gather_forward``) が S 行の和集合を取ってから
        ``mx.take_along_axis`` で KV 列を集める。

        ``__call__`` 末尾の早期救済 (``offset < compress_ratio - 1`` のとき
        可視ブロックが無い行をフル causal で救う分岐) には対応しない —
        呼び出し側がその条件を避けてから呼ぶ規約 (Attention 側で判定済み)。
        ``kv_len <= token_budget`` で疎化が要らないときは ``__call__`` と
        同じく ``None`` を返す。
        """
        res = self._pooled_and_top(x, rope, cache, offset, positions)
        if res is None:
            return None
        keep_block, n_blocks, kv_len, q_col = res
        return QSABlockSelection(
            keep_block=keep_block,
            n_blocks=n_blocks,
            kv_len=kv_len,
            tail=kv_len - n_blocks * self.compress_ratio,
            q_col=q_col,
        )


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj also carries the output gate: n_heads * head_dim * 2
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)

    def _positions(self, cache, S: int):
        """``(offset, positions)`` を返すシーム。

        ``offset`` はキャッシュの列位置 (QSA のブロック格子・可視判定、および
        sdpa 壁分割の causal マスク組みに使う)、``positions`` は系列の先頭から
        数えた真の位置 (rope の角度に使う)。パディングの無い単一系列では
        両者は一致する。バッチ経路 (`mlxturbo/batch.py` の左パディング、
        `mlxturbo/batch_spec.py` の dead slot 台帳) はここだけを差し替える。
        """
        offset = cache.offset if cache is not None else 0
        return offset, mx.arange(offset, offset + S)[None]

    def _final_mask(self, mask, sparse, cache, S: int, dtype):
        """sdpa に渡す最終的な mask を組むシーム。

        既定は「sparse があれば causal を捨てて sparse だけを見る」という
        本家の規約。バッチ経路は左パディングとの連言を取ったり、dead slot
        台帳から組み直したりするためにここを差し替える。
        """
        if sparse is None:
            return mask
        neg = mx.finfo(dtype).min if hasattr(mx, "finfo") else -1e9
        add = mx.where(sparse, mx.array(0, dtype), mx.array(neg, dtype))
        if mask is None or isinstance(mask, str):
            return add
        return mask + add

    def _qkv(self, x, positions, rope, cache):
        """q/k/v/gate 射影から rope・KV キャッシュ更新まで。

        ``__call__`` の本体と ``_gather_forward`` (段 3(b)、
        ``MLXTURBO_GATHER_ATTN=1``) の共通部として切り出した。計算内容は
        ``__call__`` に元々あったものと同一。
        """
        B, S, _ = x.shape
        wide = getattr(self, "_wide_qkv", None)
        if wide is None:
            qg, kk, vv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            w, sc, bi, gs, bits, (c1, c2) = wide
            out = mx.quantized_matmul(
                x, w, sc, bi, transpose=True, group_size=gs, bits=bits)
            qg, kk, vv = out[..., :c1], out[..., c1:c2], out[..., c2:]
        q, gate = mx.split(qg.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(kk.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = vv.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = rope(positions)
        cos, sin = cos[:, None], sin[:, None]
        q, k = _rope_partial(q, cos, sin), _rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        return q, k, v, gate

    def _gather_tile_attn(self, q, k, v, cache, blocks, cr: int, B: int, t0: int, t1: int):
        """`_gather_forward` の 1 タイルぶん (クエリ行 ``[t0, t1)``) の
        gather + dense sdpa。

        段 3(b) の元々の (タイル無し) 本体そのもので、``blocks`` の S 行を
        ``[t0, t1)`` に絞って同じ手順 (union → 添字集め → 小 bool マスク) を
        やり直すだけ。``q`` は呼び出し側で既に ``[:, :, t0:t1]`` へ切ってある。

        **タイルごとに union を取り直す**ので、同じ K/V ブロックが複数タイルに
        跨って重複して読まれることがある (タイル分割の代償。K/V 全体を読む
        密の sdpa よりは軽いはずだが、タイル数が増えるほど重複も増えるので
        タイル幅は掃引して決める --- `docs/research/KERNEL-PROGRAM.md` 段 P1a)。
        """
        keep_block = blocks.keep_block[:, t0:t1]  # (B, T, n_blocks)
        n_blocks = blocks.n_blocks
        tail = blocks.tail
        q_col = blocks.q_col[t0:t1]  # (T,)
        T = t1 - t0

        # T クエリの選択ブロックの和集合 (union)。上限は T * block_topk
        # (KERNEL-PROGRAM.md 段 3(b)/P1a)。隣り合うクエリは似たブロックを引く
        # ので、実際の和集合はこれよりずっと小さいことが多い。タイル幅 T を
        # S より小さく切るほど、この上限自体が縮む (P1b の狙いそのもの)。
        k_top = min(self.indexer.block_topk, n_blocks)
        U = min(n_blocks, T * k_top)

        union = mx.any(keep_block, axis=1)  # (B, n_blocks)
        # True を先頭に寄せてから先頭 U 個を取る。1 バッチあたりの True 数は
        # 定義上 U 以下なので、これで実在する選択ブロックを取りこぼさない。
        # 残りの詰め物スロットはどの行の keep_block でも False (union の
        # 定義そのもの) なので、下のクエリごとマスクで自然に False になる。
        order = mx.argsort(-union.astype(mx.int32), axis=-1)
        sel_blocks = order[:, :U]  # (B, U) ブロック添字、重複なし

        sel_tok = sel_blocks[:, :, None] * cr + mx.arange(cr)[None, None, :]
        sel_tok = sel_tok.reshape(B, U * cr)
        if tail:
            # 端数列 (最後の未満ブロック) は block 添字を持たないので、
            # union の対象外のまま常に列末尾へ足す。可視かどうかは下の
            # クエリごとマスクが因果窓 (tail_col <= q_col) で決める。
            # タイル分割の下でも tail 自体はブロック格子の外なので全タイル
            # 共通 --- どのタイルも同じ tail 列を読み直す (重複読みの一種)。
            tail_idx = mx.arange(n_blocks * cr, blocks.kv_len)[None, :]
            sel_tok = mx.concatenate(
                [sel_tok, mx.broadcast_to(tail_idx, (B, tail))], axis=1
            )
        n_sel = sel_tok.shape[1]

        gather_idx = mx.broadcast_to(
            sel_tok[:, None, :, None], (B, self.n_kv_heads, n_sel, self.head_dim)
        )
        k_sel = mx.take_along_axis(k, gather_idx, axis=2)
        v_sel = mx.take_along_axis(v, gather_idx, axis=2)

        # クエリごとの差は、集めた列に対する小さい bool マスクで表す
        # (union 幅であって元の kv_len 幅ではない)。完全なブロックは
        # select_blocks の時点で「block_end <= q_col」を満たすものだけが
        # 候補なので、選ばれたブロックの中身は追加の因果チェックなしに
        # そのまま可視 (元の keep と同じ集合になる)。
        idx_bsu = mx.broadcast_to(sel_blocks[:, None, :], (B, T, U))
        keep_sel = mx.take_along_axis(keep_block, idx_bsu, axis=-1)  # (B,T,U)
        keep_sel = mx.repeat(keep_sel, cr, axis=-1)  # (B,T,U*cr)
        if tail:
            tail_col = n_blocks * cr + mx.arange(tail)
            tail_keep = mx.broadcast_to(
                (tail_col[None, :] <= q_col[:, None])[None], (B, T, tail)
            )
            keep_sel = mx.concatenate([keep_sel, tail_keep], axis=-1)

        stats = getattr(self, "_gather_stats", None)
        if stats is not None:
            # 計測・検証専用のフック (既定 None、`_wide_qkv` と同じ属性注入の
            # 作法)。`tools/verify_gather_attn.py` が union の実サイズを見るのに
            # 使う。union_ratio = U/n_blocks (ブロック単位の刈り込み率)、
            # kv_frac = U*cr/kv_len (元の kv_len に対する、集めた列の割合。
            # tail は分母にだけ含み分子には含めない --- tail は端数ぶんの
            # 小さな定数なので比の主要項ではない)。段 P1a のタイル掃引と
            # 同じ量をタイル分割後にも取れるようにするための拡張。
            union_ratio = (U / n_blocks) if n_blocks else 0.0
            kv_frac = (U * cr / blocks.kv_len) if blocks.kv_len else 0.0
            stats.append((T, n_blocks, U, n_sel, union_ratio, kv_frac))

        out = scaled_dot_product_attention(
            q, k_sel, v_sel, cache=cache, scale=self.scale,
            mask=keep_sel[:, None],
        )
        return out

    def _gather_forward(self, x, rope, cache, idx_cache, offset: int, positions):
        """段 3(b)/P1a: 選ばれたブロックだけ集めてから、mask 無しの dense sdpa に渡す経路。

        `docs/research/KERNEL-PROGRAM.md` 段 3(b) の出し口。
        `mx.fast.scaled_dot_product_attention` は加算マスクを渡されても全 KV を
        読んで全スコアを計算する (段 1 の実測) ので、``_final_mask`` が組む
        加算マスク経由では QSA の疎性が sdpa 側の節約になっていなかった。
        ここでは選ばれた列だけを `mx.take_along_axis` で集めた小さな K/V に、
        同じく小さい bool マスクを渡す。選ぶ集合は元の `keep` と同じだが、
        和の順序が変わるので出力はビット一致しない (呼び出し側はこの前提で
        KLD / tok-step の in-model 計測を経てから採否を決めること)。

        ``MLXTURBO_GATHER_ATTN=1`` (`mlxturbo/gather_attn.py`) のときだけ
        `__call__` から呼ばれる。次のいずれかに該当すると ``None`` を返し、
        呼び出し側に通常経路 (加算マスク) を任せる — いずれの判定も
        cache を更新する前に済ませるので、``None`` を返しても
        ``idx_cache``/``cache`` は二重更新されない:

        - 疎化が要らない (``offset + S <= token_budget``)
        - ``QSAIndexer.__call__`` 末尾の早期救済に該当する
          (``offset < compress_ratio - 1``。select_blocks は非対応)
        - 量子化 KV キャッシュ (``hasattr(cache, "bits")``)。gather は
          k/v がプレーンな配列であることを前提にしている

        段 P1a (タイル分割、`docs/research/KERNEL-PROGRAM.md` 段 P1): decode
        (S=2) では union が ``S * block_topk`` で頭打ちになり、選ぶ列数が
        小さいまま済む。だが prefill のように S が大きいと (S=2048 など)、
        S クエリ全体の和集合はほぼ全ブロックになって gather の得が消える ---
        ただしこれは union がタイル幅の関数だからであって、経路そのものが
        不成立なわけではない。隣り合うクエリのブロック選択は強く相関する
        (局所窓 + 少数のグローバルブロック) ので、クエリ行を幅 ``tile`` の
        タイルに割り、タイルごとに ``_gather_tile_attn`` を呼んで union を
        取り直す。タイル幅は `_gather_attn_tile` 属性 (`mlxturbo/gather_attn.py`
        の `enable_gather_attn(..., tile=...)`、既定 0) で渡す。``0`` または
        ``S <= tile`` なら S 全体を 1 タイルとして扱う --- これは従来の
        (タイル無し) 経路とビット単位で同じ計算になる (decode の S<=8 は
        普段このまま)。新しいカーネルは要らない。`take_along_axis` + 小 bool
        マスク sdpa という既存の型をタイルの数だけ回すだけ。ただし
        **タイルごとの gather は重複読みになる** --- 同じブロックを複数の
        タイルがそれぞれ集め直すので、sdpa 呼び出し回数も列の読み出し延べ量も、
        タイル無しの 1 回呼び出しより増える (union が縮む分と天秤にかける
        代償。詳しくは `_gather_tile_attn` のコメント)。実 17k/50k での
        タイル幅 {0, 128, 256, 512} の壁時計掃引は段 P1b (別途、実モデル向け)。
        """
        if hasattr(cache, "bits"):
            return None
        cr = self.indexer.compress_ratio
        S = x.shape[1]
        if offset + S <= self.indexer.token_budget or offset < cr - 1:
            return None

        # 集めるだけの価値があるかを**ホスト側の算数だけ**で見る。
        # **キャッシュに触る前に判定すること** -- ここから先の
        # `select_blocks` と `_qkv` はどちらもキャッシュを進めるので、
        # 触ったあとに None を返すと呼び手のフォールバックが二重に更新する
        # (実際にやってみて `verify_gather_attn` が max|diff|=0.33 で落ちた)。
        #
        # union <= T*block_topk という上限は、T が小さい (decode の 2-4 行)
        # ときしか締まっていない。prefill のタイル (128-2048 行) では上限が
        # n_blocks を軽く超えるので「比 100%」としか言えない。実際の union は
        # ブロックが強く重なるので遥かに小さい (合成モデルで 1.00 -> 0.27、
        # 実機の掃引でも tile=256 が prefill -1.5%)。
        # よって**上限が締まっていて、かつそれでも比が大きいときだけ**弾く。
        #
        # 比の既定 0.20 は M3 Max の実測 (17k/25k/50k、幅 2 で
        # 比 24%/16%/8% -> +1.1%/-6.7%/-15.4%、ゼロ交差 23%) から安全側に
        # 倒した値。**マシン依存**なので `MLXTURBO_GATHER_MAX_RATIO` で
        # 上書きでき、測り直し方は docs/research/KERNEL-PROGRAM.md の段 C。
        kv_len = offset + S
        n_blocks = kv_len // cr
        tile = getattr(self, "_gather_attn_tile", 0)
        rows = tile if 0 < tile < S else S
        bound = rows * (self.indexer.token_budget // cr)
        if bound < n_blocks and bound * cr > _gather_max_ratio(self.head_dim) * kv_len:
            return None

        # 集めるだけの価値があるかを**ホスト側の算数だけ**で判定する。
        #
        # タイルごとの union はクエリ行数 T と block_topk の積で上に抑えられる
        # ので、集める列数の上限は `T * block_topk * compress_ratio`
        # = `T * token_budget`。これを kv_len で割った比が、この呼び出しで
        # 読む列の割合の上限になる。**選択を計算しなくても分かる**ので、
        # 判定に GPU の仕事は要らない。
        #
        # 比が大きいと gather は割に合わない -- 集める読み (飛び飛びなので
        # 帯域効率が悪い) + 書き + sdpa の読み、で密の 1 回読みに近づく。
        # M3 Max の実測 (17k/25k/50k、幅 2):
        #
        #   比 24% -> +1.1%   比 16% -> -6.7%   比 8% -> -15.4%
        #
        # ほぼ直線で、0 と交わるのが比 ~23%。安全側に倒して 0.20 を既定にする
        # (少し保守的だと長文で取り分を少し落とすだけだが、逆に攻めすぎると
        # 数の多い中尺のリクエストで損をする)。
        #
        # **この値はマシン依存。**分子側 (飛び飛びの読みとカーネル起動) と
        # 分母側 (連続読み) のスケールの仕方が機種で違う。別のマシンでは
        # `MLXTURBO_GATHER_MAX_RATIO` で上書きし、
        # `tools/decode_ab.py --knob gather-attn` を文脈長を振って掃引して
        # 直線が 0 と交わる比を取り直すこと (実測は M3 Max のみ)。

        blocks = self.indexer.select_blocks(x, rope, idx_cache, offset, positions)
        if blocks is None:
            return None

        q, k, v, gate = self._qkv(x, positions, rope, cache)
        B = x.shape[0]


        if tile <= 0 or S <= tile:
            tile = S  # 従来どおり S 全体を 1 タイルとして 1 回で処理する

        if tile >= S:
            out = self._gather_tile_attn(q, k, v, cache, blocks, cr, B, 0, S)
        else:
            outs = [
                self._gather_tile_attn(
                    q[:, :, t0 : t0 + tile], k, v, cache, blocks, cr, B,
                    t0, min(t0 + tile, S),
                )
                for t0 in range(0, S, tile)
            ]
            out = mx.concatenate(outs, axis=2)

        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    def __call__(self, x, rope, mask, cache, idx_cache) -> mx.array:
        B, S, _ = x.shape
        offset, positions = self._positions(cache, S)

        if getattr(self, "_gather_attn", False) and cache is not None:
            gathered = self._gather_forward(
                x, rope, cache, idx_cache, offset, positions
            )
            if gathered is not None:
                return gathered

        sparse = self.indexer(x, rope, idx_cache, offset, positions)
        q, k, v, gate = self._qkv(x, positions, rope, cache)

        mask = self._final_mask(mask, sparse, cache, S, q.dtype)

        # MLX sdpa の速い vector カーネルは q 行数 (S * n_heads) 32 まで。
        # 越えると全 KV にスコアを実体化する経路に落ち、長い文脈で跳ねる
        # (docs/research/SDPA-WIDTH-WALL.md)。マスク付きの注意は q の行どうしが
        # 独立なので、壁に収まる幅で切って繋いでも数値は同一。列を切れない
        # 文字列マスク ("causal") のときは触らない。
        gqa = self.n_heads // self.n_kv_heads
        split_mask = None
        if S * gqa > 32 and S <= 8:
            if isinstance(mask, mx.array):
                split_mask = mask
            elif mask == "causal" and k.shape[2] >= 512:
                # 文字列のままでは列を切れないので、同じ意味のブールマスクを組む
                # (q は絶対位置 offset..offset+S-1、k は 0..kv-1)。文字列 causal は
                # sdpa の最速経路なので、壁の代償が小さい短い kv では触らない
                # (kv=200 で壁 0.7ms < マスク実体化の代償、kv=1024 で壁 6.2ms)
                kv_len = k.shape[2]
                split_mask = (
                    mx.arange(kv_len)[None, :]
                    <= (offset + mx.arange(S))[:, None]
                )[None, None]
        if split_mask is not None:
            step = max(1, 32 // gqa)
            out = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        q[:, :, i : i + step], k, v, cache=cache,
                        scale=self.scale, mask=split_mask[..., i : i + step, :],
                    )
                    for i in range(0, S, step)
                ],
                axis=2,
            )
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# ------------------------------------------------------------------- gated deltanet


def _tail_window(cache, arr: mx.array, n_keep: int) -> mx.array:
    """``arr`` の axis=1 から、次に持ち越す ``n_keep`` 列を取る。

    既定は末尾。ただしバッチの prefill は右パディングで走るので末尾は
    パディングであり、行ごとの実長 ``cache.lengths`` があればその直後の
    ``n_keep`` 列を取る (上流の qwen3_next が同じことをしている)。
    ``lengths`` を持たないキャッシュでは末尾を取る従来どおりの動きで、
    単一系列の経路は 1 ビットも変わらない。

    再帰系の状態はどれもこの形に落ちる: GDN の conv 窓、PLE の short conv 窓、
    n-gram の直前文脈。3 階 (B, L, C) と 2 階 (B, L) の両方を受ける。
    """
    lengths = getattr(cache, "lengths", None)
    if lengths is None:
        return mx.contiguous(arr[:, -n_keep:])
    ends = mx.clip(lengths, 0, arr.shape[1] - n_keep)
    pos = ends[:, None] + mx.arange(n_keep)
    if arr.ndim == 3:
        pos = pos[..., None]
    return mx.take_along_axis(arr, pos, axis=1)


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        # unlike qwen3-next, the projections are split
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def _project_in(self, x):
        """4 本の入力射影。mlxturbo.fused.enable_wide_projections が
        ``_wide_in`` を置くと 1 本の qmm に連結される (数値は不変)。
        spec_flash.capture の GDN 差し替えもここを通る。"""
        wide = getattr(self, "_wide_in", None)
        if wide is None:
            return (self.in_proj_qkv(x), self.in_proj_z(x),
                    self.in_proj_b(x), self.in_proj_a(x))
        w, sc, bi, gs, bits, (c1, c2, c3) = wide
        out = mx.quantized_matmul(
            x, w, sc, bi, transpose=True, group_size=gs, bits=bits)
        return out[..., :c1], out[..., c1:c2], out[..., c2:c3], out[..., c3:]

    def _store_conv_state(self, cache, conv_input) -> None:
        """次ステップに持ち越す conv 状態 (K-1 列) を書く。窓の取り方は
        `_tail_window` (右パディングのときは実長基準)。"""
        cache[0] = _tail_window(cache, conv_input, self.conv_kernel_size - 1)

    def __call__(self, x, mask, cache) -> mx.array:
        B, S, _ = x.shape
        mixed_qkv, z, b, a = self._project_in(x)
        z = z.reshape(B, S, self.n_v, self.dv)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            self._store_conv_state(cache, conv_input)
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


# ------------------------------------------------------------------------- MoE


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        r513 = getattr(self, "_router513", None)
        if r513 is not None:
            # mlxturbo.fused.enable_moe_shared_fold: shared expert は
            # バンクの 513 番目。router の行列積 1 本から選択重みと
            # shared のゲートの両方を取る
            logits = x.astype(mx.float32) @ r513.T
            lr = logits[..., :512]
            sg = mx.sigmoid(logits[..., 512:])
            idx = mx.argpartition(-lr, self.top_k - 1, axis=-1)[..., : self.top_k]
            w = mx.softmax(mx.take_along_axis(lr, idx, axis=-1), axis=-1, precise=True)
            idx = mx.concatenate(
                [idx, mx.full((*idx.shape[:-1], 1), 512, dtype=idx.dtype)], axis=-1)
            w = mx.concatenate([w, sg], axis=-1)
            return (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        wide = getattr(self, "_wide_shared", None)
        if wide is None:
            return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        wq, sc, bi, gs, bits, h = wide
        gus = mx.quantized_matmul(
            x, wq, sc, bi, transpose=True, group_size=gs, bits=bits)
        g, u, sg = gus[..., :h], gus[..., h : 2 * h], gus[..., 2 * h :]
        shared = self.shared_expert.down_proj(nn.silu(g) * u)
        return out + mx.sigmoid(sg) * shared


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ------------------------------------------------------ hyper-connections (residual)


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None
        )

    def __call__(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject


# -------------------------------------------------------------- n-gram / PLE


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramEmbedding(nn.Module):
    """N-gram hash table, sharded into `split_ngram_parts` pieces.

    ~51B parameters: a dense lookup is never performed. Indices are sorted by shard
    on the host side, as in the llama.cpp implementation.
    """

    def __init__(self, args: TextArgs, embed_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes

        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        self.n_shards = args.split_ngram_parts
        self.rows_per_shard = math.ceil(padded / self.n_shards)
        self.ngram_embedding = _ShardedEmbedding(
            self.n_shards, self.rows_per_shard, head_dim
        )

        # buffers taken as-is from the checkpoint
        mults = []
        max_long = (1 << 63) - 1
        half = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + _PRIME_1 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(
                2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
            )
        # What we build here is only the initial value for the case where there is
        # no checkpoint. load_weights overwrites it with the tensor of the same
        # name, so what actually gets used is always that one (mlxturbo change).
        # The original port used `_`-prefixed recomputed copies in __call__, but
        # the recomputed layer_multipliers do not match the actual values in the
        # public checkpoint (vocab_sizes/offsets do match). With different
        # multipliers every hash lands somewhere else, which makes the whole
        # n-gram embedding meaningless and breaks generation.
        # int64 survives the bf16 conversion because Module.set_dtype only touches
        # floating point (confirmed to stay I64 in the converted safetensors).
        self.layer_multipliers = mx.array(mults, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)

    def _shift_right(self, ids: mx.array, shift: int) -> mx.array:
        """Shift right by `shift`, without crossing an EOS boundary."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = mx.arange(T)
        eos_pos = mx.where(ids == self.eos_token_id, pos, -1)
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=prev_incl.dtype), prev_incl[:, :-1]], axis=1
        )
        in_segment = pos[None] - (prev + 1)
        src = pos - shift
        gathered = mx.take_along_axis(
            ids, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1
        )
        ok = (in_segment >= shift) & (src[None] >= 0)
        return mx.where(ok, gathered, self.eos_token_id)

    def __call__(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        n_new = ids.shape[1]
        history = mx.concatenate([prev_context, ids], axis=1).astype(mx.int64)
        shifted = [self._shift_right(history, s) for s in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            gid = mixed[..., None] % self.ngram_heads_vocab_sizes[lo:hi].reshape(
                1, 1, -1
            )
            blocks.append(self.ngram_heads_offsets[lo:hi].reshape(1, 1, -1) + gid)

        gid = mx.concatenate(blocks, axis=-1)[:, -n_new:]
        return self.ngram_embedding(gid).reshape(*gid.shape[:2], -1)


class _ShardedEmbedding(nn.Module):
    """N embedding tables concatenated logically, addressed by global index."""

    def __init__(self, n_shards: int, rows: int, dim: int):
        super().__init__()
        self.n_shards = n_shards
        self.rows = rows
        self.dim = dim
        if NGRAM_ON_DISK:
            # mlxturbo: do not hold the table. mlxturbo.ngram_stream.install
            # swaps in a sidecar-backed implementation. Allocating it here would
            # take 102GB.
            return
        for i in range(n_shards):
            setattr(self, f"shard_{i}", nn.Embedding(rows, dim))

    def __call__(self, gid: mx.array) -> mx.array:
        if NGRAM_ON_DISK:
            raise RuntimeError(
                "n-gram がディスク運用のまま差し替えられていない。"
                "mlxturbo.ngram_stream.install(model, <サイドカー>) を呼ぶこと"
            )
        flat = gid.reshape(-1)
        shard_of = flat // self.rows
        row_of = flat % self.rows

        # which shards are actually touched: decided host-side, as llama.cpp does
        touched = np.unique(np.array(shard_of, copy=False))
        out = mx.zeros((flat.size, self.dim), dtype=mx.float32)
        for s in touched.tolist():
            sel = mx.array(np.nonzero(np.array(shard_of, copy=False) == s)[0])
            emb = getattr(self, f"shard_{s}")(mx.take(row_of, sel))
            out = mx.put_along_axis(out, sel[:, None], emb.astype(mx.float32), axis=0)
        return out.reshape(*gid.shape, self.dim)


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(args, args.ple_embed_dim, ple_layer_index)
        k = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=self.dilation,
            bias=False,
        )

    def _store_short_conv_state(self, cache, full) -> None:
        """PLE の short conv 状態を書く。窓の取り方は `_tail_window`。"""
        cache[2] = _tail_window(cache, full, self.short_conv_state_len)

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        S = x.shape[1]
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            self._store_short_conv_state(cache, full)
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def __call__(
        self, hidden: mx.array, ids: mx.array, prev_ctx: mx.array, cache
    ) -> mx.array:
        emb = self.ple_embedding(ids, prev_ctx).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        return gated + self._short_conv(self.norm_conv(gated), cache)


# ------------------------------------------------------------------- decoder / model


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_idx = (
            args.ple_layer_ids.index(layer_idx + 1)
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.ple = PLELayer(args, ple_idx) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    @staticmethod
    def _combine(hyper, x, inject):
        """hyper-connection の合成。分岐出力 x を inject で hc 本に配り、
        持ち越した hyper に足す。"""
        return hyper + (x[..., None, :] * inject[..., None]).reshape(
            *x.shape[:-1], -1
        )

    def pre_mlp(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        """PLE と attention までを進め、MoE の入力 ``(x, hyper, inject)`` を返す。

        `mlxturbo/spec_flash.py` の `_group_prefill_forward` (layer-major
        prefill) が、G チャンクぶんの x を concat して MoE を 1 回で呼ぶために
        ここで切る。あちらの本物の差分は「MoE の呼び出し粒度」だけで、層の
        中身は本家のこれを使う。
        """
        if self.ple is not None:
            h = h + self.ple(h, ids, prev_ctx, cache)

        x, hyper, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, rope, mask, cache, idx_cache)
        h = self._combine(hyper, x, inject)
        return self.mlp_hyper_connection(h)

    def __call__(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        x, hyper, inject = self.pre_mlp(
            h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx
        )
        return self._combine(hyper, self.mlp(x), inject)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # no final `norm` in this model: this mixer carries it
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, args.rope_theta)
        self.ple_layers = [
            i for i in range(args.num_hidden_layers) if (i + 1) in args.ple_layer_ids
        ]

    def _make_masks(self, h, cache):
        """``(mask, conv_mask)`` を返すシーム。

        ``mask`` は full attention 用、``conv_mask`` は再帰系 (GDN) の入力から
        列を落とすためのもので、単一系列では要らない (None)。バッチ経路
        (`mlxturbo/batch.py`) は左パディングの配列マスクと、右パディングを
        再帰状態から外す conv_mask を返すためにここを差し替える。
        """
        full_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "full_attention"
        ]
        attn_cache = cache[full_idx[0]] if full_idx else None
        mask = create_attention_mask(
            h, [attn_cache] if attn_cache is not None else None
        )
        return mask, None

    def _store_ngram_ctx(self, pc, cat, ctx_len: int) -> None:
        """n-gram の直前文脈 (ctx_len トークン) を書く。窓の取り方は
        `_tail_window`。"""
        pc[3] = _tail_window(pc, cat, ctx_len)

    def _prelude(self, ids, h, cache):
        """層ループの前段。``(mask, conv_mask, prev_ctx)`` を返す。

        `mlxturbo/spec_flash.py` の `_staged_forward` (2 層ごとに async_eval を
        挟む段階投入版) と共有する。あちらの本物の差分はループ骨格だけで、
        前段は本家と同一なので、ここから呼ばせる。
        """
        mask, conv_mask = self._make_masks(h, cache)

        prev_ctx = None
        if self.ple_layers:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            pc = cache[self.ple_layers[0]]
            prev = pc[3] if pc is not None else None
            prev_ctx = (
                prev
                if prev is not None
                else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            )
            if pc is not None:
                self._store_ngram_ctx(
                    pc, mx.concatenate([prev_ctx, ids], axis=1), ctx_len
                )
        return mask, conv_mask, prev_ctx

    def __call__(self, ids: mx.array, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        mask, conv_mask, prev_ctx = self._prelude(ids, h, cache)

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache):
            idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
            h = layer(h, self.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        return self.hyper_connection_mixer(h)


class _IndexerCache(_BaseCache):
    """Holds the indexer raw keys (one per token, not pooled).

    KVCache と同じくブロック単位で確保して埋める。以前は毎更新
    ``mx.concatenate`` で、ラウンドごとに全長を読み書きし直していた
    (17k で 1 層 4.3MB x 12 層 = 52MB/フォワード)。値はビット不変。

    ``keys`` は**論理配列** (offset までの view) を返す。呼び手
    (``spec_flash.trim_attn_cache``、``batch.py`` の ``_trim_indexer``、
    ``batch_spec`` の compaction、検証ツール) はどれも「今ある鍵そのもの」を
    期待していて、確保済みバッファの尻を見せると静かに 0 を掴む。KVCache は
    ``keys`` が生バッファで ``state`` が view という逆の約束なので、そちらの
    書き方をそのまま持ってこないこと。

    段 X1 (``docs/research/KERNEL-PROGRAM.md``): rope 済み pooled キーも
    ここで持ち回る (``_pooled``、確定済みブロック数は ``_pooled_n``)。
    ``update`` (通常の追記) だけが増分で伸ばす。``keys`` の setter は
    trim / rollback / batch の filter・extend・extract・merge・state 復元の
    どれもが通る「外から論理配列を差し込む」経路で、縮み・並べ替えの
    どちらもあり得るので、そこでは無条件に pooled を捨てて作り直す
    (古いブロックを静かに使い回すよりは、作り直しの分だけ遅い方がまし)。
    """

    step = 256

    def __init__(self):
        self._buf = None
        self.offset = 0
        self._pooled = None
        self._pooled_n = 0

    @property
    def keys(self):
        return None if self._buf is None else self._buf[:, : self.offset]

    @keys.setter
    def keys(self, v):
        # 外から論理配列を差し込まれたら、それをそのままバッファにする
        # (確保の余裕は失うが、次の update で取り直す)
        self._buf = v
        self.offset = 0 if v is None else v.shape[1]
        # 縮み・並べ替えの可能性がある経路 (この setter しかない)。
        # pooled はブロック分割に対応しているので、古いものを引きずるくらい
        # なら丸ごと捨てて次回に作り直す。
        self._pooled = None
        self._pooled_n = 0

    def pooled(self, n_blocks: int, make_new) -> mx.array:
        """rope 済み pooled キーを、確定済みブロックは使い回して返す。

        ``make_new(start, end)`` は新規に確定したブロック範囲
        ``[start, end)`` の rope 済み pooled (``(B, end-start, head_dim)``) を
        計算するコールバック。``n_blocks`` が既知より減っていたら
        (``keys`` の setter を経由しない縮みは無いはずだが、念のため)
        捨てて作り直す。
        """
        if self._pooled is not None and self._pooled_n > n_blocks:
            self._pooled = None
            self._pooled_n = 0
        if self._pooled_n < n_blocks:
            new = make_new(self._pooled_n, n_blocks)
            self._pooled = (
                new
                if self._pooled is None
                else mx.concatenate([self._pooled, new], axis=1)
            )
            self._pooled_n = n_blocks
        return self._pooled

    def update(self, k: mx.array) -> mx.array:
        prev = self.offset
        if self._buf is None or prev + k.shape[1] > self._buf.shape[1]:
            B, S, D = k.shape
            n_steps = (self.step + S - 1) // self.step
            new = mx.zeros((B, n_steps * self.step, D), k.dtype)
            if self._buf is None:
                self._buf = new
            else:
                self._buf = mx.concatenate([self._buf[:, :prev], new], axis=1)
        self.offset = prev + k.shape[1]
        self._buf[:, prev : self.offset] = k
        return self._buf[:, : self.offset]

    @property
    def state(self):
        return self.keys

    @state.setter
    def state(self, v):
        self.keys = v


class _AttnCache(KVCache):
    def __init__(self):
        super().__init__()
        self.indexer = _IndexerCache()


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i, t in enumerate(self.args.text.layer_types):
            if t == "full_attention":
                caches.append(_AttnCache())
            else:
                # 0: deltanet conv, 1: ssm state, 2: PLE conv, 3: n-gram context
                caches.append(ArraysCache(4))
        return caches

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("model.language_model."):
                # The public checkpoint has Qwen4ExpForConditionalGeneration (VLM)
                # shape, with the text side under model.language_model.*. The MLX
                # Model expects model.*, so fold away the intermediate
                # language_model.
                k = "model." + k[len("model.language_model.") :]
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            if k.startswith("vision_tower.") or k.startswith("model.visual."):
                continue  # text-only pour l'instant
            if NGRAM_ON_DISK and "ngram_embedding.shard_" in k:
                # mlxturbo: in on-disk mode, don't keep this in the model itself
                continue
            if k.startswith("mtp."):
                # mlxturbo: MTP is extracted into a sidecar and loaded separately
                # (see the header)
                continue
            if k.endswith("mlp.experts.gate_up_proj"):
                # The ckpt is fused, (E, 2*moe_inter, H). As with the Qwen3-Next
                # family in transformers, the convention is to chunk(2) the linear
                # output, so the first half of the rows is gate and the second
                # half is up. MLX's SwitchGLU holds the two as separate matrices.
                base = k[: -len("experts.gate_up_proj")] + "switch_mlp."
                gate, up = mx.split(v, 2, axis=1)
                out[base + "gate_proj.weight"] = gate
                out[base + "up_proj.weight"] = up
                continue
            if k.endswith("mlp.experts.down_proj"):
                base = k[: -len("experts.down_proj")] + "switch_mlp."
                out[base + "down_proj.weight"] = v
                continue
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                # (C, 1, K) torch -> (C, K, 1) mlx
                if v.shape[1] == 1:
                    v = v.transpose(0, 2, 1)
            out[k] = v
        return out

    @property
    def quant_predicate(self):
        def fn(path, module, _):
            # only the MoE router stays in full precision (norms and conv1d are
            # never quantized anyway)
            return not path.endswith("mlp.gate")

        return fn
