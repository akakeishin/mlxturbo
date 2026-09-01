"""バッチ x 投機 (B 本同時の MTP 投機デコード) の中核部品。

## 設計 (advisor 監査で確定済み。詳細な理由はここに畳む)

同期ラウンド: B 本全員が同じラウンドで (B, T+1) の verify フォワードを 1 回
踏む。行ごとに受理数 ``keep_b`` (1..T+1) が違う。

full attention 側は行別 trim をしない。全行を一律 T+1 スロット前進させ、
不採用位置は **dead slot** として、以後のフォワードで mask により恒久的に
除外する (物理的には残るが二度と見えない)。ずれは各ラウンドあたり最大 T
なので、paged KV のような複雑なメモリ管理は要らない -- 帳簿 (``RaggedLedger``)
が「行ごとにどの物理列が生きているか」を持ち、次ラウンドのマスクを組む。

GDN / PLE / n-gram 側は逆に、行別 **take** で済ませる。これらは
``mlxturbo.spec_flash.capture()`` が集める ``states_all`` (B, T+1, ...) /
``conv_input`` から、行 b は位置 ``keep_b-1`` (または ``keep_b`` からの窓) を
取り出せばよい -- 既存の単一行 ``rollback()`` (mlxturbo/spec_flash.py) の
per-row 一般化。attention 側と違って「巻き戻し」そのものは要らない
(dead な過去を引きずらない設計そのものなので)。

## アーキテクチャ非依存の切り方 (docs/BACKLOG.md 2026-09-01 追記)

GDN (Gated DeltaNet) は Flash-Next 独自ではない — 線形注意/再帰系は
Qwen3.8-27B のハイブリッドにも GLM-5.3-Flash の KDA にもあり、今後の主流側。
そこで「再帰状態を持つ層を行別 take で戻す」は族として汎用の部品として書き、
qwen4_exp に閉じるのは状態テンソルの形とフックの当て先だけにする。このモジュールの
``RaggedLedger``/``batched_rollback`` は「trimmable な KV」と「untrimmable な
ArraysCache (行ごとに take できる状態を持つキャッシュ)」という一般的な
キャッシュ種別に対する操作として書いてあり、qwen4_exp のクラス名や
フィールド名には (``GatedDeltaNet``/``PLELayer`` の内部構造を除いて) 依存
しない。

唯一の例外が ``ragged_attention()``: RoPE の位置と mask を行ごとに正しく
供給するには ``Attention.__call__`` を差し替える必要がある。これは
``mlxturbo/spec_flash.py`` の ``capture()`` が ``GatedDeltaNet.__call__`` を
差し替えるのと同じ「capture 相当のフック」であって、qwen4_exp 版の
Attention 実装がまだ行別 (配列) offset を受け付けないから要るだけ
(``mlxturbo/batch.py`` が全く同じ理由で ``Attention.__call__`` を差し替えて
いる -- 継続バッチングの左パディングとは前提が違うので、そちらの実装は
再利用せず独立に書いた)。他アーキテクチャに移植する際は、そちらの
Attention 実装が同じフックを必要とするかどうかを個別に確認すること。

## indexer / QSA はバッチ対象外 (既知の未対応)

QSA のブロック選択は絶対列位置で切られる。dead slot を挟んだバッチ列で
これを正しく動かすには、行ごとにブロック境界を作り直す必要があり
(``mlxturbo/batch.py`` の "Remaining limitation" と同種の問題)、今回はやって
いない。このモジュールは ``indexer_budget`` を kv 長が超えない構成でだけ
正しく動く (超えた場合 ``ragged_attention()`` は ``NotImplementedError`` で
落ちる -- 黙って間違った結果を返すよりはっきり止める方を選んだ)。
``mlxturbo/batch.py`` の "solo tier" と同じ考え方で、QSA が活性化しうる
リクエストは常にこの機構の対象外にすること。

## 未対応 (このモジュールの範囲外)

- サーバー配線・admission・スケジューラは書いていない (依頼どおり)。
- indexer / QSA (上記)。
- サンプリング (温度 > 0) の分布保証はここでは検証していない (貪欲のみ)。
  spec_flash.py 側の議論 (位置ごとの分布は独立) はそのまま当てはまるはずだが
  未確認。

## 検証

正しさは ``tools/verify_batch_spec.py`` (合成モデル、CPU のみ) にある。
結果 (誤差値) はそちらのファイル冒頭に記録。
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx


def _arch():
    import mlx_lm.models.qwen4_exp as Q

    return Q


# --------------------------------------------------------------- ledger


class RaggedLedger:
    """B 行の物理長 (全行共通) と行別の dead slot 帳簿。

    表現: ``self._alive[b]`` は行 b の各物理列が生きているか (bool) の
    Python リスト。列数はどの行も ``self.L`` で共通 (毎ラウンド T+1 だけ
    全行一律に伸びる)。``self._valid_len[b]`` は行 b の論理長 (dead を除いた
    本当の系列長 = RoPE の位置として使う値)。

    ブックキーピングは全て Python 側 (CPU) -- B・ラウンド数は小さく、GPU 側の
    ホットパスではない (実際 compaction は「たまに」呼ぶ想定で、ここを速く
    する動機がない)。
    """

    def __init__(self, batch_size: int):
        self.B = batch_size
        self.L = 0
        self._alive: list[list[bool]] = [[] for _ in range(batch_size)]
        self._valid_len: list[int] = [0] * batch_size

    def valid_len_array(self) -> mx.array:
        """行ごとの論理長 (B,)。RoPE の位置 (= このラウンドの新規列の
        位置は ``valid_len + arange(T)``) と mask 生成の両方に使う。"""
        return mx.array(self._valid_len)

    def next_round_mask(self, T: int) -> mx.array:
        """次に足す T 列を含めた bool マスク (B, 1, T, L+T) を返す。
        True = 見える。

        構成は 2 ブロックの結合:
          - 既存の L 列: 行ごとに alive かどうかだけで決まる (dead は
            ラウンドをまたいで恒久的に不可視)
          - 今回の新規 T 列: 通常の因果マスク (verify 対象の T+1 列同士は
            まだ受理/棄却が決まっていないので、単純な下三角)
        """
        B, L = self.B, self.L
        if L:
            prev = mx.array(self._alive, dtype=mx.bool_)  # (B, L)
            prev = mx.broadcast_to(prev[:, None, :], (B, T, L))
        else:
            prev = mx.zeros((B, T, 0), dtype=mx.bool_)
        causal = mx.arange(T)[None, :] <= mx.arange(T)[:, None]  # (T, T)
        new = mx.broadcast_to(causal[None], (B, T, T))
        return mx.concatenate([prev, new], axis=-1)[:, None]

    def commit_round(self, keeps: list[int], total: int) -> None:
        """検証後、行 b は先頭 ``keeps[b]`` 列 (1..total) を alive として
        確定し、残りは dead のまま恒久的に不可視にする。``total`` はこの
        ラウンドで実際に新規追加した列数 (verify なら T+1、prefill なら
        プロンプト長 -- prefill は「全行 100% 受理の 1 ラウンド」として
        同じ扱いで表現できる)。
        """
        for b, keep in enumerate(keeps):
            assert 1 <= keep <= total, (keep, total)
            self._alive[b].extend([True] * keep + [False] * (total - keep))
            self._valid_len[b] += keep
        self.L += total

    def compact(self, model, caches) -> None:
        """dead slot を物理的に詰める (mx.take_along_axis で KV/indexer を
        詰め、帳簿をリセット)。

        全行の物理長を ``max(valid_len)`` に揃える。短い行は詰めたあとも
        差分だけ「左側が不可視」という単純な形で残る (mlxturbo/batch.py の
        左詰めパディングと同じ収束先 -- ここでは compaction 後の帳簿が
        「どこまでが本当に有効か」という一点だけになる、というだけの意味で
        合流していて、batch.py のクラス自体は使っていない)。attention の
        KV/indexer だけを触る -- GDN/PLE/n-gram 側は dead な過去を最初から
        引きずらない設計なので compaction の対象にならない。
        """
        Q = _arch()
        new_L = max(self._valid_len) if self._valid_len else 0
        gather_idx = []
        for b in range(self.B):
            alive_cols = [i for i, a in enumerate(self._alive[b]) if a]
            assert len(alive_cols) == self._valid_len[b]
            pad = new_L - len(alive_cols)
            # pad 分のダミー列インデックス (0) は mask 側で常に不可視になる
            # ので値そのものはどうでもよい
            gather_idx.append([0] * pad + alive_cols)
        idx = mx.array(gather_idx)  # (B, new_L)

        for layer, c in zip(model.model.layers, caches):
            if layer.layer_type != "full_attention":
                continue
            c.keys = _gather_cols(c.keys, idx, axis=2)
            c.values = _gather_cols(c.values, idx, axis=2)
            if c.indexer.keys is not None:
                c.indexer.keys = _gather_cols(c.indexer.keys, idx, axis=1)

        self.L = new_L
        self._alive = [
            [False] * (new_L - self._valid_len[b]) + [True] * self._valid_len[b]
            for b in range(self.B)
        ]


def _gather_cols(arr: mx.array, idx: mx.array, axis: int) -> mx.array:
    """``idx`` (B, new_L) で指定した列だけを ``axis`` から集める。バッチ軸
    (axis 0) 以外は take_along_axis の暗黙ブロードキャストに任せる。"""
    shape = [1] * arr.ndim
    shape[0] = idx.shape[0]
    shape[axis] = idx.shape[1]
    return mx.take_along_axis(arr, idx.reshape(shape), axis=axis)


# --------------------------------------------------------- attention cache


class RaggedAttnCache:
    """full attention 用のバッチキャッシュ。

    ``mlxturbo/batch.py`` の ``BatchAttnCache`` (継続バッチング用、不揃いな
    プロンプト長を左パディングで揃える) とは前提が違うので継承しない --
    ここでは物理長が毎ラウンド全行同じだけ伸びる (dead slot はラウンドの
    "中" に穴を作らず、末尾に作るだけ)。行別の可視性・論理位置は全て
    ``RaggedLedger`` に一元化してあり、このクラスは物理列を積むだけ。
    """

    def __init__(self, ledger: RaggedLedger):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.ledger = ledger
        self.indexer = _arch()._IndexerCache()

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values

    def size(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    @property
    def offset(self) -> mx.array:
        """(B,) の論理位置。帳簿から毎回引く (手で同期する箇所を作らない)。"""
        return self.ledger.valid_len_array()

    def is_trimmable(self) -> bool:
        # 行別 trim をしない設計そのもの (dead slot が代わりに扱う)
        return False


def make_ragged_cache(model, batch_size: int):
    """``model.make_cache()`` のバッチ版。full attention 層には
    ``RaggedAttnCache`` (帳簿を共有)、それ以外には無変更の ``ArraysCache(4)``
    を積む -- GDN/PLE/n-gram のテンソルは先頭次元がバッチ次元になるだけで、
    キャッシュクラス自体は継続バッチング側と同じく手を入れる必要がない。
    """
    from mlx_lm.models.cache import ArraysCache

    ledger = RaggedLedger(batch_size)
    caches = []
    for lt in model.args.text.layer_types:
        if lt == "full_attention":
            caches.append(RaggedAttnCache(ledger))
        else:
            caches.append(ArraysCache(4))
    return caches, ledger


# ------------------------------------------------------------ attention hook


@contextmanager
def ragged_attention():
    """このラウンドの検証フォワード限定で ``Attention.__call__`` を差し替える。

    本家 (``mlxturbo/_vendor/qwen4_exp.py`` の ``Attention.__call__``) の写しで、
    異なるのは 2 点だけ:

    1. RoPE の位置: ``cache.offset`` を (B,) の論理位置として扱う (本家は
       python int の物理列位置)。``RaggedAttnCache.offset`` が
       ``RaggedLedger.valid_len_array()`` を返すので、ここで区別は要らない --
       ``rope(offset[:, None] + arange(S))`` を常に使うだけでよい。
    2. mask: ``Qwen4ExpModel.__call__`` が渡す ``mask`` 引数は
       ``create_attention_mask(h, [attn_cache])`` の結果で、単一キャッシュで
       はなくリストを渡すため必ず "causal"/None に潰れる
       (mlxturbo/batch.py の docstring 項目 4 と同じ理由)。dead slot を
       知っているのは ``cache.ledger`` だけなので、渡された mask は無視して
       ``cache.ledger.next_round_mask(S)`` で組み直す。

    QSA (indexer) は本家のまま素通しする。このモジュールが対象にする構成は
    kv 長が ``indexer_budget`` を超えないことが前提 (モジュール docstring の
    "indexer / QSA はバッチ対象外" を参照) で、その前提の下では
    ``QSAIndexer.__call__`` は必ず ``None`` を返す (早期 return は
    ``raw_k.shape[1]`` だけで決まり、offset の型に依存しない)。前提が破れて
    ``sparse`` が None でなくなった場合は、黙って間違えるより
    ``NotImplementedError`` で止める。
    """

    Q = _arch()
    orig = Q.Attention.__call__

    def call(self, x, rope, mask, cache, idx_cache):
        B, S, _ = x.shape
        offset = cache.offset  # (B,) 論理位置
        sparse = self.indexer(x, rope, idx_cache, cache.size())
        if sparse is not None:
            raise NotImplementedError(
                "QSA (indexer) はこのモジュールの対象外 -- indexer_budget を "
                "kv 長が超えない構成でのみ使うこと (モジュール docstring参照)"
            )

        qg = self.q_proj(x)
        q, gate = mx.split(qg.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        positions = offset[:, None] + mx.arange(S)[None, :]  # (B, S)
        cos, sin = rope(positions)
        cos, sin = cos[:, None], sin[:, None]
        q, k = Q._rope_partial(q, cos, sin), Q._rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        ledger_mask = cache.ledger.next_round_mask(S)
        out = Q.scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=ledger_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    Q.Attention.__call__ = call
    try:
        yield
    finally:
        Q.Attention.__call__ = orig


@contextmanager
def batched_capture(model):
    """1 ラウンドぶんの検証フォワードに要るフックをまとめて張る:
    ``mlxturbo.spec_flash.capture()`` (GDN/PLE/GatedResidual、無変更で再利用)
    と ``ragged_attention()`` (このモジュール、full attention 側) を同時に。
    """
    from .spec_flash import capture

    with capture(model) as cap, ragged_attention():
        yield cap


# -------------------------------------------------------------- rollback


def _take_rows(arr: mx.array, starts: mx.array, width: int) -> mx.array:
    """``arr`` の axis=1 から、行 b ごとに ``[starts[b], starts[b]+width)`` を
    切り出す。``mlxturbo.spec_flash.rollback`` の単一行スライス
    (``x[:, keep:keep+w, :]`` 系) を行別に一般化した共通部品 -- GDN の
    conv window、GDN の状態 (width=1)、PLE の conv window、n-gram の文脈が
    全てこの形に落ちる。
    """
    idx = starts[:, None] + mx.arange(width)[None, :]  # (B, width)
    shape = [idx.shape[0], width] + [1] * (arr.ndim - 2)
    return mx.take_along_axis(arr, idx.reshape(shape), axis=1)


def snapshot_pre_ctx(model, caches) -> list:
    """検証フォワードの**前**に、n-gram 文脈 (``cache[3]``) をレイヤーごとに
    控えておく。``mlxturbo.spec_flash.snapshot_pre`` の相当品だが、こちらは
    attention 側 (``pre["kv"]``) を持たない -- 行別 trim をしない設計なので
    KV 側に「戻すために控える」ものが無いため。
    """
    return [
        (None if layer.layer_type == "full_attention" else c[3])
        for layer, c in zip(model.model.layers, caches)
    ]


def batched_rollback(model, caches, cap, keeps: list[int], pre_ctx=None, pair=None) -> None:
    """(B, T+1) の verify 後、GDN/PLE/n-gram 系キャッシュを行別 keep に戻す。

    ``mlxturbo.spec_flash.rollback`` が単一行に対してやっている 4 対象
    (GDN 状態・GDN conv window・PLE conv window・n-gram 文脈) を、
    ``keeps[b]`` (行 b の受理数、1..total) ごとに ``_take_rows`` で一般化した
    だけ。attention 側は一切触らない -- dead slot として ``RaggedLedger`` が
    扱う設計そのものなので、呼び出し側で別途 ``ledger.commit_round(keeps,
    total)`` を呼ぶこと (ここでは呼ばない: このモジュールの他の部品と同じく
    「1 つのことだけをする」ほうを選んだ)。

    ``pre_ctx``/``pair`` を渡さない場合は n-gram 文脈の更新をスキップする
    (``mlxturbo.spec_flash.rollback`` が ``ids_kept=None`` のとき ``c[3]`` を
    更新しないのと同じ扱い)。
    """
    keep_arr = mx.array(keeps)
    for layer, c in zip(model.model.layers, caches):
        if layer.layer_type == "full_attention":
            continue
        la = layer.linear_attn
        conv_input, states_all = cap.gdn[id(la)]
        win = la.conv_kernel_size - 1
        c[0] = _take_rows(conv_input, keep_arr, win)
        c[1] = _take_rows(states_all, keep_arr - 1, 1)[:, 0]
        if layer.ple is not None:
            full = cap.ple.get(id(layer.ple))
            if full is not None:
                n = layer.ple.short_conv_state_len
                c[2] = _take_rows(full, keep_arr, n)

    if pre_ctx is None or pair is None:
        return
    ctx_len = model.args.text.ngram_size - 1
    for layer, c, ctx in zip(model.model.layers, caches, pre_ctx):
        if layer.layer_type == "full_attention" or ctx is None:
            continue
        cat = mx.concatenate([ctx, pair], axis=1)
        c[3] = _take_rows(cat, keep_arr, ctx_len)


__all__ = [
    "RaggedLedger",
    "RaggedAttnCache",
    "make_ragged_cache",
    "ragged_attention",
    "batched_capture",
    "snapshot_pre_ctx",
    "batched_rollback",
]
