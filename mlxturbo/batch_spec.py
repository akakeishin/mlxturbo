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

## サーバー配線

このファイルの末尾「coordinator」節にある (``--max-batch-spec``)。admission と
スケジューラの決めごと 4 点はそこに書いてある。走行中のバッチに後から行を
足すことだけはやっていない (理由も同じ節)。

## 未対応 (このモジュールの範囲外)

- indexer / QSA (上記)。
- サンプリング (温度 > 0) の分布保証は実測していない。全位置を先に引く形は
  spec_flash.py 側の議論 (位置ごとの分布は独立) がそのまま当てはまり、
  ``BatchSpecGenerator._sample_rows`` はその B 行版だが、KLD などで確かめた
  わけではない。

## 検証

正しさは ``tools/verify_batch_spec.py`` (合成モデル、CPU のみ) にある。
結果 (誤差値) はそちらのファイル冒頭に記録。
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx

from . import arch as _archmod
from .arch import qwen4_arch as _arch


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
        self._prefill_lengths: list[int] | None = None
        # round_mask の memo (キーは T)。同じラウンドの中では帳簿が動かないので
        # 全 full attention 層 (このモデルで 12 層) が同じ配列を使い回せる。
        # memo が無いと層ごとに (B, L) の Python リストを mx.array に変換し直す
        # ことになり、L が伸びるほど decode の 1 ラウンドが重くなる (B=1 でも
        # 効くので、単独リクエストの無劣化に直接効く)。帳簿を動かす操作
        # (commit_round / compact / filter_rows / prefill_lengths の付け外し) が
        # 全て _invalidate を通る。
        self._mask_memo: dict[int, mx.array] = {}

    def _invalidate(self) -> None:
        self._mask_memo.clear()

    @property
    def prefill_lengths(self) -> "list[int] | None":
        """prefill の 1 回だけ立てる。右パディングで流すので、そのラウンドの
        マスクは「因果 かつ 実長より手前」であって、通常ラウンドの
        「生きている過去 + 因果」とは別物 (round_mask 参照)。"""
        return self._prefill_lengths

    @prefill_lengths.setter
    def prefill_lengths(self, value: "list[int] | None") -> None:
        self._prefill_lengths = value
        self._invalidate()

    def max_valid_len(self) -> int:
        """行の論理長の最大 (= compaction 後の物理列数)。"""
        return max(self._valid_len) if self._valid_len else 0

    def filter_rows(self, rows: list[int]) -> None:
        """``rows`` の行だけを残す (帳簿側)。キャッシュ側は
        ``RaggedAttnCache.filter_rows`` / ``ArraysCache.filter`` が同じ添字で
        揃えること (``BatchSpecGenerator.retire`` が両方を呼ぶ)。"""
        self.B = len(rows)
        self._alive = [self._alive[b] for b in rows]
        self._valid_len = [self._valid_len[b] for b in rows]
        if self._prefill_lengths is not None:
            self._prefill_lengths = [self._prefill_lengths[b] for b in rows]
        self._invalidate()

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

    def round_mask(self, T: int) -> mx.array:
        """このラウンドのフォワードに渡す bool マスク (B, 1, T, L+T)。

        prefill 中 (``prefill_lengths`` が立っている) は、まだ過去が無く、
        右パディングを隠す必要があるので「因果 かつ 実長より手前」。
        それ以外は ``next_round_mask``。

        同じラウンドの中では帳簿が動かないので、T ごとに 1 回だけ作って
        memo から返す (__init__ の _mask_memo を参照)。
        """
        got = self._mask_memo.get(T)
        if got is not None:
            return got
        if self._prefill_lengths is None:
            out = self.next_round_mask(T)
        else:
            lens = mx.array(self._prefill_lengths)
            cols = mx.arange(T)
            causal = cols[None, :] <= cols[:, None]  # (T, T)
            valid = cols[None, :] < lens[:, None]  # (B, T)
            out = (causal[None] & valid[:, None, :])[:, None]
        self._mask_memo[T] = out
        return out

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
        self._invalidate()

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
        self._invalidate()


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

    def round_mask(self, T: int) -> mx.array:
        """このラウンドの mask。``ragged_attention`` の ``_final_mask`` から
        呼ばれる (MTP のドラフトキャッシュも同じ名前を持つので、Attention 側は
        キャッシュの種類を知らなくてよい)。"""
        return self.ledger.round_mask(T)

    @property
    def offset(self) -> mx.array:
        """(B,) の論理位置。帳簿から毎回引く (手で同期する箇所を作らない)。"""
        return self.ledger.valid_len_array()

    def is_trimmable(self) -> bool:
        # 行別 trim をしない設計そのもの (dead slot が代わりに扱う)
        return False

    def filter_rows(self, idx: mx.array) -> None:
        """バッチ軸 (axis 0) から ``idx`` の行だけを残す。列の意味は行ごとに
        独立なので、列側の詰め直しは要らない (残る行の物理列はそのまま)。
        帳簿は共有物なので、ここでは触らず ``BatchSpecGenerator.retire`` が
        1 回だけ ``RaggedLedger.filter_rows`` を呼ぶ。"""
        if self.keys is not None:
            self.keys = mx.contiguous(self.keys[idx])
            self.values = mx.contiguous(self.values[idx])
        if self.indexer.keys is not None:
            self.indexer.keys = mx.contiguous(self.indexer.keys[idx])


class RaggedDraftCache:
    """MTP ドラフト用のバッチ KV キャッシュ。行ごとの論理位置を持つ。

    列は全行そろって伸びる。priming は**各行の末尾に揃えた一定幅** w で行う
    (w = min(PRIME_WINDOW, 最短プロンプト長-1))。こうすると全列が実データに
    なり、パディング列を隠す必要が無くなる。

    ``offset`` は本家 Attention の ``_positions`` シームが読む値なので、
    (B,) の配列を返す。

    受理数が行ごとに違っても列の進み方は変わらない (`_draft_chain` は
    ラウンドごとに 1 列だけ残す) ので、行がずれることは無い。
    """

    def __init__(self, batch_size: int):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        # base は全行 0。MTP ヘッドの位置は**priming 窓の先頭からの相対**で、
        # 本物のトークン位置ではない (単一系列の `_prime_draft_cache` も空の
        # キャッシュから始めて 0.. と数える)。窓幅を全行そろえれば列の意味も
        # そろうので、行ごとのずれは生じない。配列で持つのは
        # `Attention._positions` のシームが (B,) を期待するからで、
        # 単一系列と同じ角度になる。
        self._base = mx.zeros(batch_size, dtype=mx.int32)
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
        return self._base + self.size()

    def round_mask(self, T: int):
        """全列が実データなので、必要なのは新規 T 列どうしの因果性だけ。
        T==1 (ドラフトは 1 段ずつ) なら mask 自体が要らない。"""
        if T == 1:
            return None
        total = self.size()
        cols = mx.arange(total)
        q = total - T + mx.arange(T)
        return (cols[None, :] <= q[:, None])[None, None]

    def trim(self, n: int) -> int:
        """末尾 n 列を落とす。

        **切り出しは実体化する (mx.contiguous)。**遅延スライスのまま次の
        ラウンドで concat すると、その先のグラフが壊れた形で評価される
        (実測: ドラフト連鎖 2 周目の sdpa で q の head_dim が半分になり
        `[matmul] ... (1,2,2,1,8) と (1,2,1,7,16)` で落ちた。落ちるのは
        評価時なので、traceback は無関係な場所を指す)。本家の KVCache は
        offset を減らすだけでスライスしないので、この罠を踏まない。
        """
        n = min(self.size(), n)
        if n:
            keep = self.size() - n
            self.keys = mx.contiguous(self.keys[:, :, :keep])
            self.values = mx.contiguous(self.values[:, :, :keep])
        return n

    def is_trimmable(self) -> bool:
        return True

    def filter_rows(self, idx: mx.array) -> None:
        """バッチ軸から ``idx`` の行だけを残す。列は全行そろって伸びるので
        (どの行も priming 窓 + ラウンド数)、列側は触らなくてよい。"""
        if self.keys is not None:
            self.keys = mx.contiguous(self.keys[idx])
            self.values = mx.contiguous(self.values[idx])
        if self.indexer.keys is not None:
            self.indexer.keys = mx.contiguous(self.indexer.keys[idx])
        self._base = self._base[idx]


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
    """このラウンドの検証フォワード限定で ``Attention`` の 2 つのシームを
    差し替える。フォワード本体は本家 (``mlxturbo/_vendor/qwen4_exp.py``) の
    ままで、違うのは次の 2 点だけ。

    1. RoPE の位置 (``_positions``): ``cache.offset`` を (B,) の論理位置として
       扱う (本家は python int の物理列位置)。``RaggedAttnCache.offset`` が
       ``RaggedLedger.valid_len_array()`` を返すので、ここでは
       ``offset[:, None] + arange(S)`` を常に使うだけでよい。QSA に渡す列位置は
       ``cache.size()`` (物理列数) のまま。
    2. mask (``_final_mask``): ``Qwen4ExpModel.__call__`` が渡す ``mask`` 引数は
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
    orig_positions = Q.Attention._positions
    orig_final_mask = Q.Attention._final_mask
    orig_make_masks = Q.Qwen4ExpModel._make_masks

    def make_masks(self, h, cache):
        """prefill (右パディング) の間だけ conv_mask を立てる。

        再帰系 (GDN) はパディング列を状態に取り込んでしまうので、入力から
        落とす必要がある。窓の取り出し (`_tail_window`) だけでは足りない --
        あちらは「どの列を持ち越すか」で、こちらは「どの列を計算に入れるか」。
        """
        mask, conv_mask = orig_make_masks(self, h, cache)
        led = next((c.ledger for c in cache if hasattr(c, "ledger")), None)
        if led is not None and led.prefill_lengths is not None:
            lens = mx.array(led.prefill_lengths)
            conv_mask = mx.arange(h.shape[1])[None] < lens[:, None]
        return mask, conv_mask

    def positions(self, cache, S):
        # 列位置 (QSA 用) は物理列数、rope の位置は行別の論理位置。
        # RaggedAttnCache と RaggedDraftCache が同じ形を返すので、
        # Attention はキャッシュの種類を知らなくてよい
        return cache.size(), cache.offset[:, None] + mx.arange(S)[None, :]

    def final_mask(self, mask, sparse, cache, S, dtype):
        if sparse is not None:
            raise NotImplementedError(
                "QSA (indexer) はこのモジュールの対象外 -- indexer_budget を "
                "kv 長が超えない構成でのみ使うこと (モジュール docstring参照)"
            )
        return cache.round_mask(S)

    Q.Attention._positions = positions
    Q.Attention._final_mask = final_mask
    Q.Qwen4ExpModel._make_masks = make_masks
    try:
        yield
    finally:
        Q.Attention._positions = orig_positions
        Q.Attention._final_mask = orig_final_mask
        Q.Qwen4ExpModel._make_masks = orig_make_masks


@contextmanager
def batched_capture(model, light: bool = False):
    """1 ラウンドぶんの検証フォワードに要るフックをまとめて張る:
    ``mlxturbo.spec_flash.capture()`` (GDN/PLE/GatedResidual、無変更で再利用)
    と ``ragged_attention()`` (このモジュール、full attention 側) を同時に。

    ``light`` は素通しで ``capture(model, light=light)`` に渡る。既定の
    ``light=False`` (full capture) は ``states_all`` ((B, T, Hv, Dv, Dk) fp32、
    層あたり ~3MiB/token) を 36 層ぶん確保するため、verify 幅 (T<=2 程度) では
    無視できても **prefill 幅 (T がプロンプト長やチャンク幅) で使うと macOS の
    memorystatus killer にプロセスごと消される** (`spec_flash.capture` の
    docstring 参照。実測済みの既知の壊れ方で、B 行ぶん確保するバッチはさらに
    悪化する)。prefill 幅のフォワードをこの関数越しに流す呼び手は
    ``light=True`` を渡すこと (F1、Opus 正しさレビュー指摘)。
    """
    from .spec_flash import capture

    with capture(model, light=light) as cap, ragged_attention():
        yield cap


# -------------------------------------------------------------- rollback


def snapshot_pre_ctx(model, caches) -> list:
    """検証フォワードの**前**に、n-gram 文脈をレイヤーごとに控えておく。
    ``mlxturbo.spec_flash.snapshot_pre`` の相当品だが、こちらは attention 側
    (``pre["kv"]``) を持たない -- 行別 trim をしない設計なので KV 側に
    「戻すために控える」ものが無いため。

    スロットの取り出しは直添字 (``c[3]``) ではなく ``arch.recurrent_layers``
    の名前付き ``ngram`` スロット経由 (mlxturbo/arch.py 参照)。
    """
    slots = {rl.index: rl for rl in _archmod.recurrent_layers(model)}
    out = []
    for i, c in enumerate(caches):
        rl = slots.get(i)
        out.append(c[rl.ngram] if (rl is not None and rl.ngram is not None) else None)
    return out


def batched_rollback(model, caches, cap, keeps: list[int], pre_ctx=None, pair=None) -> None:
    """(B, T+1) の verify 後、GDN/PLE/n-gram 系キャッシュを行別 keep に戻す。

    実体は ``mlxturbo/arch.py`` の ``rollback_recurrent_rows`` -- 単一行版
    ``mlxturbo.spec_flash.rollback`` との共通部分 (再帰状態・conv window・
    PLE conv window・n-gram 文脈) を名前付きスロット経由の共通部品に畳んで
    ある。attention 側は一切触らない -- dead slot として ``RaggedLedger`` が
    扱う設計そのものなので、呼び出し側で別途 ``ledger.commit_round(keeps,
    total)`` を呼ぶこと (ここでは呼ばない: このモジュールの他の部品と同じく
    「1 つのことだけをする」ほうを選んだ)。

    ``pre_ctx``/``pair`` を渡さない場合は n-gram 文脈の更新をスキップする
    (``mlxturbo.spec_flash.rollback`` が ``ids_kept=None`` のとき ``c[3]`` を
    更新しないのと同じ扱い)。
    """
    _archmod.rollback_recurrent_rows(
        model, caches, cap, keeps, ngram_ctx=pre_ctx, pair=pair
    )


# ------------------------------------------------------------ admission


def bucket_batches(
    lengths: list[int], max_batch: int, max_ratio: float = 1.5
) -> list[list[int]]:
    """待っているリクエストを、長さの近いものだけでバッチにまとめる。

    右パディングの無駄は行ごとに ``max_len - len`` 列で、そこは dead slot と
    同じく**計算はするが使わない**。長さ比が開くほど無駄が増えるので、
    バッチ内の最長/最短が ``max_ratio`` を超えるものは別バッチに割る。

    引数は長さのリスト、返り値は元の添字のリストのリスト (入力順は保つ)。
    長さでソートしてから詰めるので、隣り合う長さが同じバッチに入る。

    ``max_ratio`` の既定 1.5 は**未測**。無駄列の割合の上限がおよそ
    1 - 1/1.5 = 33% になる値として置いた。スループットを測ってから詰める。
    """
    if max_batch < 1:
        raise ValueError("max_batch は 1 以上")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (
            len(cur) >= max_batch
            or lengths[i] > lengths[cur[0]] * max_ratio
        ):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------ generator


class BatchSpecGenerator:
    """B 行同時の MTP 投機デコード (同期ラウンド)。

    ラウンドごとに B 行そろって ``(B, T+1)`` の検証フォワードを 1 回踏み、
    行ごとの受理数 ``keep_b`` だけ論理長を進める。不採用位置は dead slot
    として ``RaggedLedger`` が恒久的に隠す (物理列は全行そろって伸びる)。

    ## プロンプト長

    右パディングでそろえる。マスクは「因果 かつ 実長より手前」
    (`RaggedLedger.round_mask` の prefill モード)、再帰状態の窓は本家の
    `_tail_window` が ``cache.lengths`` を見て実長基準で取る。GDN の入力からは
    ``conv_mask`` でパディング列を落とす。

    ## MTP ドラフトキャッシュ

    priming の幅を全行そろえる (各行の末尾に揃えた w 列)。MTP ヘッドの位置は
    priming 窓の先頭からの相対なので、幅がそろえば行の意味もそろう。受理数が
    行ごとに違っても列の進み方は変わらない (`_draft_chain` はラウンドごとに
    1 列だけ残す) ので、単一系列と同じ挙動になる。

    ## 対象外

    QSA (indexer)。kv 長が ``indexer_budget`` を超える構成では
    ``ragged_attention`` が ``NotImplementedError`` で止まる (モジュール
    docstring 参照)。呼び手は長いリクエストをバッチから外すこと。

    ## サンプリング

    既定は貪欲 (``temp=0`` / ``sampler=None``) で、これは以前と 1 ビットも
    変わらない。``temp>0`` / ``sampler`` を渡すと ``mlxturbo.spec_flash`` の
    ``FlashSpecEngine._verify`` と同じ形で全位置を先に引く -- あちらが
    ``(1, T+1)`` に対してやっていることを ``(B, T+1)`` に広げただけで、
    「位置 j のサンプルは lg[:, j] にしか依存しない」という独立性の議論
    (``FlashSpecEngine._verify`` の注記) はそのまま成り立つ。**バッチ内の
    全行が同じサンプラーを共有する**前提なので (1 回の呼び出しで全行ぶんを
    引く)、パラメータの違う要求を同じバッチに入れてはいけない -- admission
    側の責務 (``BatchSpecCoordinator._sampling_key``)。
    """

    def __init__(
        self,
        engine,
        prompts: list[list[int]],
        depth: int | None = None,
        temp: float = 0.0,
        sampler=None,
    ):
        from .spec_flash import PRIME_WINDOW

        if not prompts:
            raise ValueError("prompts が空")
        self.eng = engine
        self.model = engine.model
        self.B = len(prompts)
        self.lengths = [len(p) for p in prompts]
        if min(self.lengths) < 2:
            raise ValueError("プロンプトは 2 トークン以上 (priming に 1 対要る)")
        self.depth = depth or engine.depth
        self.L = max(self.lengths)
        pad = 0
        self.ids = mx.array(
            [list(p) + [pad] * (self.L - len(p)) for p in prompts]
        )
        self.caches, self.ledger = make_ragged_cache(self.model, self.B)
        self.prime_window = min(PRIME_WINDOW, min(self.lengths) - 1)
        self.out: list[list[int]] = [[] for _ in range(self.B)]
        self.rounds = 0
        self.accepted = 0
        self.temp = float(temp)
        self.sampler = sampler
        self._cur = None
        self._hyper_prev = None
        self._mtp_cache = None

    def _sample_rows(self, lg: mx.array) -> mx.array:
        """検証フォワードの logits (B, S, vocab) から各位置のトークン (B, S)。

        ``FlashSpecEngine._verify`` / ``_sample`` の B 行版。貪欲のときは
        以前と同じ ``mx.argmax`` をそのまま通る。
        """
        if self.sampler is None and self.temp <= 0:
            return mx.argmax(lg, axis=-1)
        b, s, v = lg.shape
        flat = lg.reshape(b * s, v)
        if self.sampler is not None:
            return self.sampler(flat).reshape(b, s)
        return mx.random.categorical(flat.astype(mx.float32) / self.temp).reshape(b, s)

    # ---- prefill ----------------------------------------------------

    def _row_take(self, arr: mx.array, idx: list[int]) -> mx.array:
        """行 b の位置 idx[b] を 1 つずつ取る (B, 1, ...)。"""
        pos = mx.array(idx).reshape(self.B, 1, *([1] * (arr.ndim - 2)))
        return mx.take_along_axis(
            arr, mx.broadcast_to(pos, (self.B, 1, *arr.shape[2:])), axis=1
        )

    def prefill(self) -> None:
        from .spec_flash import capture

        for c in self.caches:
            if not hasattr(c, "ledger"):
                # 上流 (ArraysCache) の API で入れる。``lengths`` は advance が
                # 読んで毎回減らす作業用の値でもあるので、生の代入と後始末を
                # 自前でやるとズレる。本家の _tail_window もこれを見て実長
                # 基準で窓を取る
                c.prepare(lengths=self.lengths)
        self.ledger.prefill_lengths = self.lengths
        try:
            with ragged_attention(), capture(self.model, light=True) as cap:
                logits = self.model(self.ids, cache=self.caches)
                mx.eval(logits, cap.hyper)
        finally:
            self.ledger.prefill_lengths = None
            for c in self.caches:
                if not hasattr(c, "ledger"):
                    c.finalize()
        self.ledger.commit_round(self.lengths, self.L)

        last = [n - 1 for n in self.lengths]
        first = self._sample_rows(self._row_take(logits, last))[:, 0]
        mx.eval(first)
        for b, t in enumerate(first.tolist()):
            self.out[b].append(int(t))
        self._cur = first.reshape(self.B, 1)
        self._hyper_prev = self._row_take(cap.hyper, last)
        self._prime(cap.hyper)

    def _prime(self, hyper: mx.array) -> None:
        """MTP ヘッドに各行の末尾 w 対を流す (幅は全行そろえる)。"""
        w = self.prime_window
        cache = RaggedDraftCache(self.B)
        if w < 1:
            self._mtp_cache = cache
            return
        # 行 b: トークン位置 n_b-w .. n_b-1 と、その 1 つ前の hyper
        tok_idx = [[n - w + j for j in range(w)] for n in self.lengths]
        hyp_idx = [[n - w - 1 + j for j in range(w)] for n in self.lengths]
        toks = mx.take_along_axis(self.ids, mx.array(tok_idx), axis=1)
        hy = mx.take_along_axis(
            hyper,
            mx.broadcast_to(mx.array(hyp_idx)[..., None], (self.B, w, hyper.shape[2])),
            axis=1,
        )
        embeds = self.model.model.embed_tokens(toks)
        with ragged_attention():
            # mask は渡さない。`ragged_attention` の `_final_mask` が
            # `cache.round_mask` で組み直すので、ここで作っても捨てられる
            # (しかも update_and_fetch の前なので列数が 0 の壊れた形になる)
            out = self.eng.mtp(
                embeds, hy, self.eng.rope, None, cache, cache.indexer
            )
            mx.eval(out)
        self._mtp_cache = cache

    # ---- rounds -----------------------------------------------------

    def step(self, truncate=None) -> list[list[int]]:
        """1 ラウンド進めて、行ごとの新規トークンを返す。

        ``truncate`` (省略可) は ``(b, toks) -> 残す個数 (1 以上)``。eos や
        残り max_tokens で行 b の出力をこのラウンドの途中で打ち切るための口で、
        単独経路 (``FlashSpecEngine.generate_stream``) が ``cut``/``remaining``
        で ``vals`` を切り詰めてから ``rollback(keep=len(vals))`` を呼ぶのと
        同じ形 -- 打ち切った分は受理しなかったことにするので、キャッシュと
        出力が食い違わない。
        """
        from .spec_flash import _staged_forward

        eng = self.eng
        with ragged_attention():
            drafts = eng._draft_chain(
                self._cur, self._hyper_prev, self._mtp_cache, self.depth
            )
        pair = mx.concatenate([self._cur] + drafts, axis=1)
        total = pair.shape[1]
        pre_ctx = snapshot_pre_ctx(self.model, self.caches)
        with batched_capture(self.model) as cap:
            # 段階投入 (spec_flash._staged_forward)。単独経路の検証
            # フォワードはこれを通っていて、グラフ構築中の GPU 泡を刈る分
            # だけ速い。バッチ側が `model(...)` を直接呼んでいると、B=1 でも
            # 単独に負ける (実測 0.82x)。計算内容は同じ。
            lg = _staged_forward(self.model, pair, self.caches)
            nxt = self._sample_rows(lg)            # (B, T+1)
            dv = mx.concatenate(drafts, axis=1)   # (B, T)
            mx.eval(nxt, dv, cap.hyper)
        self.rounds += 1

        nxt_l, dv_l = nxt.tolist(), dv.tolist()
        keeps, new = [], []
        for b in range(self.B):
            hit = 0
            while hit < total - 1 and nxt_l[b][hit] == dv_l[b][hit]:
                hit += 1
            self.accepted += hit
            toks = nxt_l[b][: hit + 1]
            if truncate is not None:
                n_keep = truncate(b, toks)
                if n_keep < len(toks):
                    toks = toks[:n_keep]
            keeps.append(len(toks))
            new.append(toks)
            self.out[b].extend(toks)

        batched_rollback(self.model, self.caches, cap, keeps, pre_ctx, pair)
        self.ledger.commit_round(keeps, total)
        last = [k - 1 for k in keeps]
        self._cur = self._row_take(nxt[..., None], last)[:, :, 0]
        self._hyper_prev = self._row_take(cap.hyper, last)
        return new

    def retire(self, rows: list[int]) -> None:
        """終わった行を物理的に落とし、``rows`` の行だけを残す (入力は残す側の
        添字、昇順)。

        残さない選択肢 (終わった行を計算し続けて、全行が終わったらバッチごと
        畳む) もあるが、そちらは 20 トークンで終わった行が 500 トークンの行に
        付き合って 480 ラウンドぶんの前進計算と KV を占め続ける。行の可視性は
        元から行ごと (`RaggedLedger._alive`) で、キャッシュもバッチ軸が先頭に
        揃っているので、落とすのは軸 0 のスライスで済む。
        """
        if len(rows) == self.B:
            return
        idx = mx.array(rows)
        for c in self.caches:
            if hasattr(c, "ledger"):
                c.filter_rows(idx)
            else:
                # 上流 (ArraysCache.filter) がそのまま使える。GDN/PLE/n-gram の
                # テンソルは先頭次元がバッチ次元
                c.filter(idx)
        if self._mtp_cache is not None:
            self._mtp_cache.filter_rows(idx)
        self.ledger.filter_rows(rows)
        self.lengths = [self.lengths[b] for b in rows]
        self.out = [self.out[b] for b in rows]
        self._cur = self._cur[idx]
        self._hyper_prev = self._hyper_prev[idx]
        self.B = len(rows)

    def maybe_compact(self, hard_limit: int | None = None, waste_ratio: float = 1.5) -> bool:
        """dead slot が溜まってきたら詰める。詰めたら True。

        2 つの理由で呼ぶ:

        - 硬い上限 (``hard_limit``): 物理列数が QSA の境界 (``indexer_budget``)
          に届くと ``ragged_attention`` が ``NotImplementedError`` で止まる。
          次のラウンドで足す ``depth+1`` 列を含めて境界の手前に収まるよう、
          越えそうなら必ず詰める。論理長がこの境界より短いことは admission が
          保証しているので (``spec_batchable``)、詰めれば必ず収まる。
        - 無駄 (``waste_ratio``): 物理列が論理長の ``waste_ratio`` 倍を超えたら
          詰める。受理率が 1.6 tok/round 程度なので、詰めないと物理列は論理長の
          およそ ``(depth+1)/1.6`` 倍で伸び続け、attention の費用がそのぶん
          丸ごと無駄になる。
        """
        valid = self.ledger.max_valid_len()
        need = (
            hard_limit is not None
            and self.ledger.L + self.depth + 1 > hard_limit
        ) or (valid and self.ledger.L > valid * waste_ratio)
        if not need or self.ledger.L <= valid:
            return False
        self.ledger.compact(self.model, self.caches)
        return True

    def generate(self, max_tokens: int) -> list[list[int]]:
        """全行が ``max_tokens`` 個に達するまで回す (eos は見ない)。"""
        self.prefill()
        while min(len(o) for o in self.out) < max_tokens:
            self.step()
        return [o[:max_tokens] for o in self.out]


# ------------------------------------------------------------- coordinator
#
# 上の部品をサーバーに配線する層 (--max-batch-spec)。形は
# `mlxturbo/batch.py` の BatchCoordinator に寄せてある: inbox に Admission を
# 積み、駆動ループを 1 本だけ executor (モデルを読んだ唯一の MLX ワーカー
# スレッド) に投げ、live な仕事が無くなったら抜ける。違いは駆動する対象
# だけ -- あちらは mlx_lm の BatchGenerator (投機なし)、こちらは上の
# BatchSpecGenerator (MTP 投機つき)。
#
# ## 決めたこと 4 点 (依頼の admission / スケジューラの論点)
#
# 1. まとめる条件。同じ**サンプリング設定**を持ち、長さが近いものだけを
#    まとめる。長さは `bucket_batches` (既存) に任せる -- 右パディングの無駄は
#    行あたり `max_len - len` 列で、比が開くほど丸ごと捨てる計算が増えるため。
#    サンプリングを揃えるのは、`BatchSpecGenerator._sample_rows` が全行ぶんを
#    1 回の呼び出しで引くから (行ごとに違うサンプラーを混ぜると分布が壊れる)。
#    seed 指定つきの要求は混ぜない (`mx.random.seed` はプロセス全体の状態で、
#    同居する他の行の乱数まで動かしてしまうため) -- 直列経路に回す。
#
# 2. 途中で終わった列。終わった行はその場で完了させ、`retire()` で物理的に
#    落とす (バッチ軸のスライス)。dead slot の帳簿は元から行ごとなので、
#    行を抜くのに列の詰め直しは要らない。落とさない場合、短い行が長い行に
#    最後まで付き合って前進計算と KV を占め続ける。
#
# 3. 新しい要求。**走行中のバッチには入れない (closed batch)。**次のバッチの
#    編成は、走行中のバッチが空になった時点で行う。走行中に入れるには、
#    新入りの KV を走行中の物理列数に左詰めで揃え、MTP の priming 窓も
#    揃えた上で軸 0 に連結する必要がある (`compact()` が作る形と同じ形に
#    寄せれば筋は通るが、`ArraysCache` の内部オフセットと priming 窓の
#    整合は実機でしか確かめられない)。GPU を使わない今回の範囲では検証
#    できないので入れていない。今の直列キュー (1 リクエストずつ) と比べれば
#    待ち時間は決して増えない -- 増えるのは同時到着の並列度だけ。
#
# 4. メモリ。下の `plan_batch` を参照。上限は 3 つの積み重ねで、どれも
#    設定値ではなくモデルの形から計算する。
#
# ## B=1 は既存の単独経路のまま
#
# 待ち行列に 1 本しか無いときは `BatchSpecGenerator` を使わず、
# `FlashSpecRunner.generate` (= `FlashSpecEngine.generate_stream`) をそのまま
# 呼ぶ。単独経路には楽観パイプライン・次ラウンドの draft 先行投入・適応的
# depth・チャンク prefill が乗っていて、バッチ経路にはどれも無い。単独
# リクエストをバッチ経路に流すと、それだけで短 decode が落ちる
# (`bench/batch_b1_gate.py` が固定している線)。同時到着を拾うために、
# 1 本しか無いときだけ `wait_ms` (既定 15ms) だけ相方を待つ -- 待つのは
# TTFT の手前だけで、decode の ms/token には入らない。

import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def spec_batchable(model, prompt_len: int, max_tokens: int, depth: int) -> bool:
    """この要求をバッチ x 投機に入れてよいか。

    条件は QSA (indexer) が最後まで活性化しないこと。`ragged_attention` は
    QSA を通さず `NotImplementedError` で止める (モジュール docstring の
    「indexer / QSA はバッチ対象外」) ので、**物理列数**が `indexer_budget`
    を超えない保証が要る。物理列は論理長より速く伸びるが、
    `BatchSpecGenerator.maybe_compact` が境界の手前で必ず詰めるので、
    「論理長 + 次のラウンドで足す depth+1 列」が境界に収まれば足りる。

    プロンプトが 2 トークン未満のものも弾く (priming に 1 対要る)。
    """

    if prompt_len < 2 or max_tokens < 1:
        return False
    budget = _archmod.indexer_budget(model)
    if budget is None:
        # QSA を持たない族。この機構自体が qwen4_exp 向けなのでここには
        # 来ない想定だが、来たら長さの上限は無いものとして扱う
        return True
    return prompt_len + max_tokens + depth + 1 <= budget


def kv_bytes_per_token(model) -> int | None:
    """1 行 1 トークンあたりの KV バイト数 (full attention の k/v + indexer)。

    再帰系 (GDN/PLE/n-gram) はトークン数に依らない固定サイズなので含めない。
    """

    t = getattr(getattr(model, "args", None), "text", None)
    types = getattr(t, "layer_types", None)
    if not types:
        return None
    n_full = sum(1 for lt in types if lt == "full_attention")
    elem = 2  # bf16 / fp16
    kv = n_full * t.num_key_value_heads * t.head_dim * 2 * elem
    idx = n_full * getattr(t, "indexer_kv_heads", 0) * getattr(t, "indexer_head_dim", 0) * elem
    return kv + idx


def capture_bytes_per_token(model) -> int | None:
    """検証フォワード 1 位置ぶんの `states_all` (fp32) の総バイト数 (1 行)。

    `mlxturbo.spec_flash.capture` が rollback のために層ごとに確保する
    (B, T, Hv, Dv, Dk) fp32 のこと。KV と違って**ラウンドごとに作り直す**
    一時領域だが、1 位置あたりが KV の数千倍あるので、同時に何行流せるかを
    決めているのは実際にはこちら (capture の docstring 参照)。
    """

    t = getattr(getattr(model, "args", None), "text", None)
    types = getattr(t, "layer_types", None)
    if not types:
        return None
    n_lin = sum(1 for lt in types if lt != "full_attention")
    return (
        n_lin
        * t.linear_num_value_heads
        * t.linear_value_head_dim
        * t.linear_key_head_dim
        * 4
    )


def free_bytes(reserve: float = 0.10) -> int | None:
    """今この瞬間に使ってよい残りバイト数の見積もり。

    Metal の推奨作業セット上限から、モデルの重みを含む現在の常駐量を引く。
    `reserve` は MoE の一時活性など、ここで数えていないものへの取り置き。
    """

    try:
        rec = mx.metal.device_info()["max_recommended_working_set_size"]
    except Exception:  # noqa: BLE001  Metal が無い環境では上限を掛けない
        return None
    return int(rec * (1.0 - reserve)) - mx.get_active_memory()


def plan_batch(
    model,
    lengths: list[int],
    max_tokens: list[int],
    depth: int,
    max_batch: int,
    prefill_token_budget: int,
) -> int:
    """先頭から何行までを 1 つのバッチに入れてよいかを返す (1 以上)。

    上限は 3 つ。いずれも設定値ではなくモデルの形と実際の空きから計算する。

    1. ``max_batch`` (--max-batch-spec)。運用者が置く硬い上限。
    2. prefill の幅。バッチの prefill は `(B, 右パディング後の最長)` を 1 回で
       流す。単独経路が 1 チャンクで流している幅は `PREFILL_STEP_SIZE`
       (2048) で、そこは実測が乗っている安全な水準なので、その 2 倍を
       ``prefill_token_budget`` の既定に置いた (`B * L <= budget`)。
    3. 常駐と一時。`B * (KV + capture)` が `free_bytes()` に収まること。
       KV は行の最終的な長さ (プロンプト + max_tokens) で、capture は
       1 ラウンドぶん (depth+1 位置) で数える。

    実機 (128GB、重み 91GB) では 3 が効くのは B が数十のときで、実際に効く
    のは 1 と 2 -- それでも式に書いてあるのは、`indexer_budget` の制約が
    将来外れて長い要求が入るようになったときに、黙って落ちる代わりにここで
    止まるようにしておくため。
    """

    n = min(len(lengths), max(1, max_batch))
    # 2. prefill の幅
    while n > 1 and n * max(lengths[:n]) > prefill_token_budget:
        n -= 1
    # 3. 常駐と一時
    kv_per_tok = kv_bytes_per_token(model)
    cap_per_tok = capture_bytes_per_token(model)
    room = free_bytes()
    if kv_per_tok is not None and cap_per_tok is not None and room is not None:
        while n > 1:
            final_len = max(lengths[i] + max_tokens[i] for i in range(n))
            need = n * (final_len * kv_per_tok + (depth + 1) * cap_per_tok)
            if need <= room:
                break
            n -= 1
    return n


@dataclass
class SpecAdmission:
    """バッチ x 投機の待ち行列に入っている 1 リクエスト。

    ``on_tokens`` / ``on_done`` / ``future`` の約束は
    ``mlxturbo.batch.Admission`` と同一 (server.py の
    ``_build_streaming_pipeline`` が作る 3 点セットをそのまま受ける)。
    ``sampling`` は ``FlashSpecRunner.generate`` にそのまま渡す辞書で、単独
    経路に落ちたときはこれが丸ごと使われる。
    """

    prompt_ids: list[int]
    max_tokens: int
    temp: float
    sampling: dict
    eos_ids: set
    on_tokens: Callable[[list[int]], None] | None
    on_done: Callable[[str, Any], None] | None
    cancel_event: "threading.Event | None"
    future: "Any"  # concurrent.futures.Future
    tokens: list = field(default_factory=list)
    t0: float | None = None
    ttft: float | None = None
    steps: int = 0
    done: bool = False


class BatchSpecCoordinator:
    """バッチ x 投機のスケジューラ。

    上の「決めたこと 4 点」がこのクラスの中身。走行中は executor を占有する
    ので、駆動ループは仕事が無くなったら必ず抜ける (`mlxturbo.batch.
    BatchCoordinator` と同じ約束)。
    """

    def __init__(
        self,
        runner,
        executor,
        max_batch: int,
        eos_ids,
        wait_ms: int = 15,
        max_ratio: float = 1.5,
        prefill_token_budget: int = 4096,
    ):
        self.runner = runner
        self.engine = runner.engine
        self.model = runner.engine.model
        self.executor = executor
        self.max_batch = max(1, max_batch)
        self.eos_ids = set(eos_ids)
        self.wait_ms = max(0, wait_ms)
        self.max_ratio = max_ratio
        self.prefill_token_budget = prefill_token_budget
        self._inbox: "_queue.SimpleQueue[SpecAdmission]" = _queue.SimpleQueue()
        self._guard = threading.Lock()
        self._active = False

    # ---- 受付 (任意のスレッドから) --------------------------------------

    def submit(self, admission: SpecAdmission) -> None:
        self._inbox.put(admission)
        with self._guard:
            if not self._active:
                self._active = True
                self.executor.submit(self._drive)

    # ---- 以下は全て単一の MLX ワーカースレッド上 --------------------------

    def _complete(self, adm: SpecAdmission, res=None, cancelled=False, error=None) -> None:
        """`mlxturbo.batch.BatchCoordinator._complete` と同じ約束。"""
        if adm.done:
            return
        adm.done = True
        if adm.on_done is not None:
            if error is not None:
                adm.on_done("error", error)
            elif cancelled:
                adm.on_done("cancelled", None)
            else:
                adm.on_done("done", res)
            adm.future.set_result(res)
        else:
            if error is not None:
                adm.future.set_exception(error)
            else:
                adm.future.set_result(res)

    def _build_res(self, adm: SpecAdmission) -> dict:
        decode_time = 0.0
        if adm.t0 is not None and adm.ttft is not None:
            decode_time = max(0.0, time.perf_counter() - adm.t0 - adm.ttft)
        n_decode = max(len(adm.tokens) - 1, 0)
        return {
            "tokens": adm.tokens,
            "ttft_s": adm.ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            # バッチ経路はセッションを持たない (毎回まっさらな prefill)。
            # mlxturbo/batch.py の継続バッチングと同じ割り切り
            "prefill_reused": 0,
            "prefill_new": len(adm.prompt_ids),
            # 定義は mlxturbo.spec.SpecEngine / FlashSpecSession と同じ
            # (n_decode / この行が参加したラウンド数)
            "tokens_per_step": (n_decode / adm.steps) if adm.steps else 0.0,
        }

    def _cancelled(self, adm: SpecAdmission) -> bool:
        return adm.cancel_event is not None and adm.cancel_event.is_set()

    def _deliver(self, adm: SpecAdmission, toks: list[int]) -> None:
        if not toks:
            return
        if adm.ttft is None:
            adm.ttft = time.perf_counter() - (adm.t0 or time.perf_counter())
        adm.tokens.extend(toks)
        if adm.on_tokens is not None:
            adm.on_tokens(toks)

    @staticmethod
    def _sampling_key(adm: SpecAdmission):
        """同じバッチに入れてよいサンプリング設定か (全行で 1 回の呼び出しに
        まとめるため、揃っている必要がある)。``seed`` 指定つきは常に単独。"""
        s = adm.sampling
        if s.get("seed") is not None:
            return None  # 常に単独 (プロセス全体の乱数状態を動かすため)
        bias = s.get("logit_bias") or {}
        return (
            round(adm.temp, 6),
            round(float(s.get("top_p") or 0.0), 6),
            int(s.get("top_k") or 0),
            round(float(s.get("min_p") or 0.0), 6),
            tuple(sorted((int(k), float(v)) for k, v in bias.items())),
        )

    def _drive(self) -> None:
        pending: list[SpecAdmission] = []
        try:
            while True:
                self._drain(pending)
                if len(pending) == 1 and self.wait_ms:
                    # 同時到着を拾うための待ち。1 本しか無いときだけで、
                    # decode ではなく TTFT の手前に乗る
                    self._wait_for_company(pending)
                pending = [a for a in pending if not self._reject_if_dead(a)]
                if not pending:
                    self._drain(pending)
                    if not pending:
                        break
                    continue
                group = self._take_group(pending)
                if len(group) == 1:
                    self._run_solo(group[0])
                else:
                    self._run_batch(group)
        except BaseException as exc:  # noqa: BLE001 - 内部の不具合で Future を宙吊りにしない
            for adm in pending:
                self._complete(adm, error=exc)
        finally:
            with self._guard:
                self._active = False
            # 空判定と _active を落とす間に届いた要求が取り残されないよう、
            # 錠の中で見直す (mlxturbo/batch.py と同じ)
            if not self._inbox.empty():
                with self._guard:
                    if not self._active:
                        self._active = True
                        self.executor.submit(self._drive)

    def _drain(self, pending: list) -> None:
        while True:
            try:
                pending.append(self._inbox.get_nowait())
            except _queue.Empty:
                return

    def _wait_for_company(self, pending: list) -> None:
        deadline = time.perf_counter() + self.wait_ms / 1000.0
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                return
            try:
                pending.append(self._inbox.get(timeout=left))
            except _queue.Empty:
                return
            self._drain(pending)
            if len(pending) >= self.max_batch:
                return

    def _reject_if_dead(self, adm: SpecAdmission) -> bool:
        if self._cancelled(adm):
            self._complete(adm, cancelled=True)
            return True
        return False

    def _take_group(self, pending: list) -> list:
        """先頭 (最古) の要求を必ず含む 1 バッチぶんを `pending` から抜き取る。

        最古を軸にするのは飢餓を作らないため -- `bucket_batches` は長さ順に
        詰めるので、それだけに任せると短い要求ばかりが選ばれ続けうる。
        """
        head = pending[0]
        key = self._sampling_key(head)
        if key is None:
            pending.pop(0)
            return [head]
        cand = [a for a in pending if self._sampling_key(a) == key]
        buckets = bucket_batches(
            [len(a.prompt_ids) for a in cand], self.max_batch, self.max_ratio
        )
        chosen = next(b for b in buckets if 0 in b)
        # 到着順に戻してから、入るところまでを取る
        rows = [cand[i] for i in sorted(chosen)]
        n = plan_batch(
            self.model,
            [len(a.prompt_ids) for a in rows],
            [a.max_tokens for a in rows],
            self.engine.depth,
            self.max_batch,
            self.prefill_token_budget,
        )
        rows = rows[:n]
        for a in rows:
            pending.remove(a)
        return rows

    # ---- 単独 (B=1) -----------------------------------------------------

    def _run_solo(self, adm: SpecAdmission) -> None:
        """既存の単独経路をそのまま呼ぶ。`_start_generation` の worker()
        (server.py) と同じ 3 分岐で結果を返す。session は渡さない --
        バッチ経路と揃えた割り切りで、`mlxturbo/batch.py` の継続バッチングも
        同じ (毎回まっさらな prefill)。"""
        adm.t0 = time.perf_counter()
        try:
            res = self.runner.generate(
                adm.prompt_ids,
                max_tokens=adm.max_tokens,
                temp=adm.temp,
                eos_ids=adm.eos_ids,
                on_tokens=adm.on_tokens,
                session=None,
                **adm.sampling,
            )
        except BaseException as exc:  # noqa: BLE001 - on_tokens 由来の中断もここに来る
            if self._cancelled(adm):
                self._complete(adm, cancelled=True)
            else:
                self._complete(adm, error=exc)
            return
        self._complete(adm, res=res)

    # ---- バッチ (B>=2) ---------------------------------------------------

    def _run_batch(self, rows: list) -> None:
        try:
            head = rows[0]
            sampler = _position_local_sampler_for(head)
            gen = BatchSpecGenerator(
                self.engine,
                [list(a.prompt_ids) for a in rows],
                temp=head.temp,
                sampler=sampler,
            )
            now = time.perf_counter()
            for a in rows:
                a.t0 = now
            gen.prefill()
            # prefill が出す 1 個目は単独経路と同じくラウンドに数えない
            # (tokens_per_step の定義が n_decode / decode ラウンド数のため)
            rows = self._after_round(gen, rows, [o[-1:] for o in gen.out], count_step=False)
            budget = _archmod.indexer_budget(self.model)
            while rows:
                new = gen.step(
                    truncate=lambda b, toks, r=rows: self._truncate(r[b], toks)
                )
                rows = self._after_round(gen, rows, new)
                if rows:
                    gen.maybe_compact(hard_limit=budget)
        except BaseException as exc:  # noqa: BLE001 - 1 行の失敗でも全行の Future を必ず閉じる
            for a in rows:
                self._complete(a, cancelled=self._cancelled(a),
                               error=None if self._cancelled(a) else exc)

    def _truncate(self, adm: SpecAdmission, toks: list[int]) -> int:
        """このラウンドで行 b が受け取ってよい個数。単独経路の
        `generate_stream` が eos (`cut`) と残り (`remaining`) で `vals` を
        切り詰めるのと同じ判定。"""
        cut = next((i for i, t in enumerate(toks) if t in self.eos_ids), None)
        n = len(toks) if cut is None else cut + 1
        return max(1, min(n, adm.max_tokens - len(adm.tokens)))

    def _after_round(self, gen, rows: list, new: list, count_step: bool = True) -> list:
        """1 ラウンドぶんを配り、終わった行を落として残りを返す。"""
        keep: list[int] = []
        for b, adm in enumerate(rows):
            if count_step:
                adm.steps += 1
            try:
                self._deliver(adm, new[b])
            except BaseException as exc:  # noqa: BLE001 - 1 行のコールバック失敗を他の行に伝染させない
                cancelled = self._cancelled(adm)
                self._complete(adm, cancelled=cancelled, error=None if cancelled else exc)
                continue
            finished = (
                len(adm.tokens) >= adm.max_tokens
                or (new[b] and new[b][-1] in self.eos_ids)
                or self._cancelled(adm)
            )
            if finished:
                if self._cancelled(adm):
                    self._complete(adm, cancelled=True)
                else:
                    self._complete(adm, res=self._build_res(adm))
                continue
            keep.append(b)
        if len(keep) != len(rows):
            gen.retire(keep)
        return [rows[b] for b in keep]


def _position_local_sampler_for(adm: SpecAdmission):
    """行が共有する位置局所サンプラー。`FlashSpecRunner.generate` が組むものと
    同じ (`mlxturbo.runner._position_local_sampler`) -- 単独経路とバッチ経路で
    サンプリングの形をずらさないため。"""
    from .runner import _position_local_sampler

    s = adm.sampling
    return _position_local_sampler(
        adm.temp,
        float(s.get("top_p") or 0.0),
        int(s.get("top_k") or 0),
        float(s.get("min_p") or 0.0),
        s.get("logit_bias"),
    )


__all__ = [
    "RaggedLedger",
    "RaggedAttnCache",
    "RaggedDraftCache",
    "make_ragged_cache",
    "ragged_attention",
    "batched_capture",
    "snapshot_pre_ctx",
    "batched_rollback",
    "BatchSpecGenerator",
    "bucket_batches",
    "BatchSpecCoordinator",
    "SpecAdmission",
    "spec_batchable",
    "plan_batch",
]
