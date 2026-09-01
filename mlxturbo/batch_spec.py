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
        # prefill の 1 回だけ立てる。右パディングで流すので、そのラウンドの
        # マスクは「因果 かつ 実長より手前」であって、通常ラウンドの
        # 「生きている過去 + 因果」とは別物 (round_mask 参照)
        self.prefill_lengths: list[int] | None = None

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
        """
        if self.prefill_lengths is None:
            return self.next_round_mask(T)
        lens = mx.array(self.prefill_lengths)
        cols = mx.arange(T)
        causal = cols[None, :] <= cols[:, None]  # (T, T)
        valid = cols[None, :] < lens[:, None]  # (B, T)
        return (causal[None] & valid[:, None, :])[:, None]

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

    貪欲のみ。温度つきサンプリングは行ごとに受理判定が変わるだけで構造は
    同じだが、まだ書いていない。
    """

    def __init__(self, engine, prompts: list[list[int]], depth: int | None = None):
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
        self._cur = None
        self._hyper_prev = None
        self._mtp_cache = None

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
        first = mx.argmax(self._row_take(logits, last)[:, 0], axis=-1)
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

    def step(self) -> list[list[int]]:
        """1 ラウンド進めて、行ごとの新規トークンを返す。"""
        from .spec_flash import capture  # noqa: F401  (batched_capture 経由)

        eng = self.eng
        with ragged_attention():
            drafts = eng._draft_chain(
                self._cur, self._hyper_prev, self._mtp_cache, self.depth
            )
        pair = mx.concatenate([self._cur] + drafts, axis=1)
        total = pair.shape[1]
        pre_ctx = snapshot_pre_ctx(self.model, self.caches)
        with batched_capture(self.model) as cap:
            lg = self.model(pair, cache=self.caches)
            nxt = mx.argmax(lg, axis=-1)          # (B, T+1)
            dv = mx.concatenate(drafts, axis=1)   # (B, T)
            mx.eval(nxt, dv, cap.hyper)
        self.rounds += 1

        nxt_l, dv_l = nxt.tolist(), dv.tolist()
        keeps, new = [], []
        for b in range(self.B):
            hit = 0
            while hit < total - 1 and nxt_l[b][hit] == dv_l[b][hit]:
                hit += 1
            keeps.append(hit + 1)
            self.accepted += hit
            new.append(nxt_l[b][: hit + 1])
            self.out[b].extend(new[-1])

        batched_rollback(self.model, self.caches, cap, keeps, pre_ctx, pair)
        self.ledger.commit_round(keeps, total)
        last = [k - 1 for k in keeps]
        self._cur = self._row_take(nxt[..., None], last)[:, :, 0]
        self._hyper_prev = self._row_take(cap.hyper, last)
        return new

    def generate(self, max_tokens: int) -> list[list[int]]:
        """全行が ``max_tokens`` 個に達するまで回す (eos は見ない)。"""
        self.prefill()
        while min(len(o) for o in self.out) < max_tokens:
            self.step()
        return [o[:max_tokens] for o in self.out]


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
]
