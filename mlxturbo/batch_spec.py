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

## indexer / QSA (2026-09-02 に対応。長さの上限は外れた)

QSA のブロック選択は本家では**物理列**で切られる。dead slot を挟んだバッチ列
でそれをやると、ブロックの平均に棄却済みの鍵が混ざり、pooled 鍵の rope 角が
ずれ、top-k が dead slot を選ぶ。3 つとも「行ごとにどの列が生きているか」が
違うことから来る (``mlxturbo/batch.py`` の "Remaining limitation" と同種)。

そこで ``_ragged_indexer_call`` がブロック境界を**行の論理列** (その行の
生きている列を並べ直したもの) の上で引き直す。dead slot は論理列に現れない
ので、選ばれようがない。論理列の上ではブロック格子が全行そろうため、pooled の
rope 角は本家と同じ 1 本で済み、行ごとに違うのは「何ブロックまで実在するか」
だけになる。穴も右パディングも無いラウンドは本家をそのまま呼ぶ。

**この結果、``spec_batchable`` の長さの上限は無くなった。**残る条件は
プロンプト 2 トークン以上と max_tokens 1 以上だけ。段 3(b) の gather 経路
(``MLXTURBO_GATHER_ATTN``、既定 off) だけは行別境界に未対応で、バッチ経路の
間は通常経路 (加算マスク) に落ちる (``ragged_attention`` の項目 4)。

## サーバー配線

このファイルの末尾「coordinator」節にある (``--max-batch-spec``)。admission と
スケジューラの決めごとはそこに書いてある。スケジューラは 1 ステップに
トークン予算を 1 つ持つ chunked prefill 方式で、走行中のバッチにも後から行を
足す (``SpecPrefillLane`` -> ``BatchSpecGenerator.join``)。

## 未対応 (このモジュールの範囲外)

- 段 3(b) の gather 経路 (上記)。
- サンプリング (温度 > 0) の分布保証は実測していない。全位置を先に引く形は
  spec_flash.py 側の議論 (位置ごとの分布は独立) がそのまま当てはまり、
  ``BatchSpecGenerator._sample_rows`` はその B 行版だが、KLD などで確かめた
  わけではない。

## 検証

正しさは ``tools/verify_batch_spec.py`` (合成モデル、CPU のみ) にある。
結果 (誤差値) はそちらのファイル冒頭に記録。
"""

from __future__ import annotations

import math
import os
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
        # alive の (B, L) bool 配列と、QSA の論理列対応表の memo。どちらも
        # 帳簿が動くまで不変なので _invalidate で一緒に捨てる。
        self._alive_memo: mx.array | None = None
        self._qsa_memo: dict[tuple[int, int], "_QSAMap"] = {}

    def _invalidate(self) -> None:
        self._mask_memo.clear()
        self._alive_memo = None
        self._qsa_memo.clear()

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

    def next_round_mask(self, T: int, lens: "mx.array | None" = None) -> mx.array:
        """次に足す T 列を含めた bool マスク (B, 1, T, L+T) を返す。
        True = 見える。

        構成は 2 ブロックの結合:
          - 既存の L 列: 行ごとに alive かどうかだけで決まる (dead は
            ラウンドをまたいで恒久的に不可視)
          - 今回の新規 T 列: 通常の因果マスク (verify 対象の T+1 列同士は
            まだ受理/棄却が決まっていないので、単純な下三角)

        ``lens`` を渡すと、新規 T 列のうち行 b の実長 ``lens[b]`` より後ろも
        落とす (右パディングで流す prefill 用)。過去のブロックは同じ式のまま
        なので、**チャンクに刻んだ prefill** (2 回目以降は過去がある) も
        この 1 本で表せる。
        """
        B, L = self.B, self.L
        if L:
            prev = mx.broadcast_to(self._alive_bool()[:, None, :], (B, T, L))
        else:
            prev = mx.zeros((B, T, 0), dtype=mx.bool_)
        cols = mx.arange(T)
        causal = cols[None, :] <= cols[:, None]  # (T, T)
        if lens is None:
            new = mx.broadcast_to(causal[None], (B, T, T))
        else:
            valid = cols[None, :] < lens[:, None]  # (B, T)
            new = causal[None] & valid[:, None, :]
        return mx.concatenate([prev, new], axis=-1)[:, None]

    def round_mask(self, T: int) -> mx.array:
        """このラウンドのフォワードに渡す bool マスク (B, 1, T, L+T)。

        prefill 中 (``prefill_lengths`` が立っている) は右パディングを隠す
        必要があるので「生きている過去 かつ (因果 かつ 実長より手前)」。
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
            out = self.next_round_mask(T, mx.array(self._prefill_lengths))
        self._mask_memo[T] = out
        return out

    # --------------------------------------------------------- QSA 用

    def _alive_bool(self) -> mx.array:
        """帳簿の (B, L) bool 配列。Python リスト -> mx.array の変換は L に
        比例するので、帳簿が動くまで使い回す (mask と QSA の両方が読む)。"""
        if self._alive_memo is None:
            self._alive_memo = mx.array(self._alive, dtype=mx.bool_)
        return self._alive_memo

    def _cur_lens(self, S: int) -> list[int]:
        """このラウンドで行 b が実際に足す列数。右パディングで流す prefill の
        間だけ行ごとに違い、それ以外は全行 S。"""
        if self._prefill_lengths is None:
            return [S] * self.B
        return list(self._prefill_lengths)

    def qsa_uniform(self, S: int) -> bool:
        """dead slot も右パディングも 1 つも無いか。

        True なら物理列と論理列が全行で一致するので、QSA は**本家のまま**
        走らせてよい (`_ragged_indexer_call` の速い道)。判定は Python の
        帳簿だけで済むので、QSA が働かない短い文脈でも GPU の仕事は増えない。
        """
        return all(v == self.L for v in self._valid_len) and all(
            c == S for c in self._cur_lens(S)
        )

    def qsa_max_len(self, S: int, kv_phys: int) -> int:
        """行ごとの論理 kv 長の最大。QSA を発火させるかの判定に使う
        (単独実行なら行 b は kv 長 ``valid_len[b] + cur[b]`` で判定される
        ので、その最大が budget 以下なら**どの行も**疎化が要らない)。"""
        if not self._valid_len:
            return 0
        return max(v + c for v, c in zip(self._valid_len, self._cur_lens(S)))

    def qsa_logical(self, S: int, kv_phys: int) -> "_QSAMap":
        """物理列 <-> 論理列の対応表 (memo つき)。

        論理列 = その行の生きている物理列を並べ直したもの。dead slot は
        論理列に**現れない**ので、これを通した先ではブロックの内容にも
        top-k の候補にも dead slot が混ざりようがない。
        """
        key = (S, kv_phys)
        got = self._qsa_memo.get(key)
        if got is not None:
            return got
        cur = self._cur_lens(S)
        assert kv_phys == self.L + S, (kv_phys, self.L, S)
        new = mx.arange(S)[None, :] < mx.array(cur)[:, None]  # (B, S)
        alive = (
            mx.concatenate([self._alive_bool(), new], axis=1)
            if self.L
            else new
        )
        out = _qsa_map_from_alive(alive, self.qsa_max_len(S, kv_phys))
        self._qsa_memo[key] = out
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

    def extend_rows(self, new_valid: list[int], new_L: int) -> None:
        """走行中のバッチに行を足す (帳簿側)。

        物理列数を ``new_L`` にそろえてから、論理長 ``new_valid[j]`` の行を
        末尾に足す。足りない列は**左側**に dead として付ける -- ``compact()``
        が作る形 (``[False]*(new_L - valid) + [True]*valid``) と同じなので、
        キャッシュ側も左詰めのゼロ埋めで揃えればよい (``RaggedAttnCache.
        extend_rows``)。既存の行も ``new_L`` に届かなければ左に伸ばす
        (新入りの方が長いとき)。

        RoPE の位置は ``_valid_len`` から引くので、左に何列足しても回転は
        動かない -- 物理列の番号は attention の可視性にしか使っていない。
        """
        grow = new_L - self.L
        if grow < 0:
            raise ValueError(f"new_L={new_L} は現在の L={self.L} より小さい")
        if grow:
            for b in range(self.B):
                self._alive[b] = [False] * grow + self._alive[b]
        for v in new_valid:
            if not 0 <= v <= new_L:
                raise ValueError(f"論理長 {v} が物理列数 {new_L} に収まらない")
            self._alive.append([False] * (new_L - v) + [True] * v)
            self._valid_len.append(v)
        self.L = new_L
        self.B = len(self._alive)
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


# ------------------------------------------------------------- QSA の列対応


class _QSAMap:
    """QSA を行ごとに組み直すための列対応表。

    - ``log2phys`` (B, max_len): 行 b の論理列 c が物理列のどこか。
      その行の論理長より後ろは詰め物 (物理列 0 を指す)。
    - ``phys2log`` (B, kv_phys): 物理列 p が論理列のどこか。dead slot と
      右パディングは ``max_len`` (= sink) を指す。
    - ``kv_lens`` (B,): 行ごとの論理 kv 長。
    - ``max_len``: その最大 (Python int。配列の形を決めるのに要る)。
    """

    __slots__ = ("log2phys", "phys2log", "kv_lens", "max_len")

    def __init__(self, log2phys, phys2log, kv_lens, max_len: int):
        self.log2phys = log2phys
        self.phys2log = phys2log
        self.kv_lens = kv_lens
        self.max_len = max_len


def _qsa_map_from_alive(alive: mx.array, max_len: int) -> _QSAMap:
    """(B, kv_phys) の alive bool から対応表を作る。

    ``phys2log`` は「生きている列の通し番号」なので cumsum で出る。
    ``log2phys`` はその逆写像で、単調なので並べ替え無しに散布で作れる
    (sink 列を 1 本余分に用意して、そこへ捨てる)。どちらも 1 ラウンドに
    1 回 (全 full attention 層で共有) しか作らない。
    """
    B, kv_phys = alive.shape
    ranks = mx.cumsum(alive.astype(mx.int32), axis=1) - 1
    phys2log = mx.where(alive, ranks, mx.array(max_len, mx.int32))
    cols = mx.broadcast_to(mx.arange(kv_phys, dtype=mx.int32)[None], (B, kv_phys))
    l2p = mx.zeros((B, max_len + 1), dtype=mx.int32)
    l2p = mx.put_along_axis(l2p, phys2log, cols, axis=1)
    kv_lens = mx.sum(alive.astype(mx.int32), axis=1)
    return _QSAMap(l2p[:, :max_len], phys2log, kv_lens, max_len)


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
        # `Attention.__call__` が QSA に渡すのは KV キャッシュではなく
        # この `_IndexerCache` なので、行別の列対応を引ける先をここに
        # 挿しておく (`_wide_qkv` / `_gather_stats` と同じ属性注入の作法)。
        # `_ragged_indexer_call` はこれが無ければ本家に素通しする。
        self.indexer.qsa_owner = self

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

    # QSA の列対応は帳簿がそのまま持っている (`_ragged_indexer_call` が読む)。
    # ドラフトキャッシュも同名の 3 つを持つので、QSA 側はキャッシュの種類を
    # 知らなくてよい。呼ばれ方は `__init__` の `qsa_owner` を参照。
    def qsa_uniform(self, S: int) -> bool:
        return self.ledger.qsa_uniform(S)

    def qsa_max_len(self, S: int, kv_phys: int) -> int:
        return self.ledger.qsa_max_len(S, kv_phys)

    def qsa_logical(self, S: int, kv_phys: int) -> _QSAMap:
        return self.ledger.qsa_logical(S, kv_phys)

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

    def extend_rows(self, others: list["RaggedAttnCache"], new_L: int) -> None:
        """``others`` の行を軸 0 に足す。列は ``new_L`` へ**左詰め**で揃える。

        新入りの KV は自分の実位置で回転済みなので、左にゼロ列を足しても
        回転は動かない。足したゼロ列は帳簿側が dead として恒久的に隠す
        (``RaggedLedger.extend_rows``) ので、値そのものは読まれない。
        列位置に意味を持たせているのは QSA だけで、そちらは帳簿から論理列を
        引き直すので (``_ragged_indexer_call``) 左詰めの影響を受けない。
        """
        self.keys = _cat_left_padded([self] + others, "keys", new_L, axis=2)
        self.values = _cat_left_padded([self] + others, "values", new_L, axis=2)
        idx = [c.indexer for c in [self] + others]
        if any(c.keys is not None for c in idx):
            self.indexer.keys = _cat_left_padded(idx, "keys", new_L, axis=1)


def _pad_left(arr: mx.array, pad: int, axis: int) -> mx.array:
    if pad <= 0:
        return arr
    spec = [(0, 0)] * arr.ndim
    spec[axis] = (pad, 0)
    return mx.pad(arr, spec)


def _cat_left_padded(caches: list, attr: str, width: int, axis: int) -> mx.array:
    """``caches`` の ``attr`` を ``axis`` 方向に ``width`` 列へ左詰めしてから
    軸 0 で連結する。

    片方だけ空のキャッシュは受けない -- 同じモデルで prefill を通った行なら
    埋まっているかどうかは必ずそろうので、そろっていなければ帳簿とキャッシュ
    のどちらかが壊れている。黙ってゼロで埋めると、そのずれが「たまに品質が
    落ちる」形で出て追えなくなる。
    """
    parts = []
    for c in caches:
        arr = getattr(c, attr)
        if arr is None:
            raise ValueError(f"join する行の {attr} が空 (キャッシュの不整合)")
        parts.append(_pad_left(arr, width - arr.shape[axis], axis))
    return mx.contiguous(mx.concatenate(parts, axis=0))


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
        #
        # 走行中の join (`extend_rows`) だけは、priming 窓の幅が行ごとに違う
        # ので列数がそろわない。そのときは短い側を**左**にゼロで詰め、
        # 詰めた列数を `_pad` に持って mask で隠す。base は `-pad` になるので
        # 論理位置 (offset) は詰める前と同じまま。
        self._pad = [0] * batch_size
        self._base = mx.zeros(batch_size, dtype=mx.int32)
        self.indexer = _arch()._IndexerCache()
        # `Attention.__call__` が QSA に渡すのは KV キャッシュではなく
        # この `_IndexerCache` なので、行別の列対応を引ける先をここに
        # 挿しておく (`_wide_qkv` / `_gather_stats` と同じ属性注入の作法)。
        # `_ragged_indexer_call` はこれが無ければ本家に素通しする。
        self.indexer.qsa_owner = self

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
        """必要なのは新規 T 列どうしの因果性と、join で左に詰めた列を隠すこと。

        詰めていない (`_pad` が全部 0) ときは全列が実データなので、T==1
        (ドラフトは 1 段ずつ) なら mask 自体が要らない -- join を使わない
        経路では以前と 1 ビットも変わらない。
        """
        padded = any(self._pad)
        if T == 1 and not padded:
            return None
        total = self.size()
        cols = mx.arange(total)
        q = total - T + mx.arange(T)
        causal = cols[None, :] <= q[:, None]  # (T, total)
        if not padded:
            return causal[None, None]
        live = cols[None, :] >= mx.array(self._pad)[:, None]  # (B, total)
        return (causal[None] & live[:, None, :])[:, None]

    # ---- QSA の列対応 (`RaggedAttnCache` と同じ 3 つ) ----------------
    #
    # こちらは穴が空かない (受理数が行ごとに違ってもドラフトの列は全行そろって
    # 伸びる) ので、生きていない列は join で左に詰めたぶんだけ。よって
    # 論理列 c は物理列 ``_pad[b] + c`` で、対応表は cumsum を経由せずに
    # 直接書ける。プロンプトが長いと priming 窓が PRIME_WINDOW (= 2048、
    # indexer_budget と同値) に達するので、**ここでも QSA は発火する**。
    def qsa_uniform(self, S: int) -> bool:
        return not any(self._pad)

    def qsa_max_len(self, S: int, kv_phys: int) -> int:
        return kv_phys - min(self._pad) if self._pad else 0

    def qsa_logical(self, S: int, kv_phys: int) -> _QSAMap:
        max_len = self.qsa_max_len(S, kv_phys)
        pad = mx.array(self._pad, dtype=mx.int32)[:, None]
        cols_p = mx.arange(kv_phys, dtype=mx.int32)[None]
        phys2log = mx.where(cols_p >= pad, cols_p - pad, mx.array(max_len, mx.int32))
        log2phys = mx.minimum(
            mx.arange(max_len, dtype=mx.int32)[None] + pad, kv_phys - 1
        )
        kv_lens = mx.array([kv_phys - p for p in self._pad], dtype=mx.int32)
        return _QSAMap(log2phys, phys2log, kv_lens, max_len)

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
        rows = idx.tolist()
        self._pad = [self._pad[b] for b in rows]
        self._base = self._base[idx]

    def extend_rows(self, others: list["RaggedDraftCache"]) -> None:
        """``others`` の行を軸 0 に足す (走行中の join)。

        列数は行ごとに「priming 窓 + 消化したラウンド数」なので、新入りと
        走行中でそろわない。長い方に合わせて短い方を**左**にゼロで詰め、
        詰めた列数を ``_pad`` に足して mask で隠す。位置は詰める前のまま
        (base を同じだけ下げる) なので、ヘッドから見た角度は動かない。

        ここが多少ずれても**出力は変わらない** -- このキャッシュはドラフトを
        引くためだけのもので、出すトークンは本体の検証フォワードの logits から
        来る (`FlashSpecEngine._prime_draft_cache` の注記)。ずれるのは受理率
        だけ。それでも隠すのは、ゼロ列を見に行かせる理由が無いから。
        """
        caches = [self] + list(others)
        sizes = [c.size() for c in caches]
        n = max(sizes)
        # 詰める列数は「連結する前」の列数から決める (self は下で書き換わる)
        pads = [
            [p + (n - s) for p in c._pad] for c, s in zip(caches, sizes)
        ]
        if n:
            self.keys = _cat_left_padded(caches, "keys", n, axis=2)
            self.values = _cat_left_padded(caches, "values", n, axis=2)
            idx = [c.indexer for c in caches]
            if any(c.keys is not None for c in idx):
                self.indexer.keys = _cat_left_padded(idx, "keys", n, axis=1)
        self._pad = [p for row in pads for p in row]
        self._base = mx.array([-p for p in self._pad], dtype=mx.int32)


def _extend_arrays_rows(dst, srcs: list) -> None:
    """``ArraysCache`` (GDN/PLE/n-gram) の行を軸 0 に足す。

    ``ArraysCache.extend`` を使わないのは、あちらが空スロットをゼロで埋める
    ため -- ここでは埋まり方がそろっていないこと自体が不整合なので、
    ``_cat_left_padded`` と同じ理由で黙って通さない。列を持たない固定サイズの
    状態 (GDN の再帰状態と conv 窓、PLE の conv 窓、n-gram の直前文脈) だけ
    なので、そろえる作業は要らない。
    """
    for j in range(len(dst.cache)):
        parts = [dst.cache[j]] + [s.cache[j] for s in srcs]
        if all(p is None for p in parts):
            continue
        if any(p is None for p in parts):
            raise ValueError(f"join する行の再帰状態 slot{j} がそろっていない")
        dst.cache[j] = mx.contiguous(mx.concatenate(parts, axis=0))


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


# ------------------------------------------------------------ QSA (indexer)


def _ragged_indexer_call(self, x, rope, cache, offset, positions, orig):
    """``QSAIndexer.__call__`` の行別版。ブロック境界を**行の論理列**の上で
    引き直す。

    本家はブロック格子を物理列で切る (``block_starts = arange(n) * cr``)。
    dead slot を挟んだバッチ列でそれをやると 3 つ同時に壊れる:

    1. **プーリングの中身**が壊れる。ブロックの平均に dead 列 (棄却された
       ドラフトの鍵) が混ざる。混ざり方は行ごとに違う。
    2. **rope の角度**がずれる。pooled 鍵は ``block_starts`` そのもので
       回すが、行 b の物理列 i の本当の位置は i ではない。
    3. **top-k が dead 列を選ぶ。**ブロックが選ばれた時点でその cr 列は
       まるごと可視になるので、棄却したはずのトークンが見える。

    直し方は「行 b の生きている列を並べ直した論理列」の上でブロックを
    切ること (``_QSAMap``)。dead slot は論理列に現れないので、1 と 3 は
    構成上起きない。2 は論理位置 = ``positions`` (帳簿の論理長 + s) で回す。
    ブロック格子は論理列の上では**全行そろう** (どの行も論理 0 から cr 刻み)
    ので、pooled の rope 角は本家と同じ 1 本で済む。

    行ごとに違うのは「何ブロックまでが実在するか」(``kv_len // cr``、
    ``row_blocks``) だけで、そこから先の端数列は本家と同じく因果窓で見せる。
    行の論理 kv 長が ``token_budget`` 以下の行は ``row_blocks`` を 0 に潰す
    --- ブロック選択には一切乗せず、行全体を「格子の外」として因果窓
    フォールバックに落とす。ブロック選択に乗せたまま「候補が block_topk
    以下だから top-k が全部拾う」という理屈は成り立たない -- クエリ自身が
    属するブロックは ``block_end <= q_col`` を満たさない限り (端の列で
    ない限り) 候補にすら入らないので、budget 以下の短い行でも自分の直近
    列が見えなくなっていた。単独実行のその行が疎化されない (``__call__``
    が None を返す) のと一致させるには、行ごと causal に落とすしかない。

    ``dead slot も右パディングも無いラウンド``では物理列と論理列が全行
    一致するので、**本家をそのまま呼ぶ** (pooled の増分キャッシュもそのまま
    効く)。判定は帳簿の Python 値だけなので、QSA が働かない短い文脈で
    余計な仕事は増えない。
    """
    from mlxturbo import qsa_tail as _qsa_tail

    Q = _arch()
    # ``cache`` は KV キャッシュではなく ``_IndexerCache``。行別の列対応は
    # そこに挿してある ``qsa_owner`` (RaggedAttnCache / RaggedDraftCache) が持つ。
    owner = getattr(cache, "qsa_owner", None) if cache is not None else None
    if owner is None or positions is None or owner.qsa_uniform(x.shape[1]):
        # ここは**キャッシュに触る前**でなければならない (触ってから本家に
        # 渡すと update が二重に走る)
        return orig(self, x, rope, cache, offset, positions)

    B, S, _ = x.shape
    cr = self.compress_ratio
    qk = self.index_qk_proj(x)
    split = self.n_heads * self.head_dim
    q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
    raw_k = qk[..., split:].reshape(B, S, self.head_dim)
    raw_k = cache.update(raw_k)
    kv_phys = raw_k.shape[1]

    # 疎化が要るかは**行ごとの論理 kv 長**で決める (物理列数ではない)。
    # どの行も budget に収まるなら本家と同じく None -- 呼び出し側の
    # `final_mask` が帳簿のマスクだけで組む。
    if owner.qsa_max_len(S, kv_phys) <= self.token_budget:
        return None

    m = owner.qsa_logical(S, kv_phys)
    n_len = m.max_len
    n_blocks = n_len // cr

    # 論理順に並べ直した raw 鍵。dead slot はここで落ちる。
    gidx = mx.broadcast_to(m.log2phys[:, :, None], (B, n_len, self.head_dim))
    raw_log = mx.take_along_axis(raw_k, gidx, axis=1)

    # pooled: 本家の非キャッシュ経路と同じ式 (増分キャッシュは使わない --
    # あちらは物理ブロック番号で持つので論理格子と噛み合わない。ブロックを
    # 毎ラウンド組み直す分だけ indexer が重くなるが、行別の論理 pooled を
    # 持ち回るのは別の仕事なのでここではやらない)。
    pooled = raw_log[:, : n_blocks * cr].reshape(B, n_blocks, cr, self.head_dim)
    pooled = self.k_layernorm(
        pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
    )
    block_starts = mx.arange(n_blocks) * cr
    cos_k, sin_k = rope(block_starts[None, :])
    pooled = Q._rope_partial(pooled, cos_k, sin_k)

    q_col = positions  # (B, S) 行ごとの論理位置
    cos_q, sin_q = rope(q_col)
    q = self.q_layernorm(q)
    q = Q._rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])

    scores = mx.einsum(
        "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
    )
    scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

    # 候補は「そのクエリの過去に丸ごと収まっているブロック」かつ「その行に
    # 実在するブロック」。後者は本家に無い条件で、行の論理 kv 長が短いぶん
    # だけブロック数も少ないことを表す (実在しないブロックの pooled は
    # 詰め物から作られた値なので、候補に入れてはいけない)。
    block_end = block_starts + cr - 1
    # budget 以下の行は単独実行なら疎化されない (`__call__` が None を返す)。
    # ここでは複数行をまとめて処理するので、そういう行だけ実在ブロック数を
    # 0 に潰す (B-1)。`in_block` (下) が全列 False になり、素の causal
    # フォールバックが行全体に効く。潰さないと、クエリ自身が属するブロックは
    # `block_end <= q_col` を満たせず候補にも入らないので、budget 以下でも
    # 直近の列が消える。
    row_blocks = mx.where(
        m.kv_lens <= self.token_budget,
        mx.zeros_like(m.kv_lens),
        m.kv_lens // cr,
    )  # (B,)
    visible = (block_end[None, None, :] <= q_col[:, :, None]) & (
        mx.arange(n_blocks)[None, None, :] < row_blocks[:, None, None]
    )
    if _qsa_tail.TIEBREAK:
        # 単一系列側 (`QSAIndexer._pooled_and_top`) と同じ同点 bias。ここを
        # 抜かすと TIEBREAK=1 のとき batch と solo でブロック選択がずれる。
        scores = scores - (
            mx.arange(n_blocks, dtype=mx.float32) * _qsa_tail.TIEBREAK_EPS
        )
    scores = mx.where(visible, scores, -mx.inf)

    k = min(self.block_topk, n_blocks)
    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
    top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
    keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[
        ..., :n_blocks
    ]

    # ブロック -> 論理トークン列。行 b の ``row_blocks[b] * cr`` から後ろは
    # その行の端数 (ブロック格子の外) なので、本家と同じく因果窓で見せる。
    # 行によってその境目が違うのがここの肝。
    keep_log = mx.repeat(keep_block, cr, axis=-1)  # (B, S, n_blocks*cr)
    pad = n_len - n_blocks * cr
    if pad:
        keep_log = mx.concatenate(
            [keep_log, mx.zeros((B, S, pad), dtype=mx.bool_)], axis=-1
        )
    cols = mx.arange(n_len)
    causal = cols[None, None, :] <= q_col[:, :, None]
    in_block = cols[None, None, :] < (row_blocks * cr)[:, None, None]
    # 可視ブロックが 1 つも無い行の救済 (本家 `__call__` 末尾と同じ)。
    # sparse が非 None のとき Attention は causal を捨てる規約なので、
    # ここで開けないとその行の mask が全面 -inf になって未来まで見える。
    need = (q_col < cr - 1)[:, :, None]
    if _qsa_tail.MODE == "query":
        # 論理列の上でのクエリごと tail (`mlxturbo/qsa_tail.py`)。
        # 列 [cr*floor((q+1)/cr), q] を必ず可視にする。ブロック格子の外
        # (端数域) だけでなく**格子の途中にも出る**ので、`in_block` の外を
        # まるごと causal に開ける global 側とは開け方が違う。
        # `row_blocks == 0` の行 (論理 kv 長が budget 以下 = 単独実行なら
        # 疎化されない行、B-1) は global 側と同じく行全体を causal に落とす
        # --- そこだけは tail の話ではなく「疎化しない」という別の規約。
        own = ((q_col + 1) // cr) * cr                       # (B, S)
        own_keep = (cols[None, None, :] >= own[:, :, None]) & causal
        keep_log = mx.where(in_block, keep_log, mx.array(False)) | own_keep
        dense_row = (row_blocks == 0)[:, None, None]
        keep_log = mx.where(need | dense_row, causal, keep_log)
    else:
        keep_log = mx.where(in_block & ~need, keep_log, causal)
    keep_log = keep_log & (cols[None, None, :] < m.kv_lens[:, None, None])

    # 論理 -> 物理。dead slot と右パディングは sink 列 (常に False) を指す。
    keep_log = mx.concatenate(
        [keep_log, mx.zeros((B, S, 1), dtype=mx.bool_)], axis=-1
    )
    pidx = mx.broadcast_to(m.phys2log[:, None, :], (B, S, kv_phys))
    keep = mx.take_along_axis(keep_log, pidx, axis=-1)
    return keep[:, None]  # (B, 1, S, kv_phys)


# ------------------------------------------------------------ attention hook


@contextmanager
def ragged_attention():
    """このラウンドの検証フォワード限定で ``Attention`` の 2 つのシームと
    ``QSAIndexer`` の 2 つを差し替える。フォワード本体は本家
    (``mlxturbo/_vendor/qwen4_exp.py``) のままで、違うのは次の 4 点だけ。

    1. RoPE の位置 (``Attention._positions``): ``cache.offset`` を (B,) の
       論理位置として扱う (本家は python int の物理列位置)。
       ``RaggedAttnCache.offset`` が ``RaggedLedger.valid_len_array()`` を
       返すので、ここでは ``offset[:, None] + arange(S)`` を常に使うだけで
       よい。QSA に渡す列位置は ``cache.size()`` (物理列数) のまま。
    2. mask (``Attention._final_mask``): ``Qwen4ExpModel.__call__`` が渡す
       ``mask`` 引数は ``create_attention_mask(h, [attn_cache])`` の結果で、
       単一キャッシュではなくリストを渡すため必ず "causal"/None に潰れる
       (mlxturbo/batch.py の docstring 項目 4 と同じ理由)。dead slot を
       知っているのは ``cache.ledger`` だけなので、渡された mask は無視して
       ``cache.round_mask(S)`` で組み直す。QSA が活性なら、そこへ
       ``sparse`` を連言で重ねる (本家の「sparse があれば causal を捨てる」
       規約は使えない -- dead slot を知っているのは帳簿側だけなので)。
    3. QSA (``QSAIndexer.__call__``): ブロック境界を行の論理列で引き直す
       (``_ragged_indexer_call``)。
    4. ``QSAIndexer.select_blocks``: 常に None を返す。段 3(b) の gather 経路
       (``MLXTURBO_GATHER_ATTN=1``、既定 off) は物理列でブロックを集めるので
       行別の境界に対応していない。``Attention._gather_forward`` は None を
       受けると通常経路 (加算マスク) に落ちるので、黙って間違えることは無い
       (キャッシュにも触らずに返すので二重更新も起きない)。
    """

    Q = _arch()
    orig_positions = Q.Attention._positions
    orig_final_mask = Q.Attention._final_mask
    orig_make_masks = Q.Qwen4ExpModel._make_masks
    orig_indexer_call = Q.QSAIndexer.__call__
    orig_select_blocks = Q.QSAIndexer.select_blocks

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
        # 帳簿のマスクが「生きている過去 + 因果」で、sparse が QSA の
        # ブロック選択。dead slot は帳簿側にしか出てこないので、本家のように
        # sparse だけを見ることはできない。
        led = cache.round_mask(S)
        if sparse is None:
            return led
        return sparse if led is None else (led & sparse)

    def indexer_call(self, x, rope, cache, offset, positions=None):
        return _ragged_indexer_call(
            self, x, rope, cache, offset, positions, orig_indexer_call
        )

    def select_blocks(self, x, rope, cache, offset, positions=None):
        return None

    Q.Attention._positions = positions
    Q.Attention._final_mask = final_mask
    Q.Qwen4ExpModel._make_masks = make_masks
    Q.QSAIndexer.__call__ = indexer_call
    Q.QSAIndexer.select_blocks = select_blocks
    try:
        yield
    finally:
        Q.Attention._positions = orig_positions
        Q.Attention._final_mask = orig_final_mask
        Q.Qwen4ExpModel._make_masks = orig_make_masks
        Q.QSAIndexer.__call__ = orig_indexer_call
        Q.QSAIndexer.select_blocks = orig_select_blocks


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


def sample_positions(lg: mx.array, temp: float, sampler) -> mx.array:
    """logits (B, S, vocab) から各位置のトークン (B, S) を引く。

    ``FlashSpecEngine._verify`` / ``_sample`` の B 行版。貪欲のときは以前と
    同じ ``mx.argmax`` をそのまま通る。**バッチ内の全行が同じサンプラーを
    共有する**前提 (1 回の呼び出しで全行ぶんを引く) なので、パラメータの
    違う要求を同じバッチに入れてはいけない -- admission 側の責務
    (``BatchSpecCoordinator._sampling_key``)。

    chunked prefill の車線 (``SpecPrefillLane``) も 1 個目のトークンをこれで
    引く -- 単独経路とバッチ経路でサンプリングの形をずらさないため。
    """
    if sampler is None and temp <= 0:
        return mx.argmax(lg, axis=-1)
    b, s, v = lg.shape
    flat = lg.reshape(b * s, v)
    if sampler is not None:
        return sampler(flat).reshape(b, s)
    return mx.random.categorical(flat.astype(mx.float32) / temp).reshape(b, s)


def prime_draft_rows(eng, ids: mx.array, hyper: mx.array, lengths: list[int], w: int):
    """MTP ヘッドに各行の末尾 w 対を流した ``RaggedDraftCache`` を返す。

    ``ids`` (B, L) と ``hyper`` (B, L, H) は同じ位置をそろえて渡すこと。
    行 b が使うのはトークン位置 ``n_b-w .. n_b-1`` と、その 1 つ前の hyper
    (``FlashSpecEngine._prime_draft_cache`` の対の取り方と同じ)。幅 w は全行
    そろえる -- MTP ヘッドの位置は priming 窓の先頭からの相対なので、幅が
    そろえば行の意味もそろう。
    """
    B = len(lengths)
    cache = RaggedDraftCache(B)
    if w < 1:
        return cache
    tok_idx = [[n - w + j for j in range(w)] for n in lengths]
    hyp_idx = [[n - w - 1 + j for j in range(w)] for n in lengths]
    toks = mx.take_along_axis(ids, mx.array(tok_idx), axis=1)
    hy = mx.take_along_axis(
        hyper,
        mx.broadcast_to(mx.array(hyp_idx)[..., None], (B, w, hyper.shape[2])),
        axis=1,
    )
    embeds = eng.model.model.embed_tokens(toks)
    with ragged_attention():
        # mask は渡さない。`ragged_attention` の `_final_mask` が
        # `cache.round_mask` で組み直すので、ここで作っても捨てられる
        # (しかも update_and_fetch の前なので列数が 0 の壊れた形になる)
        out = eng.mtp(embeds, hy, eng.rope, None, cache, cache.indexer)
        mx.eval(out)
    return cache


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

    段 3(b) の gather 経路 (``MLXTURBO_GATHER_ATTN``)。QSA そのものは
    行ごとにブロック境界を引き直して対応済み (モジュール docstring 参照)。

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
        """検証フォワードの logits (B, S, vocab) から各位置のトークン (B, S)。"""
        return sample_positions(lg, self.temp, self.sampler)

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
        self._mtp_cache = prime_draft_rows(
            self.eng, self.ids, hyper, self.lengths, self.prime_window
        )

    # ---- 走行中の join -----------------------------------------------

    @classmethod
    def from_prefilled(cls, engine, rows: list, temp: float = 0.0, sampler=None):
        """chunked prefill を終えた行 (``PrefilledRow``) からバッチを起こす。

        1 行目のキャッシュと帳簿を**そのまま引き取る** -- 車線
        (``SpecPrefillLane``) は最初からこのクラスと同じ
        ``make_ragged_cache`` で組んであるので、作り直す理由が無い。2 行目
        以降は ``join`` に回す。
        """
        if not rows:
            raise ValueError("rows が空")
        first, rest = rows[0], rows[1:]
        self = cls.__new__(cls)
        self.eng = engine
        self.model = engine.model
        self.B = 1
        self.depth = engine.depth
        self.temp = float(temp)
        self.sampler = sampler
        self.lengths = [first.length]
        self.L = first.length
        self.ids = None  # 一括 prefill を通らないので持たない
        self.caches, self.ledger = first.caches, first.ledger
        self.prime_window = first.window
        self.out = [[first.first_token]]
        self.rounds = 0
        self.accepted = 0
        self._cur = first.cur
        self._hyper_prev = first.hyper_prev
        self._mtp_cache = first.mtp_cache
        if rest:
            self.join(rest)
        return self

    def join(self, rows: list) -> None:
        """走行中のバッチに、prefill を終えた行を足す。

        新入りの KV は自分の実位置で回転済みの ``(1, H, P, D)``、走行中は
        ``(B, H, L, D)`` で、P と L はそろっていない。そろえ方は
        ``compact()`` が作る形と同じ「**左詰め**、足りない列は dead」:

        - 物理列数を ``max(L, max P)`` に取り、短い側を左にゼロで詰める
          (``RaggedAttnCache.extend_rows``)。
        - 帳簿は行ごとに ``[False]*(new_L - valid) + [True]*valid``
          (``RaggedLedger.extend_rows``)。以後のマスクがゼロ列を隠す。
        - RoPE の位置は ``_valid_len`` から引くので、左に何列足しても回転は
          動かない。物理列の番号を見ているのは QSA だけで、そちらは帳簿から
          論理列を引き直す。
        - 再帰系 (GDN/PLE/n-gram) は列を持たない固定サイズの状態なので、
          バッチ軸に連結するだけ。
        - MTP のドラフトキャッシュは列数がそろわないので、同じく左詰め
          (``RaggedDraftCache.extend_rows``)。

        先に ``compact()`` を通して dead slot を落としてから足す。新入りに
        走行中の無駄を相続させない、というだけでなく、``new_L`` が小さくなる
        ぶん join そのものの pad も減る。
        """
        if not rows:
            return
        if self.ledger.L > self.ledger.max_valid_len():
            self.ledger.compact(self.model, self.caches)
        new_L = max([self.ledger.L] + [r.length for r in rows])
        for i, layer in enumerate(self.model.model.layers):
            src = [r.caches[i] for r in rows]
            if layer.layer_type == "full_attention":
                self.caches[i].extend_rows(src, new_L)
            else:
                _extend_arrays_rows(self.caches[i], src)
        self.ledger.extend_rows([r.length for r in rows], new_L)
        self._mtp_cache.extend_rows([r.mtp_cache for r in rows])
        self._cur = mx.concatenate([self._cur] + [r.cur for r in rows], axis=0)
        self._hyper_prev = mx.concatenate(
            [self._hyper_prev] + [r.hyper_prev for r in rows], axis=0
        )
        self.lengths = self.lengths + [r.length for r in rows]
        self.out = self.out + [[r.first_token] for r in rows]
        self.B = len(self.out)

    # ---- rounds -----------------------------------------------------

    def _depth_for_round(self, depth: "int | None") -> int:
        """このラウンドのドラフト数。文脈長連動の政策をここで被せる。

        単独経路の ``FlashSpecEngine._effective_depth`` と同じ ``choose_depth``
        を通す。論理長が疎注意の境界 (``indexer_budget``) を越えたら 1 に
        落とす政策で、単独経路の実測は depth 2 を基準に 2.6k で -3.3% /
        4k で -3.1% / 17k で -10.9% (``mlxturbo/spec_flash.py`` の
        ``DEPTH_CONTEXT_LIMIT`` の注記、複数プロンプト x 512 の回文順掃引)。

        バッチではラウンドが ``(B, T+1)`` の矩形なので、1 位置足す費用は行数
        ぶん重なる一方、受理が増える利得は行ごとに独立。**効く向きは単独と
        同じで、大きさはむしろ大きい。**行ごとに T を変えられないので、位置は
        **論理長の最大**で代表する。全行が境界の内側にいる限り (短いプロンプト
        のバッチ) この規則は発火しないので、そちらの振る舞いは変わらない。

        被せ方は上限としてだけ。``choose_depth`` は境界の内側では引数の
        ``depth`` をそのまま返すが、境界の外では 1 を返すので、スケジューラが
        予算の都合で 0 (素の decode) を指示したラウンドを 1 に押し上げて
        しまわないよう ``min`` を取る。

        ``MLXTURBO_BATCH_DEPTH_CTX=1`` で有効。**既定 off = 移す前のバッチの
        挙動そのまま。**env を**毎ラウンド読む**のは、1 つのサーバープロセスの
        中で同じ負荷を on/off 交互に流せるようにするため -- プロセスを分けた
        比較は熱とメモリ状態が変わり、狙っている差 (数 %) と同じ桁のドリフトが
        乗る (``MLXTURBO_PIPELINE`` が ``spec_flash.py`` の decode ループで
        毎ラウンド読んでいるのと同じ形)。読むのはラウンドに 1 回で、
        1 ラウンドは数十 ms なので費用は無視できる。

        **効果は未確認 (2026-09-02)。**コーディネータ経由 (chunked prefill +
        走行中の join)、1880 トークン x 2 本 x 512 生成、1 プロセス内で回文順
        (on/off/off/on)、4 ラウンドとも joins=1:

            壁時計    on 23.20s  off 23.03s  (+0.7%)
            decode 中央値  on 31.8  off 32.5 tok/s

        採用の線は「2% 以上縮む」だったので届かない。理由は発火の窓が狭い
        こと -- 1880 + 512 = 2392 なので、規則が効くのは decode の後ろ 2/3
        だけで、壁時計はプロンプト 2 本の prefill が半分を占める。
        4k でも測ったが、**そちらはドリフトで判定不能**だった (同じ条件の
        ラウンドが 33.94s と 61.35s、回文順を逆にすると符号も反転)。

        一方**閉じたバッチ** (``BatchSpecGenerator`` を直接、一括 prefill で
        1 ラウンド目から全行そろっている形) では、同じ 1 プロセス回文順で
        1880x2 が -3.3%、3962x2 が -3.1% (256 生成) / -4.1% (512 生成)、
        ラウンド数も 248→294 / 134→159 / 265→319 と規則の発火が確認できた。
        単独経路の記録とも大きさが一致する。**機構としては効いているが、
        実運用の経路 (コーディネータ) では取り分が窓に埋もれる。**
        """
        from .spec_flash import choose_depth

        base = self.depth if depth is None else max(0, depth)
        if os.environ.get("MLXTURBO_BATCH_DEPTH_CTX", "0") != "1":
            return base
        return min(base, choose_depth(
            self.ledger.max_valid_len(), base, self.eng.depth_ctx_limit,
            batch_size=self.B,
        ))

    def step(self, truncate=None, depth: int | None = None) -> list[list[int]]:
        """1 ラウンド進めて、行ごとの新規トークンを返す。

        ``truncate`` (省略可) は ``(b, toks) -> 残す個数 (1 以上)``。eos や
        残り max_tokens で行 b の出力をこのラウンドの途中で打ち切るための口で、
        単独経路 (``FlashSpecEngine.generate_stream``) が ``cut``/``remaining``
        で ``vals`` を切り詰めてから ``rollback(keep=len(vals))`` を呼ぶのと
        同じ形 -- 打ち切った分は受理しなかったことにするので、キャッシュと
        出力が食い違わない。

        ``depth`` (省略可) はこのラウンドだけのドラフト数。スケジューラが
        矩形 ``B*(1+k)`` とトークン予算に合わせて毎ラウンド決める
        (``BatchSpecCoordinator._round_depth``)。``0`` を渡すとドラフトを
        引かない素の decode になる -- そのラウンドはドラフトキャッシュに
        1 列も足さないので、ヘッドから見た履歴に穴が 1 つ空く。受理率が
        少し動くだけで、出す**トークンは変わらない** (出力は本体の検証
        フォワードの logits から来る)。

        そのうえで文脈長連動の上限を被せる (``_depth_for_round``)。
        """
        from .spec_flash import _staged_forward

        eng = self.eng
        d = self._depth_for_round(depth)
        drafts = []
        if d:
            with ragged_attention():
                drafts = eng._draft_chain(
                    self._cur, self._hyper_prev, self._mtp_cache, d
                )
        pair = mx.concatenate([self._cur] + drafts, axis=1) if drafts else self._cur
        total = pair.shape[1]
        pre_ctx = snapshot_pre_ctx(self.model, self.caches)
        with batched_capture(self.model) as cap:
            # 段階投入 (spec_flash._staged_forward)。単独経路の検証
            # フォワードはこれを通っていて、グラフ構築中の GPU 泡を刈る分
            # だけ速い。バッチ側が `model(...)` を直接呼んでいると、B=1 でも
            # 単独に負ける (実測 0.82x)。計算内容は同じ。
            lg = _staged_forward(self.model, pair, self.caches)
            nxt = self._sample_rows(lg)            # (B, T+1)
            dv = mx.concatenate(drafts, axis=1) if drafts else None  # (B, T)
            if dv is None:
                mx.eval(nxt, cap.hyper)
            else:
                mx.eval(nxt, dv, cap.hyper)
        self.rounds += 1

        nxt_l = nxt.tolist()
        dv_l = dv.tolist() if dv is not None else [[]] * self.B
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
        # 空の list をそのまま渡すと float32 の配列になって gather が落ちる
        # (全行が同じラウンドで終わる / 最後の 1 行を退避する場合に通る)
        idx = mx.array(rows, dtype=mx.int32)
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

    def maybe_compact(self, waste_ratio: float = 1.5) -> bool:
        """dead slot が溜まってきたら詰める。詰めたら True。

        理由は無駄の 1 つだけ: 物理列が論理長の ``waste_ratio`` 倍を超えたら
        詰める。受理率が 1.6 tok/round 程度なので、詰めないと物理列は論理長の
        およそ ``(depth+1)/1.6`` 倍で伸び続け、attention の費用がそのぶん
        丸ごと無駄になる。

        以前はもう 1 つ「硬い上限」があった。物理列数が ``indexer_budget`` に
        届くと QSA が発火して ``ragged_attention`` が止まったので、境界の手前で
        必ず詰めていた。QSA を行ごとに引き直した今 (``_ragged_indexer_call``)
        物理列数に上限は無く、逆に境界で毎ラウンド詰めると長文で丸損になる
        (17k の物理列を毎ラウンド gather することになる) ので外した。
        """
        valid = self.ledger.max_valid_len()
        need = bool(valid) and self.ledger.L > valid * waste_ratio
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


# ------------------------------------------------------- chunked prefill


class PrefilledRow:
    """prefill を終えて、走行中のバッチに入れる用意ができた 1 行。

    ``BatchSpecGenerator.join`` / ``from_prefilled`` がそのまま受ける形で、
    中身は「1 行ぶんのキャッシュ一式 + 1 個目のトークン + 直前の hyper +
    priming 済みの MTP ドラフトキャッシュ」。
    """

    __slots__ = (
        "caches", "ledger", "length", "window",
        "first_token", "cur", "hyper_prev", "mtp_cache",
    )

    def __init__(self, caches, ledger, length, window, first_token, cur,
                 hyper_prev, mtp_cache):
        self.caches = caches
        self.ledger = ledger
        self.length = length
        self.window = window
        self.first_token = first_token
        self.cur = cur
        self.hyper_prev = hyper_prev
        self.mtp_cache = mtp_cache


class SpecPrefillLane:
    """1 本の要求のプロンプトを刻んで流す prefill 車線 (chunked prefill)。

    スケジューラは 1 ステップのトークン予算を、走行中の decode ラウンドと
    この車線で分け合う (``BatchSpecCoordinator._step``)。**フェーズを分けない**
    のが要点で、prefill が終わるまで decode を止めない代わりに、prefill も
    1 ステップで食う量が ``chunk`` に抑えられる。

    ## 1 行ずつなのはなぜか

    行を並べて刻む (B_p 行 x chunk の矩形) 形にはしていない。MLX のフォワードは
    矩形なので、1 ステップに違うトークン数の行を混ぜると全行が最大幅ぶんの
    計算を払う -- decode の行 (1+k 列) を 512 列の prefill と同じ矩形に入れた
    時点で、decode が 512 列ぶんを払う。だから 1 ステップは「decode の矩形」と
    「prefill の矩形」の 2 回に分かれ、prefill 側を複数行にしても
    (右パディングの無駄を別にすれば) FLOP は減らない。減るのはディスパッチ
    回数だけで、prefill は元から計算律速なので取り分が無い。1 行なら右
    パディングも dead slot も生じないので、帳簿は「全列 alive」のまま済む。

    ## 刻んでも結果が変わらない根拠

    2 回目以降のチャンクのマスクは「生きている過去 + 新規 T 列の因果」で、
    これは ``RaggedLedger.next_round_mask`` そのもの。再帰系は元から状態を
    持ち越して進む (単独経路の ``generate_stream`` も
    ``PREFILL_STEP_SIZE`` 幅で刻んでいる)。チャンク境界で変わるのは浮動小数の
    まとめ方だけ。
    """

    def __init__(self, engine, prompt_ids: list[int], temp: float = 0.0, sampler=None):
        from .spec_flash import PRIME_WINDOW

        if len(prompt_ids) < 2:
            raise ValueError("プロンプトは 2 トークン以上 (priming に 1 対要る)")
        self.eng = engine
        self.model = engine.model
        self.ids = list(prompt_ids)
        self.pos = 0
        self.temp = float(temp)
        self.sampler = sampler
        self.caches, self.ledger = make_ragged_cache(self.model, 1)
        # priming に要るのは末尾 w+1 対だけ。チャンクをまたいで hyper の
        # 尻尾だけを持ち回る (単独経路の HYPER_KEEP_CHUNKS と同じ狙い)
        self._keep = min(PRIME_WINDOW + 1, len(self.ids))
        self._hyper_tail = None
        self._row = None

    @property
    def remaining(self) -> int:
        return len(self.ids) - self.pos

    @property
    def finished(self) -> bool:
        return self._row is not None

    def result(self) -> PrefilledRow:
        if self._row is None:
            raise RuntimeError("まだ prefill が終わっていない")
        return self._row

    def advance(self, n: int) -> None:
        """次の ``n`` トークンを流す。最後のチャンクなら ``result()`` が揃う。"""
        from .spec_flash import capture

        n = max(1, min(int(n), self.remaining))
        last = self.remaining == n
        chunk = mx.array(self.ids[self.pos : self.pos + n])[None]
        logits = None
        # light=True: 幅がチャンク幅まで伸びるので、rollback 用の states_all を
        # 取ると macOS の memorystatus killer に消される (capture の docstring)
        with ragged_attention(), capture(self.model, light=True) as cap:
            if last:
                logits = self.model(chunk, cache=self.caches)
                mx.eval(logits, cap.hyper)
            else:
                # 途中のチャンクは lm_head を通す理由が無い (単独経路の
                # generate_stream も同じ切り方)
                mx.eval(self.model.model(chunk, cache=self.caches), cap.hyper)
        self.ledger.commit_round([n], n)
        self.pos += n
        self._push_hyper(cap.hyper)
        if last:
            self._finish(logits)

    def _push_hyper(self, hyper: mx.array) -> None:
        if self._hyper_tail is None:
            self._hyper_tail = hyper
        else:
            self._hyper_tail = mx.concatenate([self._hyper_tail, hyper], axis=1)
        if self._hyper_tail.shape[1] > self._keep:
            self._hyper_tail = mx.contiguous(self._hyper_tail[:, -self._keep :])

    def _finish(self, logits: mx.array) -> None:
        first = sample_positions(logits[:, -1:], self.temp, self.sampler)
        mx.eval(first)
        m = self._hyper_tail.shape[1]
        toks = mx.array(self.ids[-m:])[None]
        # w は一括 prefill 側と同じ式 (min(PRIME_WINDOW, プロンプト長-1))。
        # 尻尾を w+1 対ぶん持っているので m-1 がそれに一致する
        mtp = prime_draft_rows(self.eng, toks, self._hyper_tail, [m], m - 1)
        self._row = PrefilledRow(
            caches=self.caches,
            ledger=self.ledger,
            length=len(self.ids),
            window=m - 1,
            first_token=int(first[0, 0].item()),
            cur=first,
            hyper_prev=mx.contiguous(self._hyper_tail[:, -1:]),
            mtp_cache=mtp,
        )


# ------------------------------------------------------------- coordinator
#
# 上の部品をサーバーに配線する層 (--max-batch-spec)。形は
# `mlxturbo/batch.py` の BatchCoordinator に寄せてある: inbox に Admission を
# 積み、駆動ループを 1 本だけ executor (モデルを読んだ唯一の MLX ワーカー
# スレッド) に投げ、live な仕事が無くなったら抜ける。違いは駆動する対象
# だけ -- あちらは mlx_lm の BatchGenerator (投機なし)、こちらは上の
# BatchSpecGenerator (MTP 投機つき)。
#
# ## スケジューラ: 1 ステップに予算 1 つ (chunked prefill)
#
# フェーズを分けない。1 ステップは「トークン予算」を 1 つ持ち、
#
#   1. 先に RUNNING を進める。decode の準備ができた行 (残り 1 トークン) は
#      MTP の投機ラウンドで `1+k` トークン、prefill の途中の行は
#      `min(残り, chunk, 予算)` トークン。
#   2. 予算とスロットが余っていれば WAITING を許容する。許容した行はその場で
#      最初のチャンクを流すので、短いプロンプトなら 1 ステップで prefill を
#      終えて走行中のバッチに join する。
#
# prefill を刻むから、走行中に join できる。「まとめて 1 回の prefill」だと
# 途中参加のたびにバッチ全体が止まるので、閉じたバッチにするしかなかった
# (docs/BACKLOG.md の A-3)。**A-1 (2048 の上限) は別原因**だった -- 刻んでも
# KV は伸びるので、刻んだことでは解けない。そちらは QSA のブロック境界を
# 行ごとに引き直して外した (2026-09-02、`_ragged_indexer_call`)。
#
# ## 1 ステップが 2 回のフォワードに割れること (MLX 側の制約)
#
# vLLM-V1 は 1 ステップを 1 回の varlen フォワードで流すが、MLX のフォワードは
# 矩形で、この族の再帰系 (GDN/PLE/n-gram) も (B, S, ...) の矩形しか受けない。
# 行ごとにトークン数が違うと、全行が最大幅ぶんの計算を払う -- decode の行を
# 512 列の prefill と同じ矩形に入れた瞬間に decode が 512 列ぶんを払う。
# そこで**トークン数の同じ行だけを 1 つの矩形にまとめる**。実際には
# 「decode の矩形 (`B*(1+k)`)」と「prefill の矩形 (chunk)」の 2 回になる。
# 予算はその合計に掛かるので、スケジューラの意味論 (フェーズを分けない、
# 1 ステップの総トークン数を抑える) はそのまま。
#
# ## 決めたこと (admission / スケジューラの論点)
#
# 1. まとめる条件。同じ**サンプリング設定**であること。
#    `BatchSpecGenerator._sample_rows` が全行ぶんを 1 回の呼び出しで引くので、
#    行ごとに違うサンプラーを混ぜると分布が壊れる。seed 指定つきの要求は
#    混ぜない (`mx.random.seed` はプロセス全体の状態で、同居する他の行の乱数
#    まで動かしてしまうため) -- 直列経路に回す。待ち行列は**先頭だけ**を
#    許容の候補にする (FIFO)。設定の合う要求を先に拾うと、合わない要求が
#    いつまでも進まないため。
#
#    長さの比では割らなくなった (`bucket_batches` は削除)。まとめて 1 回の
#    prefill をしないので、右パディングの矩形がそもそも立たない。残るのは
#    「長い行と短い行が同じ物理列数を共有する」ぶんの attention で、これは
#    `maybe_compact` が論理長の最大まで詰めるところまでが上限
#    (`mlxturbo/batch.py` の左パディングと同じ性質のコスト)。
#
# 2. 途中で終わった列。終わった行はその場で完了させ、`retire()` で物理的に
#    落とす (バッチ軸のスライス)。dead slot の帳簿は元から行ごとなので、
#    行を抜くのに列の詰め直しは要らない。落とさない場合、短い行が長い行に
#    最後まで付き合って前進計算と KV を占め続ける。
#
# 3. 新しい要求。**走行中のバッチに入れる。**車線 (`SpecPrefillLane`) で
#    prefill を終えた行を `BatchSpecGenerator.join` が軸 0 に連結する。
#    KV のそろえ方は `compact()` が作る形と同じ左詰めで、詰めた列は帳簿が
#    dead として隠す。RoPE の位置は帳簿の論理長から引くので、左に何列足しても
#    回転は動かない。MTP の priming 窓も同じく左詰め (`RaggedDraftCache.
#    extend_rows`)。正しさは `tools/verify_batch_spec.py` (合成モデル、CPU) が
#    見る -- 一括 prefill と刻んだ prefill、join の前後で、出力が 1 本ずつの
#    貪欲デコードと一致すること。
#
# 4. メモリと preemption。`rows_fit` が「この行たちが最後まで走るのに要る
#    追加バイト数」を空きと比べる。足りなければ**優先度最低 = 最若手**の行を
#    退避する: 生成済みのトークンは保持したまま、`プロンプト + 生成済み` を
#    新しいプロンプトとして待ち行列の**先頭**へ戻す (vLLM の recompute 型)。
#    残り max_tokens は同じだけ減るので、`spec_batchable` の条件は退避の
#    前後で動かない (長さの上限が外れた今はそもそも長さを見ていない)。
#
# ## B=1 は既存の単独経路のまま (二重に守る)
#
# **入口で守る (1 段目)。**まとめる相手がいない要求は、そもそもこの機構に
# 入れない -- `server._resolve_batch_route` が `is_idle()` を見て通常経路
# (STATE.lock + セッション) に落とす。ここを守らないと、単独の要求が
# **セッション (プロンプトキャッシュの再利用) だけを失う**。バッチ経路が
# セッションを持たないのは B>1 では必然だが、B=1 では丸損で、実測で
# TTFT +0.32s / decode +4% として出た (2026-09-02、`is_idle` の docstring)。
#
# **駆動ループでも守る (2 段目)。**1 段目をすり抜けた場合 (入口の判定と
# 駆動の間に他の要求が終わった等) でも、走行中のバッチも車線も空で
# 待ち行列に 1 本しか無いときは
# `BatchSpecGenerator` を使わず、`FlashSpecRunner.generate`
# (= `FlashSpecEngine.generate_stream`) をそのまま呼ぶ。単独経路には次ラウンドの
# draft 先行投入が乗っていて、バッチ経路には無い (単体で測った記録が無いので
# 移していない -- 導入時の commit で段階投入と束ねて測られている)。文脈長連動の
# depth は 2026-09-02 に `_depth_for_round` として移したが、**既定 off**で、
# 1815 トークン x 2 本では効果が雑音に埋もれた (下の docstring 参照)。
# 楽観パイプラインは単独経路でも既定 off (短 +49.3% / 長 +34.6% の負け)。
# 単独リクエストをバッチ経路に流すと、それだけで
# 短 decode が落ちる (`bench/batch_b1_gate.py` が固定している線)。同時到着を
# 拾うために、1 本しか無いときだけ `wait_ms` (既定 15ms) だけ相方を待つ --
# 待つのは TTFT の手前だけで、decode の ms/token には入らない。
#
# 逆に言うと、**単独で走り出した要求に後から join はできない**。単独経路は
# 走り切るまで executor を占有し、その途中のキャッシュを外から触る口が無い。
# 走行中の join が効くのは「2 本以上でバッチが立ったあと」で、そこから先は
# 何本でも途中参加できる。単独で走っている間に来た要求を拾うには、単独側を
# preemption と同じ形で打ち切って (生成済みを保持して) バッチに組み直す
# ことになるが、それは B=1 の経路に手を入れる話なのでここではやらない。
import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def spec_batchable(model, prompt_len: int, max_tokens: int, depth: int) -> bool:
    """この要求をバッチ x 投機に入れてよいか。

    **長さの上限は無い (2026-09-02)。**以前は「`prompt_len + max_tokens +
    depth+1 <= indexer_budget`」を要求していた。QSA が活性化すると
    `ragged_attention` が `NotImplementedError` で止まったためで、実運用の
    要求 (プロンプト 2000 + 生成 128 = 2130 > 2048) が 1 本も通らなかった
    (`docs/BACKLOG.md`「同時実行の比較は A-1 が外れるまで測れない」)。
    QSA のブロック境界を行ごとに引き直したので (`_ragged_indexer_call`)、
    この上限は外れた。物理列数が境界を超えてもよい -- dead slot は論理列に
    現れないので、ブロック選択に混ざらない。

    残る条件は 2 つだけ。プロンプトが 2 トークン以上あること (priming に
    1 対要る) と、生成が 1 トークン以上あること。メモリの余裕は別口
    (`rows_fit` / `BatchSpecCoordinator._ensure_room`) で見る。

    ``model`` と ``depth`` は呼び出し側の互換のために残してある (長さの
    判定に使わなくなった)。
    """

    return prompt_len >= 2 and max_tokens >= 1


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


def rows_fit(model, remaining: list[int], depth: int, room: "int | None" = None) -> bool:
    """この行たちが最後まで走るのに要る**追加**バイト数が、いま空いているか。

    ``remaining`` は行ごとの残り生成トークン数 (max_tokens - 出した数)。
    ここから先に新しく確保するのは 2 つだけで、どちらも設定値ではなく
    モデルの形から計算する。

    1. KV。残りトークンぶんの full attention の k/v (+ indexer)。物理列は
       論理長より速く伸びるが、``maybe_compact`` が論理長の最大まで詰めるので
       上限はこれで足りる。
    2. capture。検証フォワードが 1 ラウンドぶん確保する ``states_all``
       (行あたり depth+1 位置)。1 位置あたりが KV の数千倍あるので、同時に
       何行流せるかを実際に決めているのはこちら (``capture`` の docstring)。

    ``free_bytes()`` は現在の常駐 (書き終わった KV を含む) を引いた値なので、
    ここで数えるのは**これから増える分**だけ。数えられない環境
    (Metal が無い等) では ``True`` -- 上限を掛けない、が以前からの扱い。

    実機 (128GB、重み 91GB) でここが効くのは B が数十のときだったが、
    ``indexer_budget`` の長さ制約が外れて 17k 級の要求が入るようになったので
    (2026-09-02)、B が小さくてもここが効く。黙って落ちる代わりに preemption へ
    倒すのがこの式の役目。
    """

    kv_per_tok = kv_bytes_per_token(model)
    cap_per_tok = capture_bytes_per_token(model)
    if room is None:
        room = free_bytes()
    if kv_per_tok is None or cap_per_tok is None or room is None:
        return True
    need = sum(max(0, r) for r in remaining) * kv_per_tok
    need += len(remaining) * (depth + 1) * cap_per_tok
    return need <= room


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
    # 到着順の通し番号。preemption の「優先度最低 = 最若手」の判定に使う。
    # 付けるのは駆動スレッド (`BatchSpecCoordinator._drain`)
    seq: int = 0
    # 退避された回数。退避のたびに「プロンプト + 生成済み」で prefill を
    # やり直すので、TTFT の意味が変わることを /health 等から見えるようにする
    preempted: int = 0
    # 復帰prefillが実際に完了した分だけ数える。退避直後やcancelされたままの
    # admissionは加算しない。
    preemption_recomputed_tokens: int = 0
    recompute_events: int = 0


# 1 ステップのトークン予算。decode の矩形と prefill のチャンクで分け合う
TOKEN_BUDGET = 2048
# prefill を 1 ステップで進める上限。走行中の decode を止める長さの上限でも
# ある (長いプロンプトはこの幅で刻まれ、間に decode ラウンドが挟まる)
PREFILL_CHUNK = 512
# 1 ステップで許容を検討する待ち行列の長さ。これを超えた分は次のステップに
# 回るだけで、捨てはしない (サーバー側の上限は --max-queue)
MAX_WAITING = 64
# 検証フォワードの矩形 B*(1+k) の上限。B が増えたら depth を削る
RECT_LIMIT = 8


class BatchSpecCoordinator:
    """バッチ x 投機のスケジューラ (chunked prefill)。

    上の「スケジューラ」節がこのクラスの中身。走行中は executor を占有するので、
    駆動ループは仕事が無くなったら必ず抜ける (`mlxturbo.batch.
    BatchCoordinator` と同じ約束)。
    """

    def __init__(
        self,
        runner,
        executor,
        max_batch: int,
        eos_ids,
        wait_ms: int = 15,
        token_budget: int = TOKEN_BUDGET,
        prefill_chunk: int = PREFILL_CHUNK,
        max_waiting: int = MAX_WAITING,
    ):
        self.runner = runner
        self.engine = runner.engine
        self.model = runner.engine.model
        self.executor = executor
        self.max_batch = max(1, max_batch)
        self.eos_ids = set(eos_ids)
        self.wait_ms = max(0, wait_ms)
        self.token_budget = max(1, token_budget)
        self.prefill_chunk = max(1, prefill_chunk)
        self.max_waiting = max(1, max_waiting)
        self._inbox: "_queue.SimpleQueue[SpecAdmission]" = _queue.SimpleQueue()
        self._guard = threading.Lock()
        self._active = False
        # 空きの見積もり。テストから差し替えられるように属性で持つ
        self._free_bytes = free_bytes
        # 観測用 (bench / tools/verify_batch_spec.py が見る)
        self.joins = 0
        self.preemptions = 0
        self.preemption_recomputed_tokens = 0
        self.solo_runs = 0
        # 以下は駆動スレッドだけが触るスケジューラの状態
        self._waiting: list[SpecAdmission] = []
        self._gen = None
        self._rows: list[SpecAdmission] = []
        self._lane = None
        self._lane_adm: "SpecAdmission | None" = None
        self._temp = 0.0
        self._sampler = None
        self._seq = 0

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
            "preemptions": adm.preempted,
            "preemption_recomputed_tokens": adm.preemption_recomputed_tokens,
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

    # ---- 駆動ループ -----------------------------------------------------

    def _drive(self) -> None:
        self._waiting = []
        self._gen = None
        self._rows = []
        self._lane = None
        self._lane_adm = None
        try:
            while True:
                self._drain()
                self._waiting = [
                    a for a in self._waiting if not self._reject_if_dead(a)
                ]
                if self._idle():
                    if not self._waiting:
                        break
                    if self.wait_ms and len(self._waiting) == 1:
                        # 同時到着を拾うための待ち。1 本しか無いときだけで、
                        # decode ではなく TTFT の手前に乗る
                        self._wait_for_company()
                        self._waiting = [
                            a for a in self._waiting if not self._reject_if_dead(a)
                        ]
                    if not self._waiting:
                        continue
                    if len(self._waiting) == 1 or self._sampling_key(
                        self._waiting[0]
                    ) is None:
                        # 1 本だけ / seed 付き -> 単独経路。B=1 無劣化の線
                        self._run_solo(self._waiting.pop(0))
                        continue
                progressed = self._step()
                if not progressed and self._idle() and self._waiting:
                    # 許容が通らないまま空回りしない (メモリが足りない等)。
                    # 単独経路が一番何も抱えないので、そこへ落とす
                    self._run_solo(self._waiting.pop(0))
        except BaseException as exc:  # noqa: BLE001 - 内部の不具合で Future を宙吊りにしない
            for adm in self._live_admissions():
                self._complete(adm, error=exc)
        finally:
            self._gen = None
            self._lane = None
            with self._guard:
                self._active = False
            # 空判定と _active を落とす間に届いた要求が取り残されないよう、
            # 錠の中で見直す (mlxturbo/batch.py と同じ)
            if not self._inbox.empty():
                with self._guard:
                    if not self._active:
                        self._active = True
                        self.executor.submit(self._drive)

    def is_idle(self) -> bool:
        """駆動ループが動いていないか (別スレッドから読んでよい)。

        True = 「いま投げても相方がいない」。サーバー側の入口
        (`server._resolve_batch_route`) がこれを見て、**まとめる相手がいない
        要求はコーディネータに入れない**ようにする。

        入れてしまうと、その要求はセッション (プロンプトキャッシュ) を失う。
        バッチ経路はセッションを持たない割り切りで、それは B>1 では必然
        (1 本の会話が KV を占有する仕組みと、複数行が 1 つの KV を共有する
        仕組みは同居しない) だが、**B=1 では丸損**になる。実測 (2026-09-02、
        67 トークンのプロンプトを 12 回):

            off          TTFT 0.35s / 15.51 ms/tok  (cached_tokens=67)
            コーディネータ TTFT 0.67s / 16.16 ms/tok  (cached_tokens=0)

        差の全部がこれで、単独経路そのものは無罪だった -- 毎回ちがう
        プロンプト (どちらも再利用不可) でそろえると
        off 15.85 / on 15.84 ms/tok、TTFT 0.685s / 0.682s で一致する。

        ``_active`` は駆動ループの生存そのもの (走行中のバッチ・prefill 車線・
        待ち行列のどれかがある間は True) なので、これ 1 つで足りる。
        """
        with self._guard:
            active = self._active
        return not active and self._inbox.empty()

    def _live_admissions(self) -> list:
        out = list(self._rows) + list(self._waiting)
        if self._lane_adm is not None:
            out.append(self._lane_adm)
        return out

    def _idle(self) -> bool:
        return self._gen is None and self._lane is None

    def _drain(self) -> None:
        while True:
            try:
                adm = self._inbox.get_nowait()
            except _queue.Empty:
                return
            self._seq += 1
            adm.seq = self._seq
            self._waiting.append(adm)

    def _wait_for_company(self) -> None:
        deadline = time.perf_counter() + self.wait_ms / 1000.0
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                return
            try:
                adm = self._inbox.get(timeout=left)
            except _queue.Empty:
                return
            self._seq += 1
            adm.seq = self._seq
            self._waiting.append(adm)
            self._drain()
            if len(self._waiting) >= self.max_batch:
                return

    def _reject_if_dead(self, adm: SpecAdmission) -> bool:
        if self._cancelled(adm):
            self._complete(adm, cancelled=True)
            return True
        return False

    # ---- 1 ステップ ------------------------------------------------------

    def _step(self) -> bool:
        """1 ステップ進める。何かしら前に進んだら True。

        順序は「RUNNING が先、余った予算で WAITING を許容」。RUNNING を先に
        するのは、走っている行を新入りの prefill で待たせないため。
        """
        budget = self.token_budget
        progressed = False

        if self._gen is not None:
            self._ensure_room()
        if self._gen is not None:
            depth = self._round_depth(len(self._rows), budget)
            budget -= len(self._rows) * (1 + depth)
            self._decode_round(depth)
            progressed = True

        # 車線 (chunked prefill)。1 要求に渡すのは 1 チャンクまでで、それが
        # 終わって予算が残っていれば次の要求を許容する。短いプロンプトが
        # 並んでいれば 1 ステップで何本も prefill を終えて join するし、
        # 長いプロンプト 1 本なら 1 ステップに 1 チャンクずつ進む
        while budget > 0:
            if self._lane is not None and self._lane_adm is not None:
                if self._reject_if_dead(self._lane_adm):
                    self._lane = None
                    self._lane_adm = None
            if self._lane is None and not self._admit_next():
                break
            n = min(self._lane.remaining, self.prefill_chunk, budget)
            budget -= n
            self._lane.advance(n)
            progressed = True
            if not self._lane.finished:
                # 1 ステップで 1 要求に渡すのは 1 チャンクまで。ここで回し
                # 続けると、長いプロンプト 1 本が予算を丸ごと食って走行中の
                # decode を止める -- 刻んだ意味が無くなる
                break
            self._join_lane()
        return progressed

    def _round_depth(self, n_rows: int, budget: int) -> int:
        """このラウンドのドラフト数 k。

        矩形の上限 ``B*(1+k) <= RECT_LIMIT`` と、残りのトークン予算の両方に
        収める。どちらにも収まらなければ 0 (素の decode) -- 小さい予算を
        投機で食い潰さない。走行中の行は必ず 1 トークンは進めるので、
        予算が 0 でもラウンド自体は回す。

        ここで見るのは矩形と予算だけ。文脈長で 1 に落とす規則
        (``MLXTURBO_BATCH_DEPTH_CTX``、既定 off) は
        ``BatchSpecGenerator._depth_for_round`` が上限として被せる -- 帳簿を
        持っている側で引くほうが素直なため。落ちたぶん予算を多めに引くことに
        なるが、2 行なら 6 対 4 トークンの差で 2048 の予算には響かない。
        """
        if n_rows <= 0:
            return 0
        d = min(self.engine.depth, max(0, RECT_LIMIT // n_rows - 1))
        while d > 0 and n_rows * (1 + d) > budget:
            d -= 1
        return d

    def _decode_round(self, depth: int) -> None:
        gen = self._gen
        rows = self._rows
        new = gen.step(
            truncate=lambda b, toks, r=rows: self._truncate(r[b], toks),
            depth=depth,
        )
        self._settle(new)
        if self._gen is not None:
            self._gen.maybe_compact()

    def _settle(self, new: list, count_step: bool = True) -> None:
        """1 ラウンドぶんを配り、終わった行を落とす。"""
        self._rows = self._after_round(self._gen, self._rows, new, count_step)
        if not self._rows:
            self._gen = None

    # ---- 許容と join -----------------------------------------------------

    def _effective_prompt(self, adm: SpecAdmission) -> list[int]:
        """prefill に流すトークン列。退避された行は生成済みを後ろに付ける。"""
        return list(adm.prompt_ids) + list(adm.tokens)

    def _remaining(self, adm: SpecAdmission) -> int:
        return max(0, adm.max_tokens - len(adm.tokens))

    def _record_completed_recompute(self, adm: SpecAdmission) -> None:
        """退避行の復帰prefillが完了した時点で、再計算token数を加算する。"""

        if adm.preempted <= adm.recompute_events:
            return
        n_tokens = len(self._effective_prompt(adm))
        adm.preemption_recomputed_tokens += n_tokens
        adm.recompute_events = adm.preempted
        self.preemption_recomputed_tokens += n_tokens

    def _batch_key(self):
        """いまのバッチが決めているサンプリング設定 (無ければ None)。

        持ち回さずに毎回引く。走行中の行が全部抜けたり退避されたりする経路が
        複数あるので、覚えておくと必ずどれかで古くなる。
        """
        if self._rows:
            return self._sampling_key(self._rows[0])
        if self._lane_adm is not None:
            return self._sampling_key(self._lane_adm)
        return None

    def _admit_next(self) -> bool:
        """待ち行列の先頭を 1 本、車線に入れる。入れたら True。

        候補にするのは**先頭だけ** (FIFO)。設定の合うものを探しに行くと、
        合わない要求がいつまでも進まない。
        """
        self._drain()
        while self._waiting and self._reject_if_dead(self._waiting[0]):
            self._waiting.pop(0)
        if not self._waiting:
            return False
        if len(self._rows) >= self.max_batch:
            return False
        head = self._waiting[0]
        key = self._sampling_key(head)
        if key is None:
            # seed 付き。走行中が空くのを待って単独経路へ
            return False
        cur = self._batch_key()
        if cur is not None and key != cur:
            return False
        remaining = [self._remaining(a) for a in self._rows]
        remaining.append(self._remaining(head) + len(self._effective_prompt(head)))
        if not self._fits(remaining):
            return False
        self._waiting.pop(0)
        if cur is None:
            # この行がバッチの設定を決める。以後 join できるのは同じ設定の
            # 要求だけ (`_sample_rows` が全行ぶんを 1 回の呼び出しで引くため)
            self._temp = head.temp
            self._sampler = _position_local_sampler_for(head)
        if head.t0 is None:
            head.t0 = time.perf_counter()
        self._lane_adm = head
        self._lane = SpecPrefillLane(
            self.engine,
            self._effective_prompt(head),
            temp=self._temp,
            sampler=self._sampler,
        )
        return True

    def _join_lane(self) -> None:
        """prefill を終えた行を走行中のバッチに入れる。"""
        row = self._lane.result()
        adm = self._lane_adm
        self._record_completed_recompute(adm)
        self._lane = None
        self._lane_adm = None
        if self._gen is None:
            self._gen = BatchSpecGenerator.from_prefilled(
                self.engine, [row], temp=self._temp, sampler=self._sampler
            )
            self._rows = [adm]
        else:
            self._gen.join([row])
            self._rows.append(adm)
            self.joins += 1
        # prefill が出す 1 個目は単独経路と同じくラウンドに数えない
        # (tokens_per_step の定義が n_decode / decode ラウンド数のため)
        new = [[] for _ in self._rows]
        new[-1] = [row.first_token]
        self._settle(new, count_step=False)

    # ---- preemption ------------------------------------------------------

    def _fits(self, remaining: list[int]) -> bool:
        """空きの見積もりだけ差し替えられる形の `rows_fit` (テスト用)。"""
        room = self._free_bytes()
        if room is None:
            return True  # 数えられない環境では上限を掛けない
        return rows_fit(self.model, remaining, self.engine.depth, room=room)

    def _ensure_room(self) -> None:
        """走行中の行が最後まで走れないなら、入る所まで退避する。"""
        while self._rows and not self._fits([self._remaining(a) for a in self._rows]):
            if not self._preempt_one():
                return

    def _preempt_one(self) -> bool:
        """優先度最低 = 最若手 (到着が最も新しい) の 1 行を退避する。

        生成済みのトークンは**保持**したまま、`プロンプト + 生成済み` を
        新しいプロンプトとして待ち行列の先頭に戻す (vLLM の recompute 型)。
        KV は捨てるので、復帰時は prefill をやり直す -- 逆に言えば、退避に
        必要な追加のメモリが無い。残り max_tokens は生成済みのぶん減るので、
        `spec_batchable` の条件は退避の前後で動かない。

        既に配ったトークンは `adm.tokens` に入ったままなので、復帰後に配る
        のは続きだけ (`_deliver` は追記しかしない)。
        """
        if not self._rows:
            return False
        b = max(range(len(self._rows)), key=lambda i: self._rows[i].seq)
        adm = self._rows[b]
        keep = [i for i in range(len(self._rows)) if i != b]
        self._gen.retire(keep)
        self._rows = [self._rows[i] for i in keep]
        if not self._rows:
            self._gen = None
        adm.preempted += 1
        self.preemptions += 1
        self._waiting.insert(0, adm)
        return True

    # ---- 単独 (B=1) -----------------------------------------------------

    def _run_solo(self, adm: SpecAdmission) -> None:
        """既存の単独経路をそのまま呼ぶ。`_start_generation` の worker()
        (server.py) と同じ 3 分岐で結果を返す。session は渡さない --
        バッチ経路と揃えた割り切りで、`mlxturbo/batch.py` の継続バッチングも
        同じ (毎回まっさらな prefill)。"""
        self.solo_runs += 1
        adm.t0 = time.perf_counter()
        try:
            res = self.runner.generate(
                self._effective_prompt(adm),
                max_tokens=self._remaining(adm),
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
        self._record_completed_recompute(adm)
        if adm.tokens:
            # 退避から戻ってきた行。単独経路が返す列は続きぶんだけなので、
            # 既に配った分の後ろに足して 1 本の結果に見せる
            res = dict(res)
            res["tokens"] = list(adm.tokens) + list(res.get("tokens") or [])
        else:
            res = dict(res)
        res["preemptions"] = adm.preempted
        res["preemption_recomputed_tokens"] = adm.preemption_recomputed_tokens
        self._complete(adm, res=res)

    # ---- ラウンドの後始末 -------------------------------------------------

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
            if count_step and new[b]:
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
    "PrefilledRow",
    "SpecPrefillLane",
    "prime_draft_rows",
    "sample_positions",
    "BatchSpecCoordinator",
    "SpecAdmission",
    "spec_batchable",
    "rows_fit",
]
