"""prefill チャンク境界の checkpoint まわりで、Flash-Next (spec_flash.py) と
dense (spec.py) の両エンジンに共通する薄い部品だけを置く。

## 背景: BPE 境界 checkpoint (コミット 0b1d938, spec_flash.py で先行導入)

会話 2 ターン目の retemplate では、前ターンのプロンプト末尾トークンが後続文字と
マージされて別トークンに化ける (BPE は末尾かどうかでマージが変わる)。すると
LCP は必ず checkpoint の 1 トークン手前に落ち、復元条件 pos <= lcp が恒久的に
外れて毎回ほぼ全再 prefill になる (spec_flash.py 側の実測: lcp=1074 vs
checkpoint=1075)。対策は「最終チャンクの末尾 1 トークンを切り離し、その手前
(n-1) にも checkpoint を積む」。これを両エンジンで書き方が揃うように関数化
したのがこのモジュール。

## 設計方針: 厚い抽象化はしない

チャンクの forward の呼び方はエンジンごとに違う (spec_flash.py は
capture(model, light=True) 経由で model.model() を呼び cap.hyper を持ち帰る、
spec.py は self._hidden_forward(..., capture=False) を呼び h だけを持ち帰る)。
なので forward 自体はコールバック (forward_head) として呼び手に渡してもらい、
このモジュールは「分割するかどうかの判定」「head を forward させて出力を
eval する」「checkpoint を積んで retention を切る」という、両エンジンで
一字一句同じだった後処理だけをまとめる。
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


def split_and_checkpoint_tail(
    chunk: mx.array,
    checkpoints: list | None,
    pos: int,
    caches,
    retention: int,
    snapshot_fn: Callable[[Any], Any],
    forward_head: Callable[[mx.array], Any],
    tail_size: int = 1,
) -> tuple[mx.array, tuple]:
    """最終チャンクの末尾 ``tail_size`` トークンを切り離し、手前を forward して
    そこにも checkpoint を積む。

    ``checkpoints`` が None (checkpoint 機構が無効 = サーバー経路以外の
    呼び出し) か、``chunk`` の長さが 1 以下 (これ以上割れない) のときは
    何もせず ``(chunk, ())`` を返す -- 呼び手は従来どおり ``chunk`` を
    まるごと 1 回で forward すればよい (no-op)。

    分割する場合:
      1. ``forward_head(head)`` を呼ぶ -- head = chunk の末尾 ``tail_size``
         トークンを除いた部分。戻り値はエンジンごとに違う中間結果 (例:
         spec_flash.py の ``(h0, cap0.hyper)``、spec.py の ``h0`` 単体)。
         tuple でなければ 1 要素の tuple に包む。
      2. その戻り値をまとめて ``mx.eval`` する (元の各エンジンの実装が
         forward 直後にしていたのと同じタイミング)。
      3. ``caches`` のロールバック不能な状態 (``ArraysCache.state``) を
         eval し、``mx.clear_cache()`` で一時バッファを解放する (これも
         両エンジンで同一だった後処理)。
      4. ``pos + head.shape[-1]`` (chunk 内の相対位置ではなく絶対位置) に
         ``snapshot_fn(caches)`` を積み、``retention`` 件を超えた古い
         checkpoint を捨てる。

    戻り値は ``(残りの chunk, forward_head の戻り値の tuple)``。残りの
    chunk は分割時は末尾 1 トークンのみ、no-op 時は元の ``chunk`` そのもの
    -- 呼び手はこの戻り値をこれまでの「最終チャンク」として通常どおり
    forward すればよい。

    ``chunk``/``head``/``tail`` の軸は問わない -- 1D (S,) でも 2D (B, S)
    でも、系列軸は常に末尾の軸というだけの前提で ``chunk[..., :-1]`` /
    ``chunk[..., -1:]`` を使う (spec_flash.py の ids は (B, S)、spec.py の
    tokens は (S,) で軸数が違うため)。

    注意: 分割は正しい計算だが、一括処理とビット一致とは限らない (チャンク
    割りが変わると量子化行列積の丸めが動くのは既知の性質。
    docs/research/PREFILL-CHUNKING-DETERMINISM.md 参照)。checkpoints が有効なのは
    サーバー経路だけなので、generate() や検証プローブ (checkpoints=None)
    はこの分岐を通らない。
    """
    if tail_size < 1:
        raise ValueError("tail_size must be at least 1")
    if checkpoints is None or chunk.shape[-1] <= tail_size:
        return chunk, ()

    head = chunk[..., :-tail_size]
    tail = chunk[..., -tail_size:]
    result = forward_head(head)
    if not isinstance(result, tuple):
        result = (result,)
    mx.eval(*result)
    for c in caches:
        state = getattr(c, "state", None)
        if state is not None:
            mx.eval(state)
    mx.clear_cache()
    checkpoints.append((pos + head.shape[-1], snapshot_fn(caches)))
    del checkpoints[:-retention]
    return tail, result
