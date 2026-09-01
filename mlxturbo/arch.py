"""アーキ能力レイヤ: モデル族に依存しない形で「再帰状態を持つ層」を扱うための
薄い能力関数レジストリ。

## 経緯・決定

設計は docs/BACKLOG.md の「アーキ能力レイヤの設計 (2026-09-01 決定、advisor
判断)」で確定済み。根拠・やらないこと・反転条件はそちらを参照 (ここには
再掲しない)。要点だけ:

- 切っているのは 2 能力だけ: (1) 層トポロジ + 名前付きキャッシュスロット、
  (2) それを使った rollback ループ。indexer/疎注意の trim は族ごとに有無が
  違うため、別能力 (`has_indexer`) として rollback 本体の外に出してある。
- 切るのは **`model -> 能力`** であって **`model -> モジュール`** ではない。
  `qwen4_arch()` はこのモジュールに一本化するが、あくまで qwen4_exp
  固有のヘルパーのままにしてある (他族向けの分岐は入れない)。model_type で
  分岐して「モジュールを返す」汎用ヘルパーにすると、対応外の族を渡したときの
  失敗が import 時の ImportError から実行途中の AttributeError に劣化する
  (BACKLOG 参照)。
- 再帰層 0 個 (Gemma の sliding window など) は異常系ではなく正常系。
  `recurrent_layers` は空リストを返し、`rollback_recurrent`/
  `rollback_recurrent_rows` はそれに対して no-op になる。
- フォワードの写し (spec_flash.capture の GDN/PLE フック、
  batch_spec.ragged_attention の Attention フック) はここでは抽象化しない。
  「本家と一字一句同じ」であることが正しさの根拠になっており、抽象の下に
  隠すと tools/verify_prefill_bitident.py の検査対象との対応が切れる。
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


def qwen4_arch():
    """qwen4_exp (Flash-Next) 固有のモジュール解決。

    以前は spec_flash.py / batch_spec.py / mtp_flash.py に同じ内容の
    `_arch()` が 3 重定義されていた。ここに一本化するが、**qwen4 固有の
    ヘルパーのまま**にしてあることに注意 -- model_type を見て他族の
    モジュールも返す汎用ヘルパーにはしない (理由はモジュール docstring 参照)。
    他族 (qwen3_5 など) を扱うコードは、そちら側で個別に同種のヘルパーを持つ。
    """
    import mlx_lm.models.qwen4_exp as Q

    return Q


@dataclass
class RecurrentLayer:
    """再帰状態を持つ層 1 つぶんの情報。

    スロットは **名前付きで** 返す (添字のまま汎用 API にすると、汎用の名前で
    qwen4 の構造を再凍結してしまい、結合が今より見えにくくなる -- BACKLOG
    参照)。その族に無いスロットは None。

    - `conv`: 短畳み込みの窓 (GDN の conv_input 相当)
    - `state`: 再帰状態そのもの
    - `ple_conv`: PLE (Flash-Next 固有の追加畳み込み) の窓。無い族は None
    - `ngram`: n-gram 文脈。無い族は None
    """

    index: int
    module: object  # DecoderLayer 相当 (linear_attn / ple 属性を持つ)
    conv: int | None
    state: int | None
    ple_conv: int | None
    ngram: int | None


def recurrent_layers(model) -> list[RecurrentLayer]:
    """`model.model.layers` を舐めて、再帰状態を持つ層だけを名前付きスロット
    つきで返す。

    判定は層の属性から行う (model_type 分岐ではなくダックタイピング):

    - qwen4_exp (Flash-Next): `layer.layer_type == "linear_attention"`。
      スロットは 4 つ全部埋まる (`conv=0, state=1, ple_conv=2 if layer.ple
      else None, ngram=3`)。
    - qwen3_5 (27B): `layer.is_linear` で判定できる。**今回は載せ替えない**
      (BACKLOG の決定)。紙上確認だけ書いておく -- `mlxturbo/spec.py:447-482,
      531-538` を見ると qwen3_5 の `ArraysCache` は 2 スロットのみで
      (`cache[0]`=conv, `cache[1]`=状態)、PLE も n-gram も持たない。
      対応する分岐を足すなら
      ``RecurrentLayer(index=i, module=layer, conv=0, state=1,
      ple_conv=None, ngram=None)`` で過不足なく表現できるはずで、この
      dataclass の形はその形を既に受け入れられる (フィールドを増やす必要は
      ない)。GLM の KDA が同じ形 (「位置ごとの再帰状態の一括取り出し」) で
      表せないと判明したら、この能力自体を族ごとに持つ設計へ戻す
      (BACKLOG の反転条件)。

    再帰層が 0 個のモデル (Gemma の sliding window など) は空リストを返す。
    これは異常系ではなく正常系で、`rollback_recurrent`/
    `rollback_recurrent_rows` はそれに対して no-op になる。
    """
    out: list[RecurrentLayer] = []
    for i, layer in enumerate(model.model.layers):
        if getattr(layer, "layer_type", None) == "linear_attention":
            out.append(
                RecurrentLayer(
                    index=i,
                    module=layer,
                    conv=0,
                    state=1,
                    ple_conv=2 if getattr(layer, "ple", None) is not None else None,
                    ngram=3,
                )
            )
    if not out and any(getattr(layer, "is_linear", False) for layer in model.model.layers):
        # 「再帰層 0 個は正常系」(Gemma の sliding window) と区別が付かない
        # まま黙って空リストを返すと、呼び手の rollback_recurrent/
        # rollback_recurrent_rows が黙って no-op になる。qwen3_5 など
        # `layer.is_linear` で判定できる別名のマーカーを持つ層が実在するのに
        # `layer_type == "linear_attention"` 側で 1 つも拾えていないのは
        # 「0 個」ではなく「この族はまだ載せ替えていない」なので、
        # 例外にして呼び手に伝える (Opus 設計レビュー A1 指摘)。
        raise NotImplementedError(
            "recurrent_layers: layer.is_linear を持つ層が存在するが"
            " layer_type == 'linear_attention' では検出できない"
            " (qwen3_5/GLM など、未対応のモデル族の可能性)"
        )
    return out


def attention_layers(model) -> list[tuple[int, object]]:
    """full attention 層の `(index, module)` のリスト。qwen4_exp は
    `layer.layer_type == "full_attention"`。"""
    return [
        (i, layer)
        for i, layer in enumerate(model.model.layers)
        if getattr(layer, "layer_type", None) == "full_attention"
    ]


def has_indexer(cache_entry) -> bool:
    """疎注意 (QSA) の indexer キャッシュが実際に埋まっているか。

    trim の要否判定にだけ使う -- trim 自体はここに入れない (別能力として
    呼び手 (`mlxturbo/spec_flash.py` の `rollback`) の責務のまま残す)。
    """
    indexer = getattr(cache_entry, "indexer", None)
    return indexer is not None and getattr(indexer, "keys", None) is not None


def indexer_budget(model) -> int | None:
    """疎注意 (QSA) が働き始める kv 長。持たない族では None。

    「kv がこれを超えると 1 位置あたりの検証費用が跳ねる」という境界なので、
    バッチの solo tier 判定 (`mlxturbo/batch.py`) と投機の深さ切り替え
    (`mlxturbo/spec_flash.py`) が同じ値を見る。
    """
    try:
        return model.args.text.indexer_budget
    except AttributeError:
        return None


def _take_rows(arr: mx.array, starts: mx.array, width: int) -> mx.array:
    """`arr` の axis=1 から、行 b ごとに `[starts[b], starts[b]+width)` を
    切り出す。単一行の `x[:, keep:keep+w, :]` 系スライスを行別に一般化した
    共通部品 -- GDN の conv window、GDN の状態 (width=1)、PLE の conv window、
    n-gram の文脈が全てこの形に落ちる。
    """
    idx = starts[:, None] + mx.arange(width)[None, :]  # (B, width)
    shape = [idx.shape[0], width] + [1] * (arr.ndim - 2)
    return mx.take_along_axis(arr, idx.reshape(shape), axis=1)


def rollback_recurrent(
    model, caches, cap, keep: int, *, ngram_ctx=None, ids_kept=None
) -> None:
    """単一行の巻き戻し。`recurrent_layers()` の名前付きスロットを使い、
    `mlxturbo.spec_flash.rollback` と `mlxturbo.batch_spec.batched_rollback`
    の共通部分 (再帰状態・conv 窓・PLE conv・n-gram 文脈) を畳んだもの。

    `cap` は `mlxturbo.spec_flash.Capture` 相当 (`.gdn[id(module.linear_attn)]
    -> (conv_input, states_all)`, `.ple[id(module.ple)] -> full`) を要求する。
    indexer/KV の trim はここに入れない (族ごとに有無が違うため、呼び手の
    責務のまま)。

    `ngram_ctx` はレイヤーの全体インデックスに揃えた「フォワード前の n-gram
    文脈」のリスト (n-gram スロットを持たない層は None)。`ids_kept` が None
    のときは n-gram 更新自体をスキップする (元の `rollback()` が
    `ids_kept=None` のとき `c[3]` を更新しないのと同じ扱い)。
    """
    layers = recurrent_layers(model)
    for rl in layers:
        la = rl.module.linear_attn
        c = caches[rl.index]
        conv_input, states_all = cap.gdn[id(la)]
        k = la.conv_kernel_size
        if rl.conv is not None:
            c[rl.conv] = mx.contiguous(conv_input[:, keep : keep + k - 1, :])
        if rl.state is not None:
            c[rl.state] = states_all[:, keep - 1] if keep > 0 else None
        if rl.ple_conv is not None:
            full = cap.ple.get(id(rl.module.ple))
            if full is not None:
                n = rl.module.ple.short_conv_state_len
                c[rl.ple_conv] = mx.contiguous(full[:, keep : keep + n, :])

    if ids_kept is None:
        return
    ctx_len = model.args.text.ngram_size - 1
    for rl in layers:
        if rl.ngram is None:
            continue
        ctx = ngram_ctx[rl.index] if ngram_ctx is not None else None
        if ctx is None:
            continue
        caches[rl.index][rl.ngram] = mx.concatenate([ctx, ids_kept], axis=1)[
            :, -ctx_len:
        ]


def rollback_recurrent_rows(
    model, caches, cap, keeps: list[int], *, ngram_ctx=None, pair=None
) -> None:
    """行別 (バッチ) の巻き戻し。`rollback_recurrent` の per-row 一般化。

    `keeps[b]` は行 b の受理数 (1..total)。full attention 側は一切触らない
    -- dead slot として `mlxturbo.batch_spec.RaggedLedger` が扱う設計そのもの
    なので、呼び出し側で別途 `ledger.commit_round(keeps, total)` を呼ぶこと。

    `ngram_ctx`/`pair` を渡さない場合は n-gram 文脈の更新をスキップする。
    """
    layers = recurrent_layers(model)
    if layers:
        # `keeps[b]` は「行 b の受理数 (1..total)」という契約 (docstring) だが
        # 呼び手にそれを強制する型は無い。MLX の `mx.take_along_axis` は
        # 範囲外の添字を例外にせず末尾へクランプするため (実測確認済み)、
        # 契約違反を渡されると「その行だけ状態が静かにずれる」形で隠れる。
        # `RaggedLedger.commit_round` の `assert 1 <= keep <= total` は
        # rollback の後に呼ばれる順序なのでここでは防波堤にならない。
        # 単一行版 `rollback_recurrent` の `keep - 1 if keep > 0 else None`
        # という非対称なガードと揃え、ここでも明示的に落とす
        # (F2、Opus 正しさレビュー指摘)。`total` を引数で受け取っていないので
        # 捕獲済みの `states_all` の T 軸 (= このラウンドで実際に追加した
        # 列数、commit_round の `total` と同じ意味) から求める。
        _, states_all0 = cap.gdn[id(layers[0].module.linear_attn)]
        total = states_all0.shape[1]
        for b, keep in enumerate(keeps):
            if not (1 <= keep <= total):
                raise ValueError(
                    f"rollback_recurrent_rows: keeps[{b}]={keep} が"
                    f" 1..{total} の範囲外です"
                )
    keep_arr = mx.array(keeps)
    for rl in layers:
        la = rl.module.linear_attn
        c = caches[rl.index]
        conv_input, states_all = cap.gdn[id(la)]
        if rl.conv is not None:
            win = la.conv_kernel_size - 1
            c[rl.conv] = _take_rows(conv_input, keep_arr, win)
        if rl.state is not None:
            c[rl.state] = _take_rows(states_all, keep_arr - 1, 1)[:, 0]
        if rl.ple_conv is not None:
            full = cap.ple.get(id(rl.module.ple))
            if full is not None:
                n = rl.module.ple.short_conv_state_len
                c[rl.ple_conv] = _take_rows(full, keep_arr, n)

    if ngram_ctx is None or pair is None:
        return
    ctx_len = model.args.text.ngram_size - 1
    for rl in layers:
        if rl.ngram is None:
            continue
        ctx = ngram_ctx[rl.index]
        if ctx is None:
            continue
        cat = mx.concatenate([ctx, pair], axis=1)
        caches[rl.index][rl.ngram] = _take_rows(cat, keep_arr, ctx_len)


__all__ = [
    "RecurrentLayer",
    "qwen4_arch",
    "recurrent_layers",
    "attention_layers",
    "has_indexer",
    "indexer_budget",
    "rollback_recurrent",
    "rollback_recurrent_rows",
]
