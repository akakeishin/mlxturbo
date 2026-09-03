"""段 3(b)/P1a: QSA の疎性を sdpa 自身の読み出しへ伝える経路 (gather attention)。

`docs/research/KERNEL-PROGRAM.md` 段 3(b) の出し口。いまの QSA (疎注意) は
選んだブロックを加算マスクとして sdpa に渡しているが、
`mx.fast.scaled_dot_product_attention` は加算マスクを渡されても全 KV を
読んで全スコアを計算する (段 1 の実測、17k の attention 実効帯域が下限の
25% しか出ていない一因)。つまり疎性が sdpa 側の節約になっていない。

ここでは、選ばれたブロックだけを先に集めてから、マスク無しの (集めた列に
対する小さい bool マスクだけを持つ) dense sdpa に渡す:

    from mlxturbo import gather_attn
    gather_attn.enable_gather_attn(model)

実装本体は `mlxturbo/_vendor/qwen4_exp.py` の
`QSAIndexer.select_blocks` / `Attention._gather_forward` (既存のシーム
`_positions` / `_final_mask` と同じ作法で追加した別口)。ここは `fused.py`
の `enable_*` と同じ形の有効化・無効化関数を置くだけ。

既定 off。環境変数 `MLXTURBO_GATHER_ATTN=1` で `runner.enable_default_fusions`
から有効になる。出力はビット一致しない (softmax の対象集合は元の `keep`
と同じだが、加算順が変わる) ので、採否は KLD / tok-step の in-model 計測で
決める。合成モデルでの正しさ確認は `tools/verify_gather_attn.py`。

段 P1a (タイル分割、`docs/research/KERNEL-PROGRAM.md` 段 P1): decode (S=2)
では union が `S * block_topk` で頭打ちになり効くが、prefill (S=2048) の
ように S が大きいと 2048 クエリ全体の和集合がほぼ全ブロックになって効かない。
ただし隣り合うクエリの選択は強く相関する (局所窓 + 少数のグローバルブロック)
ので、クエリ行をタイルに切ればタイルごとの union は縮む。`_gather_forward`
はタイル幅 `tile` を受け取り、クエリ行をその幅で分割してタイルごとに
(既存と同じ手順で) union を取りなおす。タイル幅は
`enable_gather_attn(model, tile=...)` / 環境変数 `MLXTURBO_GATHER_TILE`
(既定 0 = 従来どおり S 全体で 1 回) で渡す。実 17k/50k でのタイル幅
{0, 128, 256, 512} の壁時計掃引は段 P1b (実装はここまでで完了、計測は別途)。

段 P1 (融合カーネル): 上の 2 つは汎用 op を 2 段重ねる形なので、選んだ列を
**書いて読み直してマスクを作る**費用が必ず乗る。`enable_prefill_attn` は
その 2 段を 1 本の Metal カーネル (`mlxturbo/kernels/prefill_attn.py`) に
畳む。既定 off、環境変数は `MLXTURBO_PREFILL_ATTN=1`。正しさの確認は
`tools/verify_prefill_attn.py` (GPU)。
"""

from __future__ import annotations


def _each_layer(model, mtp=None):
    # 層の列挙は `fused._model_layers` に寄せる (族ごとにラッパの形が違う。
    # qwen4_exp は `model.model.layers`、qwen3_5 (27B) は
    # `model.language_model.model.layers`)。見つからなければ 0 層で、
    # 呼び手は何もせず 0 を返す。
    from .fused import _model_layers

    for layer in _model_layers(model):
        yield layer
    if mtp is not None:
        for layer in mtp.layers:
            yield layer


def enable_gather_attn(
    model, mtp=None, stats: list | None = None, tile: int = 0
) -> int:
    """レイヤーの ``Attention`` に gather 経路を仕込む。戻り値は適用した層数。

    `QSAIndexer` を持つ層 (= self_attn がある層) にだけ `_gather_attn = True`
    を立てる。GDN 層 (`linear_attn`) には触らない。

    ``stats`` にリストを渡すと、gather が実際に活性化した呼び出しごとに
    ``(T, n_blocks, U, n_sel, union_ratio, kv_frac, true_u)`` を追記する
    (`_gather_stats` 属性、`_wide_qkv` と同じ注入の作法)。``T`` はそのタイルの
    クエリ行数 (タイル無効なら S そのもの)、``union_ratio = U / n_blocks``、
    ``kv_frac = U * compress_ratio / kv_len``、``true_u`` は和集合の真の大きさ
    (``U`` は上限 ``min(n_blocks, T*block_topk)`` であって和集合の実測ではない)。
    既定 None のときは 1 行も増えない (計測・検証専用、
    `tools/verify_gather_attn.py` が使う)。

    ``tile`` は段 P1a のタイル幅 (`_gather_attn_tile` 属性、`_wide_qkv` と
    同じ注入の作法)。既定 0 は従来どおり S 全体を 1 回で処理する
    (decode の S<=8 はこの既定のままで実質タイル無効)。
    """
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "indexer"):
            sa._gather_attn = True
            sa._gather_attn_tile = tile
            if stats is not None:
                sa._gather_stats = stats
            n += 1
    return n


def disable_gather_attn(model, mtp=None) -> int:
    """`enable_gather_attn` を打ち消す。戻り値は外した数。A/B で交互に測るために要る。"""
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "_gather_attn", False):
            sa._gather_attn = False
            sa._prefill_attn = False
            n += 1
    return n


def enable_prefill_attn(model, mtp=None) -> int:
    """段 P1 の融合カーネル (`mlxturbo/kernels/prefill_attn.py`) を仕込む。

    gather と softmax を 1 本の Metal カーネルに畳む
    (`Attention._gather_forward` の中で `_gather_tile_attn` の代わりに走る)。
    **gather 経路そのものの中の話**なので `_gather_attn` も一緒に立てる ---
    `Attention.__call__` は `_gather_attn` を見て `_gather_forward` に入るかを
    決めるため、こちらだけ立てても呼ばれない。

    カーネルが引き受けられない形 (dtype・head_dim・キャッシュ型) では
    `_gather_forward` が既存のタイル経路へ落ちる。落ちた理由は
    `mlxturbo/kernels/prefill_attn.py` が 1 度だけ表示する。

    既定 off。環境変数 `MLXTURBO_PREFILL_ATTN=1` で
    `runner.enable_default_fusions` から有効になる。
    """
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "indexer"):
            sa._gather_attn = True
            sa._prefill_attn = True
            n += 1
    return n


def disable_prefill_attn(model, mtp=None) -> int:
    """`enable_prefill_attn` だけを打ち消す (gather 経路は残す)。"""
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "_prefill_attn", False):
            sa._prefill_attn = False
            n += 1
    return n
