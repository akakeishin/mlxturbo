"""段 P1 T=1 gather カーネル (`MLXTURBO_PREFILL_ATTN`) が実モデルで出力分布を
大きく動かす原因を切り分けるための層別プローブ。

`tools/kld_prefill_attn.py` は最終ロジットの KLD しか見ないので、「どの層
から食い違い始めるか」「食い違いは block 選択そのもの (離散) が変わって
いるのか、それとも online softmax の加算順によるだけの浮動小数ノイズなのか」
が分からない。ここでは Attention 層ごとに次の 2 つを ``off`` (dense, 既定)
と ``kernel`` (`MLXTURBO_PREFILL_ATTN=1` 相当) の 2 回の prefill で捕まえて
突き合わせる:

1. **Attention.__call__ の戻り値** (o_proj 後、hyper-connection で混ぜる前)
   の最終チャンク末尾 ``--tail`` 行。層ごとの max|diff| を出す。1 層目
   (最初にカーネルが発火する full-attention 層) から食い違っていれば
   online softmax の加算順そのものが怪しい。1 層目はほぼ一致していて後の層
   ほど差が育つなら、下流層の離散選択 (indexer の top-k) が上流の微小な
   浮動小数差で反転している (カスケード) 可能性が高い。

2. **QSAIndexer._pooled_and_top が返す keep_block** (top-k で選ばれたブロック
   の bool、``__call__`` (dense) と ``select_blocks`` (kernel 経路) の共通部)
   を層ごとに捕まえ、``off`` と ``kernel`` で行ごとに何ブロック分違うかを
   数える。**ここが 0 のまま Attention 出力の diff が大きい層があれば、
   選択そのものは同じで online softmax の加算順だけが原因**。**ここが
   非 0 の層があれば、その層で block 選択が実際に反転している** ---
   `_gather_forward` の visible-set 構成式 (block 展開 + tail の因果窓) は
   `QSAIndexer.__call__` の dense 経路と数式上同一になるはずなので
   (`docs/research/...` の該当調査参照)、反転の原因は「同じ keep_block を
   別々に展開している」ではなく「そもそも keep_block 自体が違う」--- つまり
   この層に入ってくる隠れ状態 x が既に off/kernel で食い違っている
   (=前段の層で発生した差がここまで伝播した) ことを意味する。

読み方: 1 層目の keep_block 差が 0 で Attention 出力 diff が浮動小数の丸め
水準 (bf16 で相対 1e-3 未満) に収まっているのに、後段の層で keep_block 差が
非 0 に転じるなら「カスケードで block 選択が反転する」仮説が実モデルでも
成立する。1 層目から keep_block 差が非 0 なら、そもそも同じ入力から違う
keep_block を作っている (`_pooled_and_top` 自体か、その手前の呼び出し規約に
実装バグがある) ことになるので、そちらを先に見る。

使い方 (GPU 必須、98GB モデルを読むので biglock 経由):

    tools/biglock.sh .venv/bin/python tools/prefill_attn_layer_probe.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram --ctx 17000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="prefill attention T=1 カーネルの層別プローブ (off vs kernel)"
    )
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument(
        "--question", default="上の文書の要点を5つに整理してください。"
    )
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    import numpy as np

    import mlxturbo  # noqa: F401  (arch_registry の meta_path フックを張る)
    from mlx_lm import load
    from mlx_lm.models import qwen4_exp as QE

    from mlxturbo.gather_attn import (
        disable_gather_attn,
        disable_prefill_attn,
        enable_prefill_attn,
    )
    from mlxturbo import runner as mlxturbo_runner

    if not mx.metal.is_available():
        print("Metal が使えないのでこの検査は走らせられない")
        return 1
    mx.set_default_device(mx.gpu)

    model_path = os.path.expanduser(args.model)
    ngram_path = os.path.expanduser(args.ngram) if args.ngram else None

    model, tok = load(model_path)
    if hasattr(mlxturbo_runner, "set_wired_limit_default"):
        mlxturbo_runner.set_wired_limit_default(log_prefix="[layer-probe]")
    if ngram_path:
        from mlxturbo.ngram_stream import install

        install(model, ngram_path)
    mlxturbo_runner.enable_default_fusions(model, log_prefix="[layer-probe]")

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts  # noqa: E402
    import prefill_anatomy as PA  # noqa: E402

    # layer index を id(self_attn) から引けるようにしておく (self_attn を持つ
    # 層だけ、つまり full-attention 層だけが対象)。
    layer_idx_of: dict[int, int] = {}
    for i, layer in enumerate(model.model.layers):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "indexer"):
            layer_idx_of[id(sa)] = i
            layer_idx_of[id(sa.indexer)] = i

    # --- フック ---------------------------------------------------------
    # captured[layer_idx] = {"out": mx.array タイル末尾行, "keep": mx.array}
    # 呼ぶたびに上書きするので、最後に残るのは「最終チャンクの値」になる
    # (プロンプトはチャンク幅で複数回に分けて流すので、層 i の __call__ は
    # チャンクごとに 1 回ずつ、順に呼ばれる)。
    captured: dict[str, dict[int, dict]] = {"off": {}, "kernel": {}}
    mode_box = {"mode": "off"}
    tail = args.tail

    orig_attn_call = QE.Attention.__call__
    orig_pooled = QE.QSAIndexer._pooled_and_top

    def patched_attn_call(self, x, rope, mask, cache, idx_cache):
        out = orig_attn_call(self, x, rope, mask, cache, idx_cache)
        idx = layer_idx_of.get(id(self))
        if idx is not None:
            S = out.shape[1]
            t = min(tail, S)
            captured[mode_box["mode"]].setdefault(idx, {})["out"] = out[
                :, -t:, :
            ].astype(mx.float32)
        return out

    def patched_pooled(self, x, rope, cache, offset, positions=None):
        res = orig_pooled(self, x, rope, cache, offset, positions)
        idx = layer_idx_of.get(id(self))
        if idx is not None and res is not None:
            keep_block, n_blocks, kv_len, q_col = res
            S = keep_block.shape[1]
            t = min(tail, S)
            captured[mode_box["mode"]].setdefault(idx, {})["keep"] = keep_block[
                :, -t:, :
            ]
            captured[mode_box["mode"]][idx]["kv_len"] = kv_len
            captured[mode_box["mode"]][idx]["n_blocks"] = n_blocks
        return res

    QE.Attention.__call__ = patched_attn_call
    QE.QSAIndexer._pooled_and_top = patched_pooled

    try:
        body = long_prompts(tok, args.ctx, [args.question])[0]
        ids = mx.array(
            tok.apply_chat_template(
                [{"role": "user", "content": body}], add_generation_prompt=True
            )
        )[None]
        print(f"kv={ids.shape[1]}", flush=True)

        def run(mode_name):
            mode_box["mode"] = mode_name
            n = ids.shape[1]
            cache = model.make_cache()
            for start in range(0, n, args.chunk):
                end = min(start + args.chunk, n)
                part = ids[:, start:end]
                logits = model(part, cache=cache)
                mx.eval(logits, *PA.pending(cache))
                mx.clear_cache()
            return logits

        # off: `_gather_attn` も含めて完全にクリーンな baseline に戻す
        # (`disable_prefill_attn` だけだと `_gather_attn` が前の状態から
        # 残る --- `mlxturbo/gather_attn.py` の docstring 参照。今回は毎回
        # 新しいプロセスなので実害は無いが、流儀として揃える)。
        disable_gather_attn(model)
        run("off")
        mx.clear_cache()

        n_layers = enable_prefill_attn(model)
        fired = [0]
        from mlxturbo.kernels import prefill_attn as PAK

        orig_kernel = PAK.prefill_attn

        def counted(*a, **kw):
            fired[0] += 1
            return orig_kernel(*a, **kw)

        PAK.prefill_attn = counted
        try:
            run("kernel")
        finally:
            PAK.prefill_attn = orig_kernel
        disable_prefill_attn(model)
        disable_gather_attn(model)
        print(f"prefill_attn_layers={n_layers} kernel_fired={fired[0]}")

    finally:
        QE.Attention.__call__ = orig_attn_call
        QE.QSAIndexer._pooled_and_top = orig_pooled

    # --- 突き合わせ -------------------------------------------------
    print(f"{'layer':>5} {'out_maxdiff':>12} {'out_relmax':>11} "
          f"{'kv_len(off/kn)':>16} {'keep_flip_rows':>15} {'keep_flip_max':>14}")
    layer_ids = sorted(set(captured["off"]) | set(captured["kernel"]))
    for idx in layer_ids:
        off_d = captured["off"].get(idx, {})
        kn_d = captured["kernel"].get(idx, {})
        if "out" not in off_d or "out" not in kn_d:
            print(f"{idx:5d}  (このチャンク幅ではカーネルが発火していない)")
            continue
        a = off_d["out"]
        b = kn_d["out"]
        diff = float(mx.max(mx.abs(a - b)))
        denom = float(mx.max(mx.abs(a)))
        rel = diff / denom if denom > 0 else diff
        kv_off = off_d.get("kv_len")
        kv_kn = kn_d.get("kv_len")
        flip_rows = "-"
        flip_max = "-"
        if "keep" in off_d and "keep" in kn_d and off_d["keep"].shape == kn_d["keep"].shape:
            xor = mx.logical_xor(off_d["keep"], kn_d["keep"])
            per_row = np.array(mx.sum(xor.astype(mx.int32), axis=-1)).reshape(-1)
            flip_rows = int((per_row > 0).sum())
            flip_max = int(per_row.max())
        print(
            f"{idx:5d} {diff:12.4e} {rel:11.4e} {str(kv_off)+'/'+str(kv_kn):>16} "
            f"{str(flip_rows):>15} {str(flip_max):>14}"
        )

    print(
        "\n読み方: out_maxdiff が bf16 の丸め水準 (相対 ~1e-3 未満) に収まって"
        "いるのに keep_flip_rows が 0 でない層があれば、その層で block 選択"
        "が実際に反転している (= カスケード仮説を支持)。1 層目から"
        "out_relmax が桁違いに大きい、または 1 層目から keep_flip_rows が"
        "多いなら、その層 (=カーネル自体か visible-set 構成) を先に疑う。"
    )

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
