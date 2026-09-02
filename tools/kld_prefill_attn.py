"""prefill attention 融合カーネル (段 P1、`MLXTURBO_PREFILL_ATTN=1`) が、
長文脈 prefill の出力分布をどれだけ動かすかを見る。

`mlxturbo/gather_attn.py` の `enable_prefill_attn(model)` / `disable_prefill_attn(model)`
がこの knob の本体 (`tools/decode_ab.py --knob prefill-attn` の A/B と同じ関数)。
kv >= 12288 あたりで `mlxturbo/kernels/prefill_attn.py` の融合カーネルが発火する
(短い文脈では `eligible()` が MIN_S 未満などで弾いて既存の gather 経路へ落ちる)。

既存の `bench/quant_eval.py compare` は継続長 128 程度の短い continuation しか
通さないので、kv >= 12k のカーネル発火域を一度も踏まない。ここではその域を
直接踏んで、カーネル on/off で出力分布がどれだけ動くかを見る。

## 手順

1. `disable_prefill_attn(model)` の状態 (= 本番の既定、カーネル off) で
   prefill をチャンク幅 `--chunk` (既定 2048) で回し、最終チャンクの末尾
   `--tail` 位置の logits を fp32 で確保する -> 分布 p
2. キャッシュを作り直し、`enable_prefill_attn(model)` (= `MLXTURBO_PREFILL_ATTN=1`
   と同じ状態) で同じ prefill をもう一度回し、同じ位置の logits を取る -> 分布 q
3. 位置ごとに KL(p‖q) を出し、平均・最大・argmax 一致率・top-5 の重なりを出す

KLD は `bench/quant_eval.py` の `evaluate()` にある `kld_mean` と同じ式
(参照側 top-K 近似、既定 K=256 で揃えている): 分布 p の対数確率トップK の位置
だけを取り、``sum(p_k * (logp_k - logq_k))`` を位置ごとに計算する。式と K を
揃えているので、ここで出る数字は他所の KLD 計測 (受け入れ幅 現行比 +0.0005) と
同じ物差しで見比べられる。ここでの独自の目安は絶対値ベースで、0.001 未満なら
「不変」、0.01 未満なら「小」と表示する (それ以上は「要確認」)。

prefill 幅 (S=2048) はカーネルの `eligible()` の MIN_S 判定を通るはずだが、
kv がまだ閾値未満だったり cache の型が合わなかったりすると `_gather_forward`
が既存経路へ黙って落ちる (`mlxturbo/kernels/prefill_attn.py` が理由を 1 度だけ
表示する)。**カーネルが 1 度も発火していないのに「分布は不変」と出すのは
無意味な結果なので**、`prefill_attn()` の実行回数を数えて `kernel_fired` に
出し、0 なら警告する (`tools/verify_prefill_attn.py` の counted 手法と同じ)。

## --q-mode: 「経路自体の丸めの揺らぎ」との切り分け

既定 (``--q-mode kernel``) は上の手順どおり、q を段 P1 の T=1 カーネル
(`MLXTURBO_PREFILL_ATTN=1`) にする。カーネルの KLD (17k で mean 0.04) が
「カーネルという経路そのものの非ビット一致」に由来するのか、それとも
「dense のままでも chunk 境界の位置が変わればこの程度は揺れる」という
経路に依らない丸めの揺らぎなのかは、このカーネル単独の計測だけでは
切り分けられない。

``--q-mode chunk:<N>`` は q も p と同じ dense 経路 (prefill_attn は on に
しない) のまま、chunk 幅だけ ``N`` に変える対照群 --- p と q の意味論は
厳密に同じ causal dense attention で、prefill をどの位置で区切るかだけが
違う (chunk 境界が動くと、pooled cache の差分計算や sdpa split の呼び出し
粒度が変わるので、浮動小数の加算順もそのぶんだけ動く)。ここで出る KLD が
カーネルの KLD (0.04 級) と同じ桁なら、カーネルの KLD は経路に内在する
揺らぎの範囲内という判定になる。1 桁以上小さければ (0.001 級)、カーネル
固有の何かが効いている ("不採用" 側の根拠になる)。

    # 既定: カーネル vs dense (p は --chunk、q はカーネル・同じ --chunk)
    tools/biglock.sh .venv/bin/python tools/kld_prefill_attn.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctxs 17000,25000

    # 対照群: dense vs dense、chunk 幅だけ 2048 -> 4096 (p は --chunk のまま)
    tools/biglock.sh .venv/bin/python tools/kld_prefill_attn.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctxs 17000 --q-mode chunk:4096

**2026-09-03 追記。**上の `chunk:4096` 対照群は実測すると kld_mean 0.374と
カーネル (0.040) より 1 桁大きく出た。ただし chunk 幅は QSA の**意味論**を
変える (indexer の可視判定は「現在のチャンク内は因果で無条件に見える、
それより前はブロック単位の top-k 候補」という規約なので、chunk を
2048->4096 にすると「無条件に見える」窓そのものが倍になる)。つまり
`chunk:<N>` は「丸めだけの対照」ではなく「選択の母集団も一緒に変える対照」
になっていた --- カーネルの KLD (0.040) と比べるには荒すぎる物差し。

代わりに、**意味論を変えずに丸め (加算順・カーネル実装) だけを変える**
既知の対照を 2 つ用意した:

``--q-mode gdn-metal-off`` は Attention/QSA 側を p と全く同じ dense のまま
(prefill_attn は使わない) にして、GDN (線形注意) の再帰だけ p (既定:
`mlxturbo/kernels/gdn_blocked_metal.py` の oMLX 移植 Metal カーネル、
`GatedDeltaNet._gdn_metal`) と q (`mlx_lm` 本来の逐次実装) で切り替える。
短い continuation では KLD +0.00014 (受け入れ幅 +0.0005 のすぐ外) という
実績がある既知の小さな非ビット一致で、chunk 幅と違って可視集合や選択の
母集団は一切変えない。17k の長い prefill を通したときにこの既知の小さな
差分がカスケードでどれだけ増幅されるかの物差しになる。`_gdn_metal` は
`GatedDeltaNet` の**クラス属性**なので、呼ぶだけでモデル内の全 GDN 層に
一括で効く。

    # 対照群: dense vs dense、GDN の再帰カーネルだけ off (chunk は両方 2048)
    tools/biglock.sh .venv/bin/python tools/kld_prefill_attn.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctxs 17000 --q-mode gdn-metal-off

``--q-mode fold-tail-off`` (端数チャンクのグループ化 `MLXTURBO_PREFILL_FOLD_TAIL`、
`mlxturbo/spec_flash.py` の `_PREFILL_FOLD_TAIL`) は**このツールでは作れない**。
このフラグは `FlashSpecEngine` (同ファイル、`class FlashSpecEngine`) が中間
チャンクをグループ化して `_group_prefill_forward` に渡す**呼び出し側のループ**
だけが参照しており (`_group_prefill_forward` 自身は受け取らない)、この
グループ化判定は `FlashSpecEngine` のループ本体に書かれている。この
ツールの `_run_prefill` は `model(part, cache=cache)` を直接呼ぶだけで
`FlashSpecEngine` を一度も経由しないため、`MLXTURBO_PREFILL_FOLD_TAIL` を
どちらに設定しても `_run_prefill` の経路には何の影響も無い (= 対照に
ならない、on/off で常に同一になるだけ)。意味のある対照を作るには prefill
の駆動そのものを `FlashSpecEngine` 経由に作り直す必要があり、この差分の
範囲を超えるので保留した。``--q-mode fold-tail-off`` を指定すると、この
理由をそのまま表示して終了する。

モデルは 1 回だけ読む。GPU を使うので実行は biglock 経由。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# bench/quant_eval.py の cmd_dump/--topk の既定と揃える (kld_mean を同じ式・
# 同じ K で比べられるようにするため)。
DEFAULT_TOPK = 256


def _run_prefill(model, cache, ids, chunk: int, tail: int, pending):
    """``ids`` ((1, n) の mx.array) をチャンク幅 ``chunk`` で ``cache`` に流し、
    最終チャンクの末尾 ``tail`` 位置の logits (fp32) を返す。

    中間チャンクは logits を捨てて次へ進む。キャッシュへの書き込みは MLX が
    遅延評価するので、チャンクごとに ``pending(cache)`` (indexer のバッファを
    含めて eval を強制する、`tools/prefill_anatomy.py` の作法) を一緒に eval
    してから ``mx.clear_cache()`` で解放する。
    """

    import mlx.core as mx

    n = ids.shape[1]
    tail_logits = None
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        part = ids[:, start:end]
        logits = model(part, cache=cache)
        if end == n:
            t = min(tail, part.shape[1])
            tail_logits = logits[:, -t:, :].astype(mx.float32)
            mx.eval(tail_logits, *pending(cache))
        else:
            mx.eval(logits, *pending(cache))
        mx.clear_cache()
    if tail_logits is None:
        raise RuntimeError("ids が空 (n=0)")
    return tail_logits


def _flat(a):
    """mx.array の入れ子リストを平らな int の列にする (top-k の添字が (1, k) 形で来ても壊れない)。"""
    import mlx.core as mx
    return [int(v) for v in mx.array(a).reshape(-1).tolist()]


def _kld_stats(logits_p, logits_q, topk: int) -> dict:
    """位置ごとに ``bench/quant_eval.py`` の kld_mean と同じ式で KL(p‖q) を出す。

    ``logits_p`` / ``logits_q`` は同じ位置集合の (positions, vocab) fp32。
    p (``logits_p``) の対数確率トップ ``topk`` の位置だけを使う近似で、
    ``sum(p_k * (logp_k - logq_k))`` を位置ごとに計算する
    (``bench/quant_eval.py`` の ``evaluate()`` の ``kld_pos`` と同一式)。
    """
    import mlx.core as mx
    # (B, tail, vocab) でも (tail, vocab) でも受ける: 末尾の vocab 軸だけ残して平らにする
    logits_p = mx.array(logits_p).reshape(-1, mx.array(logits_p).shape[-1])
    logits_q = mx.array(logits_q).reshape(-1, mx.array(logits_q).shape[-1])

    import mlx.core as mx
    import numpy as np

    logp_full = logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True)
    logq_full = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)

    k = min(topk, logp_full.shape[-1])
    idx = mx.argpartition(-logp_full, k - 1, axis=-1)[..., :k]
    top_logp = mx.take_along_axis(logp_full, idx, axis=-1)
    # argpartition は順序を保証しない。降順に並べ直す
    # (bench/quant_eval.py の cmd_dump と同じ手順)。
    order = mx.argsort(-top_logp, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    top_logp = mx.take_along_axis(top_logp, order, axis=-1)
    top_logq = mx.take_along_axis(logq_full, idx, axis=-1)

    argmax_p = mx.argmax(logits_p, axis=-1)
    argmax_q = mx.argmax(logits_q, axis=-1)
    top5_p = mx.argpartition(-logits_p, 4, axis=-1)[..., :5]
    top5_q = mx.argpartition(-logits_q, 4, axis=-1)[..., :5]
    mx.eval(top_logp, top_logq, argmax_p, argmax_q, top5_p, top5_q)

    logp_np = np.array(top_logp, dtype=np.float64)
    logq_np = np.array(top_logq, dtype=np.float64)
    p_np = np.exp(logp_np)
    kld_pos = (p_np * (logp_np - logq_np)).sum(axis=-1)

    argmax_agree = np.array(argmax_p) == np.array(argmax_q)
    t5p = np.array(top5_p)
    t5q = np.array(top5_q)
    overlap = np.array(
        [len(set(_flat(t5p[i])) & set(_flat(t5q[i]))) for i in range(t5p.shape[0])],
        dtype=np.float64,
    )

    return {
        "positions": int(kld_pos.shape[0]),
        "topk": k,
        "kld_mean": float(kld_pos.mean()),
        "kld_max": float(kld_pos.max()),
        "kld_per_position": [float(x) for x in kld_pos],
        "argmax_agree_rate": float(argmax_agree.mean()),
        "top5_overlap_mean": float((overlap / 5.0).mean()),
    }


def _verdict(kld_mean: float) -> str:
    if kld_mean < 0.001:
        return "不変"
    if kld_mean < 0.01:
        return "小"
    return "要確認"


_FOLD_TAIL_UNAVAILABLE = (
    "--q-mode fold-tail-off はこのツールでは作れない: "
    "MLXTURBO_PREFILL_FOLD_TAIL (mlxturbo/spec_flash.py の _PREFILL_FOLD_TAIL) "
    "は FlashSpecEngine が中間チャンクをグループ化して _group_prefill_forward "
    "に渡す呼び出し側のループだけが参照する (_group_prefill_forward 自身は "
    "受け取らない)。tools/kld_prefill_attn.py の _run_prefill は "
    "model(part, cache=cache) を直接呼ぶだけで FlashSpecEngine を一度も "
    "経由しないので、このフラグをどちらに設定しても _run_prefill の経路には "
    "何の影響も無い (on/off で常に同一になるだけの、対照にならない対照)。"
    "意味のある対照を作るには prefill の駆動を FlashSpecEngine 経由に作り"
    "直す必要があり、この差分の範囲を超えるので保留した。"
)


def _parse_q_mode(spec: str) -> tuple[str, int | None]:
    """``--q-mode`` を解釈する。

    - ``"kernel"``: 既定。q = `enable_prefill_attn` (段 P1 T=1 カーネル)。
    - ``"chunk:<N>"``: q も p と同じ dense 経路 (prefill_attn off) のまま、
      chunk 幅だけ ``N`` にする対照群。戻り値の 2 要素目が q 側の chunk 幅。
      **注意 (2026-09-03)**: chunk 幅は QSA の可視判定の意味論そのものを
      変える (「現在のチャンク内は無条件に見える」窓が chunk 幅に比例して
      広がる) ので、丸めだけの対照にはならない (実測 kld_mean 0.374、
      カーネルの 0.040 より 1 桁大きい)。目安には使えるが、カーネルとの
      直接比較には `gdn-metal-off` の方が適する。
    - ``"gdn-metal-off"``: q は Attention/QSA 側を p と同じ dense のまま、
      GDN (線形注意) の再帰カーネルだけ off にする対照群
      (`mlxturbo.fused.disable_gdn_metal_kernel`)。可視集合や選択の母集団は
      一切変えない、意味論を保った既知の非ビット一致 (短い continuation で
      KLD +0.00014 の実績) --- 17k でどれだけ増幅されるかの物差し。
    - ``"fold-tail-off"``: 未対応。`_FOLD_TAIL_UNAVAILABLE` の理由で
      ValueError を送出する。

    戻り値は ``("kernel", None)`` / ``("chunk", N)`` / ``("gdn-metal-off", None)``。
    """
    if spec == "kernel":
        return "kernel", None
    if spec == "gdn-metal-off":
        return "gdn-metal-off", None
    if spec == "fold-tail-off":
        raise ValueError(_FOLD_TAIL_UNAVAILABLE)
    if spec.startswith("chunk:"):
        n_str = spec[len("chunk:") :]
        try:
            n = int(n_str)
        except ValueError:
            raise ValueError(
                f"--q-mode chunk:<N> の N が整数でない: {n_str!r}"
            ) from None
        if n <= 0:
            raise ValueError(f"--q-mode chunk:<N> の N は正の整数にすること: {n}")
        return "chunk", n
    raise ValueError(
        "--q-mode は 'kernel' / 'chunk:<N>' / 'gdn-metal-off' のいずれか"
        f" (got {spec!r})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="prefill attention 融合カーネル (MLXTURBO_PREFILL_ATTN) の"
        " on/off で、長文脈 prefill の出力分布がどれだけ動くかを測る"
    )
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument(
        "--ctxs",
        default="17000,25000",
        help="kv 長のカンマ区切り一覧 (カーネルは kv>=12288 あたりで発火)",
    )
    ap.add_argument(
        "--tail", type=int, default=64, help="最終チャンクの末尾何位置で分布を比べるか"
    )
    ap.add_argument(
        "--chunk", type=int, default=2048,
        help="p 側 (常に dense) の prefill チャンク幅。--q-mode kernel では"
        " q 側もこれと同じ幅を使う",
    )
    ap.add_argument(
        "--q-mode",
        default="kernel",
        help="q (比較対象) 側の作り方。'kernel' (既定): enable_prefill_attn"
        " (段 P1 T=1 カーネル、--chunk と同じ幅)。'chunk:<N>': q も dense の"
        " まま (prefill_attn off) chunk 幅だけ N にする対照群 (注意: chunk 幅は"
        " QSA の可視判定の意味論を変えるので丸めだけの対照ではない、実測"
        " kld_mean 0.374)。'gdn-metal-off': Attention/QSA は p と同じ dense の"
        " まま GDN の再帰カーネルだけ off にする対照群 (意味論を変えない既知の"
        " 非ビット一致、短い continuation で KLD +0.00014)。'fold-tail-off':"
        " 未対応 (理由を表示して終了、_FOLD_TAIL_UNAVAILABLE 参照)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="KLD 近似に使う p 側 top-K (bench/quant_eval.py の既定と揃えてある)",
    )
    ap.add_argument(
        "--question",
        default="上の文書の要点を5つに整理してください。",
        help="長文脈プロンプトの末尾に付ける質問 (tools/_bench_text.py の作法)",
    )
    ap.add_argument("--out", default="bench/results/kld-prefill-attn.json")
    args = ap.parse_args()

    ctxs = sorted({int(v) for v in args.ctxs.split(",") if v.strip() != ""})
    if not ctxs:
        print("--ctxs が空")
        return 1
    if args.tail <= 0:
        print("--tail は正の整数にすること")
        return 1
    try:
        q_mode, q_mode_chunk = _parse_q_mode(args.q_mode)
    except ValueError as e:
        print(str(e))
        return 1
    if q_mode == "gdn-metal-off" and os.environ.get("MLXTURBO_GDN_METAL") == "0":
        # `enable_gdn_metal_kernel()` 自身が MLXTURBO_GDN_METAL=0 で
        # 何もしない (mlxturbo/fused.py)。その状態でこの対照を回すと p 側も
        # 既に GDN metal off なので、p/q が同じ経路になり差が出ない
        # (対照として無意味)。先に教えて誤読を防ぐ。
        print(
            "--q-mode gdn-metal-off だが MLXTURBO_GDN_METAL=0 が既に立って"
            "いる: p 側も GDN metal off になるため対照が成立しない"
            " (この環境変数を外してから実行すること)"
        )
        return 1

    if args.ngram:
        # n-gram をディスクに置いた構成。vendored arch は import 時に旗を読む。
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx

    import mlxturbo  # noqa: F401  (arch_registry の meta_path フックを張る)
    from mlx_lm import load

    from mlxturbo.gather_attn import (
        disable_gather_attn,
        disable_prefill_attn,
        enable_prefill_attn,
    )
    from mlxturbo.kernels import prefill_attn as prefill_attn_kernel
    from mlxturbo import fused
    from mlxturbo import runner as mlxturbo_runner

    model_path = os.path.expanduser(args.model)
    ngram_path = os.path.expanduser(args.ngram) if args.ngram else None

    model, tok = load(model_path)
    # 読み込み直後に呼ぶ (常駐条件を本番と揃える。engine を直叩きなので
    # server.py の _load() を経由しないぶん、ここで自前で wire する)。
    if hasattr(mlxturbo_runner, "set_wired_limit_default"):
        mlxturbo_runner.set_wired_limit_default(log_prefix="[kld-prefill-attn]")
    if ngram_path:
        from mlxturbo.ngram_stream import install

        install(model, ngram_path)
    # 出荷経路 (build_runner) が起動時に通す融合・置き換えと同じものを当てる。
    # これで prefill_attn 以外の knob (MLXTURBO_GDN_METAL 等) が本番の既定値
    # のまま揃い、on/off の差分が prefill_attn だけに帰属する。
    mlxturbo_runner.enable_default_fusions(model, log_prefix="[kld-prefill-attn]")

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts  # noqa: E402
    import prefill_anatomy as PA  # noqa: E402  (pending() を借りる)

    q_chunk_resolved = q_mode_chunk if q_mode == "chunk" else args.chunk
    print(
        f"model={args.model} ngram={args.ngram} ctxs={ctxs} q_mode={args.q_mode}"
        f" chunk(p)={args.chunk} chunk(q)={q_chunk_resolved}"
        f" tail={args.tail} topk={args.topk}",
        flush=True,
    )

    results: dict = {
        "kind": "kld-prefill-attn",
        "model": args.model,
        "ngram": args.ngram,
        "q_mode": args.q_mode,
        "chunk": args.chunk,  # p 側 (= 従来どおりの意味)。後方互換で残す
        "q_chunk": q_chunk_resolved,
        "tail": args.tail,
        "topk": args.topk,
        "ctxs": {},
    }

    ok_overall = True
    for ctx in ctxs:
        print(f"=== ctx={ctx} ===", flush=True)
        body = long_prompts(tok, ctx, [args.question])[0]
        ids = mx.array(
            tok.apply_chat_template(
                [{"role": "user", "content": body}], add_generation_prompt=True
            )
        )[None]
        print(f"  実 kv={ids.shape[1]}", flush=True)

        # (1) 分布 p: 完全に dense へ戻す (= 本番の既定、カーネル off)。
        #
        # **2026-09-03 訂正。**以前はここで `disable_prefill_attn(model)` を
        # 呼んでいたが、`mlxturbo/gather_attn.py` の docstring どおり
        # `disable_prefill_attn` は「`enable_prefill_attn` だけを打ち消す
        # (gather 経路は残す)」規約で、`_gather_attn` を False に戻さない。
        # `--ctxs` に複数の長さを渡す通常の使い方では、1 つ前の ctx の
        # `enable_prefill_attn` (line 279 相当) が `_gather_attn=True` を
        # 立てたまま残り、2 番目以降の ctx の「p (off のはず)」が実際には
        # 汎用 gather 経路 (`_gather_tile_attn`) を通ってしまっていた
        # (先頭の ctx だけは `_gather_attn` が未設定なので無傷)。
        # `disable_gather_attn` は `_gather_attn`/`_prefill_attn` の両方を
        # 落とすので、こちらが本来の「dense へ完全に戻す」呼び方。
        disable_gather_attn(model)
        # `_gdn_metal` は GatedDeltaNet の**クラス属性**(`mlxturbo/fused.py`)
        # なので、1 つ前の ctx が `--q-mode gdn-metal-off` で q 側を off に
        # した場合、そのままだと p 側にまで漏れる (`_gather_attn` と同じ
        # クラスの漏れ)。p は常に本番の既定 (GDN metal on) であるべきなので、
        # 毎回明示的に立て直す (env `MLXTURBO_GDN_METAL=0` ならここでも off の
        # ままになるが、それは呼び手が明示的に指定した既定なので尊重する)。
        fused.enable_gdn_metal_kernel()
        cache_p = model.make_cache()
        logits_p = _run_prefill(model, cache_p, ids, args.chunk, args.tail, PA.pending)
        del cache_p
        mx.clear_cache()

        if q_mode == "kernel":
            # (2a) 分布 q: enable_prefill_attn (= MLXTURBO_PREFILL_ATTN=1 と
            # 同じ状態)。カーネルが実際に発火した回数を数える
            # (0 なら比較そのものが無意味)。
            fired = [0]
            orig_kernel_fn = prefill_attn_kernel.prefill_attn

            def _counted(*a, **kw):
                fired[0] += 1
                return orig_kernel_fn(*a, **kw)

            prefill_attn_kernel.prefill_attn = _counted
            n_layers = enable_prefill_attn(model)
            try:
                cache_q = model.make_cache()
                logits_q = _run_prefill(
                    model, cache_q, ids, args.chunk, args.tail, PA.pending
                )
            finally:
                prefill_attn_kernel.prefill_attn = orig_kernel_fn
                disable_prefill_attn(model)
            del cache_q
            mx.clear_cache()
            q_chunk = args.chunk
        elif q_mode == "chunk":
            # (2b) 対照群: q も p と全く同じ dense 経路 (prefill_attn は
            # 一度も on にしない)。違うのは chunk 幅だけ。
            #
            # **注意 (2026-09-03)**: chunk 幅は QSA の可視判定の意味論を
            # 変える (「現在のチャンク内は無条件に見える」窓が chunk 幅に
            # 比例して広がる) ので、これは厳密な意味論一致の対照ではない
            # (実測 kld_mean 0.374、カーネルの 0.040 より 1 桁大きい)。
            # 目安にはなるが、カーネルとの直接比較には q_mode=gdn-metal-off
            # の方が適する (`_parse_q_mode` の docstring 参照)。
            disable_gather_attn(model)
            fused.enable_gdn_metal_kernel()
            n_layers = 0
            fired = [0]
            q_chunk = q_mode_chunk
            cache_q = model.make_cache()
            logits_q = _run_prefill(
                model, cache_q, ids, q_chunk, args.tail, PA.pending
            )
            del cache_q
            mx.clear_cache()
        elif q_mode == "gdn-metal-off":
            # (2c) 対照群: Attention/QSA 側は p と全く同じ dense (prefill_attn
            # は一度も on にしない、chunk 幅も p と同じ args.chunk)。違いは
            # GDN (線形注意) の再帰カーネルだけ --- p は既定どおり oMLX 移植
            # の blocked-sequential Metal カーネル (`_gdn_metal`)、q は
            # `mlx_lm` 本来の逐次実装に戻す
            # (`mlxturbo.fused.disable_gdn_metal_kernel`)。可視集合や選択の
            # 母集団は一切変えない、意味論を保った既知の非ビット一致
            # (短い continuation で KLD +0.00014、受け入れ幅 +0.0005 のすぐ
            # 外の実績)。17k の長い prefill でこの既知の小さな差分がカスケード
            # でどれだけ増幅されるかの物差しになる。
            disable_gather_attn(model)
            fused.disable_gdn_metal_kernel()
            n_layers = 0
            fired = [0]
            q_chunk = args.chunk
            try:
                cache_q = model.make_cache()
                logits_q = _run_prefill(
                    model, cache_q, ids, q_chunk, args.tail, PA.pending
                )
            finally:
                # `_gdn_metal` はクラス属性なので、次の ctx (または p) に
                # 漏れないよう必ず戻す。p 側の防御的な re-assert
                # (`fused.enable_gdn_metal_kernel()`) と合わせて二重に守る。
                fused.enable_gdn_metal_kernel()
            del cache_q
            mx.clear_cache()
        else:
            raise AssertionError(
                f"unhandled q_mode {q_mode!r} (_parse_q_mode で弾いているはず)"
            )

        stats = _kld_stats(logits_p, logits_q, args.topk)
        stats["kv"] = int(ids.shape[1])
        stats["q_mode"] = args.q_mode
        stats["p_chunk"] = args.chunk
        stats["q_chunk"] = q_chunk
        stats["prefill_attn_layers"] = n_layers
        stats["kernel_fired"] = fired[0]
        stats["verdict"] = _verdict(stats["kld_mean"])
        if q_mode == "kernel" and fired[0] == 0:
            print(
                "  ★カーネルが1度も発火していない"
                " (kv が閾値未満、または eligible() が別の理由で弾いている可能性。"
                " この ctx の比較は無意味なので verdict を信用しないこと) ★",
                flush=True,
            )
            ok_overall = False
        print(
            f"  positions={stats['positions']} q_mode={args.q_mode}"
            f" p_chunk={stats['p_chunk']} q_chunk={stats['q_chunk']}"
            f" kernel_fired={fired[0]}/{n_layers} layers"
            f" kld_mean={stats['kld_mean']:.6f} kld_max={stats['kld_max']:.6f}"
            f" argmax_agree={stats['argmax_agree_rate']:.4f}"
            f" top5_overlap={stats['top5_overlap_mean']:.4f}"
            f" -> {stats['verdict']}",
            flush=True,
        )
        results["ctxs"][str(ctx)] = stats
        del logits_p, logits_q
        mx.clear_cache()

    results["ok"] = ok_overall
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"wrote {out_path}", flush=True)

    # 計測ツールなので destructor (スレッドプール等の後始末) に用は無い。
    # interpreter shutdown 待ちでプロセスが Metal のメモリを握ったまま残る
    # 前例があるので、結果を書き終えたら即 _exit で落とす (他の tools/*.py と同じ)。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
