"""full-attention 層 1 層 (indexer 込み) の時間を 4 部品に割る。

    (a) indexer   QSAIndexer._pooled_and_top -- pooled key の作成・q・k
                  スコア・可視判定・argpartition (:387-447)
    (b) mask/選択  dense 経路: QSAIndexer.__call__ のブロック→トークン展開
                  (:449-505) + Attention._final_mask の mask & sparse
                  (:565-586)。gather 経路 (段 3(b)、既定 on): select_blocks
                  の展開ぶん (ほぼ (a) と同額) + _gather_tile_attn のうち
                  sdpa 呼び出しを除いた分 (union・argsort・take_along_axis
                  による K/V の gather、keep_sel の構築)
    (c) sdpa      mx.fast.scaled_dot_product_attention 本体 (base.py 経由)。
                  head_dim 256 は MLX 0.32.2 で fallback (スコア実体化) --
                  `tools/sdpa_headdim_micro.py` に同じ形の合成マイクロがある。
                  decode 幅 (S<=8) は vector カーネル (bool マスクでキー読みを
                  スキップする経路) に入るので、prefill 幅とは別物
    (d) その他     Attention._qkv (q/k/v 射影・QK-norm・rope・KV キャッシュ
                  更新) + 末尾の o_proj(out * sigmoid(gate))

kv (= 過去の文脈 + このステップの新規トークン数 S) は `--kvs` で指定した
点を **昇順に 1 回の prefill で通過しながら** 測る -- 50k のような長い kv を
毎回ゼロから prefill すると分単位で伸びるので、共有 1 本のキャッシュを
ランプアップ (`--chunk` 幅、既定 2048 = 本番の PREFILL_STEP_SIZE) で進め、
目標 kv に達したところで `--S` 幅 (既定 2048 = prefill チャンク。decode を見る
なら 1/2/4) の 1 ステップだけ計測してから、その同じステップを本物の
(ラップしない) forward でもう一度実行してキャッシュを確定させ、次の kv へ
進む。indexer の pooled 増分キャッシュ (`_IndexerCache`) はこの一連の流れの
中で本番と同じように「まだ確定していないブロックだけ新規計算」される。

対象は 12 層 (既定構成) のうち **最初の** full-attention 層だけ。同じ層クラス
の他 11 層は素通り (計測に混ざらない) -- `Attention.__call__` / `_qkv` /
`_final_mask` / `_gather_tile_attn` / `QSAIndexer.__call__` / `.select_blocks`
/ `._pooled_and_top` を一時的にラップし、`self is <対象インスタンス>` の
ときだけ引数を控える。**QSAIndexer の中身は変えない (ラップして呼ぶだけ)。**
`Attention._positions` などのシームも同様、差し替えではなく呼び出しの前後で
時間を取るだけ。

部品ごとに「その部品だけを走らせて mx.eval」する形で `--reps` 回 (既定 5) の
中央値 (ms) を取る。(b) は 2 つの部品 (指標を構成する外側のメソッド全体 と
(a) or (c)) を独立に測ってから引き算で出す -- この引き算そのものは
`tools/prefill_anatomy.py` の MoE 内訳 (層ごと合計 vs 積み上げ) と同じ流儀で、
隠さず表に出す。(d) は `_qkv` を直接測り、末尾の gate/o_proj は sdpa の実出力
(捕まえた実配列、計測区間の外で eval 確定済み) と qkv の実 gate を使って
`o_proj(out.transpose(0,2,1,3).reshape(B,S,-1) * sigmoid(gate))` を直接測る
(qwen4_exp.py:964-965 / :909-910 と同じ式。QSAIndexer/Attention の実装は
変えていない -- 計測用に実配列で同じ式をもう一度評価するだけ)。

**部品和 ≈ 壁時計 (層 1 つを実引数で 1 回呼んだ時間) を必ず確認する**
(CLAUDE.md の計測の作法)。壁時計は `Attention.__call__` を捕まえた実引数
(x, rope, mask, cache, idx_cache) でそのまま呼び直したもので、部品側が拾って
いない host 側分岐 (`_positions`、gather 可否のホスト算数、decode の
split_mask 用 causal 配列組み --- いずれも小さいはず) はここでは部品に
割り当てず、差として出す。既知の限界として明記する。

12 層換算 (`x12`) と、12 層ぶんを 1 トークンあたりに割った `ms_per_tok_x12`
(= 部品 ms * n_attn / S) も出す -- decode の ms/tok が kv とともに伸びる分の
うち、attention のどの部品が効いているかを直接比べるための列。

## `--chain N` (decode 幅専用、既定 0 = 使わない)

decode 幅 (S<=8) で上の「部品ごとに mx.eval して 5 回中央値」をそのまま
やると、部品和が壁時計の 1.5〜2.4 倍に膨れる実測が出ている
(`bench/results/qsa-decode-split-S1.json` / `S2.json`)。原因は部品の実体が
0.05〜0.2ms 程度しかないのに、部品ごとの `mx.eval` 1 回あたりの同期コスト
(0.2〜0.3ms、ディスパッチ + 完了待ち) がそれを上回って支配すること --
帰属には使えない。prefill 幅 (S=2048) はこの同期コストが実体の 1〜2% しか
無いので ±3% で問題ない (`--chain 0` のままでよい)。

`--chain N` (S<=8 のときだけ有効。推奨 N=50) を指定すると、部品ごとに
「わずかに入力を変えた呼び出しを N 回連ねてから、最後に 1 回だけ
`mx.eval`」に切り替え、(その 1 回の壁時計 / N) を 1 回あたりの ms として
使う (表示は us/回)。同期コスト 1 回分を N 回で割るので、N が十分大きければ
実体の割合が支配的になる。**層全体も同じ N 回連鎖で測り、部品和 (連鎖) と
突き合わせる** -- 部品ごとの `--chain` の効きが均一かどうかがここで見える。

入力の変え方: sdpa / `_gather_tile_attn` / `_qkv` / 層全体 / 末尾の
gate+o_proj は、渡す q (または x、o_proj の入力) を `t * (1 + (i+1)*1e-2)`
で毎回わずかに乗じる (bf16 で 50 回とも別値になることを CPU device の合成
テンソルで実測して決めた倍率 -- 1e-4/1e-3 は bf16 の丸めで大半が同値に潰れる
ことを確認済み、1e-2 で 50/50 distinct。MLX が同一入力の重複計算を畳んでも
拾えるように)。**indexer 系 (`_pooled_and_top` /
`QSAIndexer.__call__` / `select_blocks`) だけは同じ x を N 回渡す** --
`_IndexerCache` の増分状態 (`_buf`/`offset`/`_pooled`/`_pooled_n`) は
N 回呼ぶと進んでしまうので、**呼ぶたびに `tools/prefill_anatomy.py` の
snapshot/restore で層キャッシュを直前の状態へ戻してから次を呼ぶ** (pure
Python の属性代入だけで GPU 同期は無い。「pooled を固定して
スコア+argpartition だけ N 回」という代替案もあるが、こちらは
`_pooled_and_top` の中身を計測用に書き写す必要が出るので、**スナップショット
復元の方を選んだ** -- 採用した方式は JSON の `chain_cache_strategy` と
コンソールのヘッダに出す)。`_qkv` と層全体 (`Attention.__call__`) は KV
キャッシュ/indexer キャッシュを実際に進めるので、こちらも各呼び出し前に
層キャッシュへ戻す。`_final_mask` と sdpa 本体・`_gather_tile_attn` は
キャッシュに触れないので戻す必要が無い (sdpa/`_gather_tile_attn` は q を
上の式で摂動、`_final_mask` は bool 演算のみで摂動しようが無いので同じ引数の
まま N 回呼ぶ)。

    tools/biglock.sh .venv/bin/python tools/qsa_prefill_split.py \\
        --model ~/models/ddalcu-mlxlm --S 2048 --kvs 2048,8192,16896

    tools/biglock.sh .venv/bin/python tools/qsa_prefill_split.py \\
        --model ~/models/ddalcu-mlxlm --S 2 --kvs 4096,17000,25000,50000 --chain 50
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def wrap_instance(cls, name, is_target, calls, bucket):
    """`cls.name` を一時的にラップし、`self` が対象インスタンスのときだけ
    `(self, args, 戻り値)` を `calls[bucket]` に積む。**呼ばれるたびに
    必ず元の実装を呼ぶ** (対象以外のインスタンスの挙動は変えない)。
    戻り値は素の `getattr` (= 元の未ラップ関数、再ラップ時の巻き込み防止)。
    """
    orig_fn = getattr(cls, name)

    def g(self, *a):
        r = orig_fn(self, *a)
        if is_target(self):
            calls[bucket].append((self, a, r))
        return r

    setattr(cls, name, g)
    return orig_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--S", type=int, default=2048,
                     help="計測する 1 ステップの新規トークン数。"
                          "2048=prefill チャンク、1/2/4=decode 幅")
    ap.add_argument("--kvs", default="2048,8192,16896",
                     help="kv (= 過去の文脈 + S) の一覧。昇順でなくてよい"
                          "(内部でソートする)。カンマ区切り")
    ap.add_argument("--chunk", type=int, default=2048,
                     help="kv の間を埋めるランプアップ prefill のチャンク幅"
                          " (本番の PREFILL_STEP_SIZE と同じ既定)")
    ap.add_argument("--reps", type=int, default=5, help="中央値を取るレップ数")
    ap.add_argument("--chain", type=int, default=0,
                     help="S<=8 のときだけ有効。N>0 で「部品を N 回連鎖して"
                          "1 回だけ mx.eval」に切り替え、mx.eval の同期"
                          "コストで decode 幅の部品和が壁時計より大きく"
                          "膨らむのを避ける (既定 0=使わない。推奨 N=50)")
    ap.add_argument("--out", default=None,
                     help="既定: bench/results/qsa-attn-split-S<S>.json")
    args = ap.parse_args()

    kvs = sorted({int(v) for v in args.kvs.split(",") if v.strip() != ""})
    if not kvs:
        print("--kvs が空")
        return 1
    if kvs[0] < args.S:
        print(f"最小の kv ({kvs[0]}) が S ({args.S}) 未満 -- kv = offset + S "
              f"なので kv >= S が必要")
        return 1
    for i in range(1, len(kvs)):
        if kvs[i] - kvs[i - 1] < args.S:
            print(f"--kvs の隣接差が S={args.S} 未満 ({kvs[i - 1]} -> {kvs[i]})"
                  f": 前の点を消化しきる前に次の offset に届かない")
            return 1

    out_path = Path(args.out) if args.out else Path(
        f"bench/results/qsa-attn-split-S{args.S}.json")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401  (arch_registry の meta_path フックを張る)
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts
    import prefill_anatomy as PA  # snapshot/pending/restore/med_ms を借りる

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[qsa-attn-split]")
    # engine を直叩きなので server.py の _load() を経由しない -- 常駐条件を
    # 本番と揃えるため、ここで自前で wire する
    # (mlxturbo/runner.py の set_wired_limit_default 参照)。
    set_wired_limit_default(log_prefix="[qsa-attn-split]")

    ta = model.args.text
    layer_types = ta.layer_types
    if "full_attention" not in layer_types:
        print("full_attention 層が無いアーキテクチャ")
        return 1
    target_layer_idx = layer_types.index("full_attention")
    n_attn = sum(1 for t in layer_types if t == "full_attention")
    target_attn = model.model.layers[target_layer_idx].self_attn
    target_idx = target_attn.indexer

    max_kv = kvs[-1]
    body = long_prompts(
        tok, max_kv + 256, ["上の文書の要点を 5 つに整理してください。"]
    )[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    if ids.shape[1] < max_kv:
        print(f"プロンプトが {ids.shape[1]} tok しかなく、必要な {max_kv}"
              f" に届かない (--kvs を減らすこと)")
        return 1

    cache = model.make_cache()
    layer_cache = cache[target_layer_idx]

    mode = "decode 幅 (vector カーネル想定)" if args.S <= 8 else \
        "prefill 幅 (head_dim 256 fallback 想定)"
    chain_mode = args.chain > 0 and args.S <= 8
    chain_note = (
        "chain=OFF (部品ごとに mx.eval、既定)" if args.chain <= 0 else
        (f"chain=N={args.chain} (S<=8 なので有効。snapshot/restore で"
         f" 層キャッシュを毎回戻し、q/x/out を (1+(i+1)*1e-2) 倍で摂動)"
         if chain_mode else
         f"chain=N={args.chain} だが S={args.S}>8 (prefill 幅) なので無視"
         f" (部品ごとに mx.eval のまま)")
    )
    print(f"S={args.S} [{mode}] kvs={kvs} chunk={args.chunk} reps={args.reps}"
          f" target_layer={target_layer_idx} n_attn={n_attn}"
          f" indexer_budget={target_idx.token_budget}"
          f" compress_ratio={target_idx.compress_ratio}", flush=True)
    print(f"  {chain_note}", flush=True)

    def measure_point(offset: int) -> dict:
        """kv = offset + args.S の 1 点を測る。キャッシュは非破壊 (呼び出し
        側が snapshot/restore する必要は無い -- 中で完結する)。
        """
        S = args.S
        chunk = ids[:, offset: offset + S]

        calls = {k: [] for k in (
            "pooled", "idx_call", "select_blocks", "final_mask", "qkv",
            "gather_tile",
        )}
        sdpa_calls = []  # (args, kwargs, 戻り値) の list。self を持たない
        capture_on = [False]
        whole_box = {}

        orig = {
            "pooled": wrap_instance(
                Q.QSAIndexer, "_pooled_and_top",
                lambda s: s is target_idx, calls, "pooled"),
            "idx_call": wrap_instance(
                Q.QSAIndexer, "__call__",
                lambda s: s is target_idx, calls, "idx_call"),
            "select_blocks": wrap_instance(
                Q.QSAIndexer, "select_blocks",
                lambda s: s is target_idx, calls, "select_blocks"),
            "final_mask": wrap_instance(
                Q.Attention, "_final_mask",
                lambda s: s is target_attn, calls, "final_mask"),
            "qkv": wrap_instance(
                Q.Attention, "_qkv",
                lambda s: s is target_attn, calls, "qkv"),
            "gather_tile": wrap_instance(
                Q.Attention, "_gather_tile_attn",
                lambda s: s is target_attn, calls, "gather_tile"),
        }

        orig_call = Q.Attention.__call__

        def wrapped_call(self, *a):
            if self is not target_attn:
                return orig_call(self, *a)
            capture_on[0] = True
            try:
                r = orig_call(self, *a)
            finally:
                capture_on[0] = False
            whole_box["args"] = a
            return r

        Q.Attention.__call__ = wrapped_call

        orig_sdpa = Q.scaled_dot_product_attention

        def wrapped_sdpa(*a, **kw):
            r = orig_sdpa(*a, **kw)
            if capture_on[0]:
                sdpa_calls.append((a, kw, r))
            return r

        Q.scaled_dot_product_attention = wrapped_sdpa

        pre = PA.snapshot(cache)
        out = model.model(chunk, cache=cache)
        mx.eval([out] + PA.pending(cache))

        Q.QSAIndexer._pooled_and_top = orig["pooled"]
        Q.QSAIndexer.__call__ = orig["idx_call"]
        Q.QSAIndexer.select_blocks = orig["select_blocks"]
        Q.Attention._final_mask = orig["final_mask"]
        Q.Attention._qkv = orig["qkv"]
        Q.Attention._gather_tile_attn = orig["gather_tile"]
        Q.Attention.__call__ = orig_call
        Q.scaled_dot_product_attention = orig_sdpa

        PA.restore(cache, pre)
        mx.clear_cache()

        dense = len(calls["final_mask"]) > 0
        gather = len(calls["gather_tile"]) > 0
        if not dense and not gather:
            print("  ** 警告: dense (_final_mask) でも gather"
                  " (_gather_tile_attn) でも捕まらなかった。既定外の融合"
                  " カーネル (MLXTURBO_PREFILL_ATTN など) が有効になっていないか"
                  " 確認すること **")
            branch = "unknown"
        else:
            branch = "dense" if dense else "gather"

        def bench_layer(pairs):
            """[(元関数, self, args), ...] を層キャッシュだけ退避して測る。
            (`--chain` 無効時のみ使う: 1 回呼んで即 mx.eval、を reps 回。)
            """
            if not pairs:
                return 0.0

            def run():
                st = PA.snapshot([layer_cache])
                outs = [fn(self_obj, *a) for fn, self_obj, a in pairs]
                mx.eval(outs + PA.pending([layer_cache]))
                PA.restore([layer_cache], st)
                return outs

            ms = PA.med_ms(run, args.reps)
            mx.clear_cache()
            return ms

        def bench_bucket(key):
            items = calls[key]
            fn = orig[key]
            return bench_layer([(fn, s, a) for s, a, _ in items])

        def bench_sdpa():
            if not sdpa_calls:
                return 0.0

            def run():
                st = PA.snapshot([layer_cache])
                outs = [orig_sdpa(*a, **kw) for a, kw, _ in sdpa_calls]
                mx.eval(outs + PA.pending([layer_cache]))
                PA.restore([layer_cache], st)
                return outs

            ms = PA.med_ms(run, args.reps)
            mx.clear_cache()
            return ms

        def pert(t, i):
            """t を毎回わずかに (相対 1e-2 刻みで) ずらす -- bf16 の丸めで
            消えない程度の摂動で、MLX が同一入力の重複計算を畳んでも N 回ぶんの
            本物の計算になるようにする (コーディネータ指定)。倍率は 1e-2:
            CPU device の合成テンソルで 1e-4/1e-3 は 50 回中 2/7 個しか
            別値にならず (bf16 の丸めに埋もれる)、1e-2 で初めて 50/50 distinct
            になることを確認して選んだ (bf16 の ulp は値の丸めだけで決まる
            静的な性質なので、実モデルの hidden state でも同じ余裕がある)。"""
            return t * mx.array(1.0 + (i + 1) * 1e-2, dtype=t.dtype)

        def chain_ms(build_fn, cache_objs):
            """build_fn(i) を N=args.chain 回呼んでグラフを積み、最後に
            1 回だけ mx.eval する。壁時計 (積み+eval 込み) を reps 回取って
            中央値を N で割り、1 回あたり ms を返す。`cache_objs` を渡すと
            各呼び出しの前に snapshot へ戻す (呼ぶたびに状態が進む API 用。
            snapshot/restore は Python の属性代入のみで GPU 同期は無いので
            計測区間に含めても実体を歪めない)。build_fn は配列の list を
            返す規約。"""
            N = args.chain

            def once():
                t0 = time.perf_counter()
                pre = PA.snapshot(cache_objs) if cache_objs is not None else None
                outs = []
                for i in range(N):
                    if cache_objs is not None and i > 0:
                        PA.restore(cache_objs, pre)
                    outs.extend(build_fn(i))
                pend = PA.pending(cache_objs) if cache_objs is not None else []
                mx.eval(outs + pend)
                dt = (time.perf_counter() - t0) * 1000
                if cache_objs is not None:
                    PA.restore(cache_objs, pre)
                return dt

            once()  # 温め 1 本は捨てる
            ts = [once() for _ in range(args.reps)]
            mx.clear_cache()
            return statistics.median(ts) / N

        pooled_entry = calls["pooled"][0] if calls["pooled"] else None
        idx_call_entry = calls["idx_call"][0] if calls["idx_call"] else None
        select_blocks_entry = (
            calls["select_blocks"][0] if calls["select_blocks"] else None)
        final_mask_entry = calls["final_mask"][0] if calls["final_mask"] else None
        qkv_entry = calls["qkv"][0] if calls["qkv"] else None
        gather_tile_entries = calls["gather_tile"]

        b_detail = {}
        if chain_mode:
            # -- (a) indexer: x は摂動しない (コーディネータ指定)。
            #    _IndexerCache は毎回進むので snapshot/restore で戻す。
            def bf_pooled(i):
                self_obj, a, _ = pooled_entry
                return [orig["pooled"](self_obj, *a)]

            def bf_idx_call(i):
                self_obj, a, _ = idx_call_entry
                return [orig["idx_call"](self_obj, *a)]

            def bf_select_blocks(i):
                self_obj, a, _ = select_blocks_entry
                return [orig["select_blocks"](self_obj, *a)]

            def bf_final_mask(i):
                self_obj, a, _ = final_mask_entry
                return [orig["final_mask"](self_obj, *a)]

            def bf_qkv(i):
                self_obj, a, _ = qkv_entry
                x0, positions0, rope0, kv_cache0 = a
                return list(orig["qkv"](
                    self_obj, pert(x0, i), positions0, rope0, kv_cache0))

            def bf_sdpa(i):
                return [orig_sdpa(pert(a[0], i), *a[1:], **kw)
                        for a, kw, _ in sdpa_calls]

            def bf_gather_tile(i):
                outs = []
                for self_obj, a, _ in gather_tile_entries:
                    outs.append(orig["gather_tile"](
                        self_obj, pert(a[0], i), *a[1:]))
                return outs

            a_ms = (chain_ms(bf_pooled, [layer_cache])
                    if pooled_entry else 0.0)
            c_ms = chain_ms(bf_sdpa, None) if sdpa_calls else 0.0
            qkv_ms = chain_ms(bf_qkv, [layer_cache]) if qkv_entry else 0.0

            if dense:
                idx_total = (chain_ms(bf_idx_call, [layer_cache])
                             if idx_call_entry else 0.0)
                fm_ms = (chain_ms(bf_final_mask, None)
                         if final_mask_entry else 0.0)
                b_ms = max(idx_total - a_ms, 0.0) + fm_ms
                b_detail = {"idx_call_total_ms": idx_total,
                            "final_mask_ms": fm_ms}
            elif gather:
                sel_total = (chain_ms(bf_select_blocks, [layer_cache])
                             if select_blocks_entry else 0.0)
                gta_total = (chain_ms(bf_gather_tile, None)
                             if gather_tile_entries else 0.0)
                b_ms = max(sel_total - a_ms, 0.0) + max(gta_total - c_ms, 0.0)
                b_detail = {"select_blocks_total_ms": sel_total,
                            "gather_tile_total_ms": gta_total}
            else:
                b_ms = 0.0

            sdpa_outs = [r for (_, _, r) in sdpa_calls]
            if len(sdpa_outs) == 1:
                out_val = sdpa_outs[0]
            elif len(sdpa_outs) > 1:
                out_val = mx.concatenate(sdpa_outs, axis=2)
            else:
                out_val = None
            gate_val = qkv_entry[2][3] if qkv_entry else None

            tail_ms = 0.0
            if out_val is not None and gate_val is not None:
                mx.eval(out_val, gate_val)
                B = chunk.shape[0]

                def bf_tail(i):
                    o = pert(out_val, i).transpose(0, 2, 1, 3).reshape(B, S, -1)
                    return [target_attn.o_proj(o * mx.sigmoid(gate_val))]

                tail_ms = chain_ms(bf_tail, None)

            d_ms = qkv_ms + tail_ms

            whole_ms = 0.0
            if "args" in whole_box:
                wargs = whole_box["args"]

                def bf_whole(i):
                    return [orig_call(
                        target_attn, pert(wargs[0], i), wargs[1], wargs[2],
                        wargs[3], wargs[4])]

                whole_ms = chain_ms(bf_whole, [layer_cache])
        else:
            a_ms = bench_bucket("pooled")
            c_ms = bench_sdpa()
            qkv_ms = bench_bucket("qkv")

            if dense:
                idx_total = bench_bucket("idx_call")
                fm_ms = bench_bucket("final_mask")
                b_ms = max(idx_total - a_ms, 0.0) + fm_ms
                b_detail = {"idx_call_total_ms": idx_total,
                            "final_mask_ms": fm_ms}
            elif gather:
                sel_total = bench_bucket("select_blocks")
                gta_total = bench_bucket("gather_tile")
                b_ms = max(sel_total - a_ms, 0.0) + max(gta_total - c_ms, 0.0)
                b_detail = {"select_blocks_total_ms": sel_total,
                            "gather_tile_total_ms": gta_total}
            else:
                b_ms = 0.0

            # -- (d) の後半: gate + o_proj。sdpa の実出力と qkv の実 gate を
            #    使って qwen4_exp.py:964-965 (dense) / :909-910 (gather、
            #    同じ式) と同じ計算をもう一度評価するだけ (実装を書き換えたり
            #    差し替えたりはしない)。
            sdpa_outs = [r for (_, _, r) in sdpa_calls]
            if len(sdpa_outs) == 1:
                out_val = sdpa_outs[0]
            elif len(sdpa_outs) > 1:
                out_val = mx.concatenate(sdpa_outs, axis=2)
            else:
                out_val = None
            gate_val = calls["qkv"][0][2][3] if calls["qkv"] else None

            tail_ms = 0.0
            if out_val is not None and gate_val is not None:
                mx.eval(out_val, gate_val)
                B = chunk.shape[0]

                def tail():
                    o = out_val.transpose(0, 2, 1, 3).reshape(B, S, -1)
                    return [target_attn.o_proj(o * mx.sigmoid(gate_val))]

                tail_ms = PA.med_ms(tail, args.reps)
                mx.clear_cache()

            d_ms = qkv_ms + tail_ms

            whole_ms = 0.0
            if "args" in whole_box:
                wargs = whole_box["args"]

                def run_whole():
                    st = PA.snapshot([layer_cache])
                    r = orig_call(target_attn, *wargs)
                    mx.eval([r] + PA.pending([layer_cache]))
                    PA.restore([layer_cache], st)
                    return [r]

                whole_ms = PA.med_ms(run_whole, args.reps)
                mx.clear_cache()

        parts = a_ms + b_ms + c_ms + d_ms
        gap_pct = (parts - whole_ms) / whole_ms * 100 if whole_ms else 0.0

        comps = {"indexer": a_ms, "mask": b_ms, "sdpa": c_ms, "other": d_ms}
        return {
            "S": S, "kv": offset + S, "offset": offset, "branch": branch,
            "n_sdpa_calls": len(sdpa_calls),
            "chain_n": args.chain if chain_mode else 0,
            "chain_cache_strategy": (
                "snapshot_restore_between_each_call" if chain_mode else None),
            "indexer_ms": a_ms, "mask_ms": b_ms, "sdpa_ms": c_ms,
            "other_ms": d_ms, "qkv_ms": qkv_ms, "tail_ms": tail_ms,
            "b_detail": b_detail,
            "parts_sum_ms": parts, "whole_layer_ms": whole_ms,
            "gap_pct": gap_pct,
            "x12": {k: v * n_attn for k, v in comps.items()},
            "ms_per_tok_x12": {k: v * n_attn / S for k, v in comps.items()},
        }

    def advance(lo: int, hi: int) -> None:
        """[lo, hi) を `--chunk` 幅で本物の forward で進める (ラップ無し)。"""
        i = lo
        while i < hi:
            j = min(i + args.chunk, hi)
            mx.eval([model.model(ids[:, i:j], cache=cache)] + PA.pending(cache))
            mx.clear_cache()
            i = j

    def print_point(p: dict) -> None:
        chain = p["chain_n"] > 0
        unit = "us/回" if chain else "実測 ms"
        scale = 1000.0 if chain else 1.0  # ms -> us
        print(f"\n[kv={p['kv']} offset={p['offset']} S={p['S']}"
              f" 経路={p['branch']} sdpa呼び出し={p['n_sdpa_calls']}"
              + (f" chain=N={p['chain_n']} ({p['chain_cache_strategy']})"
                 if chain else ""),
              flush=True)
        print(f"  {'部品':28s}{unit:>10s}{'x12 ms':>10s}"
              f"{'x12 ms/tok':>12s}")
        label = {"indexer": "(a) indexer (pooled+top)",
                 "mask": "(b) mask/選択構築",
                 "sdpa": "(c) sdpa 本体",
                 "other": "(d) その他 (qkv+gate+o_proj)"}
        for k in ("indexer", "mask", "sdpa", "other"):
            ms = {"indexer": p["indexer_ms"], "mask": p["mask_ms"],
                  "sdpa": p["sdpa_ms"], "other": p["other_ms"]}[k]
            print(f"  {label[k]:28s}{ms * scale:10.3f}{p['x12'][k]:10.2f}"
                  f"{p['ms_per_tok_x12'][k]:12.4f}")
        print(f"  {'  うち qkv':28s}{p['qkv_ms'] * scale:10.3f}")
        print(f"  {'  うち gate+o_proj':28s}{p['tail_ms'] * scale:10.3f}")
        if p["branch"] == "dense":
            print(f"  (b) 内訳: QSAIndexer.__call__ 全体="
                  f"{p['b_detail'].get('idx_call_total_ms', 0.0) * scale:.3f}"
                  f"{unit[:2]}"
                  f" - indexer={p['indexer_ms'] * scale:.3f}{unit[:2]}"
                  f" + final_mask="
                  f"{p['b_detail'].get('final_mask_ms', 0.0) * scale:.3f}"
                  f"{unit[:2]}")
        elif p["branch"] == "gather":
            print(f"  (b) 内訳: select_blocks 全体="
                  f"{p['b_detail'].get('select_blocks_total_ms', 0.0) * scale:.3f}"
                  f"{unit[:2]}"
                  f" - indexer={p['indexer_ms'] * scale:.3f}{unit[:2]}"
                  f" + (_gather_tile_attn 全体="
                  f"{p['b_detail'].get('gather_tile_total_ms', 0.0) * scale:.3f}"
                  f"{unit[:2]}"
                  f" - sdpa={p['sdpa_ms'] * scale:.3f}{unit[:2]})")
        print(f"  {'部品和 (a+b+c+d)':28s}{p['parts_sum_ms'] * scale:10.3f}")
        print(f"  {'壁時計 (層 1 つ、実引数で再実行)':28s}"
              f"{p['whole_layer_ms'] * scale:10.3f}")
        print(f"  部品和 - 壁時計 = "
              f"{(p['parts_sum_ms'] - p['whole_layer_ms']) * scale:+.3f} {unit}"
              f" ({p['gap_pct']:+.1f}%)")
        if abs(p["gap_pct"]) > 10:
            print("  ** 数 % を超えてずれている。host 側の分岐 (_positions、"
                  "gather 可否判定、decode の causal split_mask 組み立てなど)"
                  " が部品に入っていない可能性がある **"
                  + ("" if chain else
                     " (decode 幅は mx.eval の同期コストが支配して部品和が"
                     " 大きく膨らむ既知の現象がある -- --chain を使うこと)"))
        elif abs(p["gap_pct"]) > 3:
            print("  (数 % のずれ。host 側の未計上分岐ぶんとして許容範囲)")
        print(f"  12層換算 合計 ms/tok (indexer+mask+sdpa+other) = "
              f"{sum(p['ms_per_tok_x12'].values()):.4f}", flush=True)

    current = 0
    results = []
    for kv in kvs:
        target_offset = kv - args.S
        advance(current, target_offset)
        current = target_offset
        point = measure_point(current)
        results.append(point)
        print_point(point)
        # 実本体を進めて確定させる (ラップ無しの本物の forward)
        advance(current, current + args.S)
        current += args.S

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "model": args.model, "ngram": args.ngram, "S": args.S,
            "kvs": kvs, "chunk": args.chunk, "reps": args.reps,
            "chain": args.chain, "chain_active": chain_mode,
            "target_layer_idx": target_layer_idx, "n_attn": n_attn,
            "indexer_budget": target_idx.token_budget,
            "compress_ratio": target_idx.compress_ratio,
        },
        "points": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n書いた: {out_path}", flush=True)

    # 計測ツールなので destructor 待ちでプロセスが Metal のメモリを握ったまま
    # 残る前例がある (他の tools/*.py と同じ理由) -- 即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
