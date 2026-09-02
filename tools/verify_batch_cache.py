"""`mlxturbo.batch` の配線を、合成した小さい Flash-Next で確かめる。

91GB の実モデルは要らないし、GPU も使わない (既定で CPU)。ここで見るのは
数値の質ではなく、キャッシュの持ち回りが正しいかどうか:

    merge   1 本ずつのキャッシュを束ねる
    filter  走行中に 1 本抜く
    extract 終わった 1 本を取り出す
    extend  走行中に 1 本足す

判定基準 (測る前に決めたもの):

    貪欲デコード 24 トークンで、1 本ずつ逐次生成した列と `BatchGenerator` で
    流した列が**完全一致**すること。prefill の割り方は両者で揃える
    (`BatchGenerator` はプロンプトを (n-1, 1) に割り、前半を
    `prefill_step_size` 刻みで流す。QSA は kv_len に依存するので、割り方を
    揃えないと比較そのものが成立しない)。

    ずれたときに丸め差か破損かを分ける基準: 左パディングの無い系列で実測
    した |logit 差| は 1.5e-7 (float32 の丸め)。破損はこれより 5 桁大きい。

ケース:

    A  QSA 不活性 (budget > 全プロンプト長)。キャッシュ・マスク・rope・
       再帰状態・4 操作を全部通る。一致する。
    B  QSA 活性、プロンプト同長 (左パディング無し)。一致する。
    C  QSA 活性、プロンプト長が不揃い。B=1 と B=2 は一致するが、B=4 の
       左パディングされた系列は一致しない (mlxturbo/batch.py の「残っている
       制限」参照)。既知の不一致として扱う。

使い方:

    .venv/bin/python tools/verify_batch_cache.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# 実モデルの比率をなるべく保った縮小版。linear_*_head_dim は
# gated_delta の Metal カーネルが Dk % 32 == 0 を要求するので 32 で止める
TINY = dict(
    model_type="qwen4_exp_text",
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=512,
    rms_norm_eps=1e-6,
    full_attention_interval=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    hc_count=2,
    hc_lowrank=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=2,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1024,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=4,
    ple_embed_dim=32,
    ple_layer_ids=[2],
    ple_conv_kernel_size=4,
    seed=0,
    eos_token_id=500,
    partial_rotary_factor=0.25,
    rope_theta=10000.0,
    tie_word_embeddings=False,
)

N_GEN = 24


def build(budget: int):
    from mlx.utils import tree_map
    from mlx_lm.models import qwen4_exp as Q

    cfg = dict(TINY, indexer_budget=budget)
    mx.random.seed(0)
    model = Q.Model(Q.ModelArgs(model_type="qwen4_exp", text_config=cfg))
    # nn.Module の既定初期化は形ごとに違うので、明示的に同じ分布で置き直す
    model.update(
        tree_map(
            lambda a: mx.random.normal(a.shape) * 0.05 if a.dtype == mx.float32 else a,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    model.eval()
    return model


def seq_generate(model, prompt, chunk, n=N_GEN):
    """`BatchGenerator` と同じ割り方で 1 本だけ流す (基準)。"""
    cache = model.make_cache()
    body = prompt[:-1]
    for lo in range(0, len(body), chunk):
        model(mx.array(body[lo : lo + chunk])[None], cache=cache)
    logits = model(mx.array(prompt[-1:])[None], cache=cache)
    out = []
    cur = int(mx.argmax(logits[0, -1]))
    for _ in range(n):
        out.append(cur)
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1]))
    return out


def batch_generate(model, prompts, chunk, max_toks=None, **kw):
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(
        model, max_tokens=N_GEN, stop_tokens=[], prefill_step_size=chunk, **kw
    )
    uids = gen.insert(
        [list(p) for p in prompts], max_toks or [N_GEN] * len(prompts)
    )
    res = {u: [] for u in uids}
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != "stop":
                res[r.uid].append(r.token)
    gen.close()
    return [res[u] for u in uids]


def report(label, got, refs):
    ok = True
    for i, (g, r) in enumerate(zip(got, refs)):
        pre = 0
        for a, b in zip(g, r):
            if a != b:
                break
            pre += 1
        if pre < len(r):
            ok = False
            print(f"  NG {label} seq{i}: 先頭一致 {pre}/{len(r)}")
            print(f"       batch={g[:12]}")
            print(f"       solo ={r[:12]}")
        else:
            print(f"  OK {label} seq{i}: {pre}/{len(r)}")
    return ok


def run_case(name, budget, prompts, expect_pass=True):
    print(f"\n### {name}  (indexer_budget={budget})")
    chunk = budget
    model = build(budget)
    refs = [seq_generate(model, p, chunk) for p in prompts]

    ok = True
    ok &= report("B=1", batch_generate(model, prompts[:1], chunk), refs[:1])
    ok &= report("B=2", batch_generate(model, prompts[:2], chunk), refs[:2])
    ok &= report("B=4", batch_generate(model, prompts, chunk), refs)

    # 走行中に 1 本抜く: seq1 を 6 トークンで打ち切る
    got = batch_generate(model, prompts, chunk, max_toks=[N_GEN, 6, N_GEN, N_GEN])
    ok &= report(
        "filter", got, [refs[0], refs[1][:6], refs[2], refs[3]]
    )

    # 走行中に足す: 先に 2 本を走らせ、あとの 2 本が合流する
    got = batch_generate(
        model, prompts, chunk, completion_batch_size=2, prefill_batch_size=2
    )
    ok &= report("extend", got, refs)

    tag = "想定どおり" if ok == expect_pass else "★想定と違う★"
    print(f"  => {'一致' if ok else '不一致'} ({tag})")
    return ok == expect_pass


def run_coordinator_join_case() -> bool:
    """`mlxturbo.batch.BatchCoordinator` (レーン 5 のスケジューラ、
    docs/research/LANES-2026-09.md 「レーン 5」) の途中参加。

    `run_case` の A/B/C は下の層 (`BatchGenerator`/`BatchAttnCache` の
    merge/filter/extract/extend) がキャッシュを正しく持ち回れるかを見ている。
    ここで見るのはその 1 段上 -- `BatchCoordinator._drive()` が、走行中の
    admission (`A`) の途中で inbox に届いた新しい admission (`B`) を実際に
    `gen.insert()` へ届けているか。届けそこねても `run_case` はそもそも
    B を投入しない (`BatchGenerator.insert` を直接呼ぶだけ) ので、この段の
    欠陥はここでしか見えない。

    A を 3 トークン (3 ラウンド) 出させてから B を投入し、両方とも単独参照
    (`seq_generate`) と完全一致することを見る。budget を大きくとって
    (32) 2 行を同居させても物理列数が budget を超えない (QSA 不活性) 組み
    合わせを選ぶ -- QSA 活性下の不一致は `run_case` のケース C で別に
    確認済みの話で、ここで割ると「途中参加の配線」と「QSA の既知の制限」の
    どちらが原因か切り分けられなくなる。
    """
    import concurrent.futures
    import threading

    from mlxturbo import batch as fb

    budget = 32
    chunk = budget
    model = build(budget)

    prompt_a, max_tokens_a = [1, 2, 3], 10   # 3+10=13 <= 32 -> pool
    prompt_b, max_tokens_b = [20, 21], 6     # 2+6=8   <= 32 -> pool
    # 同居 (B-3): longest_prompt=3, longest_tokens=10, 和 13 <= 32 -> 入る
    ref_a = seq_generate(model, prompt_a, chunk, n=max_tokens_a)
    ref_b = seq_generate(model, prompt_b, chunk, n=max_tokens_b)

    def submit(coord, prompt, max_tokens, on_tokens=None):
        fut: "concurrent.futures.Future" = concurrent.futures.Future()
        adm = fb.Admission(
            prompt_ids=list(prompt),
            max_tokens=max_tokens,
            sampler=None,
            logits_processors=[],
            tier="pool",
            on_tokens=on_tokens,
            on_done=None,
            cancel_event=None,
            future=fut,
        )
        coord.submit(adm)
        return adm, fut

    print("\n### coordinator join: A が 3 トークン出してから B を投入")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        coord = fb.BatchCoordinator(
            model, executor, max_batch=4, prefill_step_size=chunk, eos_ids=[]
        )
        reached = threading.Event()
        count_a = [0]  # 別変数で数える。adm_a 自体は下の代入が終わるまで
        # 名前として存在しない -- on_tokens はその前に (submit() 内の
        # coord.submit() が駆動スレッドを起こした直後に) 呼ばれうるので、
        # ここで adm_a.tokens を直接参照すると NameError の競合になる

        def on_a(toks):
            count_a[0] += len(toks)
            if count_a[0] >= 3:
                reached.set()

        adm_a, fut_a = submit(coord, prompt_a, max_tokens_a, on_a)
        if not reached.wait(timeout=30):
            raise TimeoutError("A が 3 トークン出す前にタイムアウトした")
        # まだ A は生成中 (max_tokens_a=10 > 3) のうちに B を投げる --
        # 走行中バッチへの正真正銘の途中参加になる
        adm_b, fut_b = submit(coord, prompt_b, max_tokens_b)
        fut_a.result(timeout=30)
        fut_b.result(timeout=30)
    finally:
        executor.shutdown(wait=True)

    ok = True
    same_a = adm_a.tokens == ref_a
    same_b = adm_b.tokens == ref_b
    ok &= same_a and same_b
    print(f"  {'OK' if same_a else 'NG'} A (先行, 途中参加を経ても不変): "
          f"got={adm_a.tokens} ref={ref_a}")
    print(f"  {'OK' if same_b else 'NG'} B (途中参加): "
          f"got={adm_b.tokens} ref={ref_b}")
    print(f"  => {'一致' if ok else '不一致'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="CPU ではなく GPU で回す")
    args = ap.parse_args()

    if not args.gpu:
        mx.set_default_device(mx.cpu)
        # BatchGenerator は Metal があると wired limit を触りに行く
        mx.metal.is_available = lambda: False

    from mlxturbo import batch as fb

    fb.enable_batch_cache()

    equal = [
        list(range(1, 22)), list(range(30, 51)),
        list(range(60, 81)), list(range(90, 111)),
    ]                                                    # 全部 21
    unequal = [
        list(range(1, 21)), list(range(21, 41)),
        list(range(41, 51)), list(range(60, 95)),
    ]                                                    # 20 / 20 / 10 / 35

    results = [
        run_case("A. QSA 不活性・長さ不揃い", 64, unequal),
        run_case("B. QSA 活性・同長", 8, equal),
        run_case("C. QSA 活性・長さ不揃い (既知の不一致)", 8, unequal,
                 expect_pass=False),
        run_coordinator_join_case(),
    ]
    print("\n=== 全ケース想定どおり ===" if all(results)
          else "\n=== 想定と違うケースあり ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
