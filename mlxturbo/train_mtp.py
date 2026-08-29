"""MTP ヘッドの FastMTP 式微調整 (D2 手順 3)。

- backbone は凍結。pre-norm hidden は SpecEngine._hidden_forward と同じ経路
  (推論と同一の hidden 契約) から取る。
- 同一ヘッドを深度 1..K に position-shared で再帰適用。深度間は勾配を通す。
- 位置減衰 CE: w_k ∝ decay^(k-1) を正規化。損失は生成領域
  (target が gen 側にある位置) のみ。
- AdamW、linear warmup → cosine decay、マイクロバッチ 1 系列 × 勾配累積。

成果物は models/mtp-tuned/ に safetensors + meta.json。保存重みは norm の
+1 シフト適用済み (mlx 規約) なので、読み戻しは load_mtp_file を使うこと。
load_mtp (原本チェックポイント用) で読むと二重シフトになる。

使い方 (GPU 長時間。カーネル側セッションの計測と排他を確認してから):
  uv run python -m fastmlx.train_mtp --data data/mtp_selfgen.jsonl \
      --limit 1000 --tag ckpt1k
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from ._mlx_compat import KVCache, TextModelArgs, mlx_lm_load
from .mtp import MTPModule, find_snapshot, load_mtp, load_mtp_file
from .spec import SpecEngine

REPO = Path(__file__).resolve().parent.parent


def load_dataset(path, max_len, limit, seed):
    records = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = rec["prompt_tokens"] + rec["gen_tokens"]
            plen = len(rec["prompt_tokens"])
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            # 生成領域が薄い系列は学習信号がないので捨てる
            if len(tokens) - plen < 16:
                continue
            records.append((tokens, plen))
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit:
        records = records[:limit]
    return records


def backbone_hiddens(engine, tokens):
    """推論と同じ経路で pre-norm hidden を得る (勾配なし)。"""
    caches = engine.text.make_cache()
    h, _ = engine._hidden_forward(mx.array(tokens), caches, capture=False)
    return mx.stop_gradient(h)


def make_head_fn(model):
    text = model.language_model
    if text.args.tie_word_embeddings:
        embedding = text.model.embed_tokens
        return lambda x: embedding.as_linear(x)
    return lambda x: text.lm_head(x)


def mtp_loss(mtp, head_fn, embed_fn, tokens, plen, h_true, k_depth, weights):
    """深度 1..K の位置減衰 CE。戻り値: (loss, 深度別 (correct, count))。"""
    L = len(tokens)
    tok = mx.array(tokens)
    losses = []
    stats = []
    h_in = h_true[:, : L - 1]
    for k in range(1, k_depth + 1):
        # 深度 k のスロット i は (embed(T[i+k]), h_{k-1,i]) から T[i+k+1] を予測
        n_slots = L - k
        e = embed_fn(tok[k:][None])
        out = mtp(e, h_in, cache=None)
        logits = head_fn(mtp.norm(out[:, : n_slots - 1]))
        targets = tok[k + 1 :]
        # 生成領域のみ: target の系列位置 i+k+1 >= plen
        pos = mx.arange(n_slots - 1) + k + 1
        mask = pos >= plen
        n_valid = mask.sum()
        ce = nn.losses.cross_entropy(
            logits[0].astype(mx.float32), targets, reduction="none"
        )
        ce = mx.where(mask, ce, 0.0)
        losses.append(weights[k - 1] * ce.sum() / mx.maximum(n_valid, 1))
        pred = mx.argmax(logits[0], axis=-1)
        correct = mx.where(mask, pred == targets, False).sum()
        stats.append((correct, n_valid))
        h_in = out[:, : n_slots - 1]
    return sum(losses), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit"
    )
    parser.add_argument("--original", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--data", default="data/mtp_selfgen.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="使う系列数 (0=全部)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64, help="勾配累積数")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--k-depth", type=int, default=3)
    parser.add_argument("--decay", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=20, help="warmup 更新数")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--out", default="models/mtp-tuned")
    parser.add_argument("--tag", default="run")
    parser.add_argument(
        "--init", default="", help="学習済みヘッドから再開する場合の safetensors"
    )
    parser.add_argument("--save-every", type=int, default=50, help="更新数間隔")
    args = parser.parse_args()

    records = load_dataset(
        REPO / args.data, args.max_len, args.limit, args.seed
    )
    if not records:
        raise SystemExit(f"no usable records in {args.data}")
    n_updates = args.epochs * math.ceil(len(records) / args.batch_size)
    print(f"{len(records)} sequences, {n_updates} updates planned")

    model, _ = mlx_lm_load(args.model)
    text_args = TextModelArgs.from_dict(model.args.text_config)
    if args.init:
        mtp = load_mtp_file(REPO / args.init, text_args)
    else:
        mtp = load_mtp(find_snapshot(args.original), text_args)
    # 学習は fp32 で行い、保存時に bf16 へ戻す
    mtp.update(tree_map(lambda p: p.astype(mx.float32), mtp.parameters()))
    mtp.train()
    engine = SpecEngine(model, mtp)
    head_fn = make_head_fn(model)
    embed_fn = model.language_model.model.embed_tokens

    w = [args.decay**i for i in range(args.k_depth)]
    weights = [x / sum(w) for x in w]

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, args.lr, args.warmup),
            optim.cosine_decay(args.lr, max(n_updates - args.warmup, 1)),
        ],
        [args.warmup],
    )
    opt = optim.AdamW(learning_rate=schedule)

    def loss_fn(mtp_, tokens, plen, h_true):
        loss, stats = mtp_loss(
            mtp_, head_fn, embed_fn, tokens, plen, h_true,
            args.k_depth, weights,
        )
        return loss, stats

    vag = nn.value_and_grad(mtp, loss_fn)

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(tag_suffix, seen, update, loss_val):
        params = dict(tree_flatten(mtp.parameters()))
        params = {k: v.astype(mx.bfloat16) for k, v in params.items()}
        path = out_dir / f"mtp-{args.tag}-{tag_suffix}.safetensors"
        mx.save_safetensors(str(path), params)
        meta = {
            "norm_shift_applied": True,
            "loader": "fastmlx.mtp.load_mtp_file",
            "base_model": args.model,
            "original": args.original,
            "data": args.data,
            "sequences": len(records),
            "epochs": args.epochs,
            "k_depth": args.k_depth,
            "decay": args.decay,
            "lr_peak": args.lr,
            "batch_size": args.batch_size,
            "seen_sequences": seen,
            "update": update,
            "last_loss": loss_val,
        }
        (out_dir / f"mtp-{args.tag}-{tag_suffix}.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        print(f"saved {path}", flush=True)

    rng = random.Random(args.seed + 1)
    update = 0
    seen = 0
    t0 = time.time()
    accum = None
    accum_n = 0
    loss_ema = None
    acc_tot = [[0, 0] for _ in range(args.k_depth)]
    for epoch in range(args.epochs):
        order = list(range(len(records)))
        rng.shuffle(order)
        for idx in order:
            tokens, plen = records[idx]
            h_true = backbone_hiddens(engine, tokens)
            (loss, stats), grads = vag(mtp, tokens, plen, h_true)
            accum = (
                grads
                if accum is None
                else tree_map(mx.add, accum, grads)
            )
            accum_n += 1
            seen += 1
            mx.eval(loss, accum)
            lv = float(loss.item())
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            for k, (c, n) in enumerate(stats):
                acc_tot[k][0] += int(c.item())
                acc_tot[k][1] += int(n.item())
            if accum_n == args.batch_size:
                mean_grads = tree_map(
                    lambda g: g / args.batch_size, accum
                )
                opt.update(mtp, mean_grads)
                mx.eval(mtp.parameters(), opt.state)
                accum, accum_n = None, 0
                update += 1
                accs = "/".join(
                    f"{c / max(n, 1):.3f}" for c, n in acc_tot
                )
                rate = seen / (time.time() - t0)
                print(
                    f"ep{epoch} upd{update}/{n_updates} seen{seen}"
                    f" loss_ema {loss_ema:.4f} acc(tf) {accs}"
                    f" {rate:.2f} seq/s",
                    flush=True,
                )
                acc_tot = [[0, 0] for _ in range(args.k_depth)]
                if update % args.save_every == 0:
                    save(f"u{update}", seen, update, loss_ema)
    if accum is not None and accum_n > 0:
        mean_grads = tree_map(lambda g: g / accum_n, accum)
        opt.update(mtp, mean_grads)
        mx.eval(mtp.parameters(), opt.state)
        update += 1
    save("final", seen, update, loss_ema)


if __name__ == "__main__":
    main()
