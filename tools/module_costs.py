"""実フォワードの実入力で、モジュール単体の時間を測る。

ablate.py の積み上げ無効化は、外した部品の下流が壊れる (hidden が退化して
エキスパートアクセスが偏る等) ので、部品のコストを正しく配分できない。
ここでは 1 回の本物のフォワードで各モジュールの入力をフックで捕まえ、
その実テンソルを使って単体を反復計測する。ルーティングも本物のまま。

単体の総和が全体より小さければ、その差がモジュール間の重なり (パイプライン) と
ここで測れない結合部の時間。

    uv run python tools/module_costs.py --model <path> --ngram <sidecar> --width 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def med_ms(fn, reps=30):
    import mlx.core as mx

    for _ in range(5):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1000)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--width", type=int, default=3)
    args = ap.parse_args()

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx

    import mlxturbo  # noqa: F401
    from mlx_lm import load

    import mlx_lm.models.qwen4_exp as Q

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて詳しく説明してください。"}],
        add_generation_prompt=True,
    )

    # プロンプトを流し、実分布の decode 入力を作る
    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    W = args.width
    chunk = mx.array([[cur] * W])

    # 各モジュールの実入力をフックで捕まえる (1 回の実フォワード)
    grabbed = {"moe": [], "gdn": [], "hc": [], "attn": []}
    omoe, ogdn, ohc, oattn = (Q.SparseMoeBlock.__call__, Q.GatedDeltaNet.__call__,
                              Q.GatedResidual.__call__, Q.Attention.__call__)

    def gmoe(self, x):
        grabbed["moe"].append((self, x))
        return omoe(self, x)

    def ggdn(self, x, mask, cache_):
        grabbed["gdn"].append((self, x, mask, cache_))
        return ogdn(self, x, mask, cache_)

    def ghc(self, hyper):
        grabbed["hc"].append((self, hyper))
        return ohc(self, hyper)

    def gattn(self, x, rope, mask, cache_, idx_cache):
        grabbed["attn"].append((self, x, rope, mask, cache_, idx_cache))
        return oattn(self, x, rope, mask, cache_, idx_cache)

    Q.SparseMoeBlock.__call__ = gmoe
    Q.GatedDeltaNet.__call__ = ggdn
    Q.GatedResidual.__call__ = ghc
    Q.Attention.__call__ = gattn
    out = model(chunk, cache=cache)
    mx.eval(out)
    Q.SparseMoeBlock.__call__ = omoe
    Q.GatedDeltaNet.__call__ = ogdn
    Q.GatedResidual.__call__ = ohc
    Q.Attention.__call__ = oattn
    for k, v in grabbed.items():
        for tup in v:
            mx.eval([a for a in tup[1:] if isinstance(a, mx.array)])

    print(f"T={W}  捕まえた: " + " ".join(f"{k}={len(v)}" for k, v in grabbed.items()))

    # 単体計測。キャッシュを壊す呼び出し (gdn/attn) は状態を退避して戻す
    def bench_moe():
        return [omoe(self, x) for self, x in grabbed["moe"]]

    def bench_hc():
        outs = []
        for self, hyper in grabbed["hc"]:
            o = ohc(self, hyper)
            outs.append(o[0] if isinstance(o, tuple) else o)
        return outs

    moe = med_ms(bench_moe)
    hc = med_ms(bench_hc)

    # GDN: cache を触らないように、状態を複製した一時 cache で呼ぶのは大掛かり
    # なので、in_proj + conv + カーネル + out_proj を cache なしで流す
    # (再帰状態ゼロ始まり。読み出し量と起動回数は同じ)
    def bench_gdn():
        return [ogdn(self, x, None, None) for self, x, m, c in grabbed["gdn"]]

    gdn = med_ms(bench_gdn)

    def bench_attn():
        outs = []
        for self, x, rope, mask, cache_, idx_cache in grabbed["attn"]:
            # cache 無しで q/k/v + sdpa だけ (KV 追記なし、マスクなし)
            B, S, _ = x.shape
            q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, S, -1)
            q = self.q_norm(q).transpose(0, 2, 1, 3)
            k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            o = Q.scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=None)
            outs.append(self.o_proj((o.transpose(0, 2, 1, 3).reshape(B, S, -1)) * mx.sigmoid(gate)))
        return outs

    attn = med_ms(bench_attn)

    def bench_lm_head():
        return model.lm_head(grabbed["hc"][0][1][..., : model.args.text_config.hidden_size]
                             if False else mx.zeros((1, W, 2560), dtype=mx.bfloat16))

    lm = med_ms(bench_lm_head)

    whole = med_ms(lambda: model(chunk, cache=cache))

    print(f"MoE 48 層     {moe:6.2f} ms")
    print(f"HC 97 回      {hc:6.2f} ms")
    print(f"GDN 36 層     {gdn:6.2f} ms")
    print(f"attn 12 層    {attn:6.2f} ms")
    print(f"lm_head       {lm:6.2f} ms")
    print(f"単体の総和    {moe + hc + gdn + attn + lm:6.2f} ms")
    print(f"全体 (実測)   {whole:6.2f} ms")


if __name__ == "__main__":
    main()
