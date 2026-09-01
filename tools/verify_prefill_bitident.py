"""prefill 経路の写し (_group_prefill_forward) が本家とビット一致するかの回帰ゲート。

spec_flash.py には Model.__call__ の写しが 3 つある (_staged_forward /
_group_prefill_forward / capture の gdn)。本家 (_vendor/qwen4_exp.py) を変えて
写しを変え忘れると、エラーにならず出力が静かにずれる。このゲートは
layer-major (group=4) と chunk-major (group=0) の 17k prefill を同一プロセスで
流し、logits 尾・hyper 尾・全キャッシュ配列のビット一致を確認する。

実行 (約 4 分、GPU 占有。他の計測やダウンロードと並走させない):

    .venv/bin/python tools/verify_prefill_bitident.py \
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep

一致しなければ exit 1。_staged_forward (decode 側の写し) はここでは検査
できないので、decode まで含めた確認は複数プロンプト x 512 の tok/step 平均で行う。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", required=True)
    ap.add_argument("--tokens", type=int, default=17000,
                    help="検査するプロンプト長 (group 経路を踏むには 6144 超が必要)")
    args = ap.parse_args()

    import mlxturbo  # noqa: F401
    from mlx_lm import load
    from mlxturbo.ngram_stream import install
    from mlxturbo import mtp_flash
    import mlxturbo.spec_flash as SF

    model, tok = load(os.path.expanduser(args.model))
    install(model, os.path.expanduser(args.ngram))
    mtp = mtp_flash.load_flash_mtp(
        os.path.join(os.path.expanduser(args.model), "mtp.safetensors"),
        model.args.text,
    )
    mx.eval(mtp.parameters())

    # 長さが安定するよう、繰り返しテキストで所定トークン数まで埋める
    base = "分散システムの結果整合性について、具体例を挙げて説明する。"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": base * (args.tokens // 8)}],
        add_generation_prompt=True,
    )[: args.tokens]
    ids = mx.array(ids)[None]
    eng = SF.FlashSpecEngine(model, mtp)

    def run(group):
        SF._PREFILL_GROUP = group
        caches = model.make_cache()
        mx.clear_cache()
        gen = eng.generate_stream(ids, 0, caches=caches)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            _, _, resume = e.value
        logits_tail, hyper_tail0, _ = resume
        states = []
        for c in caches:
            if hasattr(c, "keys") and hasattr(c, "offset"):
                states.append(c.keys[..., : c.offset, :])
                states.append(c.values[..., : c.offset, :])
                if hasattr(c, "indexer") and c.indexer.keys is not None:
                    states.append(c.indexer.keys)
            else:
                for k in range(4):
                    s = c[k]
                    if isinstance(s, mx.array):
                        states.append(s)
        mx.eval(logits_tail, hyper_tail0, *states)
        return logits_tail, hyper_tail0, states

    a = run(0)
    b = run(4)
    ok = (
        bool(mx.all(a[0] == b[0]))
        and bool(mx.all(a[1] == b[1]))
        and all(
            x.shape == y.shape and bool(mx.all(x == y))
            for x, y in zip(a[2], b[2])
        )
    )
    n = ids.shape[1]
    if ok:
        print(f"OK: group=0/4 bit-identical (n={n}, cache arrays={len(a[2])})")
        return 0
    print(f"FAIL: group=0/4 の出力が食い違う (n={n})。"
          " 本家と写し (spec_flash.py の docstring 参照) の差分を疑うこと。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
