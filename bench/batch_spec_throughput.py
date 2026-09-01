"""バッチ x 投機 (BatchSpecGenerator) のスループットを測る。

## 判定基準 (測る前に宣言、既存の宣言を引き継ぐ)

`docs/research/KERNEL-BRIEF-DECODE-BW.md` の「バッチ計測の判定プロトコル」と
`docs/research/IMPROVEMENT-QUEUE.md` B5 の反転条件そのまま:

    **B=4 で 1.6 倍未満ならレーンを畳む。**紙モデルの予測は 1.95 倍
    (共有タイルの項は 2026-09-01 の実測で外した)。

土俵は短〜中尺。プロンプトは全部 1k 未満に収める (この判定基準を宣言した
ときの土俵をそのまま使う)。**2026-09-02 に QSA の長さ制限が外れた**ので
17k 級もバッチに入るようになったが、そちらは別の土俵として測り直すこと
(mlxturbo/batch_spec.py の `_ragged_indexer_call`)。

比較の取り方:

    solo   同じプロンプトを 1 本ずつ engine.generate_stream で回した合計時間
    batch  BatchSpecGenerator で B 行同時に回した時間

どちらも同じ生成長。1 プロセス内で solo -> batch -> batch -> solo の回文順に
回して熱ドリフトを相殺する。**最初の 1 本は温めなので捨てる。**

    tools/biglock.sh .venv/bin/python bench/batch_spec_throughput.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPTS = [
    "分散システムにおける結果整合性について、具体例を挙げて説明してください。",
    "Explain why speculative decoding helps when decoding is dispatch bound.",
    "Python でリストの重複を順序を保って除去する関数を書いてください。",
    "機械学習の過学習とは何か、対策とあわせて説明してください。",
    "Describe the steps to set up a Python virtual environment.",
    "秋の京都を旅行する人向けに、二泊三日の案を書いてください。",
    "Write a Python function that flattens a nested list.",
    "SQL のインデックスが効かない典型的なケースを挙げてください。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    from mlxturbo import mtp_flash, spec_flash
    from mlxturbo.batch_spec import BatchSpecGenerator
    from mlxturbo.runner import enable_default_fusions

    path = os.path.expanduser(args.model)
    model, tok = load(path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[batch-tp]")
    mtp_path = args.mtp or os.path.join(path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path),
                                   model.args.text, quantize=q)
    mx.eval(mtp.parameters())
    eng = spec_flash.FlashSpecEngine(model, mtp)

    ids = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                add_generation_prompt=True)
        for p in PROMPTS
    ]
    N = args.tokens

    def run_solo(rows):
        t0 = time.perf_counter()
        n = 0
        for p in rows:
            gen = eng.generate_stream(mx.array(p)[None], N)
            try:
                while True:
                    n += len(next(gen))
            except StopIteration:
                pass
        return time.perf_counter() - t0, n

    def run_batch(rows):
        t0 = time.perf_counter()
        g = BatchSpecGenerator(eng, [list(p) for p in rows])
        out = g.generate(N)
        return time.perf_counter() - t0, sum(len(o) for o in out)

    print("判定基準はモジュール docstring のとおり (測る前に宣言済み)。")
    print(f"生成長 {N} トークン。最初の 1 本は温めなので捨てる。\n")
    run_solo(ids[:1])  # 温め

    rows_out = []
    for B in [int(x) for x in args.batches.split(",")]:
        rows = ids[:B]
        s1, ns1 = run_solo(rows)
        b1, nb1 = run_batch(rows)
        b2, _ = run_batch(rows)
        s2, _ = run_solo(rows)
        solo = (s1 + s2) / 2
        batch = (b1 + b2) / 2
        tps_solo = ns1 / solo
        tps_batch = nb1 / batch
        rows_out.append(dict(B=B, solo_s=solo, batch_s=batch,
                             solo_tps=tps_solo, batch_tps=tps_batch,
                             speedup=tps_batch / tps_solo))
        print(f"  B={B}: solo {tps_solo:6.1f} tok/s  batch {tps_batch:6.1f} tok/s"
              f"  -> {tps_batch / tps_solo:.2f}x", flush=True)

    print("\n=== 判定 ===")
    ok = True
    for r in rows_out:
        if r["B"] == 4:
            ok = r["speedup"] >= 1.6
            print(f"  B=4 で {r['speedup']:.2f}x "
                  f"({'合格' if ok else '不合格 -- 反転条件 1.6x 未満、レーンを畳む'})")
    if args.out:
        Path(args.out).write_text(json.dumps(rows_out, ensure_ascii=False, indent=1))
        print(f"書き出し: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
