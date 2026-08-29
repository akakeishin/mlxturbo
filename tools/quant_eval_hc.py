"""bench/quant_eval.py compare を融合カーネルの有無で回す。

bench/ は触らない方針なので、モジュールとして読み込んでから
`GatedResidual` を差し替える。

**必ず tools/biglock.sh 経由で呼ぶこと。**98GB のモデルを読むので、親
セッションのジョブと同時に走ると先に走っていた方がメモリ圧で落ちる。
scratchpad に置くと biglock の「ロック無しの相手」検出
(`\\.venv/bin/python3? (tools|bench)/`) に引っかからず、親から見えない
ジョブになる。だから tools/ に置いてある。

    tools/biglock.sh uv run python tools/quant_eval_hc.py hc-base
    tools/biglock.sh uv run python tools/quant_eval_hc.py hc-fused --fused
    tools/biglock.sh uv run python tools/quant_eval_hc.py hc-sig32 --sig32

`--sig32` は融合ではなく**対照**。素と op 単位で同じで sigmoid だけ fp32 に
した、素より*正確*な版。融合版の KLD の動きが実装の劣化なのか bf16 の
丸め順の違いによる散らばりなのかは、これと比べないと決まらない。
"""

import os
import sys

# ここで先に立てること。enable_hyper_connection_kernel() は
# mlx_lm.models.qwen4_exp を import し、その arch は import 時にこの旗を読む。
# 後から立てても遅く、n-gram を本体持ちとして組んでしまい load が落ちる
os.environ["FASTMLX_NGRAM_DISK"] = "1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "bench"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
os.chdir(REPO_ROOT)

tag = sys.argv[1]
use_fused = "--fused" in sys.argv
use_sig32 = "--sig32" in sys.argv

import quant_eval  # noqa: E402

if use_fused:
    from mlxturbo import fused

    fused.enable_hyper_connection_kernel()
    print("hyper-connections を融合カーネルに差し替えた", flush=True)
elif use_sig32:
    import mlx_lm.models.qwen4_exp as Q
    from hc_equiv_test import _install_sigmoid32

    _install_sigmoid32(Q)
    print("対照: sigmoid だけ fp32 (素より正確) に差し替えた", flush=True)
else:
    print("融合なし (基準)", flush=True)

sys.argv = [
    "quant_eval.py", "compare",
    "--model", os.path.expanduser("~/models/qwen38fn-mlx-v-fast6"),
    "--ngram", os.path.expanduser("~/models/qwen38fn-ngram-4bit"),
    "--continuations", "bench/results/qe-cont.json",
    "--ref-dump", "bench/results/qe-ref-bf16.npz",
    "--tag", tag,
]
quant_eval.main()
