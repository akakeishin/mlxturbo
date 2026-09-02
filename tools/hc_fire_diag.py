"""hyper-connection (HC) の融合 Metal カーネルが実モデルで何層発火するかを
数える。

背景: `mlxturbo/runner.py` の `enable_default_fusions` は既定
(`MLXTURBO_HC=kernel`) で `fused.enable_hyper_connection_kernel()` を呼ぶ
(本番経路そのもの)。しかし修正前は、`GatedResidual.block_inject_weight` が
QuantizedLinear に変換されず bf16 の `nn.Linear` のまま残っている層
(97 層中 96 層) で `fused.py` の `_pack_quantized` が None を返し、
`down`/`up` が量子化で問題なくても `GatedResidual.__call__` 全体が毎回
素の実装 (`orig`) に落ちていた。発火するのは inject の無い 1 層 (mixer) だけ。

`kernels/hyper_connection.py` に非量子化 bf16 inject を直接読む分岐を足し、
`fused.py` の `_pack_inject_bf16` がそれを渡すようにしたことで、97 層
全部が融合カーネル (`fused_gated_residual`) を通るはず。

数え方: `Q.GatedResidual.__call__` は `enable_hyper_connection_kernel` が
`patched` に差し替え済み (build_runner 経由)。それをさらに薄くラップして
「呼ばれた総回数」(= モデル内の GatedResidual インスタンス数、97 のはず) を
数え、`mlxturbo.kernels._fire` の `hc_kernel` カウンタ (`fused_gated_residual`
が実際に融合カーネルを叩いた回数) と突き合わせる。差 (total - fused) が
素の実装へ落ちた回数。

実行 (GPU、モデル読み込みに 1〜2 分):
    uv run python scratchpad/hc_fire_diag.py
    uv run python scratchpad/hc_fire_diag.py --model ~/models/ddalcu-mlxlm

他の GPU 計測 (qsa_prefill_split / decode_ab / quant_eval /
sdpa_split_inmodel / hc_*) と並走させないこと (CLAUDE.md の計測の作法)。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
for p in (REPO_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram-sep")
    args = ap.parse_args()

    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        print("Metal not available; this diagnostic is GPU-only.")
        return 1

    from decode_ab import prefill_once
    from mlxturbo.kernels import _fire
    from mlxturbo.spec_flash import capture
    from verify_width_cost import build_pair, build_prompt_ids, build_runner

    runner_args = argparse.Namespace(
        model=args.model, ngram=args.ngram, mtp=None, mtp_bits=4,
    )
    eng, model, tok, eos_ids = build_runner(runner_args)  # enable_default_fusions 込み

    # `mlx_lm.models.qwen4_exp` は mlxturbo が import された時点 (build_runner
    # の中) でベンダー版 (mlxturbo/_vendor/qwen4_exp.py) に差し替わるので、
    # build_runner の後で import すること
    import mlx_lm.models.qwen4_exp as Q

    # build_runner の時点で GatedResidual.__call__ はすでに融合カーネル用の
    # patched に差し替わっている。それをさらに包んで「呼ばれた総回数」を数える。
    patched = Q.GatedResidual.__call__
    total_calls = {"n": 0}

    def counting_call(self, hyper):
        total_calls["n"] += 1
        return patched(self, hyper)

    Q.GatedResidual.__call__ = counting_call

    ids = build_prompt_ids(tok, 0)  # 短プロンプト
    caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
    pair, _cur = build_pair(eng, resume, 1)  # S=1 decode

    total_calls["n"] = 0
    _fire.reset()
    from decode_ab import _restore

    _restore(caches, snap)
    with capture(model) as _cap:
        lg = model(pair, cache=caches)
    mx.eval(lg)

    Q.GatedResidual.__call__ = patched  # 後始末

    fired = _fire.snapshot()
    fused_call = fired.get("hc_kernel", 0)
    total = total_calls["n"]
    orig_call = total - fused_call

    print(f"GatedResidual 呼び出し総数: {total}")
    print(f"  fused_call (融合カーネル発火) = {fused_call}")
    print(f"  orig_call  (素の実装に落ちた) = {orig_call}")
    print(f"  _fire snapshot 全体: {fired}")

    # os._exit は atexit/バッファの自動 flush を素通りするので、呼ぶ前に
    # 明示的に flush する (これを忘れて最初の実行で出力が消えた)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # モデル読み込み後の後片付けを待たない


if __name__ == "__main__":
    raise SystemExit(main())
