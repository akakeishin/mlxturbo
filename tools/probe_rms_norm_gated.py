"""実モデルの RMSNormGated 呼び出しを捕まえて、素とカーネルを直接突き合わせる。

合成テストではビット一致したのに、実モデルで logits が動いた。呼び出し時の
dtype / shape / 連続性のどれかが合成テストと違うはず。推測せずに見る。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    import mlx_lm.models.qwen4_exp as Q

    from fastmlx.kernels import rms_norm_gated as rng

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install

        install(model, args.ngram)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True,
    )

    orig = Q.RMSNormGated.__call__
    from fastmlx.kernels import rms_norm_gated as _rng

    bad = []
    total = [0]

    def spy(self, x, gate=None):
        """**その場で**素とカーネルを突き合わせる。

        捕まえてから後で eval して比べると、eval がテンソルを連続化して
        しまい「非連続なビューが来たときのバグ」を隠す。全呼び出しを見る。
        """
        out = orig(self, x, gate)
        total[0] += 1
        if _rng.eligible(x, self.weight, gate):
            k = _rng.rms_norm_gated(x, self.weight, gate, self.eps, self.activation)
            mx.eval(out, k)
            a = np.array(out.astype(mx.float32))
            b = np.array(k.astype(mx.float32))
            nz = int((a != b).sum())
            if nz:
                bad.append((total[0], x.shape, x.dtype, nz, a.size,
                            float(np.abs(a - b).max())))
        else:
            bad.append((total[0], x.shape, x.dtype, -1, 0, 0.0))
        return out

    Q.RMSNormGated.__call__ = spy
    cache = model.make_cache()
    model(mx.array(ids)[None], cache=cache)      # プレフィル
    logits = model(mx.array([[int(0)]]), cache=cache)  # デコード 1 歩
    mx.eval(logits)
    Q.RMSNormGated.__call__ = orig

    print(f"呼び出し {total[0]} 回、食い違い {len(bad)} 件")
    for n, shape, dt, nz, size, mx_d in bad[:10]:
        if nz < 0:
            print(f"  呼び出し {n}: eligible=False  shape={shape} dtype={dt}")
        else:
            print(f"  呼び出し {n}: 不一致 {nz}/{size}  最大差 {mx_d:.3e}"
                  f"  shape={shape}")
    if not bad:
        print("  全呼び出しでビット一致")


if __name__ == "__main__":
    main()
