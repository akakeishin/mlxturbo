"""hyper-connection の融合が (1) 数値一致するか (2) 速いか を確かめる。

融合カーネルを書くときの受け入れ確認に使う。速度より先に数値を見ること
(mx.compile 版は top1 が一致したまま logits が 5% ずれた)。
基準は docs/KERNEL-BRIEF-HC.md。
"""
import sys, time
sys.path.insert(0, "/Users/ht/dev/fastmlx")
import os
os.environ["FASTMLX_NGRAM_DISK"] = "1"
import mlx.core as mx
from mlx_lm import load
import mlx_lm.models.qwen4_exp as Q
from fastmlx import fused
from fastmlx.ngram_stream import install

model, tok = load("/Users/ht/models/qwen38fn-mlx-v-stream")
install(model, "/Users/ht/models/qwen38fn-ngram-4bit")
ids = tok.apply_chat_template(
    [{"role": "user", "content": "分散システムについて説明してください。"}],
    add_generation_prompt=True)

def run(n=25):
    cache = model.make_cache()
    lg = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(lg[0, -1], axis=-1))
    for _ in range(3):
        lg = model(mx.array([[cur]]), cache=cache); cur = int(mx.argmax(lg[0, -1], axis=-1))
    t = time.perf_counter()
    for _ in range(n):
        lg = model(mx.array([[cur]]), cache=cache); cur = int(mx.argmax(lg[0, -1], axis=-1))
    return (time.perf_counter() - t) / n * 1000

# 数値一致: 同じ入力で全 logits を比べる
full = mx.array(ids)[None]
a = model(full, cache=model.make_cache())[0, -1].astype(mx.float32); mx.eval(a)
fused.enable_hyper_connection()
b = model(full, cache=model.make_cache())[0, -1].astype(mx.float32); mx.eval(b)
import numpy as np
an, bn = np.array(a), np.array(b)
rel = np.linalg.norm(an - bn) / max(np.linalg.norm(an), 1e-9)
print(f"logits 相対誤差 = {rel:.8f}  top1 {'一致' if an.argmax()==bn.argmax() else '不一致'}")

fused.disable_hyper_connection()
before = run()
fused.enable_hyper_connection()
after = run()
print(f"融合前 {before:6.2f} ms/token ({1000/before:5.2f} tok/s)")
print(f"融合後 {after:6.2f} ms/token ({1000/after:5.2f} tok/s)   {before-after:+.2f} ms")
