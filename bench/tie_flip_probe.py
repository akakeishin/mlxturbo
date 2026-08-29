"""spec エンジン抜きでのバッチ数値差の決定実験。

gate と同一構築の prose プロンプトで baseline 手動ループを再生する。
2本のランは index 77 まで完全に同一の逐次計算 (決定的なので cache 状態も同一)。
  (a) 77 以降も1トークンずつ forward し、index 81 の logits を取る (gate baseline)
  (b) 77 で止め、tokens[77:81] を1回の m=4 forward で流して同じ位置の logits を取る
     (spec 検証と同じ形状)
argmax が違えば、prose の spec 不一致は縮約順の数値差で説明が付き、
エンジンの検証/巻き戻しは無罪。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from mlxturbo._mlx_compat import mlx_lm_load

PROSE = (
    "分散システムにおける結果整合性と強整合性の違いを、具体例を"
    "挙げながら詳しく説明してください。"
)
DIVERGE_AT = 81
STOP_AT = 77
REF_TOK, ACT_TOK = 11, 13

model, tokenizer = mlx_lm_load("lmstudio-community/Qwen3.8-27B-MLX-4bit")
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROSE}], add_generation_prompt=True
)


def replay(n_steps):
    """gate.manual_greedy と同じ逐次ループで n_steps トークン生成して返す。"""
    cache = model.make_cache()
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
    mx.eval(token)
    tokens = []
    while len(tokens) < n_steps:
        tokens.append(int(token.item()))
        logits = model(token[None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
        mx.eval(token)
    return tokens, cache, logits


# (a) 逐次のみで index 81 の logits
tokens_a, _, logits_a = replay(DIVERGE_AT)
lg1 = logits_a[0, -1, :].astype(mx.float32)
a1 = int(mx.argmax(lg1).item())

# (b) 同一逐次計算を 77 で止め、残り4トークンを m=4 一括 forward
tokens_b, cache_b, _ = replay(STOP_AT)
assert tokens_b == tokens_a[:STOP_AT], "逐次ランが非決定的 — 前提崩壊"
tail = mx.array(tokens_a[STOP_AT:DIVERGE_AT])
lg4_all = model(tail[None], cache=cache_b)
lg4 = lg4_all[0, -1, :].astype(mx.float32)
a4 = int(mx.argmax(lg4).item())

print("context:", tokenizer.decode(tokens_a[-10:]))
print(
    f"m=1 : argmax={a1} ({tokenizer.decode([a1])!r})  "
    f"','={float(lg1[REF_TOK].item()):.5f}  '.'={float(lg1[ACT_TOK].item()):.5f}"
)
print(
    f"m=4 : argmax={a4} ({tokenizer.decode([a4])!r})  "
    f"','={float(lg4[REF_TOK].item()):.5f}  '.'={float(lg4[ACT_TOK].item()):.5f}"
)
print(f"m=1 ',' vs '.' gap: {abs(float(lg1[REF_TOK].item()) - float(lg1[ACT_TOK].item())):.5f}")
print("flip:", a1 != a4)
