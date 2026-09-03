"""decode 1 step のグラフに現れる **op (プリミティブ) をモジュールごとに数える**。

## 何のための道具か

`tools/decode_gpu_trace.py` は「どのカーネルが何回・何 us 走ったか」までは
出すが、**その dispatch を Python のどこが作ったかは出せない** (dylib は
MLX の eval スレッドから呼ばれるので Python のスタックが無い)。
depth 0 の trace で `g1_copy` が 133 本 = 1.79 ms/round あることが分かった
(`docs/research/SESSION-2026-09-02-CATCHUP.md` 18:45) が、出所が不明のまま
だった。この道具はそこを埋める。

## どうやるか (GPU も観測 dylib も要らない)

MLX は遅延グラフなので、`mx.eval` を挟まなければ 1 forward ぶんの
グラフがまるごと残る。`mx.export_to_dot` はそのグラフを
``{ 43136691352 [label ="Concatenate", shape=rectangle]; }`` の形で吐く
(ノード id はプリミティブのポインタ、label は op 名)。

そこで各モジュールの `__call__` を包み、**戻り値から辿れるプリミティブ集合**
を取る。内側のモジュールほど先に return するので、「まだ誰にも帰属していない
id だけを自分のものにする」だけで、内側 → 外側の順に自然と分かれる
(外側に残るのは、そのモジュールが子を呼ぶ合間に直接作った op だけ)。

`--model` を渡さなければ合成の小さい Flash-Next (`tools/verify_batch_cache.py`
の `TINY`、CPU、数秒) で走る。**合成側は非量子化で GPU 融合カーネルが
1 つも発火しない**ので、構造の下見にしか使えない。本番の並びを見るには
`--model` を渡すこと (98GB を読むので `tools/biglock.sh` 経由)。

    # 下見 (GPU 不要、数秒)
    .venv/bin/python tools/decode_copy_probe.py --width 1

    # 本番の並び
    tools/biglock.sh .venv/bin/python tools/decode_copy_probe.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep --width 1

## 読むときの注意

- 数えているのは **op (MLX のプリミティブ) であって dispatch ではない**。
  1 op が 1 dispatch とは限らない (`Slice` や `Contiguous` は元が連続なら
  view で済み、copy カーネルを出さない)。trace の回数と突き合わせて読む。
- 突き合わせの実例 (2026-09-03、実モデル S=1): trace の `g1_copy` 133 本に対し
  ここが数えた `Concatenate` は 134 本で一致した。**つまり concat 1 本 =
  copy 1 dispatch** で、入力の本数ぶんには増えていない。一方 `Contiguous`
  36 本 (GDN の conv 窓の切り出し) は B=1 では連続なので dispatch を出して
  いない。内訳は Attention 84 / GatedDeltaNet 36 / QSAIndexer 12 / PLE 1。
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

_NODE_RE = re.compile(r'^\{ (\d+) \[label ="([^"]*)"')
_EDGE_RE = re.compile(r'^(\d+) -> "')


def graph_nodes(arrays):
    """`arrays` から辿れるプリミティブを {id: op 名} で返す。"""
    import mlx.core as mx

    buf = io.StringIO()
    try:
        mx.export_to_dot(buf, *arrays)
    except Exception:            # noqa: BLE001  非 mx.array が混ざった等
        return {}
    out = {}
    for line in buf.getvalue().splitlines():
        m = _NODE_RE.match(line.strip())
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def graph_outputs(arrays):
    """`arrays` から辿れる「プリミティブが作った配列」を数える。

    返すのは ``{プリミティブ id: (op 名, 出力配列の本数)}``。
    **プリミティブ id では数えないこと**: `mx.compile` の `Compiled`
    プリミティブは呼び出しをまたいで同じインスタンスが使い回されるので、
    id を数えると 48 層ぶんが 1 本に潰れる。出力配列 (``id -> "X"`` の辺)
    なら呼び出しごとに 1 本立つ。ここが dispatch 数に一番近い量。
    """
    import mlx.core as mx

    buf = io.StringIO()
    try:
        mx.export_to_dot(buf, *arrays)
    except Exception:            # noqa: BLE001
        return {}
    names, edges = {}, Counter()
    for line in buf.getvalue().splitlines():
        line = line.strip()
        m = _NODE_RE.match(line)
        if m:
            names[m.group(1)] = m.group(2)
            continue
        m = _EDGE_RE.match(line)
        if m:
            edges[m.group(1)] += 1
    return {int(k): (names.get(k, "?"), n) for k, n in edges.items()}


def _arrays(obj, acc, depth=0):
    """任意の戻り値から mx.array だけを掘り出す (tuple/list/dict/1 段の属性)。"""
    import mlx.core as mx

    if depth > 3:
        return
    if isinstance(obj, mx.array):
        acc.append(obj)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _arrays(o, acc, depth + 1)
    elif isinstance(obj, dict):
        for o in obj.values():
            _arrays(o, acc, depth + 1)


def _cache_arrays(objs, acc):
    """引数に混ざったキャッシュ (mlx_lm の *Cache) の中身を拾う。

    GDN の conv 状態や KV は戻り値ではなくキャッシュに書かれるので、
    ここを見ないとその copy が誰のものか分からない。"""
    for v in objs:
        if isinstance(v, (list, tuple)):
            _cache_arrays(v, acc)
            continue
        st = getattr(v, "state", None)
        if st is not None and not callable(st):
            _arrays(st, acc)


class Attributor:
    """モジュールごとの「新しく作られた op」を数える。"""

    def __init__(self):
        self.counts: Counter = Counter()     # (持ち主, op 名) -> 出力配列の本数
        self.calls: Counter = Counter()      # 持ち主 -> 呼ばれた回数
        self.claimed: dict[int, int] = {}    # プリミティブ id -> 帰属済みの本数
        self.enabled = False

    def wrap(self, cls, name=None):
        owner = name or cls.__name__
        orig = cls.__call__

        def wrapped(self_, *a, **kw):
            if not self.enabled:
                return orig(self_, *a, **kw)
            # **入口の祖先を先に控える。**出口のグラフからは呼び出し前の
            # ops も辿れてしまうので、差を取らないと「最後に return した
            # 内側のモジュール」が上流を丸ごと持っていく
            ins = []
            _arrays(list(a) + list(kw.values()), ins)
            _cache_arrays(list(a) + list(kw.values()), ins)
            pre = graph_outputs(ins)
            out = orig(self_, *a, **kw)
            acc = []
            _arrays(out, acc)
            _cache_arrays(list(a) + list(kw.values()), acc)
            self.claim(owner, acc, pre)
            return out

        cls.__call__ = wrapped
        return orig

    def claim(self, owner, arrays, pre=None):
        """`arrays` から辿れる出力配列のうち、まだ誰のものでもない本数を数える。

        基準は「入口で既にあった本数」と「既に他所へ帰属した本数」の大きい方。
        子は先に return するので、親の post には子の分も入っているが、
        そのぶんは `claimed` で引かれて重複しない。"""
        pre = pre or {}
        self.calls[owner] += 1
        for nid, (op, n) in graph_outputs(arrays).items():
            base = max(pre.get(nid, (op, 0))[1], self.claimed.get(nid, 0))
            if n > base:
                self.counts[(owner, op)] += n - base
                self.claimed[nid] = n

    def report(self, title, top=40):
        print(f"\n=== {title} ===")
        total = sum(self.counts.values())
        print(f"  プリミティブが作った配列 (dispatch に一番近い量) 合計 {total}")
        by_op = Counter()
        for (owner, op), n in self.counts.items():
            by_op[op] += n
        print("\n  -- op 別 (全モジュール合計) --")
        for op, n in by_op.most_common(top):
            print(f"    {op:28s} {n:6d}")
        print("\n  -- モジュール x op (多い順) --")
        for (owner, op), n in self.counts.most_common(top):
            print(f"    {owner:22s} {op:24s} {n:6d}"
                  f"   ({n / max(1, self.calls[owner]):.2f}/呼び出し)")
        # copy 系 (Metal で copy カーネルを投げうる op) は全モジュールを出す
        print("\n  -- copy を投げうる op の出所 (全件) --")
        for op in ("Concatenate", "Slice", "SliceUpdate", "Contiguous",
                   "AsType", "Copy", "ScatterAxis", "Scatter", "Pad",
                   "Transpose", "Reshape"):
            rows = [(o, n) for (o, p), n in self.counts.items() if p == op]
            if not rows:
                continue
            rows.sort(key=lambda r: -r[1])
            tail = "  ".join(f"{o}={n}" for o, n in rows)
            print(f"    {op:14s} 計 {by_op[op]:5d}   {tail}")


def install(attr, Q):
    """vendor のモジュールを包む (内側から順に帰属される)。"""
    import mlx.nn as nn

    targets = [
        (Q.RMSNorm, "RMSNorm"),
        (getattr(Q, "RMSNormGated", None), "RMSNormGated"),
        (Q.MLP, "MLP(shared)"),
        (Q.GatedResidual, "GatedResidual(HC)"),
        (Q.Attention, "Attention"),
        (Q.GatedDeltaNet, "GatedDeltaNet"),
        (Q.SparseMoeBlock, "SparseMoeBlock"),
        (getattr(Q, "QSAIndexer", None), "QSAIndexer"),
        (getattr(Q, "NGramEmbedding", None), "NGramEmbedding"),
        (getattr(Q, "PLELayer", None), "PLELayer"),
        (Q.DecoderLayer, "DecoderLayer(残り)"),
        (Q.Qwen4ExpModel, "Qwen4ExpModel(残り)"),
    ]
    for cls, name in targets:
        if cls is not None:
            attr.wrap(cls, name)
    # 線形射影は「行列積そのもの」なので別枠 (糊と混ぜない)
    attr.wrap(nn.Linear, "Linear")
    if hasattr(nn, "QuantizedLinear"):
        attr.wrap(nn.QuantizedLinear, "QuantizedLinear")


def build_synthetic():
    import mlx.core as mx

    from verify_batch_cache import build

    mx.set_default_device(mx.cpu)
    return build(8), None


def build_real(model_path, ngram_path):
    """出荷経路と同じ融合を掛けたモデルを返す (MTP は読まない --- ここで見るのは
    1 forward のグラフだけで、投機 decode のループには入らない)。"""
    from ab_bundle import load_bundle

    b = load_bundle(model_path, ngram_path=ngram_path, load_mtp=False,
                    log_prefix="[copy_probe]")
    return b.model, b


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="省略すると合成の小さいモデル (CPU)")
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--width", default="1", help="1 forward の S (カンマ区切り可)")
    ap.add_argument("--prompt-len", type=int, default=64)
    return ap


def run_with_model(argv, bundle) -> int:
    """常駐 worker の `tool` ジョブの入口 (規約は `tools/ab_bundle.py`)。

    98GB を読み直さずに済ませるためだけの分岐で、出力は CLI と同じ。
    """
    args = build_parser().parse_args(argv)
    args.model = args.model or getattr(bundle, "model_path", "(worker)")
    return _probe(args, bundle.model,
                  getattr(bundle.model.args, "text").vocab_size)


def main() -> int:
    args = build_parser().parse_args()
    import mlxturbo  # noqa: F401  arch (mlx_lm.models.qwen4_exp) の登録が先

    if args.model:
        model, _bundle = build_real(args.model, args.ngram)
        vocab = model.args.text.vocab_size if hasattr(model, "args") else 1000
    else:
        model, _bundle = build_synthetic()
        vocab = 512
    return _probe(args, model, vocab)


def _probe(args, model, vocab) -> int:
    import mlx.core as mx

    import mlxturbo  # noqa: F401
    from mlx_lm.models import qwen4_exp as Q

    install_done = False
    inner = model.model if hasattr(model, "model") else model
    for width in [int(w) for w in str(args.width).split(",")]:
        attr = Attributor()
        if not install_done:
            install(attr, Q)      # 包み直すと二重に包まれるので 1 回だけ
            install_done = attr
        else:
            attr = install_done   # 同じ Attributor を使い回す (計数はリセット)
            attr.counts.clear()
            attr.calls.clear()
            attr.claimed.clear()

        cache = model.make_cache()
        ids = mx.array([[(i * 7 + 3) % (vocab - 8)
                         for i in range(args.prompt_len)]])
        inner(ids, cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])

        attr.enabled = True
        out = inner(mx.array([[5] * width]), cache=cache)
        attr.claim("(top level)", [out])
        attr.enabled = False
        mx.eval(out)

        attr.report(f"decode forward S={width} "
                    f"({'実モデル' if args.model else '合成 (CPU)'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
