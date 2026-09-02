"""forward 1 回を build (CPU でグラフを組む) / eval (GPU) に割る。

mlx-serve の ``[fwd-ubench]`` (この機体で S=1: build 2.2 + eval 18.2 = 20.4 ms,
3940 ops/forward。S=2: build 2.6 + eval 24.0 = 26.5 ms, 8315 ops) と同じ量・
同じ切り方をこちら側 (``tools/verify_width_cost.py`` の壁時計 S=1 25.2 ms /
S=2 33.1 ms) で出し、差 5-7 ms がどこにあるか見るための道具。

S ごとに 3 通り測る:

  (a) staged (本番)  -- ``capture()`` の中で ``spec_flash._staged_forward``。
      内部で ``mx.async_eval`` するため、build 側に GPU 時間が混ざる
      (mlx-serve の ubench が測る「純粋な CPU グラフ構築」とは別物)。
  (b) plain           -- ``capture()`` の中で ``model(pair, cache=caches)``
      (段階投入なし)。これが mlx-serve の ubench と同じ切り方。
  (c) plain, no capture -- capture 無しで (b) と同じ呼び出し。mlx-serve の
      S=1 ubench は capture=false なので、S=1 でだけ素直に比較できる
      (S>1 は参考値)。

op 数は (b) の build 区間だけ、``mlx.core`` (``mx``) と ``mlx.core.fast`` の
名前空間をモンキーパッチして関数名ごとに集計する (詳細は
``_profiled_build`` の docstring)。以前は ``sys.setprofile`` の ``c_call``
イベントで拾おうとしたが、``mlx.core`` (nanobind) の関数呼び出しはこの
イベントで捕まらず合計が 0 になっていたため、この方式に変えた。
モンキーパッチも計測 (build 時間) を歪めるので、時間計測とは別の追加 1 回
で測る。

engine/model/caches の用意 (prefill 1 回、退避・復元、pair の作り方、
回文掃引) は ``tools/verify_width_cost.py`` と ``tools/decode_ab.py`` の
関数をそのまま import して使う。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# decode_ab.py と同じキャッシュ退避・復元ヘルパー
from decode_ab import _restore, prefill_once  # noqa: E402

# verify_width_cost.py と同じ engine 構築・プロンプト作り・pair 作り・
# 回文掃引・要約 (中央値、最初の 3 回捨て) をそのまま使う
from verify_width_cost import (  # noqa: E402
    build_pair,
    build_prompt_ids,
    build_runner,
    summarize,
    sweep_order,
)


def _profiled_build(model, pair, caches):
    """(b) plain の build 区間だけ、``mlx.core`` (``mx``) と ``mlx.core.fast``
    の名前空間にある呼び出し可能オブジェクト (関数・builtin) を、呼ばれた
    回数を名前ごとに数えるラッパへ一時的に差し替えて集計する。差し替えは
    計測区間の直前に行い、直後 (例外時も ``finally`` で) に必ず元へ戻す。
    計測値 (build 時間) は別の反復で取るので、ここでは時間を測らない。

    以前は ``sys.setprofile`` の ``c_call`` イベントで拾おうとしたが、
    ``mlx.core`` (nanobind) の関数呼び出しはこのイベントで捕まらず合計 0
    になっていた。この関数のモンキーパッチ方式に変えている。

    捕まえられない範囲 (計測の対象外。既知の限界であって bug ではない):
      - ``mx.array`` インスタンスのメソッド呼び出し (``.reshape()`` /
        ``.astype()`` / ``__matmul__`` など)。束縛メソッドは
        ``mx``/``mx.fast`` の名前空間上の属性ではないため、ここでの
        差し替えでは触れない。
      - ``mx.compile`` された関数の内側の呼び出し。コンパイル済みグラフの
        中はネイティブに評価され、差し替えた Python ラッパを経由しない。
      - ``mx`` / ``mx.fast`` の中でも type オブジェクト (``mx.array`` 自体
        など) は差し替えない。isinstance など互換性チェックに使われて
        おり、関数でない差し替えは壊れるため対象から外している。
      - ``from mlx.core import foo`` のように名前を直接束縛している呼び手
        (このリポジトリの現状のコードは ``import mlx.core as mx`` で統一
        されているため該当しないはずだが、構造的な限界として明記する)。
      - 差し替えを試みて例外になった名前は ``not_captured.unpatchable_*``
        にそのまま列挙する。

    層ごとの内訳: ``model.model.layers`` が共有する ``DecoderLayer`` クラス
    の ``__call__`` を同様に一時ラップして求める。層インスタンスではなく
    クラス側を差し替える必要がある -- ``layer(...)`` の暗黙呼び出しは
    特殊メソッドとして型から解決されるため、インスタンス属性の差し替えは
    効かない。各層呼び出しの直前・直後で上の名前別カウンタのスナップ
    ショット差分を取り、``layer.layer_type`` ("linear_attention" /
    "full_attention") 別に積み上げる (層は入れ子で呼ばれない前提)。層の外
    (embed_tokens / hyper_connection_mixer / lm_head など) の分は、合計から
    層内訳の合計を引いた差分で「その他」として出す。
    """
    import mlx.core as mx
    from mlxturbo.spec_flash import capture

    c_calls: dict[str, int] = {}

    def _patch_namespace(namespace, ns_label):
        patched: list[tuple[str, object]] = []
        unpatchable: list[str] = []
        for name in dir(namespace):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(namespace, name)
            except Exception:
                continue
            if not callable(obj) or isinstance(obj, type):
                continue
            key = f"{ns_label}.{name}"

            def _make_wrapper(orig, key=key):
                def _wrapper(*a, **kw):
                    c_calls[key] = c_calls.get(key, 0) + 1
                    return orig(*a, **kw)

                return _wrapper

            wrapper = _make_wrapper(obj)
            try:
                setattr(namespace, name, wrapper)
            except Exception:
                unpatchable.append(name)
                continue
            patched.append((name, obj))
        return patched, unpatchable

    patched_core, unpatchable_core = _patch_namespace(mx, "mx")
    patched_fast, unpatchable_fast = _patch_namespace(mx.fast, "mx.fast")

    # 層ごとの内訳: DecoderLayer クラス (全層が共有) の __call__ を一時ラップする
    layers = getattr(getattr(model, "model", model), "layers", None)
    layer_cls = type(layers[0]) if layers else None
    orig_layer_call = layer_cls.__call__ if layer_cls is not None else None

    layer_op_counts: dict[str, dict[str, int]] = {}
    layer_instance_counts: dict[str, int] = {}

    if layer_cls is not None:

        def _layer_call_wrapper(self, *a, **kw):
            before = dict(c_calls)
            try:
                return orig_layer_call(self, *a, **kw)
            finally:
                lt = getattr(self, "layer_type", "unknown")
                bucket = layer_op_counts.setdefault(lt, {})
                for k, v in c_calls.items():
                    d = v - before.get(k, 0)
                    if d:
                        bucket[k] = bucket.get(k, 0) + d
                layer_instance_counts[lt] = layer_instance_counts.get(lt, 0) + 1

        layer_cls.__call__ = _layer_call_wrapper

    try:
        with capture(model) as _cap:
            lg = model(pair, cache=caches)
    finally:
        for name, orig in patched_core:
            setattr(mx, name, orig)
        for name, orig in patched_fast:
            setattr(mx.fast, name, orig)
        if layer_cls is not None:
            layer_cls.__call__ = orig_layer_call

    mx.eval(lg)

    total = sum(c_calls.values())
    top30 = sorted(c_calls.items(), key=lambda kv: -kv[1])[:30]

    by_layer_type: dict[str, dict] = {}
    layer_total = 0
    for lt, bucket in layer_op_counts.items():
        n = layer_instance_counts.get(lt, 0)
        bucket_total = sum(bucket.values())
        layer_total += bucket_total
        top15 = sorted(bucket.items(), key=lambda kv: -kv[1])[:15]
        by_layer_type[lt] = {
            "n_layers": n,
            "total_calls": bucket_total,
            "calls_per_layer": (bucket_total / n) if n else 0.0,
            "top15": top15,
            "top15_per_layer": [(name, cnt / n) for name, cnt in top15] if n else [],
        }

    return {
        "total_c_calls": total,
        "top30": top30,
        "by_layer_type": by_layer_type,
        "other_total_calls": total - layer_total,
        "not_captured": {
            "mx_array_methods": (
                "mx.array インスタンスのメソッド (reshape/astype/__matmul__ 等) は"
                "名前空間の差し替えでは捕まえられないため対象外"
            ),
            "mx_compile_internals": "mx.compile された関数の内側は捕まらない",
            "direct_name_imports": (
                "from mlx.core import foo のように直接名前を束縛している呼び手は"
                "捕まえられない (現状のコードは import mlx.core as mx で統一)"
            ),
            "unpatchable_mx": unpatchable_core,
            "unpatchable_mx_fast": unpatchable_fast,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None, help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=0, help="既定 0 = 短プロンプト")
    ap.add_argument("--widths", default="1,2")
    ap.add_argument("--reps", type=int, default=20, help="S ごとの反復回数 (最初の 3 回を含む)")
    ap.add_argument("--out", default=str(REPO_ROOT / "bench" / "results" / "forward-split.json"))
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    if not widths:
        print("--widths が空")
        return 1
    if args.reps < 3:
        print("--reps は 3 より大きい必要がある (最初の 3 回を捨てる作法のため)")
        return 1

    eng, model, tok, eos_ids = build_runner(args)
    ids = build_prompt_ids(tok, args.ctx)

    import mlx.core as mx
    from mlxturbo.spec_flash import _staged_forward  # noqa: SLF001 (依頼どおり直接使う)
    from mlxturbo.spec_flash import capture

    print(f"ctx={ids.shape[1]} (--ctx {args.ctx})  widths={widths}  reps={args.reps}")

    caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
    print(f"  prefill 1 回だけ流した (n={ids.shape[1]})。以降は同じ状態から退避・復元する。")

    pair_full, _cur = build_pair(eng, resume, max(widths))

    # 回文掃引 (昇順・降順を交互に、verify_width_cost.py と同じ作法) で S を回す
    rounds = 2 if args.reps >= 2 else 1
    base = args.reps // rounds
    rem = args.reps % rounds

    raw: dict[int, dict[str, dict[str, list[float]]]] = {
        s: {
            "staged": {"build": [], "eval": []},
            "plain": {"build": [], "eval": []},
            "plain_nocap": {"build": [], "eval": []},
        }
        for s in widths
    }

    for sweep_idx, order in enumerate(sweep_order(widths, rounds)):
        n_this = base + (1 if sweep_idx < rem else 0)
        for s in order:
            pair = pair_full[:, :s]
            for _ in range(n_this):
                # (a) staged (本番)。async_eval が混ざるので build に GPU 時間が漏れる
                _restore(caches, snap)
                t0 = time.perf_counter()
                with capture(model) as _cap:
                    lg = _staged_forward(model, pair, caches)
                t_build = time.perf_counter() - t0
                t0 = time.perf_counter()
                mx.eval(lg)
                t_eval = time.perf_counter() - t0
                raw[s]["staged"]["build"].append(t_build)
                raw[s]["staged"]["eval"].append(t_eval)

                # (b) plain (段階投入なし)。mlx-serve の ubench と同じ切り方
                _restore(caches, snap)
                t0 = time.perf_counter()
                with capture(model) as _cap:
                    lg = model(pair, cache=caches)
                t_build = time.perf_counter() - t0
                t0 = time.perf_counter()
                mx.eval(lg)
                t_eval = time.perf_counter() - t0
                raw[s]["plain"]["build"].append(t_build)
                raw[s]["plain"]["eval"].append(t_eval)

                # (c) plain, capture 無し。mlx-serve の S=1 ubench (capture=false) と比較可能なのは S=1 だけ
                _restore(caches, snap)
                t0 = time.perf_counter()
                lg = model(pair, cache=caches)
                t_build = time.perf_counter() - t0
                t0 = time.perf_counter()
                mx.eval(lg)
                t_eval = time.perf_counter() - t0
                raw[s]["plain_nocap"]["build"].append(t_build)
                raw[s]["plain_nocap"]["eval"].append(t_eval)

    print("\n=== S ごとの build / eval / 合計 (中央値, ms) ===")
    result: dict = {
        "ctx": ids.shape[1],
        "ctx_arg": args.ctx,
        "widths": widths,
        "reps": args.reps,
        "note_staged_build": "staged の build は async_eval を含むため GPU 時間が混ざる",
        "note_plain_nocap": "plain_nocap は S=1 のときだけ mlx-serve の S=1 ubench (capture=false) と比較可能",
        "timing": {},
        "ops": {},
    }
    for s in widths:
        result["timing"][str(s)] = {}
        print(f"-- S={s} --")
        for variant in ("staged", "plain", "plain_nocap"):
            b = raw[s][variant]["build"]
            e = raw[s][variant]["eval"]
            total = [bi + ei for bi, ei in zip(b, e)]
            b_sum = summarize(b)
            e_sum = summarize(e)
            t_sum = summarize(total)
            result["timing"][str(s)][variant] = {"build": b_sum, "eval": e_sum, "total": t_sum}
            print(
                f"  {variant:12s}  build {b_sum['median_ms']:7.3f}"
                f"  eval {e_sum['median_ms']:7.3f}"
                f"  total {t_sum['median_ms']:7.3f}  (n={t_sum['n']})"
            )

    print("\n=== S ごとの op 数 ((b) plain の build 区間, モンキーパッチ別反復) ===")
    layer_label = {"linear_attention": "GDN 層", "full_attention": "attention 層"}
    for s in widths:
        _restore(caches, snap)
        pair = pair_full[:, :s]
        ops = _profiled_build(model, pair, caches)
        _restore(caches, snap)
        result["ops"][str(s)] = ops
        print(
            f"-- S={s}  mx/mx.fast 呼び出し合計 {ops['total_c_calls']}"
            f"  (層の外 = その他 {ops['other_total_calls']}) --"
        )
        for name, n in ops["top30"]:
            print(f"    {n:6d}  {name}")
        for lt, info in ops["by_layer_type"].items():
            title = layer_label.get(lt, lt)
            print(
                f"  -- {title} (n={info['n_layers']})  合計 {info['total_calls']}"
                f"  1 層あたり {info['calls_per_layer']:.1f} --"
            )
            for name, n in info["top15"]:
                print(f"      {n:6d}  {name}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
