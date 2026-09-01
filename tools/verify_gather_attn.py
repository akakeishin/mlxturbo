"""gather attention (段 3(b)、`MLXTURBO_GATHER_ATTN`) の正しさを、合成した
小さい Flash-Next を CPU で流して確かめる。

いまの QSA (疎注意) は選んだブロックを加算マスクとして sdpa に渡している
(`mlxturbo/_vendor/qwen4_exp.py` の `QSAIndexer.__call__` が bool の `keep`
を返し、`Attention._final_mask` がそれを加算マスクへ変換する)。
`mx.fast.scaled_dot_product_attention` は加算マスクを渡されても全 KV を
読んで全スコアを計算するため、疎性が sdpa 側の節約になっていない
(`docs/research/KERNEL-PROGRAM.md` 段 1 の実測)。

段 3(b) はこれを、選ばれたブロックだけを `mx.take_along_axis` で集めてから
マスク無しの (集めた列に対する小さい bool マスクだけを持つ) dense sdpa に
渡す経路に置き換える (`QSAIndexer.select_blocks` / `Attention._gather_forward`、
`mlxturbo/gather_attn.py` の `enable_gather_attn`)。選ぶ集合は元の `keep` と
同じはずだが、和の順序が変わるので出力はビット一致しない。

この道具が見るのは:

    1. 通常経路 (加算マスク) と gather 経路が、同じ入力に対して
       十分近い出力を返すこと (許容差つき比較。ビット一致は要求しない)
    2. 実測の max|diff| が float32 の丸め水準に収まっていること
       (収まっていなければ選択集合そのものがずれている疑いで、不合格)
    3. prefill 幅 (S が大きい、端数ブロック=tail 込み) と decode 幅
       (S=1、verify を模した S=2) の両方を踏むこと

比較は固定した入力トークン列に対する logits の直接比較で行う (argmax で
自分の出力を追いかけさせない)。gather は数値をわずかに動かすので、
自分の出力を追いかけさせると argmax の際どい tie で経路ごとに違うトークン
列へ分岐し、比較そのものが壊れる。

使い方:

    .venv/bin/python tools/verify_gather_attn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mlxturbo  # noqa: E402,F401
from mlxturbo import gather_attn  # noqa: E402
from verify_batch_cache import TINY, build  # noqa: E402

# float32 の丸め水準の目安 (`tools/vendor_fingerprint.py` の
# group_prefill_forward 判定が 5e-6 を使っている前例に合わせつつ、
# gather は並べ替える項の数が多いぶん少し緩める)。これを超えたら
# 「選択集合そのものがずれている」疑いとして不合格にする。
TOL = 1e-4


def _build_calls(vocab: int):
    """段 3(b) の想定幅を両方踏む呼び出し列を組む (offset・S・tail の内訳は
    下の docstring 相当のコメントを各呼び出しに添えて main() で出力する)。

    budget=8, compress_ratio=2 (TINY 既定) での組み立て:

    - chunk 1 (offset=0, S=8): offset < compress_ratio-1 の早期救済域。
      kv_len も budget ちょうどなので QSA 自体が不活性 (sparse=None)。
      gather は素通り (通常経路と完全に同じコード)
    - chunk 2..7 (offset=8,16,...,48、各 S=8): kv_len が budget を超えて
      QSA 活性、gather も活性。tail=0 (偶数境界)
    - chunk 8 (offset=56, S=3): 端数チャンク。kv_len=59 (奇数) になり
      tail=1 を作る。prefill 幅で tail 込みの経路を踏む
    - S=1 を 4 回 (decode 幅)
    - S=2 を 1 回 (verify 幅を模す)
    - S=1 をさらに 5 回
    """
    ids = [(i * 7 + 3) % vocab for i in range(70)]
    calls = []
    body = ids[:59]
    chunk = 8
    for lo in range(0, len(body), chunk):
        calls.append(body[lo : lo + chunk])
    for i in range(59, 63):
        calls.append([ids[i]])
    calls.append(ids[63:65])
    for i in range(65, 70):
        calls.append([ids[i]])
    return calls


def _run(model, use_gather: bool, calls, stats=None):
    if use_gather:
        gather_attn.enable_gather_attn(model, stats=stats)
    else:
        gather_attn.disable_gather_attn(model)
    cache = model.make_cache()
    out = []
    for ids in calls:
        logits = model(mx.array(ids)[None], cache=cache)
        mx.eval(logits)
        out.append(logits)
    return out


def check(budget: int) -> tuple[bool, float, list]:
    model = build(budget)
    calls = _build_calls(TINY["vocab_size"])

    base = _run(model, False, calls)
    stats: list = []
    gathered = _run(model, True, calls, stats=stats)

    diffs = []
    offset = 0
    ok = True
    for i, (ids, a, b) in enumerate(zip(calls, base, gathered)):
        S = len(ids)
        d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        diffs.append(d)
        tag = "一致(bit)" if d == 0.0 else f"max|diff|={d:.3e}"
        print(f"call {i:2d}  offset={offset:3d} S={S}  {tag}")
        if d > TOL:
            ok = False
        offset += S

    max_diff = max(diffs)
    print(f"\n全体 max|diff| = {max_diff:.3e} (許容 {TOL:.0e})")
    print(f"判定: {'合格' if ok else '不合格'}")

    if stats:
        print("\ngather が実際に活性化した呼び出し (S, n_blocks, union U, "
              "gather 列数 n_sel):")
        for S, n_blocks, U, n_sel in stats:
            print(f"  S={S:2d}  n_blocks={n_blocks:3d}  U={U:3d}  n_sel={n_sel:3d}")
    else:
        print("\n警告: gather が一度も活性化しなかった (budget/chunk の組み方を見直すこと)")
        ok = False

    return ok, max_diff, stats


def check_no_activation_is_bitexact(budget: int) -> bool:
    """kv_len <= token_budget の間 (QSA 自体が不活性) は、gather の
    on/off でコードパスが完全一致するはずなので、ビット一致を要求する。
    早期救済域 (offset < compress_ratio-1) も同様。
    """
    model = build(budget)
    ids = [(i * 13 + 5) % TINY["vocab_size"] for i in range(budget)]  # kv_len == budget

    base = _run(model, False, [ids])[0]
    gathered = _run(model, True, [ids])[0]
    mx.eval(base, gathered)
    ok = bool(mx.all(base == gathered))
    print(f"QSA 不活性域 (kv_len<=budget): gather on/off がビット一致: {ok}")
    return ok


def main() -> int:
    budget = 8
    print(f"=== gather attention 正しさ確認 (budget={budget}, "
          f"compress_ratio={TINY['indexer_compress_ratio']}) ===\n")
    ok1 = check_no_activation_is_bitexact(budget)
    print()
    ok2, max_diff, stats = check(budget)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
