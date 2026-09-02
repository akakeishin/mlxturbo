"""gather attention (段 3(b)/P1a、`MLXTURBO_GATHER_ATTN` / `MLXTURBO_GATHER_TILE`)
の正しさを、合成した小さい Flash-Next を CPU で流して確かめる。

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

段 P1a はさらに、`_gather_forward` の S 行を幅 `tile` のタイルに割り、
タイルごとに union を取り直す (`enable_gather_attn(..., tile=...)`)。
decode (S が小さい) では union が `S * block_topk` で頭打ちになるので
タイルは実質無効のままだが、prefill のように S が大きいチャンクでは
タイル無しだと union がほぼ全ブロックまで膨らむ。隣り合うクエリの選択は
強く相関するので、タイルに割ればタイルごとの union は縮む --- というのが
狙い。タイルは分割の仕方であって計算の中身ではないので、タイル幅を変えても
最終出力は変わらないはず (和の順序はタイル境界でわずかに変わるので近似一致)。

この道具が見るのは:

    1. 通常経路 (加算マスク) と gather 経路が、同じ入力に対して
       十分近い出力を返すこと (許容差つき比較。ビット一致は要求しない)。
       タイル幅 {0 (従来), 3, 5} をそれぞれこの比較にかける
    2. 実測の max|diff| が float32 の丸め水準に収まっていること
       (収まっていなければ選択集合そのものがずれている疑いで、不合格)
    3. prefill 幅 (S が大きいチャンク、端数ブロック=tail 込み) と decode 幅
       (S=1、verify を模した S=2) の両方を踏むこと
    4. タイル幅を変えても (0 と 3 と 5 のあいだで) 出力がほぼ変わらないこと
       (タイルはあくまで分割の仕方であって、選ぶ集合の意味は変わらない)

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

# 段 P1a のタイル幅掃引。0 は従来 (S 全体を 1 回)。TINY は S が高々 32 の
# 合成モデルなので、実運用の 128/256/512 ではなく小さい幅で踏む
# (3 は端数が出やすい奇数幅、5 は 2 の倍数境界からずらした幅)。
TILE_WIDTHS = (0, 3, 5)


def _build_calls(vocab: int):
    """段 3(b)/P1a の想定幅を両方踏む呼び出し列を組む (offset・S・tail の内訳は
    下の docstring 相当のコメントを各呼び出しに添えて main() で出力する)。

    budget=8, compress_ratio=2 (TINY 既定) での組み立て:

    - chunk 1 (offset=0, S=8): offset < compress_ratio-1 の早期救済域。
      kv_len も budget ちょうどなので QSA 自体が不活性 (sparse=None)。
      gather は素通り (通常経路と完全に同じコード)
    - chunk 2..7 (offset=8,16,...,48、各 S=8): kv_len が budget を超えて
      QSA 活性、gather も活性。tail=0 (偶数境界)
    - chunk 8 (offset=56, S=3): 端数チャンク。kv_len=59 (奇数) になり
      tail=1 を作る。prefill 幅で tail 込みの経路を踏む
    - chunk 9 (offset=59, S=32): 段 P1a 用の「prefill 幅」チャンク。TINY の
      中では大きい S を 1 回の呼び出しで通し、タイル幅 3/5 がそれぞれ
      11 個・7 個のタイルに割れる大きさにしてある。kv_len=91 (奇数) で
      tail=1 も引き続き踏む
    - S=1 を 4 回 (decode 幅)
    - S=2 を 1 回 (verify 幅を模す)
    - S=1 をさらに 5 回
    """
    ids = [(i * 7 + 3) % vocab for i in range(102)]
    calls = []
    body = ids[:59]
    chunk = 8
    for lo in range(0, len(body), chunk):
        calls.append(body[lo : lo + chunk])
    calls.append(ids[59:91])  # prefill 幅チャンク (S=32)
    for i in range(91, 95):
        calls.append([ids[i]])
    calls.append(ids[95:97])
    for i in range(97, 102):
        calls.append([ids[i]])
    return calls


def _run(model, use_gather: bool, calls, stats=None, tile: int = 0):
    if use_gather:
        gather_attn.enable_gather_attn(model, stats=stats, tile=tile)
    else:
        gather_attn.disable_gather_attn(model)
    cache = model.make_cache()
    out = []
    for ids in calls:
        logits = model(mx.array(ids)[None], cache=cache)
        mx.eval(logits)
        out.append(logits)
    return out


def check(budget: int, tile: int) -> tuple[bool, float, list]:
    model = build(budget)
    calls = _build_calls(TINY["vocab_size"])

    base = _run(model, False, calls)
    stats: list = []
    gathered = _run(model, True, calls, stats=stats, tile=tile)

    diffs = []
    offset = 0
    ok = True
    for i, (ids, a, b) in enumerate(zip(calls, base, gathered)):
        S = len(ids)
        d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        diffs.append(d)
        tag = "一致(bit)" if d == 0.0 else f"max|diff|={d:.3e}"
        print(f"  call {i:2d}  offset={offset:3d} S={S:2d}  {tag}")
        if d > TOL:
            ok = False
        offset += S

    max_diff = max(diffs)
    print(f"\n  全体 max|diff| = {max_diff:.3e} (許容 {TOL:.0e})")
    print(f"  判定: {'合格' if ok else '不合格'}")

    if stats:
        print(
            "\n  gather が実際に活性化した呼び出し (タイルごとに 1 行。"
            "T=タイルのクエリ行数、n_blocks、union U、gather 列数 n_sel、"
            "union_ratio=U/n_blocks、kv_frac=U*compress_ratio/kv_len、"
            "true_u=和集合の真の大きさ):"
        )
        for T, n_blocks, U, n_sel, union_ratio, kv_frac, true_u in stats:
            print(
                f"    T={T:2d}  n_blocks={n_blocks:3d}  U={U:3d}  "
                f"n_sel={n_sel:3d}  union_ratio={union_ratio:.2f}  "
                f"kv_frac={kv_frac:.2f}  true_u={true_u:3d}"
            )
    else:
        print("\n  警告: gather が一度も活性化しなかった (budget/chunk の組み方を見直すこと)")
        ok = False

    return ok, max_diff, gathered


def check_no_activation_is_bitexact(budget: int) -> bool:
    """kv_len <= token_budget の間 (QSA 自体が不活性) は、gather の
    on/off でコードパスが完全一致するはずなので、ビット一致を要求する。
    早期救済域 (offset < compress_ratio-1) も同様。

    この域は `_gather_forward` が最初のガード節で ``None`` を返して即座に
    通常経路へ委ねる (`select_blocks` すら呼ばない) ので、タイル幅は
    影響しない --- タイル幅の掃引は `check()` 側 (gather が実際に活性化する
    域) だけで行う。
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

    all_ok = ok1
    ref_gathered: list | None = None
    ref_tile: int | None = None
    for tile in TILE_WIDTHS:
        label = "従来 (タイル無し)" if tile == 0 else f"tile={tile}"
        print(f"--- gather 経路 ({label}) ---")
        ok, max_diff, gathered = check(budget, tile)
        all_ok = all_ok and ok

        if ref_gathered is None:
            ref_gathered, ref_tile = gathered, tile
        else:
            # 段 P1a: タイル幅を変えても最終出力が変わらないことを確認する。
            # タイルは分割の仕方であって計算の中身ではない (選ぶ集合の意味は
            # 同じ) ---和の順序がタイル境界でわずかに変わるので TOL 内の
            # 近似一致を見る (ビット一致は要求しない)。
            cross_diffs = [
                float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
                for a, b in zip(ref_gathered, gathered)
            ]
            cross_max = max(cross_diffs)
            cross_ok = cross_max <= TOL
            all_ok = all_ok and cross_ok
            print(
                f"  tile={ref_tile} との突き合わせ: max|diff|={cross_max:.3e} "
                f"(許容 {TOL:.0e}) -> {'合格' if cross_ok else '不合格'}"
            )
        print()

    print(f"=== 総合判定: {'合格' if all_ok else '不合格'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
