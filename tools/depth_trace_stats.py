"""MLXTURBO_DEPTH_TRACE=<path> が吐いた JSON Lines を集計する。

`mlxturbo/spec_flash.py` の `generate_stream` は (MLXTURBO_DEPTH_TRACE=<path>
のときだけ) ラウンドごとに 1 行、次の形の JSON を <path> に追記する::

    {"round": 3, "depth": 4, "hit": 2, "margins": [5.1, 2.0, 0.3, null],
     "pos": 812, "prompt_id": "short:0"}

``round`` はその ``generate_stream`` 呼び出し内でのラウンド番号 (1 始まり)、
``depth`` はそのラウンドで実際に引いた draft の本数、``hit`` は受理された
draft の本数 (0..depth)、``margins`` は draft chain の各段 (0-indexed、
位置 i は「i 個previous draft が当たった前提で引いた (i+1) 個目の draft」)
の top-1/top-2 スコア差 (float。rerank ありのバッチ行など計算できない
ときは null)、``pos`` はそのラウンドの draft を選んだ時点の文脈長、
``prompt_id`` は decode_ab.py が付けた識別子 (無ければ null)。

## 「ラン」の区切り方

同じ ``prompt_id`` でも、decode_ab.py の A/B ループは同じプロンプトに対して
``generate_stream`` を何度も呼び直す (variant ごとに毎回 fresh cache で
run_once)。つまり ``prompt_id`` だけでは「時系列として連続した 1 本の
生成」を特定できない -- 呼び出しをまたぐと ``round`` は 1 に戻る。

ここでは ``prompt_id`` を使わず、**JSON Lines の並び順そのもの**
(= 1 プロセスの中で ``generate_stream`` が実際に書き込んだ順序、必ず
呼び出し単位でひとかたまりに追記される) を見て、``round`` が前の行以下に
戻ったところを新しい「ラン」の境目として切り出す (`_split_runs`)。
これは 1 プロセスでの逐次書き込みである限り常に正しい (並行書き込みは
generate_stream 側で想定されていない)。(b)/(c) のような「直前ラウンド」を
見る集計は、ランをまたがない。

## 使い方

    .venv/bin/python tools/depth_trace_stats.py bench/results/depth-trace-short.jsonl

出力は分析 (a)〜(e) をこの順で標準出力に印字する。中身は探索用の記述統計
であって、depth を選ぶ制御則そのものではない (どの案を本番に入れるかは
別途 in-model A/B で決める -- CLAUDE.md の計測の作法どおり)。
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict

# T(S) = COST_T1 + COST_DT * S (ms)。プロンプトの指示どおり固定値を使う
# (mlxturbo/spec_flash.py._depth_cost_params の実測込みモデルとは別物 --
# ここは 3 方式の相対比較のための固定の物差し)。
COST_T1 = 24.0
COST_DT = 7.0


def round_cost_ms(depth: int) -> float:
    return COST_T1 + COST_DT * depth


# ---------------------------------------------------------------------------
# 読み込み / ラン分割
# ---------------------------------------------------------------------------


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: {path}:{line_no} を JSON として読めない ({e})、"
                      " スキップする", file=sys.stderr)
    return records


def split_runs(records: list[dict]) -> list[list[dict]]:
    """``round`` が前の行以下に戻ったところで新しいランを始める。

    モジュール docstring の「ラン」の説明を参照。空リストなら空リストを返す。
    """
    runs: list[list[dict]] = []
    cur: list[dict] = []
    prev_round = None
    for rec in records:
        r = rec.get("round")
        if cur and (prev_round is None or r is None or r <= prev_round):
            runs.append(cur)
            cur = []
        cur.append(rec)
        prev_round = r
    if cur:
        runs.append(cur)
    return runs


# ---------------------------------------------------------------------------
# (a) 位置別の受理率
# ---------------------------------------------------------------------------


def position_acceptance(records: list[dict]) -> list[tuple[int, int, float | None]]:
    """位置 i (0-indexed) ごとに (eligible数, accepted数, 受理率) を返す。

    「eligible」= そのラウンドの depth が i より大きい (位置 i まで実際に
    draft が引かれた)。「accepted」= hit が i より大きい (位置 i の draft
    が採用された、DepthController.a[i] と同じ定義)。
    """
    max_depth = max((rec.get("depth") or 0) for rec in records) if records else 0
    out = []
    for i in range(max_depth):
        eligible = [rec for rec in records if (rec.get("depth") or 0) > i]
        accepted = [rec for rec in eligible if (rec.get("hit") or 0) > i]
        rate = len(accepted) / len(eligible) if eligible else None
        out.append((len(eligible), len(accepted), rate))
    return out


# ---------------------------------------------------------------------------
# (b) 直前ラウンド全採用の効果
# ---------------------------------------------------------------------------


def full_accept(rec: dict) -> bool:
    depth = rec.get("depth") or 0
    return bool(depth) and (rec.get("hit") or 0) == depth


def prev_full_accept_effect(runs: list[list[dict]], k: int = 3) -> dict:
    """P(hit>=k | 直前ラウンドが全採用) と P(hit>=k) (無条件、直前ラウンドが
    定義できるものだけ) を返す。
    """
    cond_yes = []  # 直前ラウンド全採用のときの今ラウンドの hit>=k フラグ
    cond_no = []
    all_eligible = []  # 直前ラウンドが存在するラウンドの hit>=k フラグ
    for run in runs:
        for t in range(1, len(run)):
            cur, prev = run[t], run[t - 1]
            flag = (cur.get("hit") or 0) >= k
            all_eligible.append(flag)
            if full_accept(prev):
                cond_yes.append(flag)
            else:
                cond_no.append(flag)

    def rate(xs):
        return sum(xs) / len(xs) if xs else None

    p_cond = rate(cond_yes)
    p_marg = rate(all_eligible)
    diff = (p_cond - p_marg) if (p_cond is not None and p_marg is not None) else None
    return {
        "k": k,
        "n_cond_yes": len(cond_yes), "p_hit_ge_k_given_prev_full": p_cond,
        "n_cond_no": len(cond_no), "p_hit_ge_k_given_prev_not_full": rate(cond_no),
        "n_marginal": len(all_eligible), "p_hit_ge_k_marginal": p_marg,
        "diff": diff,
    }


# ---------------------------------------------------------------------------
# (c) 直近 k=3 ラウンドの全採用ビット列ごとの受理率
# ---------------------------------------------------------------------------


def history_bucket_stats(runs: list[list[dict]], k: int = 3) -> dict:
    """直近 k ラウンド (古い→新しいの順のビット列、1=全採用) の 2**k 通り
    ごとに、その次のラウンドの P(hit>=2)/P(hit>=3) と件数を返す。

    キーはビット列を文字列化したもの (例 "011" = 3 ラウンド前が不完全、
    2 ラウンド前と直前が全採用)。
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for t in range(k, len(run)):
            hist = tuple(full_accept(run[t - k + j]) for j in range(k))
            key = "".join("1" if b else "0" for b in hist)
            buckets[key].append(run[t])

    out = {}
    for key, recs in buckets.items():
        n = len(recs)
        p2 = sum(1 for r in recs if (r.get("hit") or 0) >= 2) / n if n else None
        p3 = sum(1 for r in recs if (r.get("hit") or 0) >= 3) / n if n else None
        out[key] = {"n": n, "p_hit_ge_2": p2, "p_hit_ge_3": p3}
    return out


# ---------------------------------------------------------------------------
# (d) 位置 1 の draft マージンと受理の AUC (順位和、sklearn 不使用)
# ---------------------------------------------------------------------------


def auc_rank_sum(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """P(正例のスコア > 負例のスコア) + 0.5*P(同点) を順位和 (Mann-Whitney U)
    で計算する。どちらかのクラスが空なら None。
    """
    n1, n2 = len(scores_pos), len(scores_neg)
    if n1 == 0 or n2 == 0:
        return None
    combined = sorted(
        [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg],
        key=lambda t: t[0],
    )
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # 1-based、[i, j] 区間の平均順位
        for idx in range(i, j + 1):
            ranks[idx] = avg_rank
        i = j + 1
    r1 = sum(r for r, (_, label) in zip(ranks, combined) if label == 1)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return u1 / (n1 * n2)


def margin_auc_at_position(records: list[dict], position: int) -> dict:
    pos_scores, neg_scores = [], []
    for rec in records:
        depth = rec.get("depth") or 0
        margins = rec.get("margins")
        if depth <= position or not margins or len(margins) <= position:
            continue
        m = margins[position]
        if m is None:
            continue
        accepted = (rec.get("hit") or 0) > position
        (pos_scores if accepted else neg_scores).append(m)
    auc = auc_rank_sum(pos_scores, neg_scores)
    return {
        "position": position,
        "n_accepted": len(pos_scores), "n_rejected": len(neg_scores),
        "mean_margin_accepted": statistics.fmean(pos_scores) if pos_scores else None,
        "mean_margin_rejected": statistics.fmean(neg_scores) if neg_scores else None,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# (f) D2 判定指標: 信号 a (直前 k ラウンド全採用条件付きの P(hit>=3)) と
#     信号 b (位置 1 マージン下位カバレッジでの棄却率)
#     -- `docs/research/IDEAS-2026-09-03.md` の Challenge 表の判定線
#     (信号 a: 0.5、信号 b: margin 下位 20% で棄却率 < 0.7 なら畳む) 用。
# ---------------------------------------------------------------------------


def full_history_hit3_rate(runs: list[list[dict]], k: int = 3) -> dict:
    """直前 k ラウンドが全部全採用だったときの P(hit>=3) と件数 (信号 a)。

    (c) と同じビット列バケット化 (`history_bucket_stats`) を使い、全ビット 1
    (= 直前 k ラウンドが全部全採用) のバケットだけを取り出す。
    """
    key = "1" * k
    st = history_bucket_stats(runs, k=k).get(key, {"n": 0, "p_hit_ge_3": None})
    return {"k": k, "n": st["n"], "p_hit_ge_3": st["p_hit_ge_3"]}


def margin_coverage_precision(
    records: list[dict], position: int, coverage_fracs: tuple[float, ...]
) -> list[dict]:
    """位置 `position` のマージンを昇順 (自信が低い順) に並べ、下位
    `coverage_fracs` (被覆率) における「その draft が棄却だった割合」
    (= 精度、信号 b) を返す。

    抽出条件は `margin_auc_at_position` と同じ (depth > position かつ
    margins[position] が数値)。母集団が空なら各カバレッジとも n=0 で返す。
    """
    pairs = []  # (margin, accepted)
    for rec in records:
        depth = rec.get("depth") or 0
        margins = rec.get("margins")
        if depth <= position or not margins or len(margins) <= position:
            continue
        m = margins[position]
        if m is None:
            continue
        accepted = (rec.get("hit") or 0) > position
        pairs.append((m, accepted))

    if not pairs:
        return [{"coverage": f, "n": 0, "threshold_margin": None, "reject_rate": None}
                for f in coverage_fracs]

    pairs.sort(key=lambda t: t[0])
    n_total = len(pairs)
    out = []
    for f in coverage_fracs:
        n_cov = min(max(1, round(f * n_total)), n_total)
        subset = pairs[:n_cov]
        reject_rate = sum(1 for _, acc in subset if not acc) / n_cov
        out.append({
            "coverage": f, "n": n_cov,
            "threshold_margin": subset[-1][0],
            "reject_rate": reject_rate,
        })
    return out


# ---------------------------------------------------------------------------
# (e) 費用シミュレーション: 定常 / 履歴表 / マージン閾値
# ---------------------------------------------------------------------------


def _simulate(depths_and_hits: list[tuple[int, int]]) -> dict:
    """[(選んだ depth, その回の実際の hit)] から ms/tok を出す。

    ``hit`` はラウンドの実測 (ある depth D で draft した結果の受理数)。
    選んだ depth d <= D なら「d しか draft しなかった場合の受理数」は
    min(hit, d) (draft の連鎖は前段の出力だけに依存する決定的な処理なので、
    後段を増やしても前段の draft トークンは変わらない --- _draft_chain の
    構造そのもの)。d > D (集めたデータより深く選ぼうとした) は観測が無い
    ので、その回は D にクリップして評価する (過大評価しない側に倒す)。
    """
    total_tokens = 0.0
    total_ms = 0.0
    for chosen_d, (observed_d, hit) in depths_and_hits:
        d = min(chosen_d, observed_d) if observed_d else chosen_d
        tokens = min(hit, d) + 1
        total_tokens += tokens
        total_ms += round_cost_ms(d)
    ms_per_tok = total_ms / total_tokens if total_tokens else None
    return {"n_rounds": len(depths_and_hits), "total_tokens": total_tokens,
            "total_ms": total_ms, "ms_per_tok": ms_per_tok}


def simulate_static(records: list[dict], depth: int = 2) -> dict:
    pairs = [(depth, (rec.get("depth") or 0, rec.get("hit") or 0)) for rec in records
              if rec.get("depth")]
    return _simulate(pairs)


def simulate_history_table(runs: list[list[dict]], k: int = 3,
                            max_depth: int | None = None,
                            fallback_depth: int = 2) -> dict:
    """(c) と同じ履歴バケットごとに、そのバケットの中で経験的に
    E[min(hit,d)] が最大の E(d)/T(d) を与える d を選ぶ (in-sample --
    同じデータでバケット統計を作って同じデータを評価するので楽観に寄る。
    「履歴が使えそうか」の一次判定用と割り切る)。履歴が 3 ラウンド分
    揃わないラン先頭は ``fallback_depth`` を使う。
    """
    if max_depth is None:
        max_depth = max((rec.get("depth") or 0) for run in runs for rec in run) or 1

    # バケットごとに depth 候補の E[min(hit,d)] を集める
    bucket_hits: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        for t in range(k, len(run)):
            hist = tuple(full_accept(run[t - k + j]) for j in range(k))
            key = "".join("1" if b else "0" for b in hist)
            bucket_hits[key].append(run[t].get("hit") or 0)

    def best_depth(hits: list[int]) -> int:
        best_d, best_score = fallback_depth, -1.0
        for d in range(1, max_depth + 1):
            if not hits:
                break
            e_tok = 1.0 + statistics.fmean(min(h, d) for h in hits)
            score = e_tok / round_cost_ms(d)
            if score > best_score:
                best_d, best_score = d, score
        return best_d

    bucket_depth = {key: best_depth(hits) for key, hits in bucket_hits.items()}

    pairs = []
    for run in runs:
        for t, rec in enumerate(run):
            if not rec.get("depth"):
                continue
            if t < k:
                chosen = fallback_depth
            else:
                hist = tuple(full_accept(run[t - k + j]) for j in range(k))
                key = "".join("1" if b else "0" for b in hist)
                chosen = bucket_depth.get(key, fallback_depth)
            pairs.append((chosen, (rec.get("depth"), rec.get("hit") or 0)))
    result = _simulate(pairs)
    result["bucket_depth"] = bucket_depth
    return result


def simulate_margin_threshold(runs: list[list[dict]],
                               thresholds: list[float]) -> list[dict]:
    """カスケード規則: margins[0] >= tau なら depth>=2 まで、margins[1] >= tau
    なら depth>=3 まで、margins[2] >= tau なら depth>=4 まで伸ばす (最初の
    1 手は必ず draft する)。static depth 4 で採取したトレースなら
    margins[0..3] が全ラウンド揃っているので、どの tau でも retrospective
    にシミュレートできる (簡易マージン AUCの (d) と同じ理由付け)。
    候補 tau ごとの結果をリストで返す (呼び手が最小 ms/tok を選ぶ)。
    """
    out = []
    for tau in thresholds:
        pairs = []
        for run in runs:
            for rec in run:
                depth = rec.get("depth") or 0
                margins = rec.get("margins") or []
                if not depth:
                    continue
                chosen = 1
                for i in range(depth - 1):
                    m = margins[i] if i < len(margins) else None
                    if m is not None and m >= tau:
                        chosen = i + 2
                    else:
                        break
                pairs.append((chosen, (depth, rec.get("hit") or 0)))
        sim = _simulate(pairs)
        sim["tau"] = tau
        out.append(sim)
    return out


# ---------------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------------


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl")
    ap.add_argument("--k", type=int, default=3, help="直近何ラウンドの履歴を見るか (既定 3)")
    ap.add_argument("--static-depth", type=int, default=2,
                    help="(e) の「定常」policy が使う depth (既定 2 = 現行の静的既定)")
    ap.add_argument("--exclude-warmup", action="store_true",
                    help="prompt_id が warmup: で始まるレコードを (a)〜(e) から除く")
    args = ap.parse_args()

    records = load_records(args.jsonl)
    if args.exclude_warmup:
        records = [r for r in records
                   if not str(r.get("prompt_id") or "").startswith("warmup:")]
    if not records:
        print("レコードが 0 件 (jsonl が空、または全部 warmup で除外された)")
        return 1

    runs = split_runs(records)
    depths = sorted({rec.get("depth") for rec in records if rec.get("depth")})
    print(f"records={len(records)}  runs={len(runs)}  depth={depths}")
    print()

    print("=== (a) 位置別の受理率 ===")
    for i, (n_elig, n_acc, rate) in enumerate(position_acceptance(records)):
        print(f"  position {i}: n={n_elig:5d}  accepted={n_acc:5d}  rate={_fmt(rate)}")
    print()

    print(f"=== (b) 直前ラウンド全採用の効果 (hit>=3) ===")
    b = prev_full_accept_effect(runs, k=3)
    print(f"  P(hit>=3 | 直前全採用)  = {_fmt(b['p_hit_ge_k_given_prev_full'])}"
          f"  (n={b['n_cond_yes']})")
    print(f"  P(hit>=3 | 直前不完全) = {_fmt(b['p_hit_ge_k_given_prev_not_full'])}"
          f"  (n={b['n_cond_no']})")
    print(f"  P(hit>=3) (周辺)       = {_fmt(b['p_hit_ge_k_marginal'])}"
          f"  (n={b['n_marginal']})")
    print(f"  差 (条件付き - 周辺)   = {_fmt(b['diff'])}")
    print()

    print(f"=== (c) 直近 {args.k} ラウンドの全採用ビット列 (旧→新、1=全採用) ===")
    hb = history_bucket_stats(runs, k=args.k)
    for bits in itertools.product("01", repeat=args.k):
        key = "".join(bits)
        st = hb.get(key, {"n": 0, "p_hit_ge_2": None, "p_hit_ge_3": None})
        print(f"  {key}: n={st['n']:5d}  P(hit>=2)={_fmt(st['p_hit_ge_2'])}"
              f"  P(hit>=3)={_fmt(st['p_hit_ge_3'])}")
    print()

    print("=== (d) 位置 1 の draft マージン と 受理 の AUC ===")
    d = margin_auc_at_position(records, position=1)
    print(f"  n_accepted={d['n_accepted']}  n_rejected={d['n_rejected']}")
    print(f"  mean margin | accepted = {_fmt(d['mean_margin_accepted'])}")
    print(f"  mean margin | rejected = {_fmt(d['mean_margin_rejected'])}")
    print(f"  AUC = {_fmt(d['auc'])}  (0.5 = マージンに情報なし)")
    print()

    print("=== (e) 費用シミュレーション T(S)=24+7*S ms ===")
    static = simulate_static(records, depth=args.static_depth)
    print(f"  定常 (depth={args.static_depth}, 現行既定): "
          f"ms/tok={_fmt(static['ms_per_tok'])}"
          f"  (n_rounds={static['n_rounds']}, tokens={static['total_tokens']:.0f})")
    hist_sim = simulate_history_table(runs, k=args.k, fallback_depth=args.static_depth)
    print(f"  履歴表 ({args.k} ラウンドのビット列 -> 経験的最良 depth、in-sample): "
          f"ms/tok={_fmt(hist_sim['ms_per_tok'])}"
          f"  (n_rounds={hist_sim['n_rounds']}, tokens={hist_sim['total_tokens']:.0f})")
    print(f"    bucket -> depth: {hist_sim['bucket_depth']}")
    all_margins = [m for rec in records for m in (rec.get("margins") or [])
                   if m is not None]
    if all_margins:
        qs = sorted(all_margins)
        n = len(qs)
        thresholds = sorted({qs[int(n * p)] for p in
                             (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
                             if int(n * p) < n})
        margin_sims = simulate_margin_threshold(runs, thresholds)
        best = min((s for s in margin_sims if s["ms_per_tok"] is not None),
                   key=lambda s: s["ms_per_tok"], default=None)
        print("  マージン閾値 (カスケード、候補 tau ごとの ms/tok):")
        for s in margin_sims:
            marker = "  <- best" if best is not None and s is best else ""
            print(f"    tau={s['tau']:.3f}: ms/tok={_fmt(s['ms_per_tok'])}"
                  f"  tokens={s['total_tokens']:.0f}{marker}")
    else:
        print("  マージン閾値: margins が 1 件も無い (want_margin が発火して"
              "いない --- trace_top2=True で _draft_chain が呼ばれていない"
              "か、rerank のバッチ行だった可能性)")
    print()

    print("=== (f) D2 判定指標 (IDEAS-2026-09-03.md Challenge 表) ===")
    sig_a = full_history_hit3_rate(runs, k=3)
    judge_a = "n/a"
    if sig_a["p_hit_ge_3"] is not None:
        judge_a = "OK (>=0.500)" if sig_a["p_hit_ge_3"] >= 0.5 else "NG (<0.500)"
    print(f"  信号a: P(hit>=3 | 直前3ラウンド全採用) = {_fmt(sig_a['p_hit_ge_3'])}"
          f"  (n={sig_a['n']})  判定線=0.500  {judge_a}")
    print()
    print("  信号b: 位置1マージン下位カバレッジでの棄却率 (判定線=0.700)")
    for cov in margin_coverage_precision(records, position=1,
                                          coverage_fracs=(0.1, 0.2, 0.3)):
        judge_b = "n/a"
        if cov["reject_rate"] is not None:
            judge_b = "OK (>=0.700)" if cov["reject_rate"] >= 0.7 else "NG (<0.700)"
        print(f"    coverage={cov['coverage'] * 100:4.0f}%  n={cov['n']:5d}"
              f"  threshold_margin={_fmt(cov['threshold_margin'])}"
              f"  reject_rate={_fmt(cov['reject_rate'])}  {judge_b}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
