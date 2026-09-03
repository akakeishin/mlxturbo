#!/usr/bin/env python3
"""オフライン集計専用 (GPU 不使用)。案 D3 (文脈 n-gram の SAM draft) の
見積もりツール。

入力: `bench/results/saveout-17k.json` / `saveout-50k.json`
(`tools/decode_ab.py --save-out` の出力。rows[] に `out` (生成トークン id の
全列) と `prompt_ids` (プロンプトのトークン id 列) を持つ行だけを使う)。

手順は依頼メッセージのとおり:
  1. 各 row について、生成の各位置 t で SAM に [0, t) (プロンプト全部 + 生成
     t 個) を流した状態で `draft(max_len=6, min_len=2)` を引き、一致長 L と、
     返った継続 6 トークンが真の続き (out[t:t+6]) と先頭から何個連続一致
     したか (R) を集計する。
  2. 一致長のバケット (2, 3, 4, 5+) 別に、k 番目まで累積で受理される確率
     P(R>=k) を出す。発火率は L>=3 の位置の割合。
  3. 費用表 T(S) = 24 + 7*S ms (S = 検証幅 = draft 本数 + 1) と現行
     (17k+: 40 ms/round, tok/round 1.8) を使って、「一致長 L が閾値 k 以上の
     位置では SAM draft を幅 k で出し (検証幅 S=k+1)、それ以外は現行」と
     したときの ms/tok を、位置ごとの発火率・平均受理数で重み付けして
     見積もる (k=2..5 を掃引)。k=5 (S=6) だけ T(S) の線形式に加えて
     T(6)=72ms (challenger 指摘の非線形値) でも出す。

このファイルは新規に置くだけで commit しない (依頼メッセージの指示)。
mlxturbo パッケージの `__init__` は import せず (`_arch_registry.install()`
などの副作用があるため)、`mlxturbo/sam.py` を単体ファイルとして
importlib で読み込む。依存は標準ライブラリのみ、GPU には一切触れない。
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM_PATH = REPO_ROOT / "mlxturbo" / "sam.py"

DEFAULT_FILES = {
    "17k": REPO_ROOT / "bench" / "results" / "saveout-17k.json",
    "50k": REPO_ROOT / "bench" / "results" / "saveout-50k.json",
}

MAX_LEN = 6
MIN_LEN = 2
SWEEP_K = (2, 3, 4, 5)

# 現行 (依頼メッセージで与えられた値。17k / 50k 共通の想定として使う)
BASELINE_MS_PER_ROUND = 40.0
BASELINE_TOK_PER_ROUND = 1.8
BASELINE_MS_PER_TOK = BASELINE_MS_PER_ROUND / BASELINE_TOK_PER_ROUND

FOLD_THRESHOLD_PCT = -3.0  # これより改善が弱ければ「畳む」


def _load_sam_module():
    spec = importlib.util.spec_from_file_location("_sam_standalone", SAM_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def cost_T(S: int) -> float:
    return 24.0 + 7.0 * S


def leading_match_run(cont: list[int], true_seg: list[int]) -> int:
    n = min(len(cont), len(true_seg))
    r = 0
    for i in range(n):
        if cont[i] == true_seg[i]:
            r += 1
        else:
            break
    return r


def analyze_row(SuffixAutomaton, prompt_ids: list[int], out: list[int]):
    """1 row 分の per-position 統計を返す。

    戻り値は position ごとの (L, R_or_None) のリスト。R は draft (cont) が
    出た位置だけ int、出なかった位置 (L<min_len、または継続が尽きている)
    は None。
    """

    sam = SuffixAutomaton()
    sam.extend_all(prompt_ids)

    n = len(out)
    records = []
    for t in range(n):
        L, _end = sam.longest_match()
        cont = sam.draft(MAX_LEN, MIN_LEN)
        if cont is None:
            records.append((L, None))
        else:
            true_seg = out[t : t + MAX_LEN]
            r = leading_match_run(cont, true_seg)
            records.append((L, r))
        sam.extend(out[t])
    return records


def bucket_label(L: int) -> str:
    if L == 2:
        return "2"
    if L == 3:
        return "3"
    if L == 4:
        return "4"
    return "5+"


def load_rows(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for row in data:
        if "out" in row and "prompt_ids" in row:
            rows.append(row)
    return rows


def fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def main() -> int:
    sam_mod = _load_sam_module()
    SuffixAutomaton = sam_mod.SuffixAutomaton

    any_found = False
    for label, path in DEFAULT_FILES.items():
        if not path.exists():
            print(f"# {label}: {path} が無いのでスキップ")
            continue
        any_found = True
        rows = load_rows(path)
        if not rows:
            print(f"# {label}: {path} に out/prompt_ids を持つ行が無い")
            continue

        print(f"\n{'=' * 70}")
        print(f"# {label} ({path.name})  行数={len(rows)}")

        # decode_ab の ABBA (A,B,B,A) 繰り返し測定は、prompt_ids/out が
        # 完全に同一な行を複数回吐く (貪欲デコードで決定的なため、同じ
        # トークン列を計測条件を変えて何度も測っているだけ)。SAM 解析は
        # 生成トークン列にしか依存しないので、重複行を集計に混ぜても比率は
        # 変わらないが、4 重に再計算するのは無駄なので de-dup する。
        seen_keys: dict[tuple, list] = {}
        for r in rows:
            key = (tuple(r["prompt_ids"]), tuple(r["out"]))
            seen_keys.setdefault(key, []).append(r)
        unique_rows = [group[0] for group in seen_keys.values()]
        print(f"  一意な (prompt, 生成列) = {len(unique_rows)} 本"
              f" (行数 {len(rows)} 本中、ABBA 重複を除いた実体)")
        for key, group in seen_keys.items():
            r0 = group[0]
            variants = [g.get("variant") for g in group]
            print(f"  - kind={r0.get('kind')} ctx={r0.get('ctx')} "
                  f"n_out={len(r0['out'])}  重複={len(group)}本 "
                  f"(variant={variants})")

        all_records = []  # (L, R_or_None) をプールした全 unique row 分
        per_row_records = []
        for r in unique_rows:
            recs = analyze_row(SuffixAutomaton, r["prompt_ids"], r["out"])
            per_row_records.append((r, recs))
            all_records.extend(recs)

        n_total = len(all_records)
        fire3 = sum(1 for L, _ in all_records if L >= 3)
        print(f"\n## 発火率 (一致長 L >= 3)  n={n_total}")
        print(f"  {fire3}/{n_total} = {fire3 / n_total * 100:.1f}%")

        # ---- バケット別の累積受理率 ----------------------------------
        bucket_records: dict[str, list[int]] = {"2": [], "3": [], "4": [], "5+": []}
        for L, R in all_records:
            if R is None:
                continue
            bucket_records[bucket_label(L)].append(R)

        print("\n## バケット別 累積受理確率 P(R>=k)  (draft が出た位置のみ)")
        header = "  bucket |     n | " + " | ".join(f"k>={k}" for k in range(1, MAX_LEN + 1))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for b in ("2", "3", "4", "5+"):
            rs = bucket_records[b]
            nb = len(rs)
            if nb == 0:
                print(f"  {b:>6} | {0:>5} | " + " | ".join("  -  " for _ in range(MAX_LEN)))
                continue
            cells = []
            for k in range(1, MAX_LEN + 1):
                p = sum(1 for r in rs if r >= k) / nb * 100
                cells.append(f"{p:5.1f}")
            print(f"  {b:>6} | {nb:>5} | " + " | ".join(cells))

        # ---- k 掃引の ms/tok 予測 --------------------------------------
        print("\n## k 掃引: ms/tok 予測 (現行 40ms/round, tok/round 1.8 "
              f"-> {BASELINE_MS_PER_TOK:.2f} ms/tok)")
        print("  k(幅) | S=k+1 | 発火率 p_k | 平均受理 a_k | tokens/round | "
              "ms/round | 予測ms/tok | 現行比 | 判定")
        for k in SWEEP_K:
            fires = [(L, R) for L, R in all_records if R is not None and L >= k]
            p_k = len(fires) / n_total if n_total else 0.0
            if fires:
                a_k = statistics.mean(min(R, k) for _, R in fires)
            else:
                a_k = 0.0
            S = k + 1

            def _predict(T_S: float):
                tok_per_round = p_k * (a_k + 1) + (1 - p_k) * BASELINE_TOK_PER_ROUND
                ms_per_round = p_k * T_S + (1 - p_k) * BASELINE_MS_PER_ROUND
                ms_per_tok = ms_per_round / tok_per_round
                pct = (ms_per_tok - BASELINE_MS_PER_TOK) / BASELINE_MS_PER_TOK * 100
                return tok_per_round, ms_per_round, ms_per_tok, pct

            tok_pr, ms_pr, ms_tok, pct = _predict(cost_T(S))
            verdict = "畳む" if pct > FOLD_THRESHOLD_PCT else "残す"
            print(f"  {k:>5} | {S:>5} | {p_k * 100:8.1f}% | {a_k:11.2f} | "
                  f"{tok_pr:11.2f} | {ms_pr:7.2f} | {ms_tok:9.2f} | "
                  f"{fmt_pct(pct):>7} | {verdict} (T(S)線形 {cost_T(S):.0f}ms)")

            if k == 5:
                tok_pr2, ms_pr2, ms_tok2, pct2 = _predict(72.0)
                verdict2 = "畳む" if pct2 > FOLD_THRESHOLD_PCT else "残す"
                print(f"  {k:>5} | {S:>5} | {p_k * 100:8.1f}% | {a_k:11.2f} | "
                      f"{tok_pr2:11.2f} | {ms_pr2:7.2f} | {ms_tok2:9.2f} | "
                      f"{fmt_pct(pct2):>7} | {verdict2} (T(6)=72ms 非線形)")

        # 行ごとの内訳 (row が複数あるときの参考、A/B の再現性チェック用)
        if len(unique_rows) > 1:
            print("\n## 一意な生成列ごとの発火率 (参考、プロンプト間のばらつき)")
            for r, recs in per_row_records:
                nt = len(recs)
                f3 = sum(1 for L, _ in recs if L >= 3)
                print(f"  ctx={r.get('ctx')}: 発火率(L>=3) "
                      f"{f3}/{nt} = {f3 / nt * 100:.1f}%")

    if not any_found:
        print("17k/50k どちらの saveout ファイルも見つからなかった。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
