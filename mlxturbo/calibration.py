"""段 C: 較正プロファイルの読み込みと、原始量から閾値を出す式。

`docs/research/KERNEL-PROGRAM.md` の「閾値をマシン非依存にする (段 C)」の
出し口。いま既定に埋まっている閾値はほぼ全部 **M3 Max 1 台で測った値**で、
別の機種では動く。ここは「定数を式と原始量に分ける」ための最小の道具で、

- 原始量を測るのは `tools/calibrate.py` (JSON を書き出す)
- その JSON を読んで式に通すのがここ

という分担。**式に入れてよいのはマシンで決まるものだけ**で、受理率に依存する
閾値 (depth の境界 / `PRIME_WINDOW` / `MLXTURBO_SORT_MIN` /
`MLXTURBO_DRAFT_RERANK`) は原始量からは出ない。それらは `tools/decode_ab.py`
の掃引でしか測り直せない (文書の「マシンでは決まらないもの」の節)。

## 原始量 (JSON の中身)

| 記号 | 中身 | 単位 |
|---|---|---|
| `B` | 連続読みの達成帯域 | バイト/秒 |
| `G` | 飛び飛び読みの効率 (`B` に対する比、head_dim ごと) | 無次元 |
| `L` | カーネル起動 1 回の費用 | 秒 |
| `F` | 密 4bit qmm の上限 | FLOP/秒 |
| `r(rows)` | `gather_qmm` の効率曲線と、その膝の行数 | 無次元 / 行 |
| `D` | グラフ構築 1 層あたりの CPU 時間 | 秒 |

## 読み込み口

環境変数 `MLXTURBO_CALIBRATION` に JSON のパスを入れたときだけ読む。
**指定が無ければ何も変わらない** (M3 Max の実測値がそのまま既定として残り、
使う側が一度だけログに出す)。既定の探索パスを持たないのは、較正が本番の
挙動を黙って変えないため -- 他の knob (`MLXTURBO_GATHER_MAX_RATIO` など) と
同じで、明示的に指定したときだけ効く。
"""

from __future__ import annotations

import json
import os
from typing import Optional

_ENV = "MLXTURBO_CALIBRATION"

_profile: Optional[dict] = None
_loaded = False


def load(path: Optional[str] = None) -> Optional[dict]:
    """較正プロファイルを読む。無ければ `None` (呼ぶ側が既定へ落ちる)。

    一度読んだら覚える。`path` を明示したときは読み直す。
    """
    global _profile, _loaded
    if path is None:
        if _loaded:
            return _profile
        path = os.environ.get(_ENV)
        _loaded = True
        if not path:
            _profile = None
            return None
    else:
        # 明示指定でもここで立てないと、次の describe()/load(None) が
        # 「まだ読んでいない」と誤認して env を読み直し、この呼び出しで
        # 読んだプロファイルを None で上書きしてしまう (D-9)。
        _loaded = True
    try:
        with open(path, "r", encoding="utf-8") as f:
            prof = json.load(f)
    except OSError as e:
        print(f"[mlxturbo] 較正プロファイル {path} が読めない ({e})。"
              " 既定値 (M3 Max の実測) を使う。")
        _profile = None
        return None
    _profile = prof
    return prof


def describe(prof: Optional[dict] = None) -> str:
    """ログ 1 行ぶんの出所。どのマシンで測った値かを黙らせない。"""
    if prof is None:
        prof = load()
    if not prof:
        return "較正プロファイル無し (M3 Max の実測値)"
    m = prof.get("machine", {})
    return (f"較正プロファイル {m.get('chip', '?')} /"
            f" {m.get('measured_at', '?')}")


# ---- 式 (docs/research/KERNEL-PROGRAM.md 段 C の「式」の節) ----------------
#
# **どれも原始量だけで書く。**実測の定数をここに混ぜない (混ぜた瞬間に
# マシン非依存でなくなる)。合わない式は捨てて掃引の値を既定にする、が
# 文書の反転条件。


def gather_max_ratio(B: float, G: float, L: float, kv_len: int,
                     bytes_per_token: float, n_launch: int = 3) -> float:
    """gather の比 `u*` (段 3(b) の「集める価値がある割合の上限」)。

        u* = (1 - n*L*B / (kv*bytes)) / (2 + 1/G)

    密は `kv*bytes/B` を 1 回読む。gather は「集める読み
    `u*kv*bytes/(B*G)` + 書き `u*kv*bytes/B` + sdpa の読み `u*kv*bytes/B`
    + 追加起動 `n*L`」。等号を解くとこの形になる。

    `bytes_per_token` は 1 トークンあたりの KV バイト数
    (= `n_kv_heads * head_dim * 2 バイト * 2 (K と V)`)。`G` は
    その `head_dim` で測った飛び飛び読みの効率。`n_launch` は gather 経路が
    増やすカーネル起動の本数 (K を集める / V を集める / 小マスク で 3)。

    起動項は kv が小さいほど効くので、**短い文脈ほど閾値が前に来る**。

    **返るのは交差点であって、安全側に倒した値ではない。**実測表の 0.20 は
    掃引のゼロ交差 0.23 から 1 割ほど手前に置いてある (攻めて外すと数の多い
    中尺で損をする、という非対称のため。`_vendor/qwen4_exp.py` の注記)。
    較正値を既定に採るなら、同じ幅を取るかどうかを別に決めること。
    """
    kv_bytes = kv_len * bytes_per_token
    if kv_bytes <= 0 or G <= 0:
        return 0.0
    return (1.0 - n_launch * L * B / kv_bytes) / (2.0 + 1.0 / G)


def prefill_group(rows_knee: int, chunk: int = 2048, top_k: int = 10,
                  n_experts: int = 512) -> float:
    """`MLXTURBO_PREFILL_GROUP`: `r(rows)` の曲線の膝そのもの。

    グループ数を上げると 1 エキスパートに集まる行数が比例して増えるので、
    `r` が飽和する行数を `chunk * top_k / n_experts` (グループ 1 のときの
    1 エキスパートあたり行数) で割ればグループ数が出る。**曲線さえ測れば
    式は要らない。**
    """
    per_group = chunk * top_k / n_experts
    if per_group <= 0:
        return 1.0
    return max(1.0, rows_knee / per_group)


def stage_every(L: float, D: float) -> float:
    """`MLXTURBO_STAGE_EVERY`: 層ループの構築時間 `D` と `async_eval` の費用 `L` の比。

    `every` 層ごとに投げると、隠せる泡は `every * D`、払う費用は `L`。
    `every* ≈ max(1, L/D)`。
    """
    if D <= 0:
        return 1.0
    return max(1.0, L / D)


def fast_qmm_m_min(B: float, F: float, flop_per_row: float,
                   weight_bytes: float) -> float:
    """`fast_qmm.M_MIN`: 密の MMA タイルが stock qmv を上回る M。

    stock は M にほぼ比例して重みを読み直すので `M * kv_bytes/B`、MMA は
    `kv_bytes/B + FLOP/F`。等号から `M* = 1/(1 - FLOP*B/(F*kv_bytes))`。
    分母が 0 以下なら「どんな M でも MMA が勝てない」= 交差点が無い
    (`inf` を返す)。

    `flop_per_row` は 1 行あたりの FLOP (= 2*K*N)、`weight_bytes` は
    量子化重みのバイト数。
    """
    if F <= 0 or weight_bytes <= 0:
        return float("inf")
    denom = 1.0 - flop_per_row * B / (F * weight_bytes)
    if denom <= 0:
        return float("inf")
    return 1.0 / denom


# ---- プロファイルから直接引く (呼び出し側の糖衣) --------------------------


def gather_ratio_from_profile(head_dim: int, kv_len: int, n_kv_heads: int = 2,
                              elem_bytes: int = 2,
                              prof: Optional[dict] = None) -> Optional[float]:
    """プロファイルの原始量から `u*` を出す。プロファイルが無ければ `None`。

    `G` は head_dim ごとに測ってある (KV は `(B, n_kv_heads, kv, head_dim)`
    なので kv 軸で集めるときの連続長が `head_dim * elem_bytes` バイトになり、
    これが短いほど `G` が落ちる)。測っていない head_dim では `None` を返す --
    **外挿しない** (機構モデルを 2 点に当てたら係数が揃わないと実測で
    分かっている、`_vendor/qwen4_exp.py` の注記)。
    """
    if prof is None:
        prof = load()
    if not prof:
        return None
    prim = prof.get("primitives", {})
    B = prim.get("B_bytes_per_s")
    L = prim.get("L_seconds")
    gs = prim.get("G_by_head_dim", {})
    G = gs.get(str(head_dim), gs.get(head_dim))
    if not B or L is None or not G:
        return None
    bytes_per_token = n_kv_heads * head_dim * elem_bytes * 2  # K と V
    return gather_max_ratio(B, G, L, kv_len, bytes_per_token)
