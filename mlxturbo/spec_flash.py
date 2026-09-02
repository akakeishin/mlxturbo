"""The foundation of Qwen3.8-Flash-Next speculative decoding: state capture and rollback.

`mlxturbo/spec.py` is tightly coupled to the 27B (qwen3_5) configuration, so we
keep this separately.

## Why capture is necessary

Speculation is "mix in a draft, verify it in one batch, and throw it away if it
misses". But 36 of Flash-Next's layers are GatedDeltaNet, and **the recurrent
state cannot be rolled back**. It is not something you can fix by cutting off the
tail as with KV, so we retain **the state immediately after finishing each
position** of the verification forward and adopt as much of it as was accepted.

There are 4 things that need rollback:

| Target | Layers | How to roll back |
|---|---|---|
| GDN recurrent state `cache[1]` | 36 | `states_all[:, keep-1]` (captured) |
| GDN conv window `cache[0]` | 36 | `conv_input[:, keep : keep+K-1]` (captured) |
| PLE conv window `cache[2]` | 1 | same as above |
| full attention KV and indexer | 12 | `KVCache.trim()` and truncating keys |
| n-gram context `cache[3]` | 1 | save the pre-forward value |

`ArraysCache.advance` only touches `lengths`/`left_padding` (both None for a
single sequence), so no offset rollback is needed on the GDN side.

## Design: do not replace the main model

We temporarily replace only `GatedDeltaNet.__call__` and `PLELayer._short_conv`,
and let everything else go through the main implementation as it is. Because we
**do not transcribe the forward path**, there is little room for the capturing
version and the plain version to diverge (transcribe it and they will drift
somewhere, without fail).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# mlxturbo-serve wiring (added 2026-08-29): share the constant used to chunk
# prefill at the same width as FallbackRunner/SpecEngine (see the
# PREFILL_STEP_SIZE docstring in mlxturbo/spec.py -- for the same reason, that a
# width differing per path makes the output diverge even for the same prompt, we
# reuse the value from that one place here too).
#
# thinking support (added 2026-08-29): partial restoration via session-reuse
# checkpoints also reuses the same machinery as the spec.py side
# (ChatSession.checkpoints / _prefill_hidden / CHECKPOINT_RETENTION). The state
# of the GDN/PLE/n-gram layers that cannot be rolled back rides on the same
# ArraysCache.state (list) on either path, so spec.py's
# snapshot_untrimmable_caches/restore_untrimmable_caches are model-independent
# (they look only at caches' is_trimmable()/state) -- we reuse them as they are.
# spec.py is only read, never modified.
from .spec import CHECKPOINT_RETENTION, PREFILL_STEP_SIZE, snapshot_untrimmable_caches
from .prefill_common import split_and_checkpoint_tail
from . import arch as _archmod
from .arch import qwen4_arch as _arch
from .kernels import _fire


# How many trailing prompt positions are fed to the MTP head before decoding
# starts (see FlashSpecEngine._prime_draft_cache). Acceptance comes from recent
# context, so a window buys most of the gain while keeping the cost independent
# of prompt length -- this model's ceiling is 262144 tokens, where priming the
# whole prompt would cost both minutes of TTFT and gigabytes of retained hyper
# state. Measured at 2048: 32k prompt, acceptance 0.574 -> 0.827.
PRIME_WINDOW = 2048
# Trailing prefill chunks whose hyper state generate_stream retains. Two chunks
# of PREFILL_STEP_SIZE always cover PRIME_WINDOW+1 positions.
HYPER_KEEP_CHUNKS = 2

# ドラフトを 1 ラウンドで何トークン引くか。1 = ヘッドを 1 回だけ回す。
# 2 以上ではヘッド自身の hyper 状態を次段に渡して連鎖させる (_draft_chain)。
# 受理率 r で depth d なら 1 ラウンドの期待トークン数は (1-r^(d+1))/(1-r) で、
# d=1 は r=0.83 でも 1.83 が上限になる。
#
# 既定 2 (2026-08-31 の掃引、ddalcu 一律 4bit・複数プロンプト x 512 トークン):
# depth 1/2/3/4 の短文脈 decode は 46.8 / 53.1-53.6 / 47-48 / 45.1。
# depth 3 は 3 本目の的中が verify の位置追加費用 (~5ms) を償却しない。
MTP_DEPTH = 2

# ここを越えたら depth を 1 に落とす。深くすると検証フォワードの位置数が増え、
# その費用は文脈長に比例する (indexer のブロック選択が長いキャッシュ全体に
# 対して位置ごとに走るため) ので、長文では利得を食い潰して逆に遅くなる。
#
# v-l / M3 Max のサーバー実測 (tok/s、温まった状態):
#
#   文脈    depth 1   depth 2   depth 3
#    1k      45.4       —        51.5
#    4k      37.0       —        45.5
#    6k      36.3      39.7      45.7
#    8k      38.1      36.6      35.4     <- ここで反転済み
#   12k      37.3      29.8      33.3
#   48k      30.8       —        17.6
#
# 反転は 6k と 8k の間。勝っている側の内側を採って 6144 (= 3 チャンク) に置く。
#
# 2026-09-01 再較正 (1): 6144 は sdpa の 32 行の壁 (qL>=3 が未融合経路に
# 落ちる) を避けるための遺物と判断し、いったん実質無効 (262144) に置いた。
#
# 2026-09-01 再較正 (2、こちらが現行): 複数プロンプト x 512 の回文順掃引で
# 測り直したところ、**長文では depth 1 が勝つ**とはっきり出た
# (tools/decode_ab.py --knob depth、bench/results/depth-*.json)。
# ms/token、depth 2 を基準にした差:
#
#   文脈    depth 1   depth 2   depth 3
#    65 tok  +5.6%     基準     +11.7%
#   900      +5.1%     基準      +2.6%
#   2.6k     -3.3%     基準      +9.9%
#   4k       -3.1%     基準      +8.2%
#   17k     -10.9%     基準      +3.2%
#
# 反転は 900 と 2.6k の間にあり、そこには QSA が働き始める境界
# (indexer_budget = 2048) がある。機構としても符合する — QSA が活性だと
# 検証フォワードに 1 位置足す費用にブロック選択と疎マスクが乗り、受理が
# 増えるぶんを償却しなくなる。tok/round は深いほど上がり続ける (17k で
# 1.64 / 1.97 / 2.33) のに壁時計は逆、というのがこの現象の形。
#
# よって既定はモデルの indexer_budget にする (下の _depth_ctx_limit)。
# 定数を持たない族では境界が無いので切り替えない。env で上書きできる。
DEPTH_CONTEXT_LIMIT = int(os.environ.get("MLXTURBO_DEPTH_CTX_LIMIT", "0") or 0) or None
_DEPTH_CTX_LIMIT_FALLBACK = 262144


# バッチの負荷連動 depth。既定 off (未測)。choose_depth の注記を参照。
_BATCH_DEPTH_ADAPT = os.environ.get("MLXTURBO_BATCH_DEPTH_ADAPT") == "1"


def _logsoftmax_rows(logits: mx.array, n: int) -> list:
    """``logits`` の先頭 n 位置を log 確率にして 1 行ずつ返す。

    形は ``(1, V)`` (prefill 末尾) と ``(1, S, V)`` (検証フォワード) の両方を
    受ける。正規化の仕方は ``mlxturbo/runner.py`` の ``_logprob_entry`` が
    前提にしているもの (logits - logsumexp) と同じにしてある -- 投機経路と
    降格経路で同じリクエストの答えが変わるのが一番悪い。
    """
    x = logits.astype(mx.float32)
    if x.ndim == 2:
        x = x[:, None]
    x = x[0, :n]
    return list(x - mx.logsumexp(x, axis=-1, keepdims=True))


def choose_depth(
    pos: int,
    depth: int,
    ctx_limit: int,
    *,
    accepted: int | None = None,
    rounds: int | None = None,
    batch_size: int = 1,
) -> int:
    """このラウンドで引くドラフト数を決める**純関数**。

    今の政策は「文脈長が境界 (既定 = indexer_budget) を超えたら 1」だけ。
    足す予定のものが 2 つあり、どちらもここに閉じる:

    - **テキスト連動**: 同じ文脈長でも tok/round は 1.6-3.4 と振れる (実測)。
      直近ラウンドの受理率 (``accepted``/``rounds``) が高ければ深く、低ければ
      浅く。遅行指標なので、静的既定に勝てるかは実測で決める (分散が大きい
      ことは「オラクルなら勝てる」ことしか意味しない)。
    - **負荷連動**: バッチでは 1 位置足す限界費用が B に比例して上がるので、
      ``batch_size`` が増えたら浅く。

    引数で全部受け取る形にしてあるのは、バッチの同期ラウンドが (B, T+1) の
    一様幅で**行ごとの depth が構造的に存在できない**ため。行別の受理信号を
    「ラウンド共有の T 1 つ」に集約するのはスケジューラの仕事で、そこから
    この関数を呼ぶ。呼び出し点が engine 側に散っていても政策はここだけ。
    """
    if pos >= ctx_limit:
        return 1
    if batch_size > 1 and _BATCH_DEPTH_ADAPT:
        # B 行同期ラウンドでは、検証フォワードに 1 位置足す費用が B にほぼ
        # 比例して上がる (行数ぶんの列が増える) 一方、受理が増える利得は
        # 行ごとに独立なので B に比例しない。よって B が増えたら浅くする、
        # が紙モデルの符号。**未測なので既定 off。**スループットを測って
        # から段数と閾値を決める (docs/research/IMPROVEMENT-QUEUE.md B5)。
        return max(1, depth - (batch_size - 1) // 4)
    return depth


def _depth_ctx_limit(model) -> int:
    """このモデルで depth を 1 に落とす文脈長。

    env の指定が最優先。無ければ疎注意の境界 (indexer_budget) を使う。
    境界を持たない族では切り替えない (モデルの文脈上限に置く)。
    """
    if DEPTH_CONTEXT_LIMIT:
        return DEPTH_CONTEXT_LIMIT
    from .arch import indexer_budget

    return indexer_budget(model) or _DEPTH_CTX_LIMIT_FALLBACK


# ---------- レーン10 (docs/research/LANES-2026-09.md): 受理率適応の depth ----
#
# 既定は上の choose_depth (文脈長だけを見る静的規則) のまま変えない。
# ここは MLXTURBO_DEPTH_ADAPT=1 のときだけ通る別分岐で、位置別の受理率の
# 実測 (EMA) からラウンドごとに depth を選び直す。
#
# 設計は相手 (mlx-serve、~/dev/mlx-serve/src/generate.zig:6215-6395、MIT) の
# EV コントローラ (索引ごとの条件付き受理率 EMA と、幅ごとのラウンド費用表から
# 期待トークン数 / 費用を最大化する) を参考にしたが、コードは写していない。

MLXTURBO_DEPTH_ADAPT = os.environ.get("MLXTURBO_DEPTH_ADAPT") == "1"

# DepthController が保持する位置別 EMA の本数。MLXTURBO_DEPTH_CAP がこれを
# 超えて指定されても、ここで頭打ちにする (相手の上限が 6 なので余裕を見て 8)。
_DEPTH_CONTROLLER_MAX = 8
_DEPTH_EMA_PRIOR = 0.85
# 位置別 a[] の EMA の既定 beta。相手と同じ 0.15 で短文脈 A/B を取ったところ
# (bench/results/depth-adapt-short.json、2026-09-03)、最初の数ラウンドの
# 不的中だけで a[0] が急落して depth 1 に張り付き、静的 depth 2 比で
# ms/tok が +3.3% 悪化した。もっと緩く動かすため既定を下げてある。
# MLXTURBO_DEPTH_BETA で上書き可。
_DEPTH_EMA_BETA_DEFAULT = 0.1
# 深さごとの実測ラウンド費用 (ms/round) の EMA の beta。位置別 a[] の beta
# (上、env で変えられる) とは別物 -- こちらは相手の受理率 EMA に合わせて
# 固定 0.15 のまま (env での上書きは今のところ要らない、必要になったら足す)。
_DEPTH_COST_EMA_BETA = 0.15


def _depth_beta_default() -> float:
    """位置別 a[] の EMA の beta。MLXTURBO_DEPTH_BETA が最優先、無ければ
    _DEPTH_EMA_BETA_DEFAULT (0.1)。"""
    env = os.environ.get("MLXTURBO_DEPTH_BETA")
    if env:
        try:
            v = float(env)
        except ValueError:
            v = 0.0
        if v > 0:
            return v
    return _DEPTH_EMA_BETA_DEFAULT


# choose() が探索する depth の上限 (cap) の既定の境界。choose_depth 側の
# DEPTH_CONTEXT_LIMIT (既定 = モデルの indexer_budget) とは別の定数 --
# こちらは相手の規則にならって文脈長 2048 に固定してある。in-model の
# 実測はまだ無く、採否は decode_ab --knob depth-adapt の A/B で親が決める。
_DEPTH_CAP_CTX_LIMIT = 2048


def _depth_cap_default(pos: int) -> int:
    """DepthController.choose がこの位置で探索する depth の上限 (m の範囲)。

    env (MLXTURBO_DEPTH_CAP) が最優先。無ければ文脈長 2048 を境に
    3 (以下) / 2 (超) にする。
    """
    env = os.environ.get("MLXTURBO_DEPTH_CAP")
    if env:
        try:
            v = int(env)
        except ValueError:
            v = 0
        if v >= 1:
            return min(v, _DEPTH_CONTROLLER_MAX)
    return 2 if pos > _DEPTH_CAP_CTX_LIMIT else 3


def _depth_cost_params(pos: int) -> tuple[float, float]:
    """線形の費用モデル T(m) = T1 + m*dT の (T1, dT) (ms)。

    2026-09-03 の短文脈 A/B (bench/results/depth-adapt-short.json) で、この
    線形モデルの絶対値 (T1=25, dT=7) が実測より深さを過大に罰っていると
    分かった (実測の 1 段あたりの費用は +4.5ms 前後)。そのため
    ``DepthController`` はもう T(m) 全体をこの式では組まない --
    実測 (``cost_ema``) が無い深さの cold start にだけこの式を使い、
    実測が育ったあとは dT を**補外の傾き**としてだけ使う
    (``DepthController._cost_for`` 参照)。

    既定は verify_width_cost の実測 (docs/research/SESSION-2026-09-02-CATCHUP.md:
    短文脈 S=1 25.2 / S=2 33.1 / S=3 39.8、17k は S=1 30.4 / S=2 39.3 --
    差分から T1≈25/dT≈7 と T1≈30/dT≈9 を採った) をそのまま初期値に使う。
    MLXTURBO_DEPTH_COST="T1,dT" で文脈長によらず固定できる。
    """
    env = os.environ.get("MLXTURBO_DEPTH_COST")
    if env:
        try:
            t1_s, dt_s = env.split(",")
            return float(t1_s), float(dt_s)
        except ValueError:
            pass
    return (30.0, 9.0) if pos > _DEPTH_CAP_CTX_LIMIT else (25.0, 7.0)


class DepthController:
    """受理率の指数移動平均から、このラウンドで引く draft の深さを選ぶ。

    位置ごとの条件付き受理率 ``a[i]`` (i = 0..max_depth-1、「ドラフトの
    チェーンが位置 i まで届いたとき、位置 i も的中する確率」) を EMA
    (既定 beta=0.1、``_depth_beta_default`` / MLXTURBO_DEPTH_BETA、未観測の
    事前値 0.85) で保持する。``choose(pos)`` は期待受理トークン数 E(m) を
    1 ラウンドの費用 T(m) で割った値が最大の m (= depth) を、
    ``m in 1..cap`` の範囲で返す:

        E(m) = 1 + sum(i = 0..m-1) prod(j = 0..i) a[j]

    ``cap`` は文脈長 (``pos``) から ``_depth_cap_default`` が決める (env で
    上書き可)。T(m) は**実測優先**: ``observe(..., round_ms=...)`` で深さ
    ``depth`` のラウンド費用を渡すたび ``cost_ema[depth]`` を EMA
    (beta=0.15 固定) で更新する。``choose`` は観測のある深さはその EMA を
    そのまま使い、観測の無い深さは「観測のある最も近い深さの EMA +
    dT * (m - 最も近い深さ)」で補外する (``_cost_for`` 参照)。線形モデル
    T1 + m*dT (``_depth_cost_params``) は、まだ何も観測が無い cold start の
    ときの初期値としてだけ使う -- 2026-09-03 の短文脈 A/B
    (bench/results/depth-adapt-short.json) で、この式の絶対値が実測より
    深さを過大に罰っていて (dT=7 は実測 +4.5ms 前後より重い)、depth 1 に
    張り付いて既定の静的 depth 2 に ms/tok で負けたため、絶対値としては
    使わなくした。

    副作用を持たない (発火の記録は呼び出し側の責務 --
    ``FlashSpecEngine._effective_depth`` を参照)。単体テストが直接構成できる
    よう、依存はコンストラクタ引数だけに閉じてある。
    """

    def __init__(self, max_depth: int = _DEPTH_CONTROLLER_MAX,
                 beta: float | None = None,
                 prior: float = _DEPTH_EMA_PRIOR):
        self.max_depth = max_depth
        self.beta = beta if beta is not None else _depth_beta_default()
        self.a = [prior] * max_depth
        # 検査用: 各位置が何回観測されたか (単体テストと分布確認に使う)。
        self.observations = [0] * max_depth
        # 深さごとの実測ラウンド費用 (ms) の EMA。キーは観測済みの depth
        # だけを持つ (「観測が無い」を空扱いで区別する)。
        self.cost_ema: dict[int, float] = {}
        self.cost_observations: dict[int, int] = {}

    def observe(self, n_accepted: int, depth: int,
                round_ms: float | None = None) -> None:
        """検証ラウンドの結果で ``a`` (と、渡されれば費用 EMA) を更新する。

        ``depth`` はそのラウンドで実際に引いたドラフト数、``n_accepted`` は
        そのうち採用された数 (0..depth)。位置 ``i < n_accepted`` は的中 (1)、
        ``i == n_accepted`` (depth 未満なら) は不的中 (0)、それより先の位置は
        「ドラフトのチェーンがそこで既に切れていて検証されていない」ので
        未観測のまま触らない。

        ``round_ms`` (省略可): このラウンドの実測費用 (ms、呼び出し側が
        draft 構築から verify の同期までを ``time.perf_counter()`` の差で
        測る)。渡されたときだけ ``depth`` の費用 EMA も更新する。
        """
        limit = min(depth, self.max_depth)
        for i in range(limit):
            if i < n_accepted:
                obs = 1.0
            elif i == n_accepted:
                obs = 0.0
            else:
                break
            self.a[i] = (1.0 - self.beta) * self.a[i] + self.beta * obs
            self.observations[i] += 1
        if round_ms is not None and round_ms >= 0:
            self._observe_cost(depth, round_ms)

    def _observe_cost(self, depth: int, round_ms: float) -> None:
        prev = self.cost_ema.get(depth)
        self.cost_ema[depth] = (
            round_ms if prev is None
            else (1.0 - _DEPTH_COST_EMA_BETA) * prev + _DEPTH_COST_EMA_BETA * round_ms
        )
        self.cost_observations[depth] = self.cost_observations.get(depth, 0) + 1

    def expected_tokens(self, m: int) -> float:
        """E(m) = 1 + sum(i<m) prod(j<=i) a[j] の閉じた式。"""
        total = 1.0
        prod = 1.0
        for i in range(m):
            prod *= self.a[i]
            total += prod
        return total

    def _cost_for(self, m: int, pos: int) -> float:
        """depth=m の 1 ラウンド費用 (ms) の見積もり。

        実測 (``cost_ema[m]``) があればそれをそのまま使う。実測が 1 つも
        無ければ (cold start) 線形モデル T1 + m*dT を使う。実測はあるが
        ``m`` そのものは未観測なら、観測済みで最も近い深さの EMA から
        ``dT * (m - 最も近い深さ)`` だけ補外する (符号込み -- m が近い深さ
        より浅ければ引く)。
        """
        t1, dt = _depth_cost_params(pos)
        if not self.cost_ema:
            return t1 + m * dt
        if m in self.cost_ema:
            return self.cost_ema[m]
        nearest = min(self.cost_ema, key=lambda k: abs(k - m))
        return self.cost_ema[nearest] + dt * (m - nearest)

    def choose(self, pos: int) -> int:
        """E(m)/T(m) を最大にする m (1..cap) を返す。"""
        cap = min(_depth_cap_default(pos), self.max_depth)
        best_m, best_score = 1, -1.0
        for m in range(1, cap + 1):
            cost = self._cost_for(m, pos)
            score = self.expected_tokens(m) / cost if cost > 0 else 0.0
            if score > best_score:
                best_score, best_m = score, m
        return best_m


class Capture:
    """The records needed to roll back one verification forward."""

    def __init__(self):
        self.gdn = {}      # id(module) -> (conv_input, states_all)
        self.ple = {}      # id(module) -> full
        self.hyper = None  # the hyper state right before entering the final mixer
        self.pre = {}      # cache state before the forward (KV offset, etc.)


@contextmanager
def capture(model, light: bool = False):
    """A context that runs the verification forward while leaving behind the
    records needed for rollback.

    ``light=True`` (an addition, False by default): record only
    ``GatedResidual`` (``cap.hyper``) and let ``GatedDeltaNet``/``PLELayer`` go
    through their plain forward (which does not capture state).

    The reason: the ``states_all`` returned by
    ``gated_delta_update_with_states`` is ``(B, T, Hv, Dv, Dk)`` fp32 --
    ``Hv*Dv*Dk*4`` bytes per layer per token (~3MiB in this model's
    configuration). For the decode loop's verification forward ``T<=2``, so the
    value is negligible, but for ``generate_stream``'s final prefill chunk ``T``
    can be the chunk width itself (up to ``PREFILL_STEP_SIZE``). There, only
    ``cap.hyper[:, -1:]`` (the hyper state at the last position) is used, yet
    ``states_all`` was being allocated and retained unconditionally for all 36
    layers (the linear_attention layer count), which demanded hundreds of GB of
    memory at around T=2000 and got the whole process killed by macOS's
    memorystatus killer (measured; this was the cause of the symptom where the
    process vanished with no traceback). Even when ``GatedDeltaNet.__call__``/
    ``PLELayer._short_conv`` go through as the plain implementation, the cache
    updates (``cache[0]``/``cache[1]``/``cache[2]``/``cache.advance``) are
    performed by the main model with the same logic, so cache consistency is
    unchanged. The only thing that changes is that ``cap.gdn``/``cap.ple``
    (unused by this caller) stay empty. Existing calls (``light`` omitted =
    False: ``generate()``, and the verification forward inside
    ``generate_stream``'s decode loop) have their behavior completely unchanged
    -- this is an addition only.
    """

    Q = _arch()
    from .kernels.gated_delta_states import (
        gated_delta_update_with_states,
        gated_delta_update_with_states_gb,
    )

    cap = Capture()
    orig_gdn = Q.GatedDeltaNet.__call__
    orig_ple = Q.PLELayer._short_conv
    orig_hc = Q.GatedResidual.__call__
    mixer = model.model.hyper_connection_mixer

    def gdn(self, x, mask, cache):
        # Transcribe the main GatedDeltaNet.__call__ as-is, replacing only the
        # kernel with the one that returns states. Do not change the logic
        B, S, _ = x.shape
        mixed_qkv, z, b, a = self._project_in(x)
        z = z.reshape(B, S, self.n_v, self.dv)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )

        # GDN 前処理の融合カーネル経路 (mlxturbo/kernels/gdn_prework.py)。
        # _vendor/qwen4_exp.py:965-997 の適格判定と完全に同じ条件
        # (同じ条件でないと A/B の解釈が壊れる)。検証フォワードは rollback 用に
        # states_all (位置ごとの再帰状態) も要るので、mlx_lm 版
        # gated_delta_kernel (最終状態だけ返す) の代わりに
        # gated_delta_update_with_states_gb を呼ぶ。conv_input (rollback で
        # conv 窓を切り出すのに使う) は前処理カーネルが返さないので、ここで
        # 組み直す (concat だけで、融合の対象である conv1d/silu/rms_norm の
        # 再計算ではない)。
        _gdn_prework_on = getattr(self, "_gdn_prework", False)
        if (
            _gdn_prework_on
            and mask is None
            and cache is not None
            and not self.training
            and getattr(cache, "lengths", None) is None
        ):
            from .kernels import gdn_prework as gp

            # 実モデルは A_log/dt_bias が bf16 で読み込まれていることがある
            # (2026-09-02、実機ログで確認)。enable_gdn_prework_kernel(model) が
            # 立てた fp32 の写しがあればそれを使う (無ければ従来どおり
            # self.A_log/self.dt_bias -- fp32 でなければ eligible() で弾かれる)。
            A_log = getattr(self, "_A_log_f32", self.A_log)
            dt_bias = getattr(self, "_dt_bias_f32", self.dt_bias)
            if gp.eligible(mixed_qkv, conv_state, self.conv1d.weight, a, b,
                            A_log, dt_bias, self.n_k, self.n_v,
                            self.dk, self.key_dim, self.value_dim):
                q, k, v, g, beta, new_conv_state = gp.fused_gdn_prework(
                    mixed_qkv, conv_state, self.conv1d.weight, a, b,
                    A_log, dt_bias, self.n_k, self.n_v,
                    self.dk, self.dv, self.key_dim, self.value_dim,
                )
                conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
                out, states_all = gated_delta_update_with_states_gb(
                    q, k, v, g, beta, cache[1], mask
                )
                cap.gdn[id(self)] = (conv_input, states_all)
                cache[0] = new_conv_state
                cache[1] = states_all[:, -1]
                cache.advance(S)
                return self.out_proj(self.norm(out, z).reshape(B, S, -1))
        elif _gdn_prework_on:
            # _vendor/qwen4_exp.py 側と同じ穴埋め (BACKLOG.md B-8 は eligible()
            # 内部の条件だけを直していて、この手前の 4 条件は無言のままだった)。
            from .kernels import gdn_prework as gp

            gp.explain_gate_miss(mask, cache, self.training)

        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, states_all = gated_delta_update_with_states(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask
        )
        cap.gdn[id(self)] = (conv_input, states_all)
        if cache is not None:
            cache[1] = states_all[:, -1]
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    def ple_conv(self, x, cache):
        n = self.short_conv_state_len
        S = x.shape[1]
        prev = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([prev, x], axis=1)
        cap.ple[id(self)] = full
        if cache is not None:
            cache[2] = mx.contiguous(full[:, -n:, :])
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def hc(self, hyper):
        if self is mixer:
            cap.hyper = hyper
        return orig_hc(self, hyper)

    if not light:
        Q.GatedDeltaNet.__call__ = gdn
        Q.PLELayer._short_conv = ple_conv
    Q.GatedResidual.__call__ = hc
    try:
        yield cap
    finally:
        if not light:
            Q.GatedDeltaNet.__call__ = orig_gdn
            Q.PLELayer._short_conv = orig_ple
        Q.GatedResidual.__call__ = orig_hc




# 段階投入の間隔 (層数)。0 で無効 = 一括構築。2026-08-31 の掃引: 16/12/8/6/4/3/2
# で 53.9/55.0/57.6/57.8/58.3/58.7 tok/s (probe 実測、出力は全て一致)。既定 2。
_STAGE_EVERY = int(os.environ.get("MLXTURBO_STAGE_EVERY", "2") or 0)

# MLXTURBO_ROUND_TRACE=1: ラウンド内の CPU 側区間を ms で刻む (調査用)
_ROUND_TRACE = os.environ.get("MLXTURBO_ROUND_TRACE") == "1"

# MLXTURBO_PREFILL_TRACE=1: prefill のフェーズ別区間を ms で刻む (調査用)。
# 既定 off で、off のときは生成コストゼロ (フラグ読み出しのみ)。
_PREFILL_TRACE = os.environ.get("MLXTURBO_PREFILL_TRACE") == "1"

# 独立レビュー A-1 の修正 (2026-09-02): 検証で確定した中間トークンを MTP
# キャッシュへ積み直すか。既定 on ("1")。_draft_chain はチェーンを引くたび
# cur 1 列まで cache を trim して戻すので、hit>=1 のラウンドでは受理済みの
# 中間トークンが一度もキャッシュに書かれず、MTP の offset (RoPE 位置) が
# 毎ラウンド hit ぶん遅れて受理率が生成長に比例して落ちていた
# (FlashSpecEngine._prime_accepted_gap、generate_stream 参照)。
# off ("0") で修正前の挙動に戻せる -- tools/decode_ab.py の knob
# `mtp-append` の B 側 (旧挙動) との比較用。
_MTP_CACHE_APPEND = os.environ.get("MLXTURBO_MTP_CACHE_APPEND", "1") != "0"


class _PrefillTracer:
    """_PREFILL_TRACE=1 のときだけ prefill の区間を ms で刻んで 1 行ずつ print する。

    区間は「既にある eval の前後で perf_counter() を挟むだけ」で作る
    (新規の mx.eval は増やさない -- 増やすと計測が経路を変えてしまう)。
    `log` は区間の長さを内部 total に積み、`summary` で合計と壁時計の
    差 (=計測できていない隙間) を出す。
    """

    def __init__(self):
        self.t0 = time.perf_counter()
        self.total = 0.0
        self.end = None

    def log(self, label, seg_t0):
        now = time.perf_counter()
        dur = (now - seg_t0) * 1e3
        self.total += dur
        print(f"[prefill] {label} t={(now - self.t0) * 1e3:.2f} "
              f"dur={dur:.2f}", flush=True)
        return now

    def mark_end(self):
        self.end = time.perf_counter()

    def summary(self):
        end = self.end if self.end is not None else time.perf_counter()
        wall = (end - self.t0) * 1e3
        print(f"[prefill] total dur_sum={self.total:.2f} wall={wall:.2f} "
              f"gap={wall - self.total:.2f}", flush=True)


def _staged_forward(model, ids, caches):
    """Model.__call__ と同じ計算を、途中の hidden を async_eval しながら組む。

    グラフを 48 層ぶん組み切ってから投げると、構築中 (7.3ms、xctrace 実測の
    泡。docs/research/KERNEL-BRIEF-DECODE-BW.md 参照) GPU が遊ぶ。既定の
    `_STAGE_EVERY=2` (2 層ごと) で投入すれば GPU は先頭から走り出し、
    CPU は残りを組み続けられる。計算内容は Qwen4ExpModel.__call__ +
    lm_head と完全に同一。

    前段 (mask 生成と PLE/n-gram の直前文脈) は本家の `_prelude` を呼ぶので
    写しではない。写しなのは層ループの骨格だけで、そこが本物の差分
    (2 層ごとの async_eval は本家に無い制御フロー)。
    """
    m = model.model
    h = m.embed_tokens(ids)
    cache = caches

    mask, conv_mask, prev_ctx = m._prelude(ids, h, cache)

    h = mx.tile(h, (1, 1, m.hc))
    step = _STAGE_EVERY
    for i, (layer, c) in enumerate(zip(m.layers, cache)):
        idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
        h = layer(h, m.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        if step and (i + 1) % step == 0 and i < len(m.layers) - 1:
            mx.async_eval(h)
    out = m.hyper_connection_mixer(h)
    if model.args.text.tie_word_embeddings:
        return m.embed_tokens.as_linear(out)
    return model.lm_head(out)


# prefill の中間チャンクを何個まとめてレイヤー主導で流すか。attention/indexer は
# チャンク幅 (2048) のまま、MoE だけグループぶんの行を concat して 1 回で流す。
# gather_qmm (affine_gather_qmm_rhs) は BM 行タイル内の expert 境界ごとに
# フル GEMM をやり直すため、効率は行数/expert に単調 (r=40/80/160 で
# 7.5/8.9/9.8 TFLOPS、密上限 11.2)。チャンク幅そのものを上げる案 (下の
# MLXTURBO_PREFILL_CHUNK) は attention/indexer の一時増と相殺したが、
# こちらは MoE の行だけ太らせるので相殺しない (in-model 実測: chunk 4096 で
# MoE 部分時間 18.4 -> 14.9s)。gather_qmm は行独立で、BM=32/64 をまたぐ
# 分割でもビット一致 (micro 確認済み) — 出力はトークン列まで不変が要件。
_PREFILL_GROUP = int(os.environ.get("MLXTURBO_PREFILL_GROUP", "4") or 0)

# group prefill を当てない末尾チャンク数。ここはチャンク主導のままなので
# checkpoint が step (2048) 粒度で立つ。グループの内側では checkpoint を刻めない
# -- レイヤー主導では「チャンク k を全層通した状態」がどの瞬間にも存在しないので、
# これは構造的な制約であって実装の手抜きではない。
#
# 2 ターン目の LCP は末尾に落ちやすいので、ここを広げると追記ターンの再 prefill が
# 減りうる (代償は prefill の MoE バッチ効率)。既定 1 は従来の挙動そのもの。
# 掃引するときは「17k TTFT の悪化 2% 以内」を反転条件にする
# (docs/research/IMPROVEMENT-QUEUE.md B2)。
_PREFILL_TAIL_CHUNKS = int(os.environ.get("MLXTURBO_PREFILL_TAIL_CHUNKS", "1") or 1)

# group prefill のグループ境界を非同期投入にする (既定 off)。詳細は使用箇所の
# コメント。取り分は小さく、メモリ側の危険は実在するので、測ってから決める。
_PREFILL_PIPELINE = os.environ.get("MLXTURBO_PREFILL_PIPELINE") == "1"

# 端数チャンク (幅 < step) をレイヤー主導グループに畳み込むか。既定 on
# (commit b80d7e2 の挙動)。"0" で畳み込み前 (端数は常に chunk-major で
# 単独処理) に戻る。A/B は tools/decode_ab.py の knob `fold-tail`。
_PREFILL_FOLD_TAIL = os.environ.get("MLXTURBO_PREFILL_FOLD_TAIL", "1") != "0"

if _PREFILL_GROUP > 1 and os.environ.get("MLXTURBO_PREFILL_CHUNK"):
    # MLXTURBO_PREFILL_CHUNK を立てると big != step になり、下の group 経路の
    # 条件が外れて layer-major prefill が**黙って無効化**される。チャンク幅の
    # 棄却記録 (-2%) は layer-major 以前のものなので、将来これを再測する人が
    # 気づかずに layer-major を落とし、偽の負けを記録する罠になっていた。
    print("[mlxturbo] 警告: MLXTURBO_PREFILL_CHUNK を立てたので layer-major"
          " prefill (MLXTURBO_PREFILL_GROUP) は無効になる。チャンク幅を測るなら"
          " その前提で読むこと (docs/research/IMPROVEMENT-QUEUE.md D2)。")


def _group_prefill_forward(model, chunks, caches):
    """中間 prefill チャンクのグループをレイヤー主導で流す。

    計算内容は「各チャンクを順に Qwen4ExpModel.__call__ に通す」のと完全に
    同一で、違いは MoE (layer.mlp) だけグループ内チャンクの行を concat して
    1 回で呼ぶこと (行独立なのでビット一致)。mixer は通さない — chunk-major
    でも中間チャンクの mixer 出力は捨てられており、呼び手が使う cap.hyper は
    mixer の入力 (= 最終レイヤー出力) だから、それをチャンクごとに返す。

    層の中身は本家の `DecoderLayer.pre_mlp` と `_combine` を呼ぶので写しでは
    ない。ここに残る差分は「二重ループ (レイヤー主導 x G チャンク) と MoE の
    呼び出し粒度」だけで、それが layer-major prefill の本体。

    キャッシュ整合性: レイヤー主導でも「レイヤー i がチャンク c を処理する
    時点のキャッシュ i の中身」は chunk-major と一致する (レイヤー i の
    キャッシュを進めるのはレイヤー i 自身だけなので、処理順の入れ替えは
    各キャッシュから見ると無関係)。mask も同じ理由で per-chunk に 1 個。
    """
    Q = _arch()
    m = model.model
    G = len(chunks)
    hs = [mx.tile(m.embed_tokens(ch), (1, 1, m.hc)) for ch in chunks]
    conv_mask = None

    masks = [None] * G  # 最初の full attention 層の走査中に生成する

    # per-chunk の prev_ctx: 本家はモデル呼び出しごとに pc[3] を進めるので、
    # チャンク列に対して同じ更新を先に畳んでおく
    prev_ctxs = [None] * G
    if m.ple_layers:
        ctx_len = m.args.ngram_size - 1
        eos_id = m.args.eos_token_id
        eos_id = eos_id[0] if isinstance(eos_id, list) else eos_id
        pc = caches[m.ple_layers[0]]
        prev = pc[3] if pc is not None else None
        if prev is None:
            prev = mx.full((chunks[0].shape[0], ctx_len), eos_id, chunks[0].dtype)
        for ci, ch in enumerate(chunks):
            prev_ctxs[ci] = prev
            prev = mx.concatenate([prev, ch], axis=1)[:, -ctx_len:]
        if pc is not None:
            pc[3] = prev

    step = _STAGE_EVERY
    for li, (layer, c) in enumerate(zip(m.layers, caches)):
        idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
        posts = []
        for ci in range(G):
            if layer.layer_type != "linear_attention" and masks[ci] is None:
                # 本家は層ごとに作るが、同じチャンクなら層をまたいで同一
                # (mask はチャンク幅とそのキャッシュのオフセットだけで決まり、
                # レイヤー i のキャッシュを進めるのはレイヤー i 自身だけ)。
                masks[ci] = Q.create_attention_mask(hs[ci], [c])
            posts.append(layer.pre_mlp(
                hs[ci], m.rope, masks[ci], conv_mask, c, idx_c,
                chunks[ci], prev_ctxs[ci],
            ))
        xcat = mx.concatenate([p[0] for p in posts], axis=1)
        ycat = layer.mlp(xcat)
        offs, acc = [], 0
        for p in posts[:-1]:
            acc += p[0].shape[1]
            offs.append(acc)
        ys = mx.split(ycat, offs, axis=1)
        for ci in range(G):
            _, hyper, inject = posts[ci]
            hs[ci] = layer._combine(hyper, ys[ci], inject)
        if step and (li + 1) % step == 0 and li < len(m.layers) - 1:
            mx.async_eval(*hs)
    return hs


def _prefetch_ngram_rows(model, ids: mx.array, caches) -> None:
    """この呼び出しで prefill する `ids` 全体ぶんの n-gram 行を先読みする。

    prefill では `ids` (プロンプト全体、またはセッション継続時はその新規
    分) の行 id が最初から全部わかっているので、GPU がチャンク 0 を計算
    している間に CPU 側でディスクから先読みしておけば、後続チャンクの
    `StreamNGram.__call__` が待たずに返る。`ngram_ids` の計算自体は 1 回
    (GPU) で済ませ、以降は `StreamNGram.prefetch` がバックグラウンドスレッド
    で pread する (呼び出しは即時に返る)。

    `ngram_embedding` が `StreamNGram` でない (RamNGram / 未 install) か
    `prefetch_enabled=False` のときは何もしない (`getattr` で判定)。

    先頭の文脈 (`prev_ctx`) の扱いは `Qwen4ExpModel._prelude` /
    `_group_prefill_forward` と揃えてある: セッション継続ならキャッシュ
    (`pc[3]`) の文脈から、そうでなければ eos x(ngram_size-1) から始める。
    ここがズレても正しさは壊れない (miss してその場で pread するだけ) が、
    先読みの分だけ効かなくなる。
    """
    if ids.shape[1] == 0:
        return
    m = model.model
    for pli in getattr(m, "ple_layers", None) or []:
        ple_emb = m.layers[pli].ple.ple_embedding
        stream = ple_emb.ngram_embedding
        if not getattr(stream, "prefetch_enabled", False):
            continue
        pc = caches[pli] if caches is not None else None
        prev = pc[3] if pc is not None else None
        if prev is None:
            prev = mx.full(
                (ids.shape[0], ple_emb.context_len), ple_emb.eos_token_id, ids.dtype
            )
        history = mx.concatenate([prev, ids], axis=1)
        gid = ple_emb.ngram_ids(history)[:, -ids.shape[1] :]
        mx.eval(gid)
        stream.prefetch(np.array(gid.reshape(-1), copy=False).astype(np.int64))


def _pipeline_snapshot(model, caches, mtp_cache):
    """楽観先組み (次ラウンドのグラフを結果を知らずに組む) 用の浅い退避。

    MLX の配列は不変ノードで、キャッシュの「更新」は Python 参照の付け替え
    (KVCache の setitem もオブジェクトを新ノードへ束縛し直すだけ) なので、
    参照とオフセットを控えておけば付け替えで完全に戻せる。飛行中のグラフは
    古いノードを掴んでいるため影響を受けない。"""
    st = []
    for layer, c in zip(model.model.layers, caches):
        if layer.layer_type == "full_attention":
            st.append(("a", c.keys, c.values, c.offset, c.indexer.keys))
        else:
            # ArraysCache は 4 スロットの参照だけ (offset を持たない。
            # advance() は lengths/left_padding 用で decode では両方 None)
            st.append(("l", c[0], c[1], c[2], c[3]))
    st.append(("m", mtp_cache.keys, mtp_cache.values, mtp_cache.offset,
               mtp_cache.indexer.keys))
    return st


def _pipeline_restore(model, caches, mtp_cache, st) -> None:
    for (layer, c), rec in zip(zip(model.model.layers, caches), st[:-1]):
        if rec[0] == "a":
            _, c.keys, c.values, c.offset, c.indexer.keys = rec
        else:
            _, c0, c1, c2, c3 = rec
            c[0], c[1], c[2], c[3] = c0, c1, c2, c3
    rec = st[-1]
    _, mtp_cache.keys, mtp_cache.values, mtp_cache.offset, mtp_cache.indexer.keys = rec


def snapshot_pre(model, caches) -> dict:
    """**Before** the forward, note down what capture cannot restore.

    n-gram context の取り出しは、直添字 (``c[3]``) ではなく
    ``arch.recurrent_layers`` の名前付き ``ngram`` スロットを経由する
    (mlxturbo/arch.py 参照 -- 族が変わってもスロット番号を汎用コード側に
    ハードコードしないため)。
    """
    slots = {rl.index: rl for rl in _archmod.recurrent_layers(model)}
    pre = {"kv": [], "ctx": []}
    for i, (layer, c) in enumerate(zip(model.model.layers, caches)):
        if layer.layer_type == "full_attention":
            # _AttnCache derives from KVCache and is not an ArraysCache (it has no .cache)
            keys = c.indexer.keys
            pre["kv"].append((c.offset, None if keys is None else keys.shape[1]))
            pre["ctx"].append(None)
        else:
            pre["kv"].append(None)
            rl = slots.get(i)
            pre["ctx"].append(c[rl.ngram] if (rl is not None and rl.ngram is not None) else None)
    return pre


def rollback(model, caches, cap: Capture, pre: dict, keep: int, total: int,
             ids_kept=None):
    """Of the `total` tokens advanced by the verification forward, keep only the
    leading `keep`.

    `ids_kept` is the adopted token sequence (B, keep). The n-gram context has to
    be a value advanced up to "the position that does not include what was thrown
    away", so we rebuild it from the pre-forward context (restoring the
    pre-forward value as-is would roll back `keep` tokens too far).
    """
    if keep == total:
        return
    drop = total - keep
    # full attention 側の trim/indexer は族ごとに有無が違う別能力
    # (arch.has_indexer) なので、再帰状態の巻き戻し (arch.rollback_recurrent)
    # には混ぜない -- ここで直接扱う。
    for i, layer in enumerate(model.model.layers):
        if layer.layer_type != "full_attention":
            continue
        c = caches[i]
        c.trim(drop)
        if _archmod.has_indexer(c):
            old_len = pre["kv"][i][1] or 0
            c.indexer.keys = c.indexer.keys[:, : old_len + keep]

    # GDN 状態・conv 窓・PLE conv・n-gram 文脈の巻き戻しは
    # mlxturbo/arch.py の名前付きスロット経由の共通部品に畳んである
    # (batch_spec.batched_rollback と共有)。
    _archmod.rollback_recurrent(
        model, caches, cap, keep, ngram_ctx=pre["ctx"], ids_kept=ids_kept
    )


def trim_attn_cache(cache, keep: int) -> None:
    """MTP のドラフトキャッシュを先頭 ``keep`` 件まで縮める。

    基準は**物理列数** (``size()``)。単一系列の ``KVCache`` では
    ``size() == offset`` なので従来と同じだが、バッチ版のドラフトキャッシュ
    (`mlxturbo/batch_spec.py`) は ``offset`` が行ごとの論理位置 (B,) の配列に
    なるので、そちらでも同じコードが通るようにしてある。
    """
    drop = cache.size() - keep
    if drop <= 0:
        return
    cache.trim(drop)
    if cache.indexer.keys is not None:
        cache.indexer.keys = cache.indexer.keys[:, :keep]


def snapshot_mtp_cache(cache):
    """Copy the MTP draft cache so it can be handed to a later resumed call.

    ``KVCache`` preallocates and writes into ``self.keys[..., i:i+n, :]`` in
    place, so the live cache cannot simply be aliased -- the decode loop would
    overwrite the copy. ``_IndexerCache`` concatenates instead (a fresh array
    per update), so its keys are safe to share. About 5MB at PRIME_WINDOW with
    this model's 2 KV heads.
    """
    if cache is None or cache.offset == 0:
        return None
    n = cache.offset
    return (
        mx.contiguous(cache.keys[..., :n, :]),
        mx.contiguous(cache.values[..., :n, :]),
        n,
        cache.indexer.keys,
    )


def restore_mtp_cache(snap):
    """Rebuild the cache saved by ``snapshot_mtp_cache`` (None -> empty)."""
    cache = _arch()._AttnCache()
    if snap is None:
        return cache
    keys, values, n, idx = snap
    cache.keys, cache.values, cache.offset = keys, values, n
    cache.indexer.keys = idx
    return cache


class FlashSpecEngine:
    """Depth-1 speculative decoding that uses the MTP as the draft.

    Decoding is dispatch-bound: a batched forward costs only 1.17x the S=1 case
    even at S=16 (docs/STATUS.md). **A width-2 verification costs roughly the
    price of a single token**, so on acceptance one forward advances 2 tokens.

    Invariant: `cur` is the token to be fed next, the caches are processed up to
    just before it, and `hyper_prev` is the hyper state at the position that
    produced `cur`.
    """

    def __init__(self, model, mtp, depth: int = MTP_DEPTH):
        self.model = model
        self.mtp = mtp
        self.rope = model.model.rope
        self.depth = max(1, int(depth))
        self.depth_ctx_limit = _depth_ctx_limit(model)
        # レーン10: 受理率 EMA で depth を選ぶ適応 (既定 off、
        # MLXTURBO_DEPTH_ADAPT=1 で有効)。controller はエンジンの生存期間中
        # 持ち回り、複数ラウンド/複数リクエストにまたがって学習する
        # (mlx-serve の EV コントローラと同じ想定 -- リクエストごとに
        # 作り直さない)。_effective_depth / generate() / generate_stream()
        # の verify 後で観測する。
        self._depth_adapt = MLXTURBO_DEPTH_ADAPT
        self._depth_controller = DepthController() if self._depth_adapt else None
        # draft-rerank (mlx-serve の設計の移植): trunk lm_head の 2bit 再量子化で
        # 全語彙を粗く読み、正確な top-32 だけを trunk のヘッドの行で再採点する。
        # 粗い top-32 に真の argmax が入っている限り draft は trunk と一致し、
        # 検証は常に trunk ヘッドなので出力分布は無条件で無傷。
        # MLXTURBO_DRAFT_RERANK=0 で無効。
        self._rerank = None
        if os.environ.get("MLXTURBO_DRAFT_RERANK", "1") != "0":
            self._build_rerank()

    RERANK_BITS = 2
    RERANK_TOP = 32

    def _head(self, x: mx.array) -> mx.array:
        """hidden から logits。``tie_word_embeddings`` のパックには
        ``lm_head`` が無く、``embed_tokens.as_linear`` が代わりになる
        (``generate`` / ``_staged_forward`` と同じ規約)。以前ここが
        ``self.model.lm_head`` 直参照で、**tie されたパックはエンジンの構築
        時点で AttributeError になっていた**。
        """
        lm = getattr(self.model, "lm_head", None)
        if lm is not None:
            return lm(x)
        return self.model.model.embed_tokens.as_linear(x)

    def _build_rerank(self) -> None:
        lm = getattr(self.model, "lm_head", None)
        if lm is None or not hasattr(lm, "scales"):
            # tie されたパック (lm_head 無し) と非量子化パックでは粗ヘッドを
            # 作れない。rerank 無しで動く (draft は trunk の argmax)。
            return
        w = mx.dequantize(lm.weight, lm.scales, lm.biases,
                          group_size=lm.group_size, bits=lm.bits)
        cw, cs, cb = mx.quantize(w, group_size=64, bits=self.RERANK_BITS)
        # 再採点用に trunk の行を bf16 で引けるよう、逆量子化した行列も保持
        # ... はメモリを食い過ぎる (2.5GB)。行の gather は量子化のまま行い、
        # その場で 32 行だけ逆量子化する。
        mx.eval(cw, cs, cb)
        self._rerank = (cw, cs, cb)
        del w
        mx.clear_cache()

    def _draft_argmax(self, out) -> mx.array:
        """draft 用の次トークン。(1, 1) を返す。

        rerank あり: 2bit 粗ヘッドで全語彙 -> top-32 -> trunk の該当行を
        逆量子化して再採点 -> argmax。無し: trunk ヘッドで argmax。
        """
        if self._rerank is None:
            return mx.argmax(self._head(out)[:, -1], axis=-1).reshape(-1, 1)
        lm = self.model.lm_head  # _rerank があるなら lm_head も必ずある
        row = out[:, -1]
        cw, cs, cb = self._rerank
        if row.shape[0] > 1:
            return self._draft_argmax_rows(row, lm, cw, cs, cb)
        coarse = mx.quantized_matmul(
            row, cw, scales=cs, biases=cb, transpose=True,
            group_size=64, bits=self.RERANK_BITS)
        top = mx.argpartition(-coarse, self.RERANK_TOP - 1, axis=-1)[..., : self.RERANK_TOP]
        rows = mx.dequantize(
            lm.weight[top[0]], lm.scales[top[0]], lm.biases[top[0]],
            group_size=lm.group_size, bits=lm.bits)
        scores = (row.astype(rows.dtype) @ rows.T)
        best = mx.argmax(scores, axis=-1, keepdims=True)
        return mx.take_along_axis(top, best, axis=-1)

    def _draft_argmax_rows(self, row, lm, cw, cs, cb) -> mx.array:
        """`_draft_argmax` の rerank 経路の B 行版 (バッチ x 投機で使う)。

        単一行の側をそのまま一般化しない理由: あちらは `row @ rows.T` の
        2 階行列積で、行を足すとバッチ行列積になる。数学的には同じでも
        加算順が変わりうるので、**実測が乗っている B=1 の経路には触らない**。
        """
        top = mx.argpartition(-mx.quantized_matmul(
            row, cw, scales=cs, biases=cb, transpose=True,
            group_size=64, bits=self.RERANK_BITS,
        ), self.RERANK_TOP - 1, axis=-1)[..., : self.RERANK_TOP]
        rows = mx.dequantize(
            lm.weight[top], lm.scales[top], lm.biases[top],
            group_size=lm.group_size, bits=lm.bits)
        scores = mx.matmul(
            row[:, None, :].astype(rows.dtype), rows.transpose(0, 2, 1))[:, 0]
        best = mx.argmax(scores, axis=-1, keepdims=True)
        return mx.take_along_axis(top, best, axis=-1)

    def _effective_depth(self, pos: int) -> int:
        """この位置で引くドラフト数。長い文脈では 1 に落とす
        (DEPTH_CONTEXT_LIMIT の注記を参照)。政策そのものは
        `choose_depth` (純関数) に閉じてある。

        `MLXTURBO_DEPTH_ADAPT=1` のときだけ、代わりに `self._depth_controller`
        (受理率 EMA) が選ぶ。既定 (off) の経路は 1 ビットも変わらない。
        """
        if self._depth_adapt:
            m = self._depth_controller.choose(pos)
            _fire.bump(f"depth_adapt_{m}")
            return m
        return choose_depth(pos, self.depth, self.depth_ctx_limit)

    def _draft_chain(self, cur, hyper_prev, cache, depth: int):
        """``self.depth`` トークンをまとめて引く。

        ヘッドは (embed(t), hyper) を受けて、mixer で潰す前に **hyper 形状の
        状態を自分で作る**。それを次段に渡すことで、1 つのヘッドで t+2 より
        先まで届く (DeepSeek-V3 と同じ形)。

        確定した (トークン, hyper) の対は 1 段目だけなので、戻る前にキャッシュを
        その 1 件まで縮める。**このキャッシュに投機的なものを入れない**という
        不変条件が、ラウンドを跨いで持ち回れる根拠になっている。
        """
        Q = _arch()
        keep = cache.size() + 1  # 物理列数 (trim_attn_cache の注記参照)
        drafts = []
        tok, hyper = cur, hyper_prev
        for step in range(depth):
            emb = self.model.model.embed_tokens(tok)
            mask = Q.create_attention_mask(emb, None)
            x = self.mtp.combine(emb, hyper)
            x = self.mtp.layers[0](
                x, self.rope, mask, None, cache, cache.indexer, None, None
            )
            out = self.mtp.hyper_connection_mixer(x)
            tok = self._draft_argmax(out)
            drafts.append(tok)
            hyper = x
            if step < depth - 1:
                # 段を組み終えるごとに投入する。GPU がこの段を回している間に
                # CPU は次の段 (と rerank) を組む。呼び出し側は末尾で全体を
                # async_eval するので、廃棄されるグラフは無い
                mx.async_eval(tok)
        trim_attn_cache(cache, keep)
        return drafts

    def _prime_accepted_gap(self, toks: list, hypers: list, cache) -> None:
        """検証で確定した中間トークンを MTP キャッシュへ追いつかせる (独立
        レビュー A-1 の修正)。

        ``_draft_chain`` は毎ラウンド、戻る前にキャッシュを ``cur`` の 1 列
        まで trim する (自身の docstring どおり)。つまり ``toks[0 .. len-2]``
        (末尾の 1 個 = 次ラウンドの ``cur`` は次の ``_draft_chain`` の 1 段目が
        自分で積む) は、これを呼ばない限り一度もキャッシュに書かれない。

        ``toks``/``hypers`` は ``_verify`` が返す対応済みの組で、
        ``hypers[j]`` は「``toks[j]`` を出した直前のトランクの hyper」
        (``_prime_draft_cache`` の規約 ``(embed(t_k), hyper_{k-1})`` と同じ
        --- ここでは配列の添字 ``j`` 自体が ``k-1`` の役を兼ねる)。
        ``_draft_chain`` 自身が使う投機的な hyper 連鎖 (MTP 自身の出力から
        作った、外れているかもしれない値) ではなく、トランクの検証フォワード
        が実際に出した hyper を使う点がここの要点 --- ドラフトが外れていても
        確定後はここで正しい履歴に差し替わる。

        呼び出し前の ``cache`` は常にクリーンな (投機的な列を含まない) 状態
        である前提 (``_draft_chain`` の不変条件そのもの)。トークンの埋め込み
        と ``mtp.layers[0]`` の呼び出しは ``_draft_chain`` の 1 段と全く同じ
        形 --- 違いは hyper の出所と、logits/argmax を作らず捨てること
        (キャッシュを進めるためだけの呼び出し) だけ。
        """
        Q = _arch()
        for i in range(len(toks) - 1):
            emb = self.model.model.embed_tokens(toks[i])
            mask = Q.create_attention_mask(emb, None)
            x = self.mtp.combine(emb, hypers[i])
            self.mtp.layers[0](
                x, self.rope, mask, None, cache, cache.indexer, None, None
            )

    def _verify(self, cap, lg, drafts, temp, precomputed=None, sampler=None):
        """検証フォワードの結果から、採用するトークンと hyper を取り出す。

        ``pair`` は [cur, d1, ..., dk]。位置 j の logits は pair[j] の次の
        トークンを与える。d1 から順に一致する限り採用し、外れたところで
        打ち切ってその位置のトークンを代わりに出す。最後まで当たれば k+1 個出る。
        """
        if not drafts:
            toks = [self._sample(lg[:, 0], temp, sampler)]
            return toks, [cap.hyper[:, 0:1]], 0
        if temp > 0 or sampler is not None:
            # 位置 j のサンプルは lg[:, j] にしか依存しない (j-1 の採否は
            # 「どこで打ち切るか」を決めるだけ) ので、全位置を先に引いて
            # 一致プレフィックスだけ採用しても分布は逐次版と同一。
            # 同期が位置ごと (最大 depth+1 回) から 1 回になる。
            #
            # **この独立性はサンプラーの形に依らない。**top_p / top_k / min_p /
            # logit_bias はどれも「その位置の logits だけを見る変換」なので、
            # 受理判定 (samples[0..j-1] にしか依存しない) で条件付けても位置 j の
            # 分布は歪まない。よって投機ありでも逐次サンプリングと厳密一致する。
            # 履歴依存のもの (repetition_penalty / presence / frequency) は
            # この形では正しく載らないので、サーバー側で非投機に降ろしてある
            # (mlxturbo/runner.py の FlashSpecRunner を参照)。
            k = len(drafts)
            if sampler is not None:
                samp = sampler(lg.reshape(k + 1, -1)).reshape(1, k + 1)
            else:
                samp = mx.random.categorical(
                    lg.astype(mx.float32) / temp).reshape(1, k + 1)
            dv = mx.concatenate(drafts, axis=1)
            mx.eval(samp, dv)
            vals = samp[0].tolist()
            dvals = dv[0].tolist()
            hit = 0
            while hit < k and vals[hit] == dvals[hit]:
                hit += 1
            toks = [samp[:, j:j + 1] for j in range(hit + 1)]
            hypers = [cap.hyper[:, j:j + 1] for j in range(hit + 1)]
            return toks, hypers, hit

        # greedy はドラフト位置ごとに .item() で同期せず、argmax と一致判定を
        # まとめて 1 回の同期で取る (1 ラウンドあたり最大 depth+1 回 -> 1 回)。
        # 呼び出し側が verify 本体と同じ eval で評価済みなら precomputed で
        # 受け取り、この同期も消える
        k = len(drafts)
        if precomputed is not None and precomputed[0] is not None:
            nxt_all, dv = precomputed
        else:
            nxt_all = mx.argmax(lg, axis=-1)          # (1, k+1)
            dv = mx.concatenate(drafts, axis=1)       # (1, k)
            mx.eval(nxt_all, dv)
        vals = nxt_all[0].tolist()
        dvals = dv[0].tolist()
        hit = 0
        while hit < k and vals[hit] == dvals[hit]:
            hit += 1
        toks = [nxt_all[:, j:j + 1] for j in range(hit + 1)]
        hypers = [cap.hyper[:, j:j + 1] for j in range(hit + 1)]
        return toks, hypers, hit

    def _prime_draft_cache(self, ids, hyper):
        """Run the tail of the prompt through the MTP head once ("priming"),
        so the first ``_draft_chain()`` of a generation already has real history.

        ``ids`` and ``hyper`` must cover the same trailing positions of the
        prompt. Only the last ``PRIME_WINDOW`` pairs are fed in: acceptance
        comes from recent context, and a window keeps the cost independent of
        prompt length (this model goes to 262144 tokens).

        The pairing follows ``_draft_chain()``: at position k the head takes
        ``(embed(t_k), hyper_{k-1})``. The prompt supplies every such pair for
        k = 1 .. N-1, and the first real draft continues at k = N, so
        there is no gap and no duplicate.

        No rollback on a rejected round: every pair fed here or by ``_draft_chain()``
        is built from values the target model's verification forward already
        confirmed. The rejected guess's embedding never enters this cache --
        only its argmax is compared. Nor can this cache change what is emitted:
        the output tokens come from the target model's own logits, which never
        read it. It moves the acceptance rate, nothing else.
        """
        Q = _arch()
        cache = Q._AttnCache()
        n = min(ids.shape[1], hyper.shape[1])
        if n > PRIME_WINDOW + 1:
            n = PRIME_WINDOW + 1
            ids, hyper = ids[:, -n:], hyper[:, -n:]
        n_pairs = n - 1
        if n_pairs < 1:
            return cache
        embeds = self.model.model.embed_tokens(ids[:, 1:])
        hyper_ctx = hyper[:, :-1]
        i = 0
        _pf_t0 = time.perf_counter() if _PREFILL_TRACE else None
        _pf_ci = 0
        while i < n_pairs:
            j = min(i + PREFILL_STEP_SIZE, n_pairs)
            chunk = embeds[:, i:j]
            if _PREFILL_TRACE:
                _pf_t = time.perf_counter()
            out = self.mtp(
                chunk, hyper_ctx[:, i:j], self.rope,
                Q.create_attention_mask(chunk, None), cache, cache.indexer,
            )
            mx.eval(out)
            mx.clear_cache()
            if _PREFILL_TRACE:
                _pf_now = time.perf_counter()
                print(f"[prefill] prime chunk ci={_pf_ci} i={i} j={j} "
                      f"t={(_pf_now - _pf_t0) * 1e3:.2f} "
                      f"dur={(_pf_now - _pf_t) * 1e3:.2f}", flush=True)
                _pf_ci += 1
            i = j
        return cache

    def generate(self, ids, max_tokens: int, caches=None):
        """Greedy generation. Returns (token sequence, accepted count, round count)."""
        model = self.model
        caches = caches or model.make_cache()
        with capture(model) as cap:
            logits = model(ids, cache=caches)
            mx.eval(logits)
        hyper_prev = cap.hyper[:, -1:]
        mtp_cache = self._prime_draft_cache(ids, cap.hyper)
        cur = mx.argmax(logits[:, -1], axis=-1).reshape(1, 1)

        # **Include the first token produced by prefill in the output too.**
        # `cur` is both "the token to be fed next" and "the most recently
        # generated token", so dropping it here shifts everything by one
        out, accepted, rounds = [int(cur.item())], 0, 0
        while len(out) < max_tokens:
            # depth-adapt の費用 EMA 用: draft 構築から verify の同期までを
            # 実測する (既存の MLXTURBO_PHASE_TIMERS のような専用の強制 eval
            # は挟まない -- 下の mx.eval(lg) が元々あった同期点なので、
            # ここで測っても新たな同期は増えない)。既定 off では
            # perf_counter() を 1 回余計に呼ぶだけで、mx.eval の追加は無い。
            _round_t0 = time.perf_counter() if self._depth_adapt else None
            drafts = self._draft_chain(
                cur, hyper_prev, mtp_cache,
                self._effective_depth(ids.shape[1] + len(out)),
            )
            pair = mx.concatenate([cur] + drafts, axis=1)
            total = pair.shape[1]
            pre = snapshot_pre(model, caches)
            with capture(model) as cap:
                lg = model(pair, cache=caches)
                mx.eval(lg)
            rounds += 1
            toks, hypers, hit = self._verify(cap, lg, drafts, 0.0)
            accepted += hit
            if self._depth_adapt:
                round_ms = (time.perf_counter() - _round_t0) * 1000.0
                self._depth_controller.observe(hit, len(drafts), round_ms)
            keep = len(toks)
            rollback(model, caches, cap, pre, keep=keep, total=total,
                     ids_kept=pair[:, :keep])
            vals = [int(t.item()) for t in toks][: max_tokens - len(out)]
            out.extend(vals)
            cur, hyper_prev = toks[-1], hypers[-1]
        return out[:max_tokens], accepted, rounds

    # ---------- mlxturbo-serve wiring (added 2026-08-29, additions only) ----------
    #
    # What follows is a new path added without changing ``generate()`` at all.
    # The reason is to absolutely not change the behavior of the existing
    # ``generate()``/``capture``/``rollback``/``snapshot_pre``, which are tied to
    # the measurements in docs/MTP-FLASH.md -- even where logic could be shared,
    # we chose to accept a little duplication over rewriting existing methods.

    @staticmethod
    def _sample(logits_row: mx.array, temp: float, sampler=None) -> mx.array:
        """Choose the next token from one position's worth of logits ((1, vocab)).

        temp<=0 is greedy (argmax, numerically identical to the existing
        generate()). temp>0 samples with temperature via
        ``mx.random.categorical`` (docs/MTP-FLASH.md, "sampling" section: as far
        as sampling from the verification-side logits goes there is no
        approximation in the correctness of the distribution -- the draft stays
        greedy). Returns (1, 1).

        ``sampler`` (省略可) は「1 位置の生 logits (N, vocab) を受けてトークン
        (N,) を返す」関数。top_p/top_k/min_p/logit_bias のような**位置局所な
        変換**を載せるための口で、渡されたときだけこちらを使う (省略時は既存の
        経路が 1 ビットも変わらない)。履歴依存のもの (repetition_penalty 系) は
        ここに載せてはいけない -- 下の `_verify` が全位置を先に引くため、
        位置 j のペナルティが「j-1 までを含む履歴」で計算できない。
        """
        if sampler is not None:
            return sampler(logits_row).reshape(1, 1)
        if temp > 0:
            return mx.random.categorical(logits_row.astype(mx.float32) / temp).reshape(1, 1)
        return mx.argmax(logits_row, axis=-1).reshape(1, 1)

    def generate_stream(
        self,
        ids: mx.array,
        max_tokens: int,
        caches=None,
        temp: float = 0.0,
        eos_ids=(),
        checkpoints: list | None = None,
        base_pos: int = 0,
        resume: tuple | None = None,
        sampler=None,
        logprob_rows: list | None = None,
    ):
        """The token-by-token version of ``generate()`` (for mlxturbo-serve's
        streaming).

        Yields the list of new tokens confirmed in one round (1 or 2 of them)
        each time, and returns ``(accepted, rounds)`` at the end of generation
        (the expectation is that you drive ``next()`` manually and pick it up
        from ``StopIteration.value`` -- ``generate()`` itself does not consume
        this generator; it is a completely independent path).

        There are 3 differences from ``generate()``, all of them additions that
        simply use the existing rollback machinery (``rollback``) as it is:

        1. When temp>0, sample with temperature from the logits at positions 0/1
           of the verification forward (the draft (MTP) itself stays greedy --
           as designed in docs/MTP-FLASH.md). Only when the sample matches the
           draft do we also sample from position 1 and advance 2 tokens. When it
           does not match, the position-1 logits are discarded outright (they are
           conditioned incorrectly, so we do not use them).
        2. Stop at a token matching ``eos_ids``. ``generate()`` does not look at
           eos at all, so this is the first place it is handled.
        3. When the number of new tokens one round produces (1 or 2) exceeds the
           remainder of ``max_tokens`` or an eos boundary, the excess is reliably
           thrown away by ``rollback`` (we just pass ``keep`` matching the number
           actually adopted --- ``rollback`` itself still has its existing branch
           that returns early when ``keep == total``, so an ordinary accept round
           (keep=total=2) is effectively a no-op). This means that even if the
           caller reuses the ``caches`` at the end of this generation as the next
           turn's session, the number of processed positions in ``caches`` always
           matches the number of tokens actually returned/yielded (it does not
           break the engine's invariant "cur is the token to be fed next, and the
           caches are processed up to just before it", even when we stop in the
           middle of a round).

        Prefill is chunked at the same width as
        ``mlxturbo.spec.PREFILL_STEP_SIZE`` (to avoid exceeding Metal's
        single-buffer limit; the same reason and the same width as
        ``_prefill_hidden`` in spec.py -- a width differing per path makes the
        output diverge even for the same prompt). Only the last chunk is
        forwarded with ``capture`` to obtain the hyper state (``hyper_prev`` uses
        only the last position, so there is no need to capture intermediate
        chunks). Intermediate chunks are forwarded with ``model.model(...)``
        (hidden only, not going through lm_head) and merely advance the cache --
        we do not repeat a vocabulary-sized matmul we will not use on every
        chunk. When the whole prompt fits in one chunk (which is nearly always
        the case for real conversational turns), this chunking is numerically
        identical to calling ``model(ids, cache=caches)`` once with capture
        (because no chunk boundary arises at all) -- it goes through the very
        same path as the existing ``generate()``.

        ``checkpoints`` (None when omitted; only FlashSpecRunner in
        mlxturbo/server.py passes it): if given, a snapshot of the layers that
        cannot be rolled back (GDN recurrent state, conv window, PLE conv window,
        n-gram context -- all of which ride on the same list returned by
        ``ArraysCache.state``) is appended in-place to this list at every chunk
        boundary. The position is absolute, i.e. with ``base_pos`` added (the
        starting position of this call from the caller's point of view = the
        number of tokens the session has already reused). Once there are more
        than ``mlxturbo.spec.CHECKPOINT_RETENTION`` entries, the oldest are
        evicted -- the same machinery and the same step size as
        ``mlxturbo.spec.ChatSession``/``_prefill_hidden`` (the prefill chunk
        boundaries themselves; we do not create a new step size). KV/indexer
        (full attention) is trimmable, so it needs no snapshot -- the restore
        side (``_try_checkpoint_restore_session_cache`` in mlxturbo/server.py)
        handles it with ``.trim()`` and by following along the indexer keys.

        ``resume`` (mlxturbo-serve wiring, added 2026-08-30): a
        ``(logits_last, hyper_prev, mtp_snap)`` triple captured by a previous
        call at *exactly* this same position (see the return value below), for when
        ``ids`` has zero new tokens (a prompt that matches a session's cache
        down to the very last position -- a resend of the same prompt, a
        regenerate). The chunk loop above never executes for an empty
        ``ids`` and ``cap``/``logits`` would stay unset, so this path skips
        the loop outright and resumes decoding straight from the saved
        state; the MTP draft cache starts cold (empty) since there is no
        freshly-prefilled tail to prime it from -- this costs a little draft
        acceptance on the first few rounds, not correctness (the target
        model's own verification is what the output distribution rests on).
        Ignored (falls back to the normal loop) unless ``ids`` is actually
        empty -- a mismatched caller never gets a shortcut.

        Returns/yields the same as before, except the final ``(accepted,
        rounds)`` is now a 3-tuple ``(accepted, rounds, (logits_last,
        hyper_prev, mtp_snap))`` -- the triple at the prefill/decode boundary
        of *this* call (untouched by the decode loop below, unlike the
        ``hyper_prev`` local variable that keeps advancing), passed straight through
        unchanged when ``resume`` was used. Callers thread it back in via
        ``resume`` on a session's next call (mlxturbo/runner.py's
        ``FlashSpecRunner``).
        """
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        eos = set(eos_ids)
        model = self.model
        caches = caches if caches is not None else model.make_cache()

        n = ids.shape[1]
        use_resume = resume is not None and n == 0
        _pf = _PrefillTracer() if (_PREFILL_TRACE and not use_resume) else None
        if use_resume:
            logits_tail, hyper_tail0, mtp_snap = resume
            # The primed draft cache is carried across too: without it a
            # resumed call would draft from an empty cache and give back the
            # acceptance that priming buys (measured: decode -7%).
            mtp_cache = restore_mtp_cache(mtp_snap)
        else:
            step = PREFILL_STEP_SIZE
            # 中間チャンク幅の knob。sorted gather_qmm の GEMM 効率は
            # 1 エキスパートあたりの行数に単調 (S=2048: 7.5 TFLOPS ->
            # S=8192: 9.8、密上限 11.2) だが、in-model では attention/indexer の
            # 一時テンソル増と相殺して 17k TTFT -2% 止まり、しかもチャンク割りで
            # 境界の丸めが動き出力が揺れる。既定は従来幅のまま。8192 は
            # n-gram RAM 常駐 (wired 108GiB 張り付き) でサーバーが Metal OOM。
            # prefill の本命は MoE gather の segment-aware GEMM カーネル
            # (KERNEL-BRIEF-DECODE-BW.md)。
            big = int(os.environ.get("MLXTURBO_PREFILL_CHUNK", "0") or 0) or step
            i = 0
            logits = None
            cap = None
            # The trailing prefill chunks' hyper state (B, chunk_len, hc*d),
            # kept to prime the MTP's own cache below. Only the last
            # HYPER_KEEP_CHUNKS are retained: priming reads at most
            # PRIME_WINDOW+1 positions, and at this model's 262144-token
            # ceiling holding all of them would be gigabytes.
            # capture(light=True) records only GatedResidual's hyper, so this
            # does not reintroduce the OOM that `light` exists to avoid.
            hyper_chunks = []
            if _pf:
                _t = time.perf_counter()
            _prefetch_ngram_rows(model, ids, caches)
            if _pf:
                _t = _pf.log("prefetch", _t)
            while i < n:
                remaining = n - i
                # 前方の等長 2048 チャンクはレイヤー主導でグループ処理する
                # (チャンク境界は従来と同一 grid なので出力はビット一致)。
                # グループと最終チャンクの間に残る端数チャンク (幅 < step) は
                # 単独だと MoE の専門家あたり行数が薄く gather_qmm の効率が
                # 落ちるので、直前のグループに余裕 (< _PREFILL_GROUP) があれば
                # そのグループの最後のチャンクとして畳み込み、余裕が無ければ
                # 端数だけの g=1 グループとして _group_prefill_forward に通す
                # (どちらも chunk-major より段階投入と MoE の呼び出し経路が
                # 揃うぶん有利なはず)。最終チャンク (BPE 境界 checkpoint 用の
                # 分割がある) だけは従来経路のまま — checkpoint の粒度が
                # 要るのはそこだけだから。
                # MLXTURBO_PREFILL_CHUNK 指定時は旧 knob を優先して無効化。
                g = min(
                    _PREFILL_GROUP,
                    (remaining - _PREFILL_TAIL_CHUNKS * step) // step,
                )
                group_chunks = None
                frac_len = 0  # 端数チャンクをこの回に含めるときの幅 (0 なら無し)
                if _PREFILL_GROUP > 1 and big == step and g >= 2:
                    group_chunks = [
                        ids[:, i + k * step : i + (k + 1) * step] for k in range(g)
                    ]
                    if _PREFILL_FOLD_TAIL and g < _PREFILL_GROUP:
                        # このグループの直後に来る幅を覗く。ちょうど端数
                        # (step < after < 2*step) なら、上限 G を超えない
                        # うちに同じ _group_prefill_forward 呼び出しへ畳み込む。
                        after = remaining - g * step
                        if step < after < 2 * step:
                            frac_len = after - step
                            group_chunks.append(
                                ids[:, i + g * step : i + g * step + frac_len]
                            )
                elif (
                    _PREFILL_FOLD_TAIL
                    and _PREFILL_GROUP > 1
                    and big == step
                    and remaining > step
                    and remaining - step < step
                ):
                    # 直前にグループが無い (無いか、上限 G で畳み込めない) 端数
                    # 単体。g=1 のグループとして _group_prefill_forward に通す。
                    frac_len = remaining - step
                    group_chunks = [ids[:, i : i + frac_len]]
                if group_chunks is not None:
                    consumed = sum(c.shape[1] for c in group_chunks)
                    gn = len(group_chunks)
                    if _pf:
                        _t = time.perf_counter()
                    hys = _group_prefill_forward(model, group_chunks, caches)
                    if _pf:
                        label = f"group build i={i} g={gn}"
                        if frac_len:
                            label += f" tokens={consumed}"
                        _t = _pf.log(label, _t)
                    hys = hys[-HYPER_KEEP_CHUNKS:]
                    states = [
                        st for c in caches
                        if (st := getattr(c, "state", None)) is not None
                    ]
                    if _PREFILL_PIPELINE:
                        # グループ境界の完全同期をやめ、非同期投入にして次の
                        # グループのグラフ構築を先に始める。全部使うグラフ
                        # なので「作って捨てる」禁則には当たらない。
                        #
                        # **既定 off。**2 グループぶんの中間が同時に生きるので、
                        # wired limit + n-gram RAM 32GB で張り付いた構成では
                        # OOM 側に倒れうる (128GB に 91GB のモデルが載っている)。
                        # clear_cache も打てない (飛行中のバッファを掴んでいる)。
                        # 取り分の見積もりは 17k で 0.5-1s (1.5-3%) と小さく、
                        # レップのばらつき (±3%) に埋もれる可能性がある。
                        # 有効にする前に in-model で測ること
                        # (docs/research/IMPROVEMENT-QUEUE.md D5)。
                        mx.async_eval(*hys, *states)
                        if _pf:
                            # 非同期投入なので、ここでの dur は GPU 完了を
                            # 待っていない (build+async の意)
                            _t = _pf.log(f"group eval i={i} g={gn} build+async", _t)
                    else:
                        mx.eval(*hys)
                        for st in states:
                            mx.eval(st)
                        if _pf:
                            _t = _pf.log(f"group eval i={i} g={gn}", _t)
                        mx.clear_cache()
                        if _pf:
                            _t = _pf.log(f"clear_cache i={i} g={gn}", _t)
                    hyper_chunks.extend(hys)
                    del hyper_chunks[:-HYPER_KEEP_CHUNKS]
                    i += consumed
                    if checkpoints is not None:
                        if _pf:
                            _t = time.perf_counter()
                        checkpoints.append(
                            (base_pos + i, snapshot_untrimmable_caches(caches))
                        )
                        del checkpoints[:-CHECKPOINT_RETENTION]
                        if _pf:
                            _pf.log(f"checkpoint i={i} g={gn}", _t)
                    continue
                if remaining > step:
                    j = i + min(big, remaining - step)
                else:
                    j = n
                chunk = ids[:, i:j]
                if j == n:
                    # BPE 境界 checkpoint (共有ヘルパー: prefill_common.py の
                    # split_and_checkpoint_tail、詳しい背景はそちらの
                    # docstring 参照)。checkpoints が有効かつ chunk が 2
                    # トークン以上のときだけ、直前 1 トークンを切り離して
                    # 手前 (head) にも checkpoint を積む -- そうしないと
                    # 会話 2 ターン目の retemplate で末尾トークンが BPE
                    # マージにより化け、LCP が checkpoint のちょうど 1
                    # トークン手前に落ちてセッション全体が使い捨てになる
                    # (実測: 診断で確認)。no-op 時 (checkpoints=None または
                    # chunk 長 1 以下、generate()/検証プローブがこちら) は
                    # head_result が空 tuple で返り、tail_split は False の
                    # まま従来の分岐に合流する。副次効果として分割時は最終
                    # チャンクの lm_head が 1 トークン分に縮む (従来は
                    # チャンク全幅の logits を作って末尾だけ使っていた)。
                    def _forward_head(head):
                        with capture(model, light=True) as cap0:
                            h0 = model.model(head, cache=caches)
                        return h0, cap0.hyper

                    if _pf:
                        _t = time.perf_counter()
                    chunk, head_result = split_and_checkpoint_tail(
                        chunk,
                        checkpoints,
                        base_pos + i,
                        caches,
                        CHECKPOINT_RETENTION,
                        snapshot_untrimmable_caches,
                        _forward_head,
                    )
                    if _pf:
                        _t = _pf.log(f"tail split i={i} j={j}", _t)
                    tail_split = bool(head_result)
                    # light=True: this chunk uses only cap.hyper[:, -1:]
                    # (referenced right below). Full capture (cap.gdn/cap.ple)
                    # unconditionally allocated memory proportional to T (this
                    # chunk's length, at most PREFILL_STEP_SIZE) for all 36
                    # layers, and OOMed the actual machine at a few thousand
                    # tokens (see the docstring for capture()'s light
                    # argument). The decode loop's verification forward
                    # (below, T<=2) stays on full capture as before
                    with capture(model, light=True) as cap:
                        logits = model(chunk, cache=caches)
                        mx.eval(logits)
                    if _pf:
                        _t = _pf.log(f"tail forward i={i} j={j}", _t)
                    if tail_split:
                        cap.hyper = mx.concatenate([head_result[1], cap.hyper], axis=1)
                else:
                    if _pf:
                        _t = time.perf_counter()
                    # light=True (added for MTP priming): only the cheap
                    # GatedResidual hook runs, so this branch's computation and
                    # cache updates are unchanged -- we additionally record
                    # cap.hyper for this chunk.
                    with capture(model, light=True) as cap:
                        h = model.model(chunk, cache=caches)
                        mx.eval(h)
                    for c in caches:
                        state = getattr(c, "state", None)
                        if state is not None:
                            mx.eval(state)
                    if _pf:
                        _t = _pf.log(f"tail forward i={i} j={j}", _t)
                    mx.clear_cache()
                    if _pf:
                        _t = _pf.log(f"clear_cache i={i} j={j}", _t)
                hyper_chunks.append(cap.hyper)
                del hyper_chunks[:-HYPER_KEEP_CHUNKS]
                i = j
                if checkpoints is not None:
                    if _pf:
                        _t = time.perf_counter()
                    checkpoints.append((base_pos + i, snapshot_untrimmable_caches(caches)))
                    del checkpoints[:-CHECKPOINT_RETENTION]
                    if _pf:
                        _pf.log(f"checkpoint i={i}", _t)
            # mx.contiguous, not a bare slice: a slice keeps its parent
            # buffer alive, and these two outlive the call (they are published
            # as the session's tail for a later diff-0 resume). The last
            # chunk's ``logits`` is (1, chunk_len, vocab) -- 2GB at
            # PREFILL_STEP_SIZE with this vocabulary -- so holding a view of it
            # in every pooled session would retain gigabytes per slot.
            hyper_tail0 = mx.contiguous(cap.hyper[:, -1:])
            logits_tail = mx.contiguous(logits[:, -1])
            if max_tokens == 0:
                # Keep the successfully prefetched cache, but do not expose the
                # first sampled ``cur``.  This mirrors SpecEngine.generate(0) and
                # lets FlashSpecRunner publish exactly the prompt as processed.
                if _pf:
                    _pf.mark_end()
                    _pf.summary()
                return 0, 0, (logits_tail, hyper_tail0, None)
            hyper_tail = (
                hyper_chunks[0] if len(hyper_chunks) == 1
                else mx.concatenate(hyper_chunks, axis=1)
            )
            if _pf:
                _t = time.perf_counter()
            mtp_cache = self._prime_draft_cache(ids[:, -hyper_tail.shape[1]:], hyper_tail)
            if _pf:
                _t = _pf.log("prime", _t)
            mtp_snap = snapshot_mtp_cache(mtp_cache)
            if _pf:
                _pf.mark_end()

        if max_tokens == 0:
            # Only reachable via ``use_resume`` -- the normal path already
            # returned above.
            return 0, 0, (logits_tail, hyper_tail0, mtp_snap)

        hyper_prev = hyper_tail0
        cur = self._sample(logits_tail, temp, sampler)

        first = int(cur.item())
        out = [first]
        if logprob_rows is not None:
            logprob_rows.extend(_logsoftmax_rows(logits_tail, 1))
        if _pf:
            _pf.log("first token", _pf.end)
            _pf.summary()
        yield [first]
        accepted, rounds = 0, 0
        if first in eos:
            return accepted, rounds, (logits_tail, hyper_tail0, mtp_snap)

        # 計測モード (MLXTURBO_PHASE_TIMERS=1): フェーズ境界で強制 eval して
        # draft / verify / post の実時間を分ける。強制 eval 自体が同期を増やす
        # ので、絶対値は少し膨らむ。比率を見るためのもの。既定では完全に素通り。
        timers = os.environ.get("MLXTURBO_PHASE_TIMERS") == "1"
        phase = {"draft": 0.0, "verify": 0.0, "post": 0.0, "rollback": 0.0}
        self.last_phase = phase if timers else None
        # 楽観パイプライン: verify の GPU 実行中に「全採用だった場合の次ラウンド」
        # のグラフを CPU で先に組む。全採用なら rollback は no-op なので先組みが
        # そのまま正しく、外れたら _pipeline_restore で参照を戻して組み直す。
        # GPU トレース実測でラウンド毎に ~7ms の泡 (グラフ構築中の GPU アイドル)
        # があり、全採用率 ~0.7 との積で泡の大半が消える。greedy のみ。
        pending = None
        next_drafts = None
        # 1=通常の楽観パイプライン, 2=組むが毎回捨てる (切り分け用), 0=無効
        pipeline = int(os.environ.get("MLXTURBO_PIPELINE", "0") or 0)
        while len(out) < max_tokens:
            # depth-adapt の費用 EMA 用 (generate() と同じ理由 -- 既存の
            # 同期点で測るだけで、新たな mx.eval は増やさない)。
            #
            # 制約: MLXTURBO_PIPELINE 有効時、`pending` 経由のラウンドは
            # このラウンドの GPU 仕事が実は**前のラウンドの余り時間**で
            # 既に終わっている (楽観先組み)。その場合ここで測る経過時間は
            # 実際の depth 依存コストを過小評価する。pipeline は既定 off
            # かつ depth-adapt との組み合わせは今回のゲート対象外なので、
            # 単純な「ループ先頭から hit 確定まで」のまま許容している。
            _round_t0 = time.perf_counter() if self._depth_adapt else None
            if timers:
                ts = time.perf_counter()
            if pending is not None:
                drafts, pair, total, pre, cap, lg, pipe_snap = pending
                pending = None
                pipe_snap = None
            else:
                if next_drafts is not None:
                    drafts = next_drafts        # 前ラウンド末尾で構築・投機済み
                    next_drafts = None
                else:
                    drafts = self._draft_chain(
                        cur, hyper_prev, mtp_cache,
                        self._effective_depth(base_pos + n + len(out)),
                    )
                    # draft は必ず使うので先に投げる。GPU が draft チェーンを
                    # 回している間に、CPU は下の検証フォワードのグラフを組む
                    mx.async_eval(drafts)
                pair = mx.concatenate([cur] + drafts, axis=1)
                total = pair.shape[1]
                pre = snapshot_pre(model, caches)
                with capture(model) as cap:
                    lg = _staged_forward(model, pair, caches)
            if timers:
                mx.eval(drafts)
                phase["draft"] += time.perf_counter() - ts
                ts = time.perf_counter()
            next_pending = None
            if pipeline > 0 and temp <= 0 and len(out) + total < max_tokens:
                mx.async_eval(lg)
                snap2 = _pipeline_snapshot(model, caches, mtp_cache)
                cur2 = mx.argmax(lg[:, total - 1], axis=-1).reshape(1, 1)
                hyper2 = cap.hyper[:, total - 1: total]
                drafts2 = self._draft_chain(
                    cur2, hyper2, mtp_cache,
                    self._effective_depth(base_pos + n + len(out) + total),
                )
                pair2 = mx.concatenate([cur2] + drafts2, axis=1)
                pre2 = snapshot_pre(model, caches)
                with capture(model) as cap2:
                    lg2 = model(pair2, cache=caches)
                next_pending = (drafts2, pair2, pair2.shape[1], pre2, cap2,
                                lg2, snap2)
            if _ROUND_TRACE:
                _rt = [("built", time.perf_counter())]
            # _verify の分岐 (`temp > 0 or sampler is not None`) と揃える --
            # sampler ありなら temp<=0 でも sampler 側に入り precomputed は
            # 無視される (独立レビュー A-6)。揃えないと argmax/concat を
            # 評価してから丸ごと捨てるだけの無駄になる。
            if temp <= 0 and sampler is None and drafts:
                nxt_all = mx.argmax(lg, axis=-1)
                dv = mx.concatenate(drafts, axis=1)
                mx.eval(lg, nxt_all, dv)
            else:
                nxt_all = dv = None
                mx.eval(lg)
            if _ROUND_TRACE:
                _rt.append(("eval_done", time.perf_counter()))
            rounds += 1
            if timers:
                phase["verify"] += time.perf_counter() - ts
                ts = time.perf_counter()
            toks, hypers, hit = self._verify(cap, lg, drafts, temp, sampler=sampler,
                                             precomputed=(nxt_all, dv))
            accepted += hit
            if self._depth_adapt:
                # このラウンドで実際に引いた深さ (drafts はここでは
                # pending/next_drafts のどちらから来ても、このラウンドで
                # 検証フォワードにかけた本数そのもの)。round_ms は
                # _round_t0 (ループ先頭、上のコメント参照) から今までの
                # 経過時間 -- ここまでに mx.eval(lg, ...) 済みなので GPU の
                # 仕事は終わっている。
                round_ms = (time.perf_counter() - _round_t0) * 1000.0
                self._depth_controller.observe(hit, len(drafts), round_ms)

            vals = [int(t.item()) for t in toks]
            cut = next((k for k, v in enumerate(vals) if v in eos), None)
            if cut is not None:
                toks, hypers, vals = toks[: cut + 1], hypers[: cut + 1], vals[: cut + 1]
            remaining = max_tokens - len(out)
            if len(vals) > remaining:
                toks, hypers, vals = toks[:remaining], hypers[:remaining], vals[:remaining]

            # When keep==total (an ordinary accept round with no truncation),
            # rollback() itself returns early, so it is fine to always call it.
            if timers:
                phase["post"] += time.perf_counter() - ts
                ts = time.perf_counter()
            full_accept = (len(vals) == total and cut is None)
            if next_pending is not None:
                if full_accept and pipeline == 1:
                    pending = next_pending
                else:
                    _pipeline_restore(model, caches, mtp_cache, next_pending[6])
            if _MTP_CACHE_APPEND and not (full_accept and pipeline == 1):
                # 独立レビュー A-1: このラウンドで確定した中間トークンを MTP
                # キャッシュへ積み直す。pipeline (既定 off) の先組みが採用され
                # た (pending が立った) ときは、その先組み自身が古い hyper 連鎖
                # で次ラウンドの cur をすでに積んでしまっているので、ここで
                # 割り込むと順序が壊れる -- 触らない (既定 off 経路を壊さない
                # ことだけが要件、そちらの受理率まで直すのはこの修正の範囲外)。
                self._prime_accepted_gap(toks, hypers, mtp_cache)
            cur, hyper_prev = toks[-1], hypers[-1]
            # 次ラウンドの draft をここで組んで投げる (rollback / yield などの
            # CPU 後処理を draft の GPU 実行に隠す)。draft はトランクの
            # キャッシュに触れず、rollback は MTP のキャッシュに触れないので
            # 順序を入れ替えても意味は変わらない。最終ラウンドでは無駄になるが
            # 1 リクエスト 1 回きり。
            if _ROUND_TRACE:
                _rt.append(("verify_done", time.perf_counter()))
            next_drafts = None
            if (pending is None and cut is None
                    and len(out) + len(vals) < max_tokens):
                next_drafts = self._draft_chain(
                    cur, hyper_prev, mtp_cache,
                    self._effective_depth(base_pos + n + len(out) + len(vals)),
                )
                mx.async_eval(next_drafts)
            if _ROUND_TRACE:
                _rt.append(("drafts_submitted", time.perf_counter()))
            rollback(model, caches, cap, pre, keep=len(vals), total=total,
                     ids_kept=pair[:, : len(vals)])
            if timers:
                mx.eval([c.state for c in caches if getattr(c, "state", None) is not None])
                phase["rollback"] += time.perf_counter() - ts
            out.extend(vals)
            if _ROUND_TRACE:
                _rt.append(("rollback_done", time.perf_counter()))
                base_t = _rt[0][1]
                print(f"[round] t={base_t * 1e3:.2f}", " ".join(
                    f"{k}={(t - base_t) * 1e3:.2f}" for k, t in _rt[1:]), flush=True)
            if logprob_rows is not None:
                # 採用位置 j の logits は pair[:j+1] に正しく条件付いている
                # (棄却位置のものは採用側に混ざらない)。呼び手はラウンドごとに
                # 引き取って entry へ畳むこと -- 語彙長のベクトルを溜め込むと
                # 512 トークンで数百 MB になる。
                logprob_rows.extend(_logsoftmax_rows(lg, len(vals)))
            yield vals
            if cut is not None:
                break

        return accepted, rounds, (logits_tail, hyper_tail0, mtp_snap)


__all__ = ["Capture", "FlashSpecEngine", "capture", "rollback", "snapshot_pre"]
