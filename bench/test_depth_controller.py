"""mlxturbo/spec_flash.DepthController の単体テスト (CPU only, モデル不要)。

レーン10 (docs/research/LANES-2026-09.md): 投機デコードの draft 深さを、
文脈長だけでなく受理率の実測 (EMA) から選び直す適応制御。GPU も実モデルも
使わない -- 純粋な python のクラス/関数を直接叩くだけ。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlxturbo.spec_flash import (
    DepthController,
    _depth_beta_default,
    _depth_cap_default,
    _depth_cost_params,
)


def test_prior_chooses_cap_at_start():
    """観測前 (全位置が事前値 0.85) は E(m)/T(m) が m について単調に増える
    ので、cap そのものを選ぶ (短文脈 cap=3、長文脈 (>2048) cap=2)。"""
    ctl = DepthController()
    assert _depth_cap_default(0) == 3
    assert _depth_cap_default(4000) == 2
    assert ctl.choose(0) == 3
    assert ctl.choose(4000) == 2


def test_high_acceptance_prefers_deeper():
    """毎ラウンド depth 全部が的中し続けると、a が 1 に近づいて cap まで
    深く選ぶようになる。"""
    ctl = DepthController()
    for _ in range(200):
        ctl.observe(n_accepted=3, depth=3)
    assert all(a > 0.99 for a in ctl.a[:3])
    assert ctl.choose(0) == 3


def test_low_acceptance_prefers_shallow():
    """毎ラウンド即座に外れ続けると、a[0] が 0 に近づいて最も浅い depth=1
    を選ぶようになる。"""
    ctl = DepthController()
    for _ in range(200):
        ctl.observe(n_accepted=0, depth=3)
    assert ctl.a[0] < 0.05
    assert ctl.choose(0) == 1


def test_observe_updates_only_up_to_first_miss():
    """的中した位置は 1 に、最初に外れた位置は 0 に更新し、それより先の
    位置は (ドラフトのチェーンがそこで既に切れていて検証していないので)
    未観測のまま事前値を保つ。"""
    ctl = DepthController()
    ctl.observe(n_accepted=2, depth=4)
    beta, prior = ctl.beta, 0.85
    assert ctl.a[0] == (1 - beta) * prior + beta * 1.0
    assert ctl.a[1] == (1 - beta) * prior + beta * 1.0
    assert ctl.a[2] == (1 - beta) * prior + beta * 0.0
    assert ctl.a[3] == prior  # 未観測
    assert ctl.observations[:4] == [1, 1, 1, 0]
    assert ctl.observations[4:] == [0] * (ctl.max_depth - 4)


def test_observe_full_accept_touches_every_drafted_position():
    """depth 全部が的中したラウンドは、引いた本数ぶんだけ位置 0..depth-1 が
    観測され、それより先 (未使用の位置) には触らない。"""
    ctl = DepthController()
    ctl.observe(n_accepted=3, depth=3)
    assert ctl.observations[:3] == [1, 1, 1]
    assert ctl.observations[3:] == [0] * (ctl.max_depth - 3)


def test_expected_tokens_matches_closed_form():
    ctl = DepthController()
    ctl.a[0], ctl.a[1], ctl.a[2] = 0.9, 0.7, 0.5
    assert ctl.expected_tokens(0) == 1.0
    assert ctl.expected_tokens(1) == 1 + 0.9
    assert abs(ctl.expected_tokens(2) - (1 + 0.9 + 0.9 * 0.7)) < 1e-12
    assert abs(
        ctl.expected_tokens(3) - (1 + 0.9 + 0.9 * 0.7 + 0.9 * 0.7 * 0.5)
    ) < 1e-12


def test_choose_matches_bruteforce_argmax():
    """choose() が探索する範囲 (cap まで) を総当たりした結果と一致する。"""
    ctl = DepthController()
    ctl.a[0], ctl.a[1], ctl.a[2] = 0.6, 0.9, 0.2
    for pos in (0, 4000):
        t1, dt = _depth_cost_params(pos)
        cap = _depth_cap_default(pos)
        scores = {
            m: ctl.expected_tokens(m) / (t1 + m * dt) for m in range(1, cap + 1)
        }
        expect = max(scores, key=scores.get)
        assert ctl.choose(pos) == expect


def test_depth_cap_env_override(monkeypatch):
    monkeypatch.setenv("MLXTURBO_DEPTH_CAP", "5")
    assert _depth_cap_default(0) == 5
    assert _depth_cap_default(4000) == 5
    # max_depth を超えて指定しても頭打ちにする
    monkeypatch.setenv("MLXTURBO_DEPTH_CAP", "99")
    assert _depth_cap_default(0) == DepthController().max_depth


def test_depth_cost_env_override(monkeypatch):
    monkeypatch.setenv("MLXTURBO_DEPTH_COST", "10,2")
    assert _depth_cost_params(0) == (10.0, 2.0)
    assert _depth_cost_params(4000) == (10.0, 2.0)


def test_depth_beta_env_override(monkeypatch):
    monkeypatch.delenv("MLXTURBO_DEPTH_BETA", raising=False)
    assert _depth_beta_default() == 0.1  # 既定 (2026-09-03 に 0.15 から下げた)
    monkeypatch.setenv("MLXTURBO_DEPTH_BETA", "0.3")
    assert _depth_beta_default() == 0.3
    assert DepthController().beta == 0.3
    # 明示引数は env より優先する
    assert DepthController(beta=0.05).beta == 0.05
    # 0 以下は無視して既定に落ちる
    monkeypatch.setenv("MLXTURBO_DEPTH_BETA", "0")
    assert _depth_beta_default() == 0.1


def test_cost_ema_shifts_choice_away_from_linear_model():
    """線形モデル (cold start) は受理率が低いと depth を 1 に抑えるが、
    実測のラウンド費用 EMA が「深くしてもさほど高くつかない」と示せば
    選択が変わる。

    2026-09-03 の短文脈 A/B (bench/results/depth-adapt-short.json) の再現:
    線形モデルの dT=7 は実測 (+4.5ms 前後) より深さを重く罰っていて、
    depth 1 に張り付いて既定の静的 depth 2 に ms/tok で負けていた。
    """
    # beta=0 で位置別 a[] を凍結し、費用 EMA だけの効果を見る
    ctl = DepthController(beta=0.0)
    ctl.a[0] = ctl.a[1] = ctl.a[2] = 0.5

    # cold start (cost_ema が空): 線形モデル T1=25, dT=7 のままだと depth 1
    assert ctl.choose(0) == 1

    # 実測: depth 2 は depth 1 よりわずかしか高くつかない (線形モデルが
    # 想定する dT=7 よりずっと軽い、実測 ~3ms 相当)
    ctl.observe(n_accepted=1, depth=1, round_ms=30.0)
    ctl.observe(n_accepted=2, depth=2, round_ms=33.0)
    assert ctl.choose(0) == 2

    # a[] は beta=0 のままなので凍結されているはず (費用 EMA だけの効果と
    # 切り分けるための前提)
    assert ctl.a[0] == ctl.a[1] == ctl.a[2] == 0.5


def test_observe_without_round_ms_leaves_cost_ema_untouched():
    """round_ms を渡さない呼び出し (既存の意味) は費用 EMA に触らない。"""
    ctl = DepthController()
    ctl.observe(n_accepted=1, depth=1)
    assert ctl.cost_ema == {}


def test_cost_ema_extrapolates_from_nearest_observed_depth():
    ctl = DepthController(beta=0.0)
    ctl.observe(n_accepted=2, depth=2, round_ms=40.0)  # depth=2 だけ観測
    t1, dt = _depth_cost_params(0)
    # depth=1 (未観測): 最も近い観測 (depth=2) から dT*(1-2) だけ補外
    assert ctl._cost_for(1, 0) == 40.0 - dt
    # depth=3 (未観測): dT*(3-2) だけ加算
    assert ctl._cost_for(3, 0) == 40.0 + dt
    # depth=2 (観測済み): EMA そのもの
    assert ctl._cost_for(2, 0) == 40.0
