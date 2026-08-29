"""n-gram lookup (SAM) だけを使う、モデル非依存の投機デコード経路。

Kimi K3 レビュー項目 12。``mlxturbo.spec.SpecEngine`` の投機は 2 つの部品
(MTP によるドラフトと、n-gram lookup (SAM) によるドラフト) からできているが、
後者は「これまでに実際に出た/読まれたトークン列の中に、今の続きの接頭辞と
同じ並びが前にも出ていないか」を見るだけの純粋な文字列 (トークン ID 列) 照合
で、モデルの重みも層構成も一切見ない。``mlxturbo/sam.py`` の
``SuffixAutomaton`` (この変更では一切書き換えていない、既存の汎用ユーティリ
ティ) がその照合を O(1) 償却で行う。

``mlxturbo/spec.py`` は触らない制約 (実測検証済みのエンジン本体) なので、
lookup 部分だけを ``SuffixAutomaton`` から直接組み立て直す、独立した runner
としてここに書く。対象は「KV キャッシュが ``trim()`` 可能なモデル」に限る:
GDN のような線形状態は途中位置へ巻き戻せない (``mlxturbo/spec.py`` の
``ChatSession`` docstring と同じ制約) が、全層 attention (KV だけ) のモデル
なら ``mlx_lm.models.cache.trim_prompt_cache`` で素直に巻き戻せる。

対応するのは貪欲 (temperature 0) だけ。temperature > 0 は「ドラフトが外れた
ときに、検証側の分布から正しく再サンプルする」導出が要る
(``mlxturbo.runner.DraftSpecRunner`` が使う ``mlx_lm.generate.
speculative_generate_step`` はこれを検証側の sampler 呼び出し1回で解決して
いるが、それは「検証モデル自身がドラフトとは独立にサンプルし、一致すれば
それを使う」という mlx_lm 側の設計に依存しており、lookup ドラフト側にはその
「検証モデル」に相当するものが無い — ドラフト自体が「モデルの外」の文字列
照合なので、同じ組み方はできない)。ここでは非対応のまま踏み込まず、
``temp > 0`` (または repetition_penalty 等、貪欲でも出力を変える logits
processor が要求された) ときは ``FallbackRunner`` へその場で委譲する
(``generate()`` 内部での降格 — server.py 側のルーティングを増やさずに済む、
このモジュール docstring 末尾の判断参照)。
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

from .runner import FallbackRunner
from .sam import SuffixAutomaton
from .spec import PREFILL_STEP_SIZE


def _prefill(model, cache, y: mx.array, step: int) -> mx.array:
    """``mlx_lm.generate.speculative_generate_step._prefill`` と同じ形
    (公開 API である ``mlx_lm.models.cache`` の上に書いた、独立した実装 —
    ``mlxturbo/spec.py`` は読んでも参照してもいない)。最後の 1 トークンだけ
    未 feed のまま残す (呼び出し側がそれを最初の「確定済みだが未 feed」の
    ペンディングトークンとして使う)。"""

    while y.size > 1:
        n = min(step, y.size - 1)
        model(y[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[n:]
    return y


def _needs_logits_processors(
    repetition_penalty: float | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    logit_bias: dict | None,
) -> bool:
    """恒等値 (分布/貪欲結果を変えない既定値) かどうか。server.py の
    ``_IDENTITY_SAMPLING_VALUES`` と同じ値の集合を使う (このモジュールは
    server.py に依存させたくないので、値そのものをここへ複製している —
    どちらかを変えたら両方直すこと)。"""

    if logit_bias:
        return True
    if repetition_penalty not in (None, 0.0, 1.0):
        return True
    if presence_penalty not in (None, 0.0):
        return True
    if frequency_penalty not in (None, 0.0):
        return True
    return False


class LookupSpecRunner:
    """n-gram lookup (SAM) だけで投機する runner。モデルのアーキテクチャを
    一切見ないので、``mlxturbo.runner.build_runner`` が spec/flash_spec の
    契約を満たさないと判定したモデル (= 通常なら ``FallbackRunner``) に
    かぶせて使う (``build_runner`` の ``--lookup-spec`` 分岐参照)。

    ``SUPPORTED_SAMPLING_PARAMS``: ``FallbackRunner`` と同じ全キーを宣言する
    — このクラスは、値の組み合わせによって「n-gram lookup 投機」か「plain
    (無投機) 生成」かを自分の中で選ぶだけで、どちらの経路でも最終的な出力
    分布は変えない (plain 経路は内部で保持する ``FallbackRunner`` インスタンス
    にそのまま委譲する、以下 ``generate()`` 参照)。あるリクエストにとって
    どちらが選ばれるかは呼び出し側から見えないし、見る必要もない。
    """

    KIND = "lookup_spec"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, model, tokenizer, max_draft: int = 8, min_match: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_draft = max_draft
        self.min_match = min_match
        self.fallback_reason = None
        # plain (無投機) 経路への委譲用。session 付きリクエストがこちらへ
        # 落ちた場合、FallbackRunner 自身の LCP 再利用がそのまま効く
        # (FallbackSession の契約どおり) — 実装を複製せず、委譲だけで済む。
        self._fallback = FallbackRunner(model, tokenizer)
        # KV キャッシュが trim 可能かどうかはモデルの層構成 (GDN 混在か
        # どうか) で決まり、リクエストごとに変わらないので構築時に 1 度だけ
        # 判定する。実際に確保するのはダミーの空キャッシュ (make_prompt_cache
        # はここでは Python オブジェクトを作るだけで GPU 計算は伴わない)。
        probe_cache = make_prompt_cache(model)
        self.trimmable = can_trim_prompt_cache(probe_cache)

    def generate(
        self,
        prompt_ids,
        max_tokens,
        temp,
        eos_ids,
        on_tokens,
        session,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        **extra,
    ):
        plain_only = (
            not self.trimmable
            or temp > 0.0
            or _needs_logits_processors(
                repetition_penalty, presence_penalty, frequency_penalty, logit_bias
            )
        )
        if plain_only:
            # temp==0 なら top_p/top_k/min_p は mlx_lm.sample_utils.make_sampler
            # 自身が argmax に短絡して無視する (mlx_lm/sample_utils.py の
            # make_sampler 46 行目) ので、ここで気にする必要はない —
            # 気にする必要があるのは logits_processors 側 (repetition_penalty
            # 等) と logit_bias だけ。
            return self._fallback.generate(
                prompt_ids,
                max_tokens,
                temp,
                eos_ids,
                on_tokens,
                session,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                seed=seed,
                **extra,
            )
        if seed is not None:
            # 貪欲なので乱数は出力に影響しないが、SpecRunner/FallbackRunner
            # と同じく引数として受けた以上は消費しておく (呼び出し側の
            # 「seed を渡したのに黙って無視された」を避ける)。
            mx.random.seed(seed)
        return self._lookup_generate(prompt_ids, max_tokens, eos_ids, on_tokens)

    def _lookup_generate(self, prompt_ids, max_tokens, eos_ids, on_tokens) -> dict:
        """貪欲、n-gram lookup 投機の本体。

        1 ラウンドにつき: (a) 直前までに確定した接頭辞と同じ並びが履歴の
        どこかに前にも出ていれば、その続きを ``draft`` として提案
        (``SuffixAutomaton.draft``)、(b) ペンディングトークン + draft を
        まとめて 1 回 forward (teacher forcing)、(c) 各位置の argmax が
        draft の次の要素と一致する間だけ受理し、最初の不一致位置の argmax
        を「ボーナストークン」として emit、(d) 受理されなかった draft 分は
        ``trim_prompt_cache`` で KV を巻き戻す。

        draft が無い/空なら (b) は 1 トークンだけの forward になり、通常の
        1 トークンずつの貪欲デコードと同じコスト・同じ出力になる — 一致が
        起きない場面で遅くならないことの根拠 (実測は別途、docs/ 配下は
        今回の変更範囲外なのでここに書く)。

        session (会話ごとの prompt cache 再利用) はここでは扱わない —
        毎回そのリクエスト専用の cache を新規に作る (呼び出し側の session
        は読みも書きもしない。次ターンでは通常どおり全量再プレフィルになる
        だけで、誤動作はしない)。
        """

        t0 = time.perf_counter()
        cache = make_prompt_cache(self.model)
        ids = mx.array(prompt_ids, dtype=mx.uint32)
        y = _prefill(self.model, cache, ids, PREFILL_STEP_SIZE)

        sam = SuffixAutomaton()
        sam.extend_all(prompt_ids)

        tokens: list[int] = []
        ttft: float | None = None
        rounds = 0
        stop = False
        while len(tokens) < max_tokens and not stop:
            rounds += 1
            budget_left = max_tokens - len(tokens)
            # ボーナストークン用に必ず 1 枠残す (draft を全部受理しても
            # 最後にちょうど 1 個「新しい」トークンが出る、DraftSpecRunner/
            # mlx_lm.speculative_generate_step と同じ構造)。
            draft_cap = max(0, min(self.max_draft, budget_left - 1))
            draft = sam.draft(draft_cap, min_len=self.min_match) if draft_cap > 0 else None
            cand = y.tolist() + (draft or [])
            cand_arr = mx.array(cand, dtype=mx.uint32)
            logits = self.model(cand_arr[None], cache=cache)
            mx.eval(logits)
            if ttft is None:
                ttft = time.perf_counter() - t0
            preds = mx.argmax(logits[0], axis=-1).tolist()

            m = len(draft) if draft else 0
            accepted = 0
            while accepted < m and preds[accepted] == cand[accepted + 1]:
                accepted += 1
            bonus = preds[accepted]
            rejected = m - accepted
            if rejected > 0:
                trim_prompt_cache(cache, rejected)

            emit = (draft[:accepted] if draft else []) + [bonus]
            batch: list[int] = []
            for t in emit:
                if len(tokens) >= max_tokens:
                    break
                tokens.append(t)
                sam.extend(t)
                batch.append(t)
                if t in eos_ids:
                    stop = True
                    break
            if batch and on_tokens:
                on_tokens(batch)
            y = mx.array([bonus], dtype=mx.uint32)

        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": 0,
            "prefill_new": len(prompt_ids),
            "tokens_per_step": (n_decode / rounds) if rounds else 0.0,
        }
