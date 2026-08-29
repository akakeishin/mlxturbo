"""SpecEngine (投機デコード) と、それが使えないモデル向けの通常生成の共通口。

mlxturbo/spec.py の SpecEngine は mlxturbo/_mlx_compat.py の
validate_spec_model_contract が要求する GDN ハイブリッド固有の形
(model.language_model.model の fa_idx/ssm_idx、各層の is_linear、linear
cache の advance/trim 等) を前提にしている。Llama/Gemma/dense Qwen は
もちろん、GDN ハイブリッドでも mlxturbo/spec.py が書かれた当時の mlx-lm 形
(qwen3_5 の language_model ラッパー) と違うレイアウトのモデル (手元で確認
できた範囲では hyper-connections を使う qwen4_exp アーキテクチャがこれに
該当し、model.language_model が無い・層に is_linear/input_layernorm が無い・
最終 norm が無いなど、単なる属性名ずれではなく _hidden_forward/_linear_capture
の再実装そのものと噛み合わない) では TypeError 等で構築に失敗する。

``build_runner`` は起動時に SpecEngine の構築を一度だけ試み、失敗したら
mlx_lm.generate.stream_generate による普通の (非投機) 生成に落とす。
どちらの経路でも呼び出し側 (cli.py / server.py) から見た形
(``Runner.generate(...)`` が返す dict) は同一にする。

``build_runner`` はここで mlxturbo.fused.enable_hyper_connection_kernel() も
有効化する (既定 on、``no_fused=True`` で無効化)。qwen4_exp (hyper-connections)
アーキテクチャの `GatedResidual.__call__` をクラス単位で Metal カーネルへ
差し替えるだけなので、SpecEngine 経路 (spec.py が独自に forward を再実装し、
この差し替えを経由しない) には影響せず、FallbackRunner 経路 (モデル自身の
__call__ をそのまま呼ぶ、つまり Flash-Next/qwen4_exp が実際に通る唯一の道)
にだけ効く。moe_route と rms_norm_gated は実測で空振り (前者は +0.34ms 遅く
なる、tools/ablate_moe.py 参照) なので有効化しない。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import mlx.core as mx

from ._mlx_compat import TextModelArgs, resolve_local_model_path
from .spec import PREFILL_STEP_SIZE, ChatSession, SpecEngine


class Runner(Protocol):
    """``SUPPORTED_SAMPLING_PARAMS`` (class attribute, set of str) は
    server.py が top_p/top_k/min_p/repetition_penalty/presence_penalty/
    frequency_penalty/logit_bias/seed のうちどれを ``**sampling_kwargs`` 経由
    でこの runner に渡してよいかを申告する。宣言に無いキーが指定された
    リクエストは server.py が生成呼び出し前に 400 で弾く (SpecRunner/
    FallbackRunner のクラス docstring 参照)。

    ``fallback_reason`` (str | None) は、この runner が build_runner に
    よってフォールバック経路として選ばれた理由。フォールバック以外の経路
    (SpecRunner/FlashSpecRunner) では常に None。server.py の /health が
    これをそのまま出す — 「黙って fallback に落ちて気づけない」を防ぐため
    のもの (build_runner のクラス docstring 参照)。
    """

    SUPPORTED_SAMPLING_PARAMS: frozenset
    fallback_reason: str | None

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temp: float,
        eos_ids: set,
        on_tokens,
        session: ChatSession | None,
        **sampling_kwargs,
    ) -> dict: ...


class SpecRunner:
    """投機デコード経路。mlxturbo.spec.SpecEngine をそのまま使う。

    fly_theta/fly_window は cli.py の --fly-theta/--fly-window 用の任意
    キーワードとして **extra 経由でそのまま SpecEngine.generate へ流す
    (未指定なら SpecEngine 側の既定 0.0/6 が効く)。server.py は渡さない。

    ``SUPPORTED_SAMPLING_PARAMS``: server.py がリクエストのサンプリング
    パラメータ (top_p/top_k/min_p/repetition_penalty/presence_penalty/
    frequency_penalty/logit_bias/seed) のうちどれをこの経路へ渡してよいか
    判定するための宣言。ここでは ``seed`` だけ。理由: SpecEngine.generate は
    temp>0 のとき Block Verification (arXiv:2403.10444, spec.py
    ``_block_verify_tau``) で棄却サンプリングと厳密同一分布を保証している。
    この保証は「ドラフトの提案分布と検証側の target 分布がどちらも生の
    ``softmax(logits/temp)`` である」ことに依存する閉形式の受理長導出
    (docs/STATUS.md) の上に立っており、top_p/top_k/min_p でロジットを
    足切りしたり repetition_penalty 等でロジットを書き換えたりすると
    target 分布そのものが変わるので、受理長の閉形式が別の式になり、実装を
    改めない限り分布保証が静かに壊れる。そこを正しく再導出して実装するのは
    このタスクの範囲外と判断し、これらのパラメータは (b) 案: server.py 側で
    400 を返して弾く。一方 ``seed`` は乱数の初期状態を変えるだけで分布その
    ものは変えないので、影響なく素通しできる (engine.generate 自身は seed
    引数を持たないため、ここで mx.random.seed() を呼んで消費する)。
    """

    KIND = "spec"
    SUPPORTED_SAMPLING_PARAMS = frozenset({"seed"})

    def __init__(self, engine: SpecEngine, n_draft: int, max_draft: int):
        self.engine = engine
        self.n_draft = n_draft
        self.max_draft = max_draft
        self.fallback_reason = None

    def generate(
        self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, seed=None, **extra
    ):
        if seed is not None:
            mx.random.seed(seed)
        return self.engine.generate(
            prompt_ids,
            max_tokens=max_tokens,
            n_draft=self.n_draft,
            max_draft=self.max_draft,
            temp=temp,
            eos_ids=eos_ids,
            on_tokens=on_tokens,
            session=session,
            **extra,
        )


class FlashSpecRunner:
    """Qwen3.8-Flash-Next (``qwen4_exp``) 向け投機デコード経路。

    ``mlxturbo.spec_flash.FlashSpecEngine`` (MTP を draft にした深さ1の投機、
    GatedDeltaNet の状態捕獲/巻き戻し) をそのまま使う。27B (``qwen3_5``) 用の
    ``SpecRunner``/``mlxturbo.spec.SpecEngine`` とは別物 — モデルの層構成
    (hyper-connections、GDN、512 expert MoE) が全く違うため、共通化はせず
    別クラスに分けてある (docs/MTP-FLASH.md 参照)。

    session は ``FallbackSession`` (``.cache`` 単数形 / ``.processed``) を
    そのまま流用する。Flash-Next も GDN ハイブリッドで線形状態を途中位置へ
    巻き戻せないため、この関数自身が持つ再利用判定は ``FallbackRunner`` と
    同じ「新プロンプトが処理済み列の純粋な追記のときだけ再利用、それ以外は
    新規に作り直す」の2択のまま —だが実際に再利用できる範囲はそれより広い:
    ``FallbackSession`` に ``mlxturbo.spec.ChatSession`` と同じ形の
    ``.checkpoints`` を持たせたことで (2026-08-29 追加、thinking 対応)、
    ``server.py`` の ``_select_session``/``_try_checkpoint_restore_session_
    cache`` が処理済み列の途中で分岐したプロンプトでも直近チェックポイント
    まで復元してから ``session`` を渡してくる。この関数はその復元の有無を
    一切知らない —``_select_session`` が復元済みの ``session.processed`` を
    渡してくる限り、ここの LCP 判定は常に「全体一致」として素通しする
    (復元後の ``processed`` は新プロンプトの真の接頭辞になっていることが
    ``_select_session`` 側で保証されている)。``checkpoints`` の記録自体は
    ``FlashSpecEngine.generate_stream`` のプレフィルチャンク境界ごとに行う
    (``mlxturbo.spec.CHECKPOINT_RETENTION``/``snapshot_untrimmable_caches`` を
    そのまま流用、KV/indexer は trim 可能なので記録不要 — spec_flash.py の
    ``generate_stream`` docstring 参照)。``KIND`` が ``"spec"`` ではない
    ため、``server.py``/``cli.py`` の ``session_factory`` 選択
    (``ChatSession if KIND == "spec" else FallbackSession``) はここを変えず
    そのまま ``FallbackSession`` を選ぶ。

    ``SUPPORTED_SAMPLING_PARAMS``: ``SpecRunner`` と同じ理由で ``seed`` のみ
    宣言する。temperature>0 は ``FlashSpecEngine.generate_stream`` が検証
    forward の位置0/1の logits から直接サンプルする形で対応済みだが、
    top_p/top_k/repetition_penalty 等でロジットを書き換えると target 分布
    そのものが変わってしまい、この経路の温度サンプリングが前提にしている
    「生の softmax(logits/temp) からサンプルする」ことと整合しなくなる。
    ``SpecRunner`` と同様、対応するための再導出はこのタスクの範囲外と判断し、
    (b) 案: server.py 側で 400 を返して弾く。``generate_stream`` 自身も
    ``**extra`` を持たない固定シグネチャなので、万一ここへ未対応キーが
    素通りしてきても TypeError で落ちる (二重の防御、``SpecRunner`` と同型)。
    """

    KIND = "flash_spec"
    SUPPORTED_SAMPLING_PARAMS = frozenset({"seed"})

    def __init__(self, engine):
        self.engine = engine
        self.fallback_reason = None

    def generate(
        self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, seed=None, **extra
    ):
        if seed is not None:
            mx.random.seed(seed)

        # FallbackRunner.generate と同じ LCP (最長共通接頭辞) 契約: この関数
        # 自身は「processed 全体が新プロンプトの接頭辞」だけを見る。実際には
        # server.py の _select_session がチェックポイント経由で processed を
        # 途中位置まで復元済みのことがあるが (このクラスの docstring 参照)、
        # 復元後の processed は新プロンプトの真の接頭辞になっていることが
        # 呼び出し側で保証されているので、ここでの判定は常にこの分岐で拾える。
        #
        # ``checkpoints`` は ``mlxturbo.spec.ChatSession`` と同じ流儀 (直接
        # 参照を引き継ぎ、コピーしない) — ``FallbackSession.invalidate()`` は
        # ``self.checkpoints`` を新しい空リストへ **再代入**するだけで、既に
        # 拾ったこのローカル変数が指す旧リストは書き換えない。cache を作り
        # 直す場合 (session が無い/新規/追記でない) は古い記録を引き継ぐ意味が
        # 無いので空にする。
        prompt_cache = None
        reused = 0
        checkpoints: list | None = None
        if session is not None:
            checkpoints = session.checkpoints
            if session.cache is not None:
                pl = session.processed
                n = min(len(pl), len(prompt_ids))
                lcp = 0
                while lcp < n and pl[lcp] == prompt_ids[lcp]:
                    lcp += 1
                if lcp == len(pl) and lcp < len(prompt_ids):
                    prompt_cache = session.cache
                    reused = lcp
                    # session.cache を以後このローカル変数が所有する。生成中の
                    # 例外はここから先、publish() されるまで公開 session を
                    # invalid のままにする (FallbackRunner と同じ理由)。
                    session.invalidate()
            if prompt_cache is None:
                prompt_cache = self.engine.model.make_cache()
                checkpoints = []

        remaining_prompt = prompt_ids[reused:]
        ids = mx.array(remaining_prompt)[None]

        tokens: list[int] = []
        t0 = time.perf_counter()
        ttft = None
        accepted = rounds = 0
        gen = self.engine.generate_stream(
            ids,
            max_tokens,
            caches=prompt_cache,
            temp=temp,
            eos_ids=eos_ids,
            checkpoints=checkpoints,
            base_pos=reused,
            **extra,
        )
        try:
            while True:
                step_tokens = next(gen)
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens.extend(step_tokens)
                if on_tokens:
                    on_tokens(step_tokens)
        except StopIteration as stop:
            if stop.value is not None:
                accepted, rounds = stop.value
        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        if session is not None:
            # FlashSpecEngine's invariant is that the final ``cur`` has been
            # produced/yielded but has not yet been fed into ``prompt_cache``.
            # Publish only the prefix the cache actually contains; the next
            # turn will prefill the trailing cur together with its new suffix.
            # ``checkpoints`` was extended in-place by ``generate_stream``
            # (or reset to ``[]`` above when this call built a fresh cache) —
            # publish it alongside so the next turn's _select_session can
            # restore to it (mlxturbo/spec.py の ChatSession.publish と同じ形)。
            session.publish(
                prompt_cache, list(prompt_ids) + tokens[:-1], checkpoints=checkpoints
            )
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            # mlxturbo.spec.SpecEngine.generate と同じ定義 (n_decode/steps):
            # プレフィルが生んだ最初の1トークンを除いた、反復あたりの実効
            # トークン数。rounds==0 (max_tokens<=1でループが一度も回らない)
            # なら SpecEngine と同じく 0.0 とする。
            "tokens_per_step": (n_decode / rounds) if rounds else 0.0,
        }


class FallbackSession:
    """FallbackRunner 用の会話ごとの mlx_lm prompt_cache 入れ物。

    spec.ChatSession と同じ契約 (新プロンプトが前回処理列の純粋な追記なら
    差分だけ prefill、そうでなければ全再構築) を、mlx_lm.models.cache の
    汎用 KV キャッシュに対して行う。FallbackRunner が実際に通る唯一のモデル
    (qwen4_exp/Flash-Next 系) は GDN ハイブリッドで、線形状態は途中位置へ
    巻き戻せない (mlxturbo/spec.py の ChatSession docstring と同じ制約) ため、
    ここでも部分巻き戻し (trim) は行わず、追記か全再構築かの二択に倒す。

    ``processed``: これまでにこのキャッシュへ実際に feed 済みのトークン列
    (prompt + それまでに生成したトークン)。mlx_lm.generate.stream_generate
    は yield されたトークンを次ステップの入力として cache へ feed してから
    出す (see FallbackRunner.generate 呼び出し側の解析)ので、生成が EOS で
    早期終了しても max_tokens で打ち切られても、``tokens`` に集まった列は
    そのままキャッシュへ feed 済みの列と一致する。

    ``checkpoints``: ``mlxturbo.spec.ChatSession.checkpoints`` と同じ形
    (``[(position, snapshot), ...]``、position 昇順) — thinking マーカーの
    再オープン等で処理済み列の途中から分岐した新プロンプトでも、直近の
    プレフィルチャンク境界まで復元してから差分だけ prefill し直せる
    (mlxturbo/server.py の ``_try_checkpoint_restore_session_cache`` が消費、
    mlxturbo/runner.py の ``FlashSpecRunner`` docstring 参照)。
    ``FallbackRunner`` (このクラスのもう一方の利用者) は ``mlx_lm.generate.
    stream_generate`` を直接使い、チャンク境界ごとのスナップショットを
    自分では作らないため、常に空のまま — ``_try_checkpoint_restore_session_
    cache`` は空リストを falsy として即座に諦めるので、その経路にとっては
    このフィールドが増えたこと自体、以前の「全体一致 or 新規スロット」の
    二択から何も変わらない。
    """

    def __init__(self):
        self.cache = None
        self.processed: list[int] = []
        self.checkpoints: list = []

    def invalidate(self):
        """Drop the published cache before it is aliased and mutated in place."""

        self.cache = None
        self.processed = []
        self.checkpoints = []

    def publish(self, cache, processed, checkpoints=None):
        self.cache = cache
        self.processed = processed
        self.checkpoints = checkpoints if checkpoints is not None else []


def _logprob_entry(resp, tokenizer, top_n: int) -> dict:
    """mlx_lm の ``GenerationResponse`` 1 個分を、OpenAI 形の logprobs
    レコードへ変換する (Kimi K3 レビュー項目 17、``FallbackRunner.generate``
    専用)。

    ``resp.logprobs`` は ``mlx_lm.generate.generate_step``/
    ``speculative_generate_step`` がサンプリングの直前に計算する、その
    デコードステップの語彙全体に対する log_softmax ベクトル
    (``logits - logsumexp(logits)``、mlx_lm/generate.py 420/549 行目参照)。
    実際にサンプルされたトークンの logprob も、top-N の代替候補も、この
    同じベクトルから読むだけで済む — 追加の forward は要らない。
    """

    token_logprob = float(resp.logprobs[resp.token])
    entry = {
        "token": tokenizer.decode([resp.token]),
        "logprob": token_logprob,
        "token_id": resp.token,
    }
    top_n = max(top_n, 0)
    if top_n > 0:
        vocab = resp.logprobs.shape[-1]
        k = min(top_n, vocab)
        top_idx = mx.argsort(resp.logprobs)[::-1][:k].tolist()
        entry["top_logprobs"] = [
            {
                "token": tokenizer.decode([idx]),
                "logprob": float(resp.logprobs[idx]),
                "token_id": idx,
            }
            for idx in top_idx
        ]
    else:
        entry["top_logprobs"] = []
    return entry


class FallbackRunner:
    """SpecEngine が受け付けないモデル向けの普通の (非投機) 生成経路。

    mlx_lm.generate.stream_generate をそのまま使う。session
    (FallbackSession) が渡されれば、前回処理列との LCP (最長共通接頭辞) が
    その session の処理済み列全体と一致する場合に限り mlx_lm の
    prompt_cache をそのまま渡して差分だけを prefill する。一致しなければ
    (会話が切り替わった・テンプレートが履歴を書き換えた等) 黙って新規
    prompt_cache を作り、prompt 全体を流し直す — 部分巻き戻しは行わない
    (FallbackSession docstring 参照)。session が None ならこの機構自体を
    素通しし、旧来どおり mlx_lm 側の一時 cache に任せる (毎ターン全量
    prefill)。n_draft/max_draft/fly_* も投機経路専用なので **extra で受け
    取って無視するだけ。

    ``SUPPORTED_SAMPLING_PARAMS``: 投機の分布保証を気にする必要が無い経路
    なので、mlx_lm.sample_utils がサポートするものは全部そのまま素通しする。

    ``logprobs``/``top_logprobs`` (Kimi K3 レビュー項目 17): 要求されたとき
    だけ (``logprobs=True``) 収集する — 常に集めるとメモリと速度に響くため、
    既定は収集しない (``logprobs=False``、結果 dict に ``"logprobs"`` キー
    自体が現れない)。収集する場合、``res["logprobs"]`` は生成トークン列
    ``res["tokens"]`` と 1:1 対応する ``list[dict]`` (各要素は
    ``_logprob_entry`` 参照)。これは ``SUPPORTED_SAMPLING_PARAMS`` の恒等値
    判定の対象ではない — server.py 側は ``_resolve_runner_for_request`` の
    別枠の ``logprobs_requested`` 引数でルーティングを決める (投機経路で
    logprobs が要求されたら、この runner へ降格させる — SpecRunner/
    FlashSpecRunner/DraftSpecRunner の投機は分布保証の数式が top_p 等と同じ
    理由でロジット処理を前提にしていない/していても閉形式が別なため、
    生成後に「これが本当にサンプルされた分布の logprob だ」と言い切れる
    のはこの非投機経路だけ)。

    ``fallback_reason``: build_runner がこの runner を選んだ理由 (str)。
    build_runner が直接構築する場合のみ渡される — 単体テストが
    ``FallbackRunner(model, tokenizer)`` を直接呼ぶ既存の使い方は省略時
    None のままで壊れない。
    """

    KIND = "fallback"
    SUPPORTED_SAMPLING_PARAMS = frozenset(
        {
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
        }
    )

    def __init__(self, model, tokenizer, fallback_reason: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.fallback_reason = fallback_reason

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
        logprobs: bool = False,
        top_logprobs: int = 0,
        **extra,
    ):
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if seed is not None:
            mx.random.seed(seed)

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        # session が渡されていれば LCP (最長共通接頭辞) が session の処理済み
        # 列全体と一致するときだけ prompt_cache を再利用する (FallbackSession
        # docstring 参照)。session が None なら旧来どおり prompt_cache 自体を
        # 渡さない — stream_generate/generate_step が毎回内部で一時 cache を
        # 作る、以前と完全に同じ経路 (単体テストが直接このメソッドを
        # session=None で叩いても壊れないようにするための分岐でもある)。
        prompt_cache = None
        reused = 0
        if session is not None:
            if session.cache is not None:
                pl = session.processed
                n = min(len(pl), len(prompt_ids))
                lcp = 0
                while lcp < n and pl[lcp] == prompt_ids[lcp]:
                    lcp += 1
                if lcp == len(pl) and lcp < len(prompt_ids):
                    prompt_cache = session.cache
                    reused = lcp
                    # session.cache を以後このローカル変数が所有する。
                    # 生成中の例外 (KeyboardInterrupt 含む) はここから先、
                    # publish() されるまで公開 session を invalid のままにする
                    # (in-place で書き換わっていく cache を半端な状態で公開
                    # しないため — spec.ChatSession と同じ理由)。
                    session.invalidate()
            if prompt_cache is None:
                from mlx_lm.models.cache import make_prompt_cache

                prompt_cache = make_prompt_cache(self.model)

        remaining_prompt = prompt_ids[reused:]

        tokens: list[int] = []
        # 要求されたときだけ集める (項目 17、このクラスの docstring 参照) —
        # None のままなら以下で一度も append されず、戻り値の dict に
        # "logprobs" キー自体が現れない (常に集めてメモリと速度に響かせない
        # ため)。
        collected_logprobs: list[dict] | None = [] if logprobs else None
        t0 = time.perf_counter()
        ttft = None
        stream_kwargs = {}
        if prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache
        # stream_generate yields exactly one GenerationResponse per generated
        # token (the very last one is folded into the finish_reason-carrying
        # wrap-up response instead of a plain per-step one, see its source),
        # so collecting .token across every yielded response is lossless: no
        # token is skipped or duplicated regardless of why generation stopped.
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            remaining_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            # mlxturbo.spec.PREFILL_STEP_SIZE と明示的に共有する。この値を
            # 渡さなければ mlx_lm.generate 側の既定 (2048、たまたま同じ) に
            # 黙って乗るだけで、どちらかを変えたときに経路ごとに prefill の
            # 刻み幅がずれる — 同じプロンプトが経路によって別のチャンク幅
            # で処理され、出力が食い違うバグを自分で作ることになる。
            prefill_step_size=PREFILL_STEP_SIZE,
            **stream_kwargs,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens.append(resp.token)
            if collected_logprobs is not None:
                collected_logprobs.append(_logprob_entry(resp, self.tokenizer, top_logprobs))
            if on_tokens:
                # stream_generate already ran this token through its own
                # internal detokenizer to produce resp.text (correctly
                # excluding eos, handling multi-byte/BPE trailing-space
                # merges). Re-detokenizing the raw id through a second,
                # independent detokenizer instance server-side was pure
                # waste at best; pass the already-correct text through so
                # there is exactly one detokenizer in the loop for this path.
                on_tokens([resp.token], resp.text)
        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        if session is not None:
            # generate_step は yield したトークンを次ステップの入力として
            # cache へ feed してから出す (EOS で早期終了しても同様) ので、
            # tokens はそのまま prompt_cache へ feed 済みの生成列と一致する。
            session.publish(prompt_cache, list(prompt_ids) + tokens)
        result = {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            # 投機なしなので 1 ステップ = 1 トークン固定。cli.py の表示行が
            # どちらの経路でも同じ res.keys() を仮定できるよう埋めておく。
            "tokens_per_step": 1.0,
        }
        if collected_logprobs is not None:
            result["logprobs"] = collected_logprobs
        return result


class DraftSpecRunner:
    """mlx_lm 自身の draft-model 投機デコード (``mlx_lm.generate.
    speculative_generate_step``、``stream_generate(..., draft_model=...)``)
    をそのまま使う、アーキテクチャ非依存の投機経路 (Kimi K3 レビュー項目 13)。

    ``mlxturbo.spec.SpecEngine``/``mlxturbo.spec_flash.FlashSpecEngine`` は
    どちらも特定のモデル形状 (GDN ハイブリッドの qwen3_5/qwen4_exp、
    ``validate_spec_model_contract`` が検査する層構成) に強く依存した専用
    実装で、Llama/Gemma/Mistral/Mixtral のような素の dense transformer には
    一切効かない (このモジュール冒頭の docstring 参照)。mlx_lm はそれとは
    独立に、「小さいドラフトモデルで提案し、本命モデルで検証する」classical
    な投機デコードを ``draft_model`` 引数として既に持っている
    (``mlx_lm/generate.py`` の ``speculative_generate_step``) —
    アーキテクチャを一切見ないので、対応表を作らずに任意の HF/MLX 変換済み
    モデルへ投機を届けられる。自前でカーネルや検証ロジックを書く必要がなく、
    upstream の実装をそのまま使う。

    ``SUPPORTED_SAMPLING_PARAMS``: ``FallbackRunner`` と同じ全キー
    (top_p/top_k/min_p/repetition_penalty/presence_penalty/
    frequency_penalty/logit_bias/seed) を宣言する。``SpecRunner``/
    ``FlashSpecRunner`` が top_p 等を弾く理由 (投機側が「生の
    softmax(logits/temp) が検証対象分布」という前提の上で閉形式の受理長を
    自前導出しているため、ロジット処理を挟むとその前提が壊れる) はここには
    存在しない。``speculative_generate_step`` (mlx_lm/generate.py 473 行目〜)
    を読むと、``sampler``/``logits_processors`` はドラフト側の
    ``_step(draft_model, draft_cache, y)`` と検証側の ``_step(model,
    model_cache, y, num_draft_tokens + 1)`` の両方に同一のクロージャで通り、
    受理判定は「検証側が (同じ sampler で) その時点の真の接頭辞に対して
    実際に独立サンプルした結果が、たまたまドラフト側の値と一致するか」だけ
    を見る (626 行目 ``if tn != dtn: break``)。一致すればドラフト値がそのまま
    emit されるが、それは検証側が独立に出したサンプルと同値なので target
    分布からのサンプルそのものであり、不一致ならその場で検証側がサンプルした
    ``tokens[n]`` を emit する (634 行目) — つまりどちらの分岐でも emit
    されるトークンは「検証モデルの、その時点の真の接頭辞に対する sampler
    適用後の 1 サンプル」であり続ける。ドラフトが当たった/外れたに関わらず
    target 分布から生成し続けるので、top_p/top_k/repetition_penalty 等で
    ロジットを変形しても分布保証は壊れない (SpecRunner/FlashSpecRunner の
    ような閉形式の受理長導出には依存していない)。``repetition_penalty``/
    ``presence_penalty``/``frequency_penalty`` が必要とする「これまでの
    生成列」の状態 (``prev_tokens``) も、ドラフト段階で先食いした分を検証
    直前に trim して二重カウントを避ける処理がライブラリ側に既にある
    (616 行目 ``prev_tokens = prev_tokens[: ...]``)。

    以上はすべて mlx_lm 側の実装 (このタスクでは書いていない、読んだだけ) の
    挙動読解に基づく判断であり、この読解自体をこのタスクで実証してはいない —
    実機で確認したのは「壊れた出力にならないこと」(貪欲一致、self-draft
    構成) までで、非貪欲サンプリングの厳密な分布一致は実測していない
    (docs/ 配下は今回の変更範囲外なので、この判断根拠はここに書く)。

    session (会話ごとの prompt cache 再利用) は当面やらない: draft/target で
    2 本の KV cache を一体で持ち運ぶ必要があり、``FallbackSession``
    (``.cache`` 単数形) の契約に合わない。session が渡されても無視し、毎回
    両モデルとも全量プレフィルする — 遅くはなるが誤動作はしない。
    """

    KIND = "draft_spec"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, model, draft_model, tokenizer, num_draft_tokens: int):
        self.model = model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_draft_tokens = num_draft_tokens
        self.fallback_reason = None

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
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if seed is not None:
            mx.random.seed(seed)

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        tokens: list[int] = []
        t0 = time.perf_counter()
        ttft = None
        n_from_draft = 0
        # stream_generate yields exactly one GenerationResponse per generated
        # token (FallbackRunner.generate と同じ前提、その docstring 参照)。
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            draft_model=self.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens.append(resp.token)
            if resp.from_draft:
                n_from_draft += 1
            if on_tokens:
                on_tokens([resp.token], resp.text)
        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        # mlx_lm の speculative_generate_step は round ごとに 0 個以上の
        # from_draft=True (ドラフト的中) と、たかだか 1 個の from_draft=False
        # (検証モデル自身がサンプルした、その round の締めのトークン) を yield
        # する。厳密な round 数は stream_generate の外から見えないので、
        # from_draft=False の個数を round 数の近似として使う (SpecRunner/
        # FlashSpecRunner の tokens_per_step と同じ「1 ラウンドあたりの実効
        # トークン数」を近似する observability 用の値であって、正しさには
        # 関わらない)。
        n_verify_rounds = max(len(tokens) - n_from_draft, 1)
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": 0,
            "prefill_new": len(prompt_ids),
            "tokens_per_step": (n_decode / n_verify_rounds) if tokens else 0.0,
        }


class _FlashMTPDiscoveryError(RuntimeError):
    """An auto-discovery candidate exists but cannot be inspected safely."""


def _discover_flash_mtp_source(model_dir: Path) -> tuple[str, dict | str] | None:
    """qwen4_exp (Flash-Next) 用、``--mtp`` 未指定のときの MTP 自動発見。

    「特化モデルが自分のアクセラレータを持ち歩く」形にして、「渡し忘れて
    フォールバック」の穴を消すためのもの。優先順位:

    1. モデル本体の safetensors シャードの中 — ``model.safetensors.index.json``
       の ``weight_map`` に ``mtp.`` で始まるキーがあれば、それを含む
       シャードだけを ``mx.load`` で読み、``mtp.*`` キーだけを集めて返す
       (全シャードは読まない — 該当シャードのみに絞る)。量子化配布では
       MTP 重みがモデル本体に同梱されているのが通例
    2. index の無い単一ファイルモデルなら ``model.safetensors`` 自体
    3. モデルディレクトリ直下の ``mtp.safetensors`` サイドカー

    見つかった場合 ``(source_label, spec)`` を返す。``spec`` は 1・2 のとき
    ``dict`` (``mtp_flash.load_flash_mtp`` の ``weights=`` にそのまま渡す)、
    3 のとき ``str`` (同 ``path``)。どれも無ければ ``None``。
    """

    discovery_errors: list[str] = []
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text())
            if not isinstance(index_data, dict):
                raise ValueError("top level must be a JSON object")
            weight_map = index_data.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError("'weight_map' must be a JSON object")

            shards: set[str] = set()
            for key, shard in weight_map.items():
                if not isinstance(key, str) or not key.startswith("mtp."):
                    continue
                if not isinstance(shard, str) or not shard:
                    raise ValueError(f"weight_map[{key!r}] must name a shard")
                shard_path = Path(shard)
                if shard_path.is_absolute() or ".." in shard_path.parts:
                    raise ValueError(f"unsafe shard path for {key!r}: {shard!r}")
                shards.add(shard)

            if shards:
                collected: dict = {}
                for shard in sorted(shards):
                    shard_path = model_dir / shard
                    if not shard_path.exists():
                        raise FileNotFoundError(f"index references missing shard: {shard_path}")
                    for key, value in mx.load(str(shard_path)).items():
                        if key.startswith("mtp."):
                            collected[key] = value
                if not collected:
                    raise ValueError("indexed MTP shards contain no mtp.* tensors")
                return ("モデル内蔵", collected)
        except Exception as exc:
            discovery_errors.append(
                f"{index_path.name}: {type(exc).__name__}: {exc}"
            )

    # Standard unsharded Hugging Face/MLX layout.  mx.load is lazy for
    # safetensors, so filtering the mapping does not materialize the unrelated
    # base-model tensors on the GPU.
    unsharded = model_dir / "model.safetensors"
    if unsharded.exists():
        try:
            collected = {
                key: value
                for key, value in mx.load(str(unsharded)).items()
                if key.startswith("mtp.")
            }
            if collected:
                return ("モデル内蔵 (model.safetensors)", collected)
        except Exception as exc:
            discovery_errors.append(
                f"{unsharded.name}: {type(exc).__name__}: {exc}"
            )

    sidecar = model_dir / "mtp.safetensors"
    if sidecar.exists():
        return ("サイドカー (mtp.safetensors)", str(sidecar))

    if discovery_errors:
        raise _FlashMTPDiscoveryError("; ".join(discovery_errors))
    return None


#: build_runner が返しうる Runner.KIND の全値。server.py の --require-runner
#: がこの集合を choices に使う (KIND 文字列がここと build_runner の実際の
#: 分岐とで散らばらないようにするための唯一の定義元)。"lookup_spec"
#: (mlxturbo.lookup_spec.LookupSpecRunner) はここでは文字列リテラルで足す —
#: 循環 import を避けるため (lookup_spec.py が runner.py の FallbackRunner を
#: import するので、逆方向の import はできない)。
RUNNER_KINDS = frozenset(
    {
        SpecRunner.KIND,
        FlashSpecRunner.KIND,
        FallbackRunner.KIND,
        DraftSpecRunner.KIND,
        "lookup_spec",
    }
)


def build_runner(
    model,
    tokenizer,
    config,
    args,
    n_draft: int = 3,
    max_draft: int = 8,
    log_prefix: str = "[mlxturbo]",
) -> Runner:
    """``args.draft_model``/``args.lookup_spec`` の 2 つを、既存の spec/
    flash_spec/fallback 選択 (``_build_base_runner``、この関数の従来の本体)
    より手前で扱う薄いラッパー (Kimi K3 レビュー項目 13/12)。

    - ``args.draft_model`` (str | None) が指定されたときだけ ``DraftSpecRunner``
      経路に入り、``_build_base_runner`` の分岐 (qwen4_exp/SpecEngine/
      FallbackRunner の選択) には一切触れない — 未指定なら従来どおりの選択
      順序をそのまま通る。ドラフトモデルは ``mlx_lm.load`` でこの関数の中で
      読み込む (本体モデルとは独立にロードする、2 本目のモデル)。トークナイザ
      の vocab_size が本体と食い違う場合は ``mlx_lm.generate`` の CLI 自身が
      する検査と同じ理由 (draft は本体と同じトークナイザである前提)
      で ``SystemExit(1)`` する — ``--mtp`` の明示指定と同じ「逃げ道なし」
      の扱い。同じモデルを draft に指定する self-draft (速くはならないが
      配線の確認にはなる) も、この検査は素通りする (vocab_size が一致する
      ため)。
    - ``args.lookup_spec`` (bool) は ``_build_base_runner`` が選んだ結果が
      ``FallbackRunner`` (= spec/flash_spec 契約に合わないモデル) のときだけ
      ``mlxturbo.lookup_spec.LookupSpecRunner`` へ差し替える (Kimi K3 レビュー
      項目 12)。spec/flash_spec が選ばれた場合は何もしない — 既にモデル
      専用の投機が効いているので、n-gram lookup を上乗せする動機が無い。

    ``args`` は ``model``/``original``/``mtp_bits``/``no_mtp``/``no_fused``/
    ``draft_model``/``num_draft_tokens``/``lookup_spec``/``lookup_max_draft``/
    ``lookup_min_match`` を持つ argparse.Namespace (cli.py / server.py の
    どちらの引数もこの形。cli.py はこれらの新規フラグを持たないため、
    ``getattr(..., None)``/``getattr(..., False)`` で欠落を許容する)。
    """

    draft_model_path = getattr(args, "draft_model", None)
    if draft_model_path:
        from mlx_lm import load as mlx_lm_load

        num_draft_tokens = getattr(args, "num_draft_tokens", None) or 4
        print(
            f"{log_prefix} --draft-model {draft_model_path}: mlx_lm draft-model 投機"
            f" (アーキテクチャ非依存、DraftSpecRunner) を構築します"
            f" (num_draft_tokens={num_draft_tokens})"
        )
        draft_model, draft_tokenizer = mlx_lm_load(draft_model_path)
        model_vocab = getattr(tokenizer, "vocab_size", None)
        draft_vocab = getattr(draft_tokenizer, "vocab_size", None)
        if draft_vocab != model_vocab:
            print(
                f"{log_prefix} --draft-model {draft_model_path} のトークナイザが本体"
                f" モデルと一致しません (vocab_size {draft_vocab} != {model_vocab})。"
                " draft モデルは本体と同じトークナイザ (同系統モデル、self-draft を"
                " 含む) である必要があります。終了します。"
            )
            raise SystemExit(1)
        print(
            f"{log_prefix} draft-model 投機デコード有効 (DraftSpecRunner, "
            f"draft={draft_model_path})"
        )
        return DraftSpecRunner(model, draft_model, tokenizer, num_draft_tokens=num_draft_tokens)

    resolved = _build_base_runner(
        model, tokenizer, config, args, n_draft=n_draft, max_draft=max_draft, log_prefix=log_prefix
    )

    if getattr(args, "lookup_spec", False) and resolved.KIND == FallbackRunner.KIND:
        from .lookup_spec import LookupSpecRunner

        wrapped = LookupSpecRunner(
            model,
            tokenizer,
            max_draft=getattr(args, "lookup_max_draft", None) or 8,
            min_match=getattr(args, "lookup_min_match", None) or 2,
        )
        # 元の FallbackRunner が持っていた理由 (「なぜ spec/flash_spec が
        # 選ばれなかったか」) は引き継ぐ — /health の fallback_reason はこの
        # 経路でも意味のある情報のまま (「黙って fallback に落ちて気づけ
        # ない」を防ぐ方針、build_runner の元の docstring 参照)。
        wrapped.fallback_reason = resolved.fallback_reason
        note = (
            "有効"
            if wrapped.trimmable
            else "無効 (このモデルの KV キャッシュが trim 不可のため、通常生成のまま)"
        )
        print(f"{log_prefix} n-gram lookup 投機 (LookupSpecRunner) を有効化: {note}")
        return wrapped

    return resolved


def _build_base_runner(
    model,
    tokenizer,
    config,
    args,
    n_draft: int = 3,
    max_draft: int = 8,
    log_prefix: str = "[mlxturbo]",
) -> Runner:
    """SpecEngine の構築を試み、モデルの形が合わなければ通常生成へ落とす。

    ``build_runner`` (この関数のもとの名前、``--draft-model``/
    ``--lookup-spec`` を横取りする薄いラッパーが追加されたので分離した)
    から呼ばれる、qwen4_exp/SpecEngine/FallbackRunner を選ぶ本体。

    ``args`` は ``model``/``original``/``mtp_bits``/``no_mtp``/``no_fused`` を
    持つ argparse.Namespace (cli.py / server.py のどちらの引数もこの形)。

    フォールバックへ落としてよいのは「このモデルのレイアウトが SpecEngine の
    契約に合わない」と判定できる場合だけに絞る。壊れた重みや Metal の確保
    失敗のような本物の障害まで「非対応アーキテクチャ」に化けさせて黙って
    フォールバックすると、原因が分からなくなる:

    - ``model.args.text_config`` が無い (= そもそも VLM ラッパー形式ですら
      ない) は ``AttributeError`` — これは正当な「対象外」判定
    - ``load_cli_mtp`` は重み欠損を自分で吸収して ``None`` を返す (失敗を
      ここまで伝播させない設計)。それでも伝播してきた例外は本物のバグ
    - ``mx.eval(mtp.parameters())`` はここでは捕まえない。落ちるなら
      Metal 確保失敗や壊れた重みで、フォールバック対象ではなく実害なので
      大声で落とす
    - ``SpecEngine(model, mtp)`` 構築時の ``validate_spec_model_contract``
      が投げる ``TypeError``/``ValueError``/``RuntimeError`` だけが
      「契約不一致 = 対象外」の正式なシグナル

    qwen4_exp (Flash-Next) の MTP は 3 段階で探す (優先順、``_discover_flash_mtp_source``
    参照):

    1. ``--mtp PATH`` 明示指定 — 最優先。運用者の明示指定なので、読めない
       まま黙って遅い (非投機の) 構成で起動することは「対象外」判定とは
       別物として扱う。ロードは 1 回だけリトライし (外付け SSD 起因の一時的
       な GPU Timeout を想定)、それでも読めなければフォールバックせず
       ``SystemExit(1)`` で終了する — 逃げ道のフラグは無い
    2. モデル本体の safetensors シャードの中 — ``model.safetensors.index.json``
       の ``weight_map`` に ``mtp.`` で始まるキーがあれば、該当シャードだけ
       読み ``mtp.*`` テンソルを集めて使う (量子化配布では MTP 重みが本体に
       同梱されているのが通例)
    3. モデルディレクトリ直下の ``mtp.safetensors`` サイドカー

    2・3 (自動発見) の失敗は明示指定ではないので exit しない — リトライは
    2・3 にも適用するが、それでも失敗すればフォールバックする。どれも
    見つからなければ従来どおりフォールバック。

    フォールバックした runner (``FallbackRunner``) は ``fallback_reason``
    (str) に理由を持つ: ``"MTP が見つからない (...)"`` /
    ``"MTP 自動発見 (<出典>) の読み込みに失敗: <理由>"`` /
    ``"spec 契約検証に失敗: <理由>"`` のいずれか。フォールバック以外の
    runner では ``fallback_reason`` は常に None (Runner Protocol 参照)。
    """

    from . import fused
    from .cli import load_cli_mtp
    from .ngram_stream import warn_if_not_installed

    # --ngram を渡し忘れると n-gram 表が初期値のまま生成に使われる。
    # 出力は最後まで出るので、ここで鳴らさないと気づけない
    warn_if_not_installed(model)

    if args.no_fused:
        print(f"{log_prefix} --no-fused: hyper-connections 融合カーネルを無効化")
    else:
        fused.enable_hyper_connection_kernel()
        print(f"{log_prefix} hyper-connections 融合カーネル有効 (moe_route/rms_norm_gated は実測で"
              " 空振りのため無効のまま)")

    # Qwen3.8-Flash-Next (qwen4_exp) + MTP (明示指定 or 自動発見) ->
    # FlashSpecEngine 経路 (docs/MTP-FLASH.md)。27B (qwen3_5) は model_type が
    # 違うので以下の分岐には一切入らず、この関数の残り (既存の
    # SpecEngine/FallbackRunner 分岐) をそのまま通る — 27B 側の経路は変えて
    # いない。``args.mtp`` が無い cli.py 呼び出し (--mtp 引数を持たない) では
    # getattr が None を返すので自動発見側に自然に落ちる。
    mtp_path = getattr(args, "mtp", None)
    if getattr(model.args, "model_type", None) == "qwen4_exp":
        from . import mtp_flash, spec_flash
        from mlx.utils import tree_flatten

        mtp_bits = getattr(args, "mtp_bits", None)
        quant = {"group_size": 64, "bits": mtp_bits} if mtp_bits else None

        explicit = bool(mtp_path)
        if explicit:
            source_label = "明示指定 (--mtp)"
            load_path, load_weights = mtp_path, None
        else:
            # ``args.model`` may be a Hugging Face repo id.  The model has
            # already been loaded by this point, so resolve the corresponding
            # local snapshot instead of treating ``org/repo`` as a relative
            # filesystem path and silently missing its embedded MTP tensors.
            model_ref = getattr(args, "model", "") or "."
            try:
                model_dir = resolve_local_model_path(model_ref)
            except Exception:
                # ``build_runner`` is also a public/testable boundary and can
                # be called with an already-constructed model plus a synthetic
                # non-cached name.  Preserve the old "no candidate" behavior
                # there; normal CLI/server flow has loaded the repo already.
                model_dir = Path(model_ref)
            try:
                discovered = _discover_flash_mtp_source(model_dir)
            except _FlashMTPDiscoveryError as exc:
                reason = f"MTP 自動発見に失敗: {exc}"
                print(f"{log_prefix} {reason} — 通常生成にフォールバックします")
                return FallbackRunner(model, tokenizer, fallback_reason=reason)
            if discovered is None:
                reason = (
                    "MTP が見つからない (--mtp で指定するか、モデルディレクトリに"
                    " mtp.safetensors を置く)"
                )
                print(f"{log_prefix} {reason} — 通常生成にフォールバックします")
                return FallbackRunner(model, tokenizer, fallback_reason=reason)
            source_label, spec = discovered
            if isinstance(spec, dict):
                load_path, load_weights = None, spec
            else:
                load_path, load_weights = spec, None
            print(f"{log_prefix} MTP を自動発見: {source_label}")

        mtp = None
        last_exc: Exception | None = None
        # ロードは最大 2 回試みる (1 回だけリトライ)。直前に GPU Timeout の
        # 実例があり (外付け SSD の mmap 読み出し起因の一時的失敗)、テンソル
        # ごとの eval で主因は潰したがそれでも起こり得る一時的失敗のための
        # 保険。明示指定 (--mtp) だけでなく自動発見 (モデル内蔵/サイドカー)
        # にも同じリトライを適用する。2 回目も失敗すれば一時的ではなく
        # 恒久的な失敗 (パス誤り・形式不一致等) とみなす。
        for attempt in range(2):
            try:
                mtp = mtp_flash.load_flash_mtp(
                    load_path, model.args.text, quantize=quant, weights=load_weights
                )
                # 壊れた重み/Metal 確保失敗はここで大声で落ちる (意図的)。
                # ただし一括 mx.eval にはしない: サイドカーは外付け SSD の
                # 5.2GB を mmap で遅延読みしており、量子化カーネルを 1 つの
                # コマンドバッファに全部積むと USB のページフォルトで GPU が
                # 止まり Metal の watchdog (GPU Timeout Error) を毎回踏む。
                # テンソルごとに eval を切れば 1 バッファが短くなり、読みの
                # 遅さは待ち時間になるだけで済む。
                for _name, p in tree_flatten(mtp.parameters()):
                    mx.eval(p)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                mtp = None
                if attempt == 0:
                    print(
                        f"{log_prefix} MTP ロードに失敗、再試行します: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if last_exc is not None:
            if explicit:
                # ``--mtp`` は運用者の明示指定なので、読めないまま黙って遅い
                # (非投機の) 構成で起動することは「対象外アーキテクチャ」
                # 判定とは別物として扱う — フォールバックせず、理由を明示
                # して終了する。逃げ道のフラグは無い。
                print(
                    f"{log_prefix} --mtp {mtp_path} (qwen4_exp) を再試行しても"
                    f"読み込めません ({type(last_exc).__name__}: {last_exc})。"
                    " この --mtp を外せば、モデル内蔵/サイドカーの MTP を"
                    " 自動で探す運用に切り替わります (見つからなければ MTP"
                    " 無しで起動)。終了します。"
                )
                raise SystemExit(1)
            # 自動発見 (モデル内蔵/サイドカー) の失敗は明示指定ではないので
            # exit しない — フォールバックへ倒す。
            reason = (
                f"MTP 自動発見 ({source_label}) の読み込みに失敗: "
                f"{type(last_exc).__name__}: {last_exc}"
            )
            print(f"{log_prefix} {reason}; 通常生成にフォールバックします")
            return FallbackRunner(model, tokenizer, fallback_reason=reason)

        engine = spec_flash.FlashSpecEngine(model, mtp)
        bits_note = f"{mtp_bits}bit" if mtp_bits else "bf16"
        print(
            f"{log_prefix} Flash-Next 投機デコード有効 (FlashSpecEngine, MTP: あり"
            f" [{source_label}], {bits_note})"
        )
        return FlashSpecRunner(engine)

    try:
        text_args = TextModelArgs.from_dict(model.args.text_config)
    except AttributeError as exc:
        reason = f"spec 契約検証に失敗: text_config なし ({type(exc).__name__}: {exc})"
        print(
            f"{log_prefix} 非対応モデルにつき通常生成にフォールバック "
            f"(text_config なし: {type(exc).__name__}: {exc})"
        )
        return FallbackRunner(model, tokenizer, fallback_reason=reason)

    mtp = load_cli_mtp(args.model, config, text_args, args.original, args.mtp_bits, args.no_mtp)
    if mtp is not None:
        mx.eval(mtp.parameters())  # 壊れた重み/Metal 確保失敗はここで大声で落ちる (意図的)

    try:
        engine = SpecEngine(model, mtp)
    except (TypeError, ValueError, RuntimeError) as exc:
        reason = f"spec 契約検証に失敗: {type(exc).__name__}: {exc}"
        print(
            f"{log_prefix} 非対応モデルにつき通常生成にフォールバック "
            f"(SpecEngine 契約検証エラー: {type(exc).__name__}: {exc})"
        )
        return FallbackRunner(model, tokenizer, fallback_reason=reason)

    mtp_note = "MTP: なし" if mtp is None else "MTP: あり"
    print(f"{log_prefix} 投機デコード有効 ({mtp_note} / lookup: 有効)")
    return SpecRunner(engine, n_draft=n_draft, max_draft=max_draft)
