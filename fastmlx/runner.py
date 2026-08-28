"""SpecEngine (投機デコード) と、それが使えないモデル向けの通常生成の共通口。

fastmlx/spec.py の SpecEngine は fastmlx/_mlx_compat.py の
validate_spec_model_contract が要求する GDN ハイブリッド固有の形
(model.language_model.model の fa_idx/ssm_idx、各層の is_linear、linear
cache の advance/trim 等) を前提にしている。Llama/Gemma/dense Qwen は
もちろん、GDN ハイブリッドでも fastmlx/spec.py が書かれた当時の mlx-lm 形
(qwen3_5 の language_model ラッパー) と違うレイアウトのモデル (手元で確認
できた範囲では hyper-connections を使う qwen4_exp アーキテクチャがこれに
該当し、model.language_model が無い・層に is_linear/input_layernorm が無い・
最終 norm が無いなど、単なる属性名ずれではなく _hidden_forward/_linear_capture
の再実装そのものと噛み合わない) では TypeError 等で構築に失敗する。

``build_runner`` は起動時に SpecEngine の構築を一度だけ試み、失敗したら
mlx_lm.generate.stream_generate による普通の (非投機) 生成に落とす。
どちらの経路でも呼び出し側 (cli.py / server.py) から見た形
(``Runner.generate(...)`` が返す dict) は同一にする。

``build_runner`` はここで fastmlx.fused.enable_hyper_connection_kernel() も
有効化する (既定 on、``no_fused=True`` で無効化)。qwen4_exp (hyper-connections)
アーキテクチャの `GatedResidual.__call__` をクラス単位で Metal カーネルへ
差し替えるだけなので、SpecEngine 経路 (spec.py が独自に forward を再実装し、
この差し替えを経由しない) には影響せず、FallbackRunner 経路 (モデル自身の
__call__ をそのまま呼ぶ、つまり Flash-Next/qwen4_exp が実際に通る唯一の道)
にだけ効く。moe_route と rms_norm_gated は実測で空振り (前者は +0.34ms 遅く
なる、tools/ablate_moe.py 参照) なので有効化しない。
"""

from __future__ import annotations

import time
from typing import Protocol

import mlx.core as mx

from ._mlx_compat import TextModelArgs
from .spec import ChatSession, SpecEngine


class Runner(Protocol):
    """``SUPPORTED_SAMPLING_PARAMS`` (class attribute, set of str) は
    server.py が top_p/top_k/min_p/repetition_penalty/presence_penalty/
    frequency_penalty/logit_bias/seed のうちどれを ``**sampling_kwargs`` 経由
    でこの runner に渡してよいかを申告する。宣言に無いキーが指定された
    リクエストは server.py が生成呼び出し前に 400 で弾く (SpecRunner/
    FallbackRunner のクラス docstring 参照)。"""

    SUPPORTED_SAMPLING_PARAMS: frozenset

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
    """投機デコード経路。fastmlx.spec.SpecEngine をそのまま使う。

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


class FallbackSession:
    """FallbackRunner 用の会話ごとの mlx_lm prompt_cache 入れ物。

    spec.ChatSession と同じ契約 (新プロンプトが前回処理列の純粋な追記なら
    差分だけ prefill、そうでなければ全再構築) を、mlx_lm.models.cache の
    汎用 KV キャッシュに対して行う。FallbackRunner が実際に通る唯一のモデル
    (qwen4_exp/Flash-Next 系) は GDN ハイブリッドで、線形状態は途中位置へ
    巻き戻せない (fastmlx/spec.py の ChatSession docstring と同じ制約) ため、
    ここでも部分巻き戻し (trim) は行わず、追記か全再構築かの二択に倒す。

    ``processed``: これまでにこのキャッシュへ実際に feed 済みのトークン列
    (prompt + それまでに生成したトークン)。mlx_lm.generate.stream_generate
    は yield されたトークンを次ステップの入力として cache へ feed してから
    出す (see FallbackRunner.generate 呼び出し側の解析)ので、生成が EOS で
    早期終了しても max_tokens で打ち切られても、``tokens`` に集まった列は
    そのままキャッシュへ feed 済みの列と一致する。
    """

    def __init__(self):
        self.cache = None
        self.processed: list[int] = []

    def invalidate(self):
        """Drop the published cache before it is aliased and mutated in place."""

        self.cache = None
        self.processed = []

    def publish(self, cache, processed):
        self.cache = cache
        self.processed = processed


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

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
            **stream_kwargs,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens.append(resp.token)
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
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            # 投機なしなので 1 ステップ = 1 トークン固定。cli.py の表示行が
            # どちらの経路でも同じ res.keys() を仮定できるよう埋めておく。
            "tokens_per_step": 1.0,
        }


def build_runner(
    model,
    tokenizer,
    config,
    args,
    n_draft: int = 3,
    max_draft: int = 8,
    log_prefix: str = "[fastmlx]",
) -> Runner:
    """SpecEngine の構築を試み、モデルの形が合わなければ通常生成へ落とす。

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

    try:
        text_args = TextModelArgs.from_dict(model.args.text_config)
    except AttributeError as exc:
        print(
            f"{log_prefix} 非対応モデルにつき通常生成にフォールバック "
            f"(text_config なし: {type(exc).__name__}: {exc})"
        )
        return FallbackRunner(model, tokenizer)

    mtp = load_cli_mtp(args.model, config, text_args, args.original, args.mtp_bits, args.no_mtp)
    if mtp is not None:
        mx.eval(mtp.parameters())  # 壊れた重み/Metal 確保失敗はここで大声で落ちる (意図的)

    try:
        engine = SpecEngine(model, mtp)
    except (TypeError, ValueError, RuntimeError) as exc:
        print(
            f"{log_prefix} 非対応モデルにつき通常生成にフォールバック "
            f"(SpecEngine 契約検証エラー: {type(exc).__name__}: {exc})"
        )
        return FallbackRunner(model, tokenizer)

    mtp_note = "MTP: なし" if mtp is None else "MTP: あり"
    print(f"{log_prefix} 投機デコード有効 ({mtp_note} / lookup: 有効)")
    return SpecRunner(engine, n_draft=n_draft, max_draft=max_draft)
