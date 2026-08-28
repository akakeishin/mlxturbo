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
    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temp: float,
        eos_ids: set,
        on_tokens,
        session: ChatSession | None,
    ) -> dict: ...


class SpecRunner:
    """投機デコード経路。fastmlx.spec.SpecEngine をそのまま使う。

    fly_theta/fly_window は cli.py の --fly-theta/--fly-window 用の任意
    キーワードとして **extra 経由でそのまま SpecEngine.generate へ流す
    (未指定なら SpecEngine 側の既定 0.0/6 が効く)。server.py は渡さない。
    """

    def __init__(self, engine: SpecEngine, n_draft: int, max_draft: int):
        self.engine = engine
        self.n_draft = n_draft
        self.max_draft = max_draft

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
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


class FallbackRunner:
    """SpecEngine が受け付けないモデル向けの普通の (非投機) 生成経路。

    mlx_lm.generate.stream_generate をそのまま使う。session (ChatSession) は
    投機経路専用の LCP prefill 再利用機構なのでここでは意味を持たず、無視
    する (毎ターン全量 prefill になる)。n_draft/max_draft/fly_* も投機経路
    専用なので **extra で受け取って無視するだけ。
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temp)
        tokens: list[int] = []
        t0 = time.perf_counter()
        ttft = None
        # stream_generate yields exactly one GenerationResponse per generated
        # token (the very last one is folded into the finish_reason-carrying
        # wrap-up response instead of a plain per-step one, see its source),
        # so collecting .token across every yielded response is lossless: no
        # token is skipped or duplicated regardless of why generation stopped.
        for resp in stream_generate(
            self.model, self.tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler
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
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": 0,
            "prefill_new": len(prompt_ids),
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
