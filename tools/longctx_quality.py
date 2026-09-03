"""prefill attention 融合カーネル (段 P1、kv >= 12288 で発火、``MLXTURBO_PREFILL_ATTN``
既定 on) の長文脈品質を、KLD ではなく**課題の正答率**で見る。

## なぜ KLD ではないか

`tools/kld_prefill_attn.py` で長文脈 (17k) の自前 dense 比 KLD を測ると 0.040 に
なるが、**丸めだけが違う既知の対照 (GDN Metal on/off、意味論は不変) でも同じ
物差しで 0.111 になる** (`docs/research/SESSION-2026-09-02-CATCHUP.md` の
2026-09-03 07:40〜08:00 の節)。QSA (indexer) の top-k 選択は境界のタイが多く、
どんな些細な丸めの違いもブロック選択のカスケードで増幅するので、長文脈の
「dense 対 変種」KLD はどんな変更でも 0.04〜0.1 のレンジに落ち、物差しとして
機能しない。本来の品質ゲート (`bench/quant_eval.py compare`、bf16 参照) は
この問題を踏まないが、それは continuation が最長 597 トークンしかなく、
このカーネルが 1 度も発火しない盲点があるため (同節)。bf16 参照そのものは
335GB でこの機体に載らないので、参照分布を用意する道も無い。

この道具は分布の近さではなく、**実際に課題を解けるか**で見る
(`docs/research/LANES-2026-09.md` のレーン 11 の宿題)。

## 課題

- **recall**: `tools/_bench_text.py` の窓の中に、ランダムな位置 (先頭
  10%〜90%) へ「合言葉は XXXX です」(XXXX は `--seed` で決まる乱数の 6 桁
  英数字) を 1 行挿入し、末尾で「合言葉は何ですか。合言葉だけ答えてください。」
  と聞く。正答 = 生成の最初の 32 トークンに XXXX が含まれる。
- **quote**: 窓の中のランダムな文 (20〜40 文字) を 1 つ選び、「次の文の
  直後に続く文をそのまま書き出してください」と聞く。正答 = 生成 64 トークン
  の中に実際の次の文の先頭 15 文字が含まれる。

各問を dense (`disable_gather_attn` で `_gather_attn`/`_prefill_attn` を両方
確実に落とす) とカーネル (`enable_prefill_attn`) の両方で解き、正答率と
両者の回答の一致率を出す。**dense 側は `disable_prefill_attn` ではなく
`disable_gather_attn` を使う** --- `tools/kld_prefill_attn.py` の
2026-09-03 の訂正がそのまま当てはまる: `disable_prefill_attn` は
「`enable_prefill_attn` だけを打ち消す (gather 経路は残す)」規約なので、
直前の問題がカーネル側で `_gather_attn=True` を立てていると、次の問題の
「dense のつもり」が実際には汎用 gather 経路を通ってしまう。

生成は `FlashSpecEngine` を経由せず、prefill (`model(chunk, cache=...)`、
チャンク幅 `--chunk` 既定 2048) + 貪欲デコードループ (`model(tok, cache=...)`
を 1 トークンずつ、最大 `--max-new` 既定 64、EOS で停止) を直に回す
(速度は測らない)。

**2026-09-03 訂正 (17k 実走で dense recall 1/6、quote 0/6)。**思考が既定 on
のモデルでは、貪欲デコード数十トークンの先頭が `<think>...</think>` の推論で
埋まり、答えそのものに辿り着く前に打ち切られていた。`mlxturbo/server.py` の
`_apply_template` が `reasoning_effort: "none"` を `enable_thinking=False` に
落としているのと同じ kwarg を `_build_ids` で常に渡すようにした
(テンプレートが受けない場合は `_apply_template` と同じ TypeError フォール
バックで無しに落とす)。

## 時間の目安

50k の prefill は 1 回あたり約 95 秒。1 問につき dense + kernel の 2 回、
課題 2 種、`--n` 問ぶん流すので、`--ctxs 50000 --n 8` は
95s x 2 x 8 x 2 ≈ 50 分かかる。手早く確かめたいときは `--ctxs` と `--n`
を両方小さくすること (例: `--ctxs 4000 --n 2`)。

## 使用例

    tools/biglock.sh .venv/bin/python tools/longctx_quality.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctxs 17000,50000 --n 8

モデルの読み込みは `tools/verify_width_cost.py` の `build_runner` を流用する
(常駐条件を本番と揃える `set_wired_limit_default` もその中で呼ばれる)。
このツールは `build_runner` が返す `FlashSpecEngine`/MTP は使わない
(生成に要らない) が、出荷経路と同じ融合 (`enable_default_fusions`) を
model に当てる副作用のためだけに `build_runner` をそのまま再利用する。

## 常駐 worker (`tool` ジョブ)

98GB を読み直さずに済むよう、`run_with_model(argv, bundle)` を持つ
(規約は `tools/ab_bundle.py` の docstring)。CLI (`main`) と worker は
同じ `parse_args` → `run` を通るので、**出力も終了コードも変わらない。**
worker から呼ばれたときだけ、借り物のモデルに残る 3 つ (経路の旗
`_gather_attn`/`_prefill_attn`/`_gather_attn_tile`、`qsa_tail.MODE`、
prefill カーネルの差し替え) を終わりに戻す。

`--qsa-tail` は `MLXTURBO_QSA_TAIL` の代わり。**worker は環境変数を
読み直さない** (`mlxturbo/qsa_tail.py` は import 時に 1 回だけ読む) ので、
モジュール属性 `qsa_tail.MODE` をその場で書き換える。省略時は今の値の
まま (CLI では環境変数どおり)。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

_PASSWORD_ALPHABET = string.ascii_uppercase + string.digits

# 全角の文末記号 (。！？) は直後に空白が無くても文の切れ目として確実
# (Markdown の `**強調**` がそのまま続く `発火していない。**次は` のような
# 形でも切る。直後に閉じ記号 (`**`/`」`/`)` 等) が続くならそれも区切りごと
# 消費する)。半角の `.!?` は `os.path.join` のような識別子を割らないよう
# 空白が続く場合だけ切る。改行 1 個以上でも切る。
#
# **2026-09-03 訂正 (17k 実走で発覚)。**旧版は全角文末記号も「直後に空白」
# を要求していたため、`発火していない。**ハーネスの...` のように句点の
# 直後に Markdown 装飾が続く箇所で切れず、意味の異なる 2 文が 1 つの
# 「文」に融合していた (`_pick_quote_pair` が 20〜40 文字という理由だけで
# それを引用文に選び、model が正しく続きを言えない実例が出た)。
_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？])[)\]）」』\"'*_`]*\s*"  # 全角文末記号: 空白不要、閉じ記号ごと消費
    r"|(?<=[.!?])\s+"  # 半角文末記号: 空白が続く場合だけ
    r"|\n+"
)


def _random_password(rng: random.Random) -> str:
    """`--seed` で決まる乱数の 6 桁英数字 (大文字+数字)。"""
    return "".join(rng.choices(_PASSWORD_ALPHABET, k=6))


def _insert_password(rng: random.Random, body: str, password: str) -> tuple[str, float]:
    """``body`` のランダムな位置 (先頭 10%〜90%) に合言葉の行を挿入する。

    改行境界に寄せる (無ければ挿入位置そのまま)。戻り値は (挿入後の本文, 使った frac)。
    """
    frac = rng.uniform(0.10, 0.90)
    idx = int(frac * len(body))
    cut = body.find("\n", idx)
    if cut == -1:
        cut = body.rfind("\n", 0, idx)
    if cut == -1:
        cut = idx
    line = f"\n合言葉は {password} です\n"
    return body[:cut] + line + body[cut:], frac


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p]


# 最終フォールバック (句読点を無視した機械的な 2 分割) が要求する、
# strip 後の本文の最低文字数。これを下回るなら「引用できる文が無い」で
# 素直に諦める (`_bench_text.long_prompts` の「足りないまま繰り返しで
# 埋めない」と同じ思想)。
_MIN_QUOTE_BODY_LEN = 10


def _pick_quote_pair(rng: random.Random, body: str) -> tuple[str, str]:
    """``body`` からランダムな引用文 (20〜40 文字) と、実際に続く文の先頭
    15 文字を選ぶ。戻り値は (引用文, 次の文の先頭 15 文字)。

    文分割は `_split_sentences` (句点・改行ベース) を使うが、20〜40 文字の
    候補が無い短い文書 (合成の小さいモデルでの CPU smoke test など、
    ``--ctxs`` の窓が数十〜百文字しかないとき) 向けに 2 段階で条件を緩める。
    最終段は句読点に関係なく本文を機械的に前後へ割るので、本文が
    `_MIN_QUOTE_BODY_LEN` 文字さえあれば失敗しない。
    """
    sentences = _split_sentences(body)
    # インラインコード (バッククォート) を含む「文」は除外する。2026-09-03
    # の実走で `矩形の上限は \`B*(1+k) <= 8\`。` のようなコード片を引用させると、
    # 「次の文」が定義しにくく model が質問文をそのまま繰り返すだけの失敗が
    # 実際に出た (`bench/results/longctx-quality-sanity.json` problem 1)。
    no_code = [i for i in range(len(sentences) - 1) if "`" not in sentences[i]]
    candidates = [i for i in no_code if 20 <= len(sentences[i]) <= 40]
    if not candidates:
        # フォールバック 1: 20〜40 文字の文が見つからない小さな文書向け。
        # 文長の条件を緩め、次の文が存在する任意の文から選ぶ (コード片除外は維持)。
        candidates = [i for i in no_code if len(sentences[i]) >= 5]
    if not candidates:
        # フォールバック 1b: コード片しか無い文書向け。除外を諦める。
        candidates = [i for i in range(len(sentences) - 1) if len(sentences[i]) >= 5]
    if candidates:
        i = rng.choice(candidates)
        return sentences[i][:40], sentences[i + 1][:15]

    # フォールバック 2: 句点・改行が無い (または 1 文しか取れない) 極端に
    # 短い文書向け。「引用文」を句読点に関係なく本文の中央付近で機械的に
    # 切り出す。課題としては不自然だが、CPU smoke test のような小さな
    # ctx でも一連の流れを最後まで通すための保険。
    stripped = body.strip()
    if len(stripped) < _MIN_QUOTE_BODY_LEN:
        raise ValueError(
            f"引用できる文が見つからない (本文が短すぎる: {len(stripped)} 文字)"
        )
    cut = len(stripped) // 2
    quoted = stripped[max(0, cut - 40) : cut]
    expected_prefix = stripped[cut : cut + 15]
    return quoted, expected_prefix


def _windows(tok, ctx: int, count: int, offset_tokens: int = 0) -> list[str]:
    """`_bench_text.long_prompts` で ``count`` 本の互いに重ならない窓を切り、
    質問部分 (常に空文字列) を剥がして本文だけを返す。"""
    from _bench_text import long_prompts

    raw = long_prompts(tok, ctx, [""] * count, offset_tokens=offset_tokens)
    suffix = "\n\n---\n\n"
    return [r[: -len(suffix)] if r.endswith(suffix) else r for r in raw]


def _build_ids(tok, text: str):
    """``text`` (文書+質問を 1 本にまとめたもの) を単発の user メッセージとして
    テンプレートに通す。``enable_thinking=False`` を明示する ---
    `mlxturbo/server.py` の `_apply_template` が `reasoning_effort: "none"` を
    ここへ落としているのと同じ kwarg (思考トークンで最初の 32〜64 トークンが
    埋まると、貪欲デコード数十トークンでは答えに辿り着かない)。この kwarg を
    受けないテンプレートもあるので、`_apply_template` と同じ TypeError
    フォールバックで無しの呼び出しに落とす。
    """
    import mlx.core as mx

    messages = [{"role": "user", "content": text}]
    try:
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    return mx.array(ids)[None]


def _set_route(model, route: str) -> int:
    """model を ``"dense"`` か ``"kernel"`` の経路に設定する。戻り値は
    (kernel のときだけ意味のある) 適用層数。

    dense は `disable_prefill_attn` ではなく `disable_gather_attn` を使う。
    `tools/kld_prefill_attn.py` の 2026-09-03 の訂正のとおり、
    `disable_prefill_attn` は「enable_prefill_attn だけを打ち消す (gather
    経路は残す)」規約なので、1 つ前の問題がカーネル側で `_gather_attn=True`
    を立てたままだと、次の問題の「dense のつもり」が実際には汎用 gather
    経路 (`_gather_tile_attn`) を通ってしまう。`disable_gather_attn` は
    `_gather_attn`/`_prefill_attn` の両方を落とすので、こちらが本来の
    「dense へ完全に戻す」呼び方。
    """
    from mlxturbo.gather_attn import disable_gather_attn, enable_prefill_attn

    if route == "dense":
        disable_gather_attn(model)
        return 0
    if route == "kernel":
        return enable_prefill_attn(model)
    raise ValueError(f"unknown route {route!r}")


def _greedy_generate(
    model, cache, ids, chunk: int, max_new: int, eos_ids, pending_fn
) -> list[int]:
    """``ids`` をチャンク幅 ``chunk`` で prefill し、続けて貪欲デコードを
    最大 ``max_new`` トークン回す。`FlashSpecEngine` は経由しない
    (`tools/kld_prefill_attn.py` の `_run_prefill` と同じ書き方)。
    """
    import mlx.core as mx

    n = ids.shape[1]
    last_logits = None
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        part = ids[:, start:end]
        logits = model(part, cache=cache)
        if end == n:
            last_logits = logits[:, -1, :].astype(mx.float32)
        mx.eval(logits, *pending_fn(cache))
        mx.clear_cache()
    if last_logits is None:
        raise RuntimeError("ids が空 (n=0)")

    cur = int(mx.argmax(last_logits[0]))
    generated = [cur]
    if not (eos_ids and cur in eos_ids):
        for _ in range(max_new - 1):
            logits = model(mx.array([[cur]]), cache=cache)
            mx.eval(logits, *pending_fn(cache))
            mx.clear_cache()
            cur = int(mx.argmax(logits[0, -1]))
            generated.append(cur)
            if eos_ids and cur in eos_ids:
                break
    return generated


def _run_route(
    model, tok, ids, route: str, chunk: int, max_new: int, eos_ids, pending_fn
) -> tuple[list[int], int, int]:
    """1 経路ぶん生成する。戻り値は (生成トークン列, カーネル発火回数, 適用層数)。

    カーネル発火回数は `tools/kld_prefill_attn.py` と同じ counted-wrapper 方式
    (0 のままカーネル経路の正答率を語るのは無意味なので、呼び出し側で警告する)。
    """
    import mlx.core as mx
    from mlxturbo.kernels import prefill_attn as prefill_attn_kernel

    n_layers = _set_route(model, route)
    fired = [0]
    orig = prefill_attn_kernel.prefill_attn
    if route == "kernel":
        def _counted(*a, **kw):
            fired[0] += 1
            return orig(*a, **kw)

        prefill_attn_kernel.prefill_attn = _counted
    try:
        cache = model.make_cache()
        generated = _greedy_generate(model, cache, ids, chunk, max_new, eos_ids, pending_fn)
        del cache
        mx.clear_cache()
    finally:
        prefill_attn_kernel.prefill_attn = orig
    return generated, fired[0], n_layers


def _eval_problem(
    model, tok, ids, correct_check, chunk: int, max_new: int, eos_ids, pending_fn
) -> dict:
    """dense/kernel 両経路で 1 問解き、経路ごとの正誤・回答テキスト・
    カーネル発火回数を積んだ dict を返す (呼び出し側が課題別のメタ情報を足す)。

    ``correct_check(route, generated: list[int]) -> tuple[bool, str]`` は
    課題ごとの正答判定 (使う生成長・照合対象が recall/quote で違うため呼び手に委ねる)。
    """
    record: dict = {"kv": int(ids.shape[1])}
    for route in ("dense", "kernel"):
        generated, fired, n_layers = _run_route(
            model, tok, ids, route, chunk, max_new, eos_ids, pending_fn
        )
        correct, answer_text = correct_check(route, generated)
        record[route] = {
            "correct": bool(correct),
            "answer": answer_text,
            "kernel_fired": fired,
            "kernel_layers": n_layers,
        }
    record["agree"] = record["dense"]["answer"] == record["kernel"]["answer"]
    return record


def eval_recall(
    model, tok, body: str, rng: random.Random, chunk: int, max_new: int, eos_ids, pending_fn
) -> dict:
    password = _random_password(rng)
    body_with_pw, frac = _insert_password(rng, body, password)
    question = "合言葉は何ですか。合言葉だけ答えてください。"
    text = f"{body_with_pw}\n\n---\n\n{question}"
    ids = _build_ids(tok, text)

    def _check(route: str, generated: list[int]) -> tuple[bool, str]:
        answer = tok.decode(generated[:32])
        return password in answer, answer

    record = _eval_problem(model, tok, ids, _check, chunk, max_new, eos_ids, pending_fn)
    record["password"] = password
    record["insert_frac"] = round(frac, 4)
    return record


def eval_quote(
    model, tok, body: str, rng: random.Random, chunk: int, max_new: int, eos_ids, pending_fn
) -> dict:
    quoted, expected_prefix = _pick_quote_pair(rng, body)
    question = f"次の文の直後に続く文をそのまま書き出してください: 『{quoted}』"
    text = f"{body}\n\n---\n\n{question}"
    ids = _build_ids(tok, text)

    def _check(route: str, generated: list[int]) -> tuple[bool, str]:
        answer = tok.decode(generated[:max_new])
        return expected_prefix in answer, answer

    record = _eval_problem(model, tok, ids, _check, chunk, max_new, eos_ids, pending_fn)
    record["quoted_sentence"] = quoted
    record["expected_prefix"] = expected_prefix
    return record


TASKS = {"recall": eval_recall, "quote": eval_quote}

# 「worker には載せられない」を表す終了コード。`tools/ab_submit.py` の
# NOT_ROUTABLE と同じ値で、`tools/biglock.sh` はこれを受けると自分でロックを
# 取り直して従来どおり別プロセスで流す (worker には 98GB を返させる)。
NOT_ROUTABLE = 64


class ArgError(ValueError):
    """引数の誤り。1 行印字して終了コード 1 (従来の main と同じ扱い)。"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="prefill attention 融合カーネル (MLXTURBO_PREFILL_ATTN) の"
        " on/off で、長文脈の課題 (recall/quote) の正答率がどれだけ動くかを測る"
        " (KLD は QSA の top-k カスケードで物差しにならないので使わない)"
    )
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument("--mtp", default=None, help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument(
        "--ctxs",
        default="17000,50000",
        help="kv 長のカンマ区切り一覧 (カーネルは kv>=12288 あたりで発火)。"
        " 50k は 1 問の prefill が約 95 秒 x dense/kernel 2 経路かかるので、"
        " --ctxs と --n を両方大きくすると数十分〜規模になる"
        " (目安: 95s x 2経路 x 8問 x 課題2種 ≈ 50 分@50k)。手早く確かめたい"
        " ときは --ctxs と --n を両方小さくすること (例: --ctxs 4000 --n 2)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=8,
        help="課題 (recall/quote) ごとの問題数 (既定 8)。時間がかかるときは"
        " 減らすこと (--ctxs の help 参照)",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="合言葉・挿入位置・引用文選択の乱数種"
    )
    ap.add_argument("--chunk", type=int, default=2048, help="prefill のチャンク幅")
    ap.add_argument(
        "--max-new", type=int, default=64, help="貪欲デコードの最大トークン数"
    )
    ap.add_argument(
        "--qsa-tail",
        choices=("query", "global"),
        default=None,
        help="QSA の端数ブロックの可視規則 (mlxturbo/qsa_tail.py の MODE)。"
        " 省略時は今の値のまま (CLI では環境変数 MLXTURBO_QSA_TAIL どおり)。"
        " 常駐 worker は環境変数を読み直さないので、切り替えはここで指定する",
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "bench" / "results" / "longctx-quality.json"))
    return ap


def parse_args(argv=None):
    """引数を解釈して ``(args, ctxs)`` を返す。誤りは `ArgError`。

    98GB を読む前に弾けるものはここで全部弾く。CLI (`main`) と常駐 worker
    (`run_with_model`) の両方がこれを通る。
    """
    args = build_parser().parse_args(argv)

    ctxs = sorted({int(v) for v in args.ctxs.split(",") if v.strip() != ""})
    if not ctxs:
        raise ArgError("--ctxs が空")
    if args.n <= 0:
        raise ArgError("--n は正の整数にすること")
    if args.max_new <= 0:
        raise ArgError("--max-new は正の整数にすること")
    if args.chunk <= 0:
        raise ArgError("--chunk は正の整数にすること")
    return args, ctxs


def run(model, tok, eos_ids, args, ctxs) -> int:
    """読み込み済みのモデルで課題を解く本体 (CLI と worker が共有する)。

    `--qsa-tail` の指定はここで `mlxturbo.qsa_tail.MODE` に当てる。
    **戻すのは呼び出し側** --- CLI はプロセスごと終わるので戻す必要が無く、
    worker (`run_with_model`) は借り物なので必ず戻す。
    """
    import prefill_anatomy as PA  # noqa: E402 (pending() を借りる)
    from mlxturbo import qsa_tail

    if args.qsa_tail is not None:
        qsa_tail.MODE = args.qsa_tail

    rng = random.Random(args.seed)

    print(
        f"model={args.model} ngram={args.ngram} ctxs={ctxs} n={args.n}"
        f" seed={args.seed} chunk={args.chunk} max_new={args.max_new}"
        f" qsa_tail={qsa_tail.MODE}",
        flush=True,
    )

    results: dict = {
        "kind": "longctx-quality",
        "model": args.model,
        "ngram": args.ngram,
        "seed": args.seed,
        "n": args.n,
        "chunk": args.chunk,
        "max_new": args.max_new,
        "qsa_tail": qsa_tail.MODE,
        "ctxs": ctxs,
        "tasks": {name: {} for name in TASKS},
    }

    for ctx in ctxs:
        print(f"=== ctx={ctx} ===", flush=True)
        bodies = _windows(tok, ctx, args.n * len(TASKS))
        task_bodies = {
            name: bodies[i * args.n : (i + 1) * args.n]
            for i, name in enumerate(TASKS)
        }

        for task_name, eval_fn in TASKS.items():
            problems = []
            for i, body in enumerate(task_bodies[task_name]):
                print(f"  [{task_name}] problem {i + 1}/{args.n}", flush=True)
                rec = eval_fn(
                    model, tok, body, rng, args.chunk, args.max_new, eos_ids, PA.pending
                )
                problems.append(rec)
                print(
                    f"    kv={rec['kv']} dense={'OK' if rec['dense']['correct'] else 'NG'}"
                    f" kernel={'OK' if rec['kernel']['correct'] else 'NG'}"
                    f" kernel_fired={rec['kernel']['kernel_fired']}"
                    f" agree={'Y' if rec['agree'] else 'N'}",
                    flush=True,
                )

            n_problems = len(problems)
            dense_correct = sum(1 for r in problems if r["dense"]["correct"])
            kernel_correct = sum(1 for r in problems if r["kernel"]["correct"])
            agree = sum(1 for r in problems if r["agree"])
            kernel_fired_total = sum(r["kernel"]["kernel_fired"] for r in problems)
            entry = {
                "n": n_problems,
                "dense": {"correct": dense_correct, "acc": dense_correct / n_problems},
                "kernel": {"correct": kernel_correct, "acc": kernel_correct / n_problems},
                "agree_rate": agree / n_problems,
                "kernel_fired_total": kernel_fired_total,
                "problems": problems,
            }
            results["tasks"][task_name][str(ctx)] = entry
            if kernel_fired_total == 0:
                print(
                    f"  ★[{task_name}] ctx={ctx}: カーネルが1度も発火していない"
                    " (kv が閾値未満、または eligible() が別の理由で弾いている"
                    " 可能性。この ctx のカーネル側正答率は無意味) ★",
                    flush=True,
                )
            print(
                f"  [{task_name}] ctx={ctx}: dense_acc={entry['dense']['acc']:.3f}"
                f" kernel_acc={entry['kernel']['acc']:.3f}"
                f" agree_rate={entry['agree_rate']:.3f}"
                f" kernel_fired_total={kernel_fired_total}",
                flush=True,
            )

    print("\n=== 課題 x 文脈長 x 経路の正答率 ===")
    header = f"{'task':<8} {'ctx':>8} {'dense_acc':>10} {'kernel_acc':>11} {'agree':>7}"
    print(header)
    for ctx in ctxs:
        for task_name in TASKS:
            e = results["tasks"][task_name][str(ctx)]
            print(
                f"{task_name:<8} {ctx:>8} {e['dense']['acc']:>10.3f}"
                f" {e['kernel']['acc']:>11.3f} {e['agree_rate']:>7.3f}"
            )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nwrote {out_path}", flush=True)
    return 0


_MISSING = object()


def _route_snapshot(model) -> list[tuple]:
    """`_set_route` が書き換える経路の旗を控える。

    `enable_prefill_attn` / `disable_gather_attn` は層の ``self_attn`` に
    ``_gather_attn`` / ``_prefill_attn`` / ``_gather_attn_tile`` を直に立てる。
    この道具は最後の 1 問を必ず kernel 経路で終えるので、**戻さないと次の
    ジョブが「本番の既定」ではなく「この道具が最後に立てた旗」で走る。**
    """
    from mlxturbo.gather_attn import _each_layer

    snap = []
    for layer in _each_layer(model):
        sa = getattr(layer, "self_attn", None)
        if sa is None or not hasattr(sa, "indexer"):
            continue
        snap.append((sa, {a: getattr(sa, a, _MISSING)
                          for a in ("_gather_attn", "_prefill_attn",
                                    "_gather_attn_tile")}))
    return snap


def _route_restore(snap: list[tuple]) -> None:
    for sa, attrs in snap:
        for name, val in attrs.items():
            if val is _MISSING:
                try:
                    delattr(sa, name)
                except Exception:
                    pass
            else:
                setattr(sa, name, val)


def run_with_model(argv, bundle) -> int:
    """読み込み済みの一式で本体を走らせる (`tool` ジョブの入口)。

    規約は `tools/ab_bundle.py` の docstring。**借り物のモデルに旗を残さない**
    のがここの仕事で、戻すのは 3 つ:

    - 経路の旗 (`_route_snapshot` の docstring)
    - `mlxturbo.qsa_tail.MODE` (`--qsa-tail` を指定したとき)
    - `prefill_attn` の差し替え (`_run_route` が自分の finally で戻すが、
      例外で抜けた場合の網としてここでも控える)

    worker が抱えているモデルと引数が食い違うときは `NOT_ROUTABLE` (64)。
    `tools/biglock.sh` はこれを受けると worker に 98GB を返させてから、
    従来どおり別プロセスで流し直す。
    """
    try:
        args, ctxs = parse_args(argv)
    except ArgError as e:
        print(e)
        return 1
    except SystemExit as e:  # argparse の --help / 引数エラー
        return int(e.code or 0)

    bad = bundle.mismatch(model_path=args.model, ngram_path=args.ngram,
                          mtp_path=args.mtp, mtp_bits=args.mtp_bits)
    if bad:
        print(f"常駐 worker には載せられない: {bad}")
        return NOT_ROUTABLE

    from mlxturbo import qsa_tail
    from mlxturbo.kernels import prefill_attn as prefill_attn_kernel

    snap = _route_snapshot(bundle.model)
    saved_mode = qsa_tail.MODE
    saved_kernel = prefill_attn_kernel.prefill_attn
    try:
        return run(bundle.model, bundle.tokenizer, bundle.eos_ids, args, ctxs)
    finally:
        prefill_attn_kernel.prefill_attn = saved_kernel
        qsa_tail.MODE = saved_mode
        _route_restore(snap)


def main() -> int:
    try:
        args, ctxs = parse_args()
    except ArgError as e:
        print(e)
        return 1

    if args.ngram:
        # n-gram をディスクに置いた構成。vendored arch は import 時に旗を読む。
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    from verify_width_cost import build_runner  # noqa: E402

    eng, model, tok, eos_ids = build_runner(args)
    del eng  # 生成に FlashSpecEngine/MTP は使わない (build_runner は出荷経路と
    # 同じ融合を model に当てる副作用のためだけに呼んでいる)。

    rc = run(model, tok, eos_ids, args, ctxs)
    if rc:
        return rc

    # 計測ツールなので destructor (スレッドプール等の後始末) に用は無い。
    # interpreter shutdown 待ちでプロセスが Metal のメモリを握ったまま残る
    # 前例があるので、結果を書き終えたら即 _exit で落とす (他の tools/*.py と同じ)。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
