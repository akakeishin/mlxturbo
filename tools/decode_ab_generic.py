"""qwen3_5 (27B) の部品差し替えを 1 プロセス内 A/B で判定する最小の道具。

`tools/decode_ab.py` は Flash-Next (qwen4_exp) 専用 --- `load_bundle` が
`mtp_flash.load_flash_mtp` を呼び、knob 群が qwen4_exp のクラスを直に触り、
生成が `FlashSpecEngine.generate_stream` に固定されている。27B は
**別のエンジン** (`mlxturbo/spec.py` の `SpecEngine`、`generate` は 1 発返しで
stream 版が無い) で動くので、そちらで回る版をここに置く。**decode_ab の写しでは
ない** (knob 表も回文の器も持たず、env 1 本の切り替えだけを扱う)。

## 走らせ方

    BIGLOCK_NO_WORKER=1 tools/biglock.sh .venv/bin/python tools/decode_ab_generic.py \
        --model ~/models/qwen38-27b-4bit --mtp ~/models/qwen38-27b-mtp \
        --knob MLXTURBO_QMM_WIDE=auto,off --ctx 0 --tokens 128 \
        --out bench/results/qmm-wide-27b-short.json

`--ctx 0` は短文脈 3 本 (`decode_ab.SHORT_PROMPTS` をそのまま共有)、`--ctx N` は
`bench/textpool-frozen.txt` から切った N トークンの窓 1 本。長文脈は
`--prefill-once` を付けて prefill を 1 回に畳む。

    ... --ctx 4000 --prefill-once --tokens 128

**常駐 worker (`tools/ab_daemon.py`) には乗らない。**あちらの `AbBundle` は
Flash-Next の一式 (`FlashSpecEngine`) を抱えるので、27B の `SpecEngine` を
渡す口が無い。`BIGLOCK_NO_WORKER=1` を付けて素の biglock で流すこと。

## knob の形

    --knob <ENV 変数名>=<値1>,<値2>[,<値3>...]

値は環境変数にそのまま入る文字列。**空文字はその変数を消す (未設定に戻す)**
意味で、`--knob MLXTURBO_GDN_METAL=,0` のように「既定 (未設定) 対 明示 off」を
測れる。基準 (まとめの分母) は既定で**最後の値** (`--baseline` で変えられる)。

各 variant の前に踏む手順は 3 つ:

1. 分かっている `disable_*` を全部呼ぶ (`_reset_fusions`)
2. `disable_*` を持たない差し替えは、控えておいた元の属性を戻し、
   **再パッチを止めている番人 (`fused._QMM_WIDE_STOCK` など) も戻す**
   (`_OrigGuard`)。番人を戻さないと、次の enable が「もう当てた」と見て
   黙って何もしない --- 対照が丸ごと死ぬ形の失敗なので、ここが要
3. env を立ててから `runner.enable_default_fusions(model)` を呼び直す

`mlxturbo/` は 1 行も変えない (`fused.py` の qwen4_exp 向け enable 群は
そのまま使うだけ)。

## 計測の作法 (CLAUDE.md に従う)

- 1 プロセス内でプロンプトごとに回文順 (値が 2 つなら A,B,B,A)
- **読み込み直後の 1 本は捨てる** (`burn_in`、`--no-burn-in` で切れる)。
  さらに variant ごと・ケースごとにも 32 トークンを 1 本捨てる
  (前者は Metal カーネルの JIT、後者はキャッシュを組み直す 1 本目の段差)
- 生成長は全条件そろえる。判定は ms/round と tok/round を分けて見る
  (ms/token は両者の比なので費用と受理が混ざる)
- 出力の先頭 24 トークン (`head`) が条件間で一致するかは常に表示する。
  ビット一致するはずの knob でここが割れたら測定は無効

## 出力

`decode_ab.py` と同じキーの行を並べる: `kind` / `ctx` / `case_idx` /
`variant` / `n_out` / `prefill_s` / `decode_s` / `ms_per_tok` /
`ms_per_round` / `accepted` / `rounds` / `tok_per_round` / `head` / `fired`。
`--prefill-once` のときの `prefill_s` は decode_ab と同じく 0.0 で埋める。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))


# ---------------------------------------------------------------- 融合の貼り直し

class _OrigGuard:
    """`disable_*` を持たない差し替えの控えと戻し。

    表の 1 行は「(オブジェクト, 属性名, `fused` 側で再パッチを止めている番人の
    名前)」。`capture()` は **最初の `enable_default_fusions` より前に**呼ぶこと
    (でないと控えるのが差し替え後の関数になる)。`restore()` は控えを書き戻し、
    番人を `None` に戻して次の enable が改めて当てられるようにする。

    今 1 行しか無いのは、27B に当たる融合のうち `disable_*` を持たないものが
    他に無いから (`enable_gather_sort` は MoE 専用で 27B では空振り、
    `enable_sdpa_rowtile` は qwen4_exp のモジュール関数を差し替えるだけ)。
    部品が増えたらここに足す。
    """

    def __init__(self):
        self._saved: list[tuple[object, str, object, str]] = []

    def capture(self) -> None:
        import mlx.nn as nn

        from mlxturbo import fused

        table = [
            # 段 P10 / HC 版の qmm_wide が共有する差し替え。番人は
            # `fused._QMM_WIDE_STOCK` (None のときだけ当て直す)
            (nn.QuantizedLinear, "__call__", fused, "_QMM_WIDE_STOCK"),
        ]
        self._saved = [
            (obj, attr, getattr(obj, attr), (mod, sentinel))
            for obj, attr, mod, sentinel in table
        ]

    def restore(self) -> None:
        for obj, attr, orig, (mod, sentinel) in self._saved:
            setattr(obj, attr, orig)
            setattr(mod, sentinel, None)


def _reset_fusions(model, guard: _OrigGuard, warned: set) -> None:
    """既定で当たっている融合を全部剥がす。

    `disable_*` は「フラグを下ろすだけ」のものが多い (差し替え自体は残す) が、
    それは A/B で交互に測るための設計なのでそのまま使う。素の対照が要る
    ものだけ `guard` が属性ごと戻す。

    存在しない / 例外を出す `disable_*` は 1 度だけ知らせて飛ばす
    (族が違えば空振りするのが正しい挙動なので、止めない)。
    """
    from mlxturbo import fused, gather_attn, indexer_lean, qsa_decode

    calls = [
        ("hyper_connection_kernel", lambda: fused.disable_hyper_connection_kernel()),
        ("hyper_connection_elem", lambda: fused.disable_hyper_connection_elem()),
        ("hyper_connection", lambda: fused.disable_hyper_connection()),
        ("hc_prefill_compiled",
         lambda: fused.disable_hyper_connection_prefill_compiled()),
        ("hc_write", lambda: fused.disable_hc_write()),
        ("hc_qmm_wide", lambda: fused.disable_hc_qmm_wide(model)),
        ("qmm_wide", lambda: fused.disable_qmm_wide()),
        ("moe_verify_gather", lambda: fused.disable_moe_verify_gather()),
        ("moe_decode_fused", lambda: fused.disable_moe_decode_fused(model)),
        ("moe_combine_fold", lambda: fused.disable_moe_combine_fold(model)),
        ("moe_grouped_gemm", lambda: fused.disable_moe_grouped_gemm()),
        ("moe_down_epilogue", lambda: fused.disable_moe_down_epilogue()),
        ("moe_route", lambda: fused.disable_moe_route()),
        ("rms_norm_gated", lambda: fused.disable_rms_norm_gated()),
        ("gdn_prework", lambda: fused.disable_gdn_prework_kernel()),
        ("gdn_decode_fused", lambda: fused.disable_gdn_decode_fused()),
        ("gdn_blocked", lambda: fused.disable_gdn_blocked_kernel()),
        ("gdn_metal", lambda: fused.disable_gdn_metal_kernel()),
        # qwen4_exp 以外の族 (27B) の GDN 3 部品。上の 4 つは
        # `_vendor/qwen4_exp.py` のシームを立てるだけで 27B には届かない
        ("gdn_port", lambda: fused.disable_gdn_port(model)),
        ("sdpa_split", lambda: fused.disable_sdpa_split()),
        ("sdpa_rowtile", lambda: fused.disable_sdpa_rowtile()),
        ("fast_rope", lambda: fused.disable_fast_rope(model)),
        ("ple_hoist", lambda: fused.disable_ple_hoist(model)),
        ("gather_attn", lambda: gather_attn.disable_gather_attn(model)),
        ("prefill_attn", lambda: gather_attn.disable_prefill_attn(model)),
        ("qsa_decode", lambda: qsa_decode.disable_qsa_decode_kernel(model)),
        ("indexer_lean", lambda: indexer_lean.disable_indexer_lean(model)),
    ]
    for name, fn in calls:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            if name not in warned:
                warned.add(name)
                print(f"  [reset] disable_{name} を飛ばした ({type(exc).__name__}: {exc})")
    guard.restore()


def make_set_variant(model, env_name: str, guard: _OrigGuard):
    """variant (env の値) を当てる関数を返す。空文字は「未設定に戻す」。"""
    from mlxturbo.runner import enable_default_fusions

    warned: set = set()
    logged: set = set()

    def set_variant(value: str) -> None:
        _reset_fusions(model, guard, warned)
        if value == "":
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = value
        buf = io.StringIO()
        with redirect_stdout(buf):
            enable_default_fusions(model, log_prefix="[ab_gen]")
        if value not in logged:
            logged.add(value)
            shown = "(未設定)" if value == "" else value
            print(f"  [{env_name}={shown}] 融合の貼り直し:")
            for line in buf.getvalue().splitlines():
                print(f"    {line}")

    return set_variant


# ---------------------------------------------------------------- キャッシュの控え

def _snap_cache(c):
    """1 つのキャッシュの状態を控える。

    KVCache は `keys`/`values`/`offset` の参照を控えるだけでよい ---
    `update_and_fetch` の代入は `keys[..., offset:new, :]` で **offset より
    後ろ**しか触らないので、offset を戻せば控えた前半はそのまま生きている
    (`tools/decode_ab.py` の `_snapshot` と同じ理屈)。ArraysCache (linear 層)
    は list のスロットが差し替わるので list ごと写す
    (`mlxturbo/spec.py` の `snapshot_untrimmable_caches` と同じ)。
    """
    if hasattr(c, "keys"):
        return ("kv", c.keys, c.values, c.offset)
    st = c.state
    return ("st", list(st) if isinstance(st, list) else st,
            getattr(c, "left_padding", None), getattr(c, "lengths", None))


def _restore_cache(c, rec) -> None:
    if rec[0] == "kv":
        _, c.keys, c.values, c.offset = rec
        return
    _, st, left_padding, lengths = rec
    c.state = list(st) if isinstance(st, list) else st
    if left_padding is not None:
        c.left_padding = left_padding
    if lengths is not None:
        c.lengths = lengths


def _snapshot_session(sess) -> dict:
    return dict(
        caches=sess.caches,
        cache_snap=[_snap_cache(c) for c in sess.caches],
        mtp_cache=sess.mtp_cache,
        mtp_snap=_snap_cache(sess.mtp_cache) if sess.mtp_cache is not None else None,
        mtp_valid=sess.mtp_valid,
        processed=list(sess.processed),
        h_last=sess.h_last,
        checkpoints=list(sess.checkpoints),
        tail=sess.tail,
    )


def _restore_session(sess, st: dict) -> None:
    for c, rec in zip(st["caches"], st["cache_snap"]):
        _restore_cache(c, rec)
    if st["mtp_cache"] is not None:
        _restore_cache(st["mtp_cache"], st["mtp_snap"])
    sess.caches = st["caches"]
    sess.mtp_cache = st["mtp_cache"]
    sess.mtp_valid = st["mtp_valid"]
    sess.processed = list(st["processed"])
    sess.h_last = st["h_last"]
    sess.checkpoints = list(st["checkpoints"])
    sess.tail = st["tail"]


# ---------------------------------------------------------------- 生成

def _result_row(res: dict, wall: float, resumed: bool) -> dict:
    """`SpecEngine.generate` の返り値を decode_ab と同じ形に均す。

    `ttft_s` は prefill + 1 トークン目まで。`--prefill-once` の再開では
    prefill を踏まないので、decode_ab の `run_resumed` に合わせて
    `prefill_s` は 0.0 で埋める (decode 秒はどちらも壁時計 - ttft)。
    """
    out = list(res["tokens"])
    rounds = int(res["steps"])
    accepted = sum(k * v for k, v in res["accept_hist"].items())
    t_dec = max(wall - res["ttft_s"], 1e-9)
    return dict(
        n_out=len(out),
        prefill_s=0.0 if resumed else res["ttft_s"],
        decode_s=t_dec,
        ms_per_tok=t_dec / max(len(out), 1) * 1000,
        ms_per_round=t_dec / max(rounds, 1) * 1000,
        accepted=accepted,
        rounds=rounds,
        tok_per_round=len(out) / max(rounds, 1),
        head=out[:24],
    )


def run_once(eng, ids, n_tokens, eos_ids, n_draft, max_draft) -> dict:
    """まっさらな状態から prefill + decode を 1 本流す。"""
    import mlx.core as mx

    mx.clear_cache()
    t0 = time.perf_counter()
    res = eng.generate(ids, max_tokens=n_tokens, n_draft=n_draft,
                       max_draft=max_draft, temp=0.0, eos_ids=eos_ids)
    return _result_row(res, time.perf_counter() - t0, resumed=False)


def prefill_once(eng, ids, n_draft, max_draft):
    """prefill を 1 回だけ流し、`(session, snapshot, 秒)` を返す。

    `SpecEngine.generate(max_tokens=0)` は prefill を流して `ChatSession` に
    publish するだけで返る。次から同じ `prompt_ids` を渡すと、LCP が全長一致し
    `session.tail` の位置も一致するので **prefill フォワードを 1 回も踏まず**
    控えた hidden から decode に入る (`mlxturbo/spec.py` の `resume_h` 分岐)。
    各条件がまったく同じ状態から始まるので、prefill をやり直すより制御が効く。

    **prefill に効く knob には使わないこと** (片方の条件で組んだキャッシュから
    もう片方が decode を再開してしまう)。
    """
    import mlx.core as mx

    from mlxturbo.spec import ChatSession

    sess = ChatSession()
    mx.clear_cache()
    t0 = time.perf_counter()
    eng.generate(ids, max_tokens=0, n_draft=n_draft, max_draft=max_draft,
                 temp=0.0, eos_ids=(), session=sess)
    took = time.perf_counter() - t0
    if sess.tail is None or sess.tail[0] != len(ids):
        raise RuntimeError(
            f"prefill-once: session.tail が位置 {len(ids)} に立たなかった"
            f" (tail={None if sess.tail is None else sess.tail[0]})。"
            " これが無いと再開で prefill を踏み直すので A/B が壊れる")
    return sess, _snapshot_session(sess), took


def run_resumed(eng, ids, sess, snap, n_tokens, eos_ids, n_draft, max_draft) -> dict:
    """控えた prefill 状態から decode だけを流す。返り値は run_once と同じ形。"""
    _restore_session(sess, snap)
    t0 = time.perf_counter()
    res = eng.generate(ids, max_tokens=n_tokens, n_draft=n_draft,
                       max_draft=max_draft, temp=0.0, eos_ids=eos_ids,
                       session=sess)
    row = _result_row(res, time.perf_counter() - t0, resumed=True)
    if res["prefill_new"] != 0:
        raise RuntimeError(
            f"prefill-once: 再開で {res['prefill_new']} トークンを prefill し直した"
            " (session の再利用が効いていない)。A/B は無効")
    return row


# ---------------------------------------------------------------- 読み込み

def load_model(args):
    """出荷経路と同じ状態の `(model, tokenizer, engine, eos_ids, guard)` を返す。

    `tools/ab_bundle.py` は Flash-Next 一式 (`mtp_flash.load_flash_mtp` +
    `FlashSpecEngine`) を作るので 27B では使えない。ここは
    `mlxturbo/runner.py` の非 qwen4_exp 分岐 (`load_cli_mtp` ->
    `SpecEngine`) と同じ順で組む。
    """
    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401  (arch 登録を先に通す)
    from mlxturbo._mlx_compat import TextModelArgs
    from mlxturbo.cli import load_cli_mtp
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default
    from mlxturbo.spec import SpecEngine

    model_path = os.path.realpath(os.path.expanduser(args.model))
    model, tok = load(model_path)

    # **enable_default_fusions より前に控える。**後だと差し替え済みの関数を
    # 「元」として掴んでしまい、素の対照が二度と作れない
    guard = _OrigGuard()
    guard.capture()

    enable_default_fusions(model, log_prefix="[ab_gen]")
    set_wired_limit_default(log_prefix="[ab_gen]")

    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:  # noqa: BLE001
        config = {}
    text_args = TextModelArgs.from_dict(model.args.text_config)

    mtp_path = args.mtp
    if mtp_path is None:
        cand = Path(model_path) / "mtp.safetensors"
        mtp_path = str(cand) if cand.exists() else None
    if mtp_path:
        mtp_path = os.path.realpath(os.path.expanduser(mtp_path))
    mtp = load_cli_mtp(model_path, config, text_args, "", args.mtp_bits,
                       no_mtp=args.no_mtp, mtp_path=mtp_path)
    if mtp is None:
        print("[ab_gen] MTP なし: lookup (SAM) だけの投機になる"
              " (--mtp を渡したか確認すること)")
    else:
        mx.eval(mtp.parameters())

    eng = SpecEngine(model, mtp)
    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    return model, tok, eng, tuple(eos) if eos else (), guard


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="qwen3_5 (27B) の env knob を 1 プロセス内 A/B で測る")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", default=None,
                    help="MTP サイドカー (ディレクトリか safetensors)。"
                         "既定は <model>/mtp.safetensors が在れば使う")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--knob", required=True, metavar="ENV=A,B",
                    help="環境変数 1 つの値を A/B で切り替える。値は 2 つ以上。"
                         "空文字はその変数を未設定に戻す意味 (例 FOO=,0)")
    ap.add_argument("--baseline", default=None,
                    help="まとめの分母にする値 (既定は --knob の最後の値)")
    ap.add_argument("--ctx", type=int, default=0,
                    help="0 = 短文脈 3 本 / N = 池から切った N トークンの窓 1 本")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=1,
                    help="1 ケースあたりの回文の本数 (既定 1 = A,B,B,A)")
    ap.add_argument("--n-draft", type=int, default=3,
                    help="本番既定 (runner.build_runner)")
    ap.add_argument("--max-draft", type=int, default=8,
                    help="本番既定 (runner.build_runner)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill-once", action="store_true",
                    help="長文脈で prefill を 1 回に畳む。"
                         "**prefill に効く knob には使わないこと**")
    ap.add_argument("--round-trace", action="store_true",
                    help="MLXTURBO_ROUND_TRACE=1 を立て、round ごとの "
                         "(検証幅, 受理数, source, ms) を集めて幅ごとの平均を "
                         "出す。生の列は JSON の `round_trace` に入る")
    ap.add_argument("--no-burn-in", action="store_true",
                    help="読み込み直後の空焼きをしない (既定は 1 本捨てる)")
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    name, sep, vals = args.knob.partition("=")
    if not sep or not name.strip():
        ap.error("--knob は ENV=A,B の形にすること")
    variants = [v.strip() for v in vals.split(",")]
    if len(variants) < 2:
        ap.error("--knob の値は 2 つ以上にすること")
    if len(set(variants)) != len(variants):
        ap.error("--knob に同じ値が 2 回入っている")
    baseline = args.baseline if args.baseline is not None else variants[-1]
    if baseline not in variants:
        ap.error(f"--baseline {baseline!r} は --knob の値に無い")
    if args.prefill_once and args.ctx <= 0:
        ap.error("--prefill-once は --ctx N (長文脈) のときだけ使う")
    return args, name.strip(), variants, baseline


def build_cases(tok, ctx: int):
    """(kind, prompt_ids) の列を作る。thinking は落とす (self_snapshot と同じ)。"""
    from _bench_text import long_prompts
    from decode_ab import LONG_QUESTIONS, SHORT_PROMPTS

    def to_ids(text):
        msgs = [{"role": "user", "content": text}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        return list(ids)

    if ctx <= 0:
        return [("short", to_ids(p)) for p in SHORT_PROMPTS]
    return [("long", to_ids(p))
            for p in long_prompts(tok, ctx, LONG_QUESTIONS[:1])]


def summarize(rows, variants, baseline) -> None:
    print("\n=== まとめ ===")
    for kind in ("short", "long"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        for metric in ("ms_per_tok", "ms_per_round", "tok_per_round", "prefill_s"):
            means = {}
            for v in variants:
                vals = [r[metric] for r in sub if r["variant"] == v]
                means[v] = sum(vals) / len(vals) if vals else 0.0
            base = means[baseline]
            if base == 0:
                continue
            cells = "  ".join(
                f"{v or '(未設定)'}={means[v]:8.3f}"
                f"({(means[v] - base) / base * 100:+5.1f}%)" for v in variants)
            print(f"  {kind:5s} {metric:14s} {cells}   [基準 {baseline or '(未設定)'}]")

    # head の一致は常に見る。ビット一致するはずの knob でここが割れたら
    # 測定は無効 (どちらが正しいかはこの道具では決められない)
    n_same = n_diff = 0
    for kc in sorted({(r["kind"], r["case_idx"]) for r in rows}):
        sub = [r for r in rows if (r["kind"], r["case_idx"]) == kc]
        if len({tuple(r["head"]) for r in sub}) == 1:
            n_same += 1
        else:
            n_diff += 1
            print(f"  head 不一致: {kc[0]} case={kc[1]} ctx={sub[0]['ctx']}")
    print(f"  head: {n_same} ケース一致 / {n_diff} ケース不一致")


def summarize_round_trace(rows: list) -> None:
    """`--round-trace` の集計: 検証幅 (= 1 + draft の本数) ごとの ms。

    「82 / 95 / 111 ms/round のばらつきが draft の本数で説明できるか」を
    見るための表。幅ごとの中央値と、幅 1 からの増分 (1 リンクあたりの ms) を
    並べる。ケース (prompt) ごとに出す -- 幅の分布が prompt で違うのが
    そもそもの論点なので、混ぜると平均が動いた理由が消える。
    """
    from statistics import median

    print("\n== round trace (検証幅ごとの ms) ==")
    for row in rows:
        trace = row.get("round_trace")
        if not trace:
            continue
        by_w: dict[int, list[float]] = {}
        acc_w: dict[int, list[int]] = {}
        for width, consumed, _source, ms in trace:
            by_w.setdefault(width, []).append(ms)
            acc_w.setdefault(width, []).append(consumed)
        base = median(by_w[1]) if 1 in by_w else None
        head = (f"ctx={row['ctx']} case={row['case_idx']} "
                f"variant={row['variant'] or '(未設定)'}")
        print(f"  {head}: rounds={len(trace)} "
              f"ms/round(全体)={median(m for ms in by_w.values() for m in ms):.1f}")
        for w in sorted(by_w):
            ms_l = by_w[w]
            per_link = ""
            if base is not None and w > 1:
                per_link = f"  幅1 からの増分 {(median(ms_l) - base) / (w - 1):+.2f} ms/リンク"
            print(f"    幅 {w}: n={len(ms_l):4d}  ms 中央 {median(ms_l):6.1f}"
                  f"  受理 平均 {sum(acc_w[w]) / len(acc_w[w]):.2f}{per_link}")


def main() -> int:
    import mlx.core as mx

    args, env_name, variants, baseline = parse_args()
    if args.round_trace:
        os.environ["MLXTURBO_ROUND_TRACE"] = "1"
    model, tok, eng, eos_ids, guard = load_model(args)
    mx.random.seed(args.seed)

    from mlxturbo.kernels import _fire

    set_variant = make_set_variant(model, env_name, guard)
    cases = build_cases(tok, args.ctx)
    nd, md = args.n_draft, args.max_draft

    print(f"\nknob {env_name}: {variants} (基準 {baseline or '(未設定)'})")
    print(f"生成長 {args.tokens} トークンで全条件そろえる。"
          f"回文 {args.reps} 本 x {len(cases)} ケース。最初の 1 本は捨てる。\n")

    # 1) プロセスの空焼き。読み込み直後の 1 本だけが 8〜9% 遅い
    #    (decode_ab.burn_in の実測)。回文はこの段差を相殺できないので
    #    (位置 1 に必ず先頭の variant が来る)、本番の前に 1 本焼いて捨てる
    if not args.no_burn_in:
        t0 = time.perf_counter()
        set_variant(baseline)
        for _kind, ids in build_cases(tok, 0):
            run_once(eng, ids, 32, eos_ids, nd, md)
        print(f"[ab_gen] 空焼き 1 本を捨てた ({time.perf_counter() - t0:.1f}s)")

    # 2) variant ごとの空焼き。mx.fast.metal_kernel は初回発火で JIT する
    for v in variants:
        set_variant(v)
        for _kind, ids in cases:
            run_once(eng, ids, 32, eos_ids, nd, md)
    set_variant(baseline)

    order = (variants + variants[::-1]) * args.reps
    rows = []
    for case_idx, (kind, ids) in enumerate(cases):
        n = len(ids)
        print(f"--- {kind} ctx={n} ---", flush=True)
        shared = None
        if args.prefill_once:
            sess, snap, took = prefill_once(eng, ids, nd, md)
            shared = (sess, snap)
            print(f"  prefill 1 回だけ流した ({took:.1f}s)。以降は decode のみ",
                  flush=True)
        # 3) ケースごとの空焼き。キャッシュを組み直す 1 本目に段差が出る
        set_variant(baseline)
        if shared is None:
            run_once(eng, ids, 32, eos_ids, nd, md)
        else:
            run_resumed(eng, ids, *shared, n_tokens=32, eos_ids=eos_ids,
                        n_draft=nd, max_draft=md)
        for v in order:
            set_variant(v)
            _fire.reset()
            if shared is None:
                row = run_once(eng, ids, args.tokens, eos_ids, nd, md)
            else:
                row = run_resumed(eng, ids, *shared, n_tokens=args.tokens,
                                  eos_ids=eos_ids, n_draft=nd, max_draft=md)
            row.update(kind=kind, ctx=n, case_idx=case_idx, variant=v)
            row["fired"] = _fire.snapshot()
            if args.round_trace:
                row["round_trace"] = list(getattr(eng, "last_round_trace", None) or [])
            rows.append(row)
            fired_s = ("  発火 " + " ".join(f"{k}={c}" for k, c in
                                            sorted(row["fired"].items()))
                       ) if row["fired"] else ""
            print(f"  {v or '(未設定)':>8s}: prefill {row['prefill_s']:6.2f}s"
                  f"  decode {row['decode_s']:6.2f}s  {row['ms_per_tok']:6.2f} ms/tok"
                  f"  {row['ms_per_round']:6.2f} ms/round"
                  f"  tok/round {row['tok_per_round']:.3f}"
                  f"  ({row['accepted']}/{row['rounds']}){fired_s}", flush=True)
        set_variant(baseline)

    summarize(rows, variants, baseline)
    if args.round_trace:
        summarize_round_trace(rows)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"\n書き出し: {p}")

    # 計測ツールなので後始末は要らない。interpreter shutdown 待ちで Metal の
    # メモリを握ったまま残った実測があるので、書き終えたら即落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
