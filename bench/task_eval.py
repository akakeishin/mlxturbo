"""Phase Q: 量子化変種を既存ベンチマークで測る (課題レーン)。

`quant_eval.py` は参照分布との距離 (KLD・top1 一致) を見る。こちらは
**答えが合っているか**を見る。両方要るのは、数値的な近さと判断の一致が
別物だからで、arXiv 2606.19558 (Displacement Is Not Direction) の結論そのもの。
分布が近くても決定が変わることがある。

採用した課題と根拠 (arXiv 2601.14277, llama.cpp 量子化の統一評価):
  gsm8k      量子化で最も差が出た (Q3_K_S で 9.32 ポイント低下)。第一指標
  humaneval  コード生成。実際にテストを走らせて通るかで判定する
  mmlu       広い知識。選択肢の対数確率で採点するので生成不要、速い
  wikitext   perplexity。標準だがノイズが大きく、単独では判断材料にしない
  (HellaSwag は同論文で量子化にほぼ反応しなかったので入れない)

データセットはすべて MIT / Apache-2.0 / CC BY-SA で、結果の公開に支障はない。

使い方:
  uv run python bench/task_eval.py fetch
  uv run python bench/task_eval.py run --model ~/models/qwen38fn-mlx-v0-95 \
      --tag v0-95 --tasks gsm8k,humaneval,mmlu
  uv run python bench/task_eval.py report

注意: humaneval はモデルが書いたコードを実行する。別プロセス + タイムアウトで
走らせるが、ネットワーク遮断まではしていない。信頼できないモデルには使わないこと。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "bench" / "data"
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "task-eval"

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "gsm8k_train": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"),
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
}


# ---------------------------------------------------------------- データ取得


def cmd_fetch(args):
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, (repo, path) in SOURCES.items():
        dest = DATA_DIR / f"{name}.parquet"
        if dest.exists() and not args.force:
            print(f"あり: {dest.name}")
            continue
        src = hf_hub_download(repo_id=repo, filename=path, repo_type="dataset")
        dest.write_bytes(Path(src).read_bytes())
        print(f"取得: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def _table(name: str) -> list[dict]:
    import pyarrow.parquet as pq

    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(f"{p} が無い。先に `fetch` を実行すること")
    return pq.read_table(p).to_pylist()


def _sample(rows: list[dict], n: int | None, seed: int = 0) -> list[dict]:
    """決定的に間引く。変種間で同じ問題を使うため乱数は固定。"""

    if n is None or n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


# ---------------------------------------------------------------- 生成


class Runner:
    def __init__(self, model_ref: str, ngram: str | None = None):
        if ngram:
            # n-gram をディスクに置いた構成。arch は import 時に旗を読む
            import os

            os.environ["FASTMLX_NGRAM_DISK"] = "1"
        from mlx_lm import load

        self.model, self.tok = load(model_ref)
        if ngram:
            import sys

            sys.path.insert(0, str(REPO_ROOT))
            from mlxturbo.ngram_stream import install

            install(self.model, ngram)

    def chat_ids(self, text: str) -> list[int]:
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True
            )

    def greedy(self, text: str, max_tokens: int, stop: tuple[str, ...] = ()) -> str:
        import mlx.core as mx

        ids = self.chat_ids(text)
        cache = self.model.make_cache()
        logits = self.model(mx.array(ids)[None], cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
        out: list[int] = []
        eos = {self.tok.eos_token_id}
        while len(out) < max_tokens and cur not in eos:
            out.append(cur)
            if stop and len(out) % 8 == 0:
                txt = self.tok.decode(out)
                if any(s in txt for s in stop):
                    break
            logits = self.model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1], axis=-1))
        return self.tok.decode(out)

    def choice_logprobs(self, text: str, choices: tuple[str, ...]) -> list[float]:
        """選択肢それぞれの先頭トークンの対数確率。生成しないので速い。"""
        import mlx.core as mx

        ids = self.chat_ids(text)
        logits = self.model(mx.array(ids)[None])
        lp = logits[0, -1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp)
        mx.eval(lp)
        out = []
        for c in choices:
            tid = self.tok.encode(c, add_special_tokens=False)
            out.append(float(lp[tid[0]]))
        return out


class HttpRunner:
    """OpenAI 互換の口ごしに測る。Runner と同じ 3 つのメソッドを持つ。

    mlx-serve を相手にするために足した。うちのエンジンと同じ土俵で比べるには、
    同じサーバー越しに同じ課題を通す必要がある。ついでに、外の API (FP8 で
    出しているところ) も `--endpoint` を変えるだけで同じ課題にかけられる。

    in-process 版との違いが 1 つある。チャットテンプレートを当てるのが
    サーバー側になるので、思考出力を止める指示を JSON で送る
    (`enable_thinking: false`)。受け取らないサーバーだと思考が混ざり、
    gsm8k の答え抽出が末尾の数字を拾い損ねる。結果に効くので、
    **結果 JSON に endpoint と served_model を必ず残す。**
    """

    def __init__(self, endpoint: str, model: str, api_key: str | None = None,
                 timeout: float = 600.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, body: dict) -> dict:
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions", data=data, headers=self.headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise SystemExit(f"{self.endpoint} が応答しない: {e}")
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    def chat_ids(self, text: str):
        raise NotImplementedError("HTTP 越しにはトークン列を触れない")

    def greedy(self, text: str, max_tokens: int, stop: tuple[str, ...] = ()) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        if stop:
            body["stop"] = list(stop)
        r = self._post(body)
        return r["choices"][0]["message"].get("content") or ""

    def choice_logprobs(self, text: str, choices: tuple[str, ...]) -> list[float]:
        """先頭 1 トークンの上位 logprob から選択肢を拾う。

        OpenAI の口では分布そのものは取れないので上位 20 件で代用する。
        選択肢が 20 件に入らなければ、その選択肢は選ばれなかったものとして
        FLOOR を返す。A-D の 4 択なら、まず外れない。
        """
        FLOOR = -30.0
        r = self._post({
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "enable_thinking": False,
            "logprobs": True,
            "top_logprobs": 20,
        })
        content = (r["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            raise SystemExit(
                f"{self.endpoint} が logprobs を返さない。"
                "mmlu は --tasks から外すか、in-process で測ること")
        top = {t["token"]: t["logprob"] for t in content[0].get("top_logprobs", [])}
        out = []
        for c in choices:
            out.append(max(
                (lp for tok, lp in top.items() if tok.strip() == c.strip()),
                default=FLOOR))
        return out


# ---------------------------------------------------------------- 課題


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(s: str) -> str | None:
    m = _NUM.findall(s.replace(",", ""))
    if not m:
        return None
    return m[-1].rstrip(".")


def task_gsm8k(run: Runner, n: int | None) -> dict:
    rows = _sample(_table("gsm8k"), n)
    ok = 0
    for i, r in enumerate(rows):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        prompt = (
            r["question"]
            + "\n\n順を追って考えたうえで、最後の行に `答え: <数値>` の形式で"
            "answer だけを書いてください。"
        )
        text = run.greedy(prompt, max_tokens=512)
        m = re.search(r"答え\s*[:：]\s*(-?[\d,]+\.?\d*)", text)
        got = (m.group(1).replace(",", "") if m else _last_number(text)) or ""
        try:
            hit = abs(float(got) - float(gold)) < 1e-6
        except ValueError:
            hit = False
        ok += hit
        if (i + 1) % 25 == 0:
            print(f"  [gsm8k] {i + 1}/{len(rows)} 正解率={ok / (i + 1):.3f}")
    return {"n": len(rows), "accuracy": ok / len(rows)}


_HE_TIMEOUT = 12


def _run_humaneval_case(program: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "case.py"
        p.write_text(program)
        try:
            r = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True,
                timeout=_HE_TIMEOUT,
                cwd=d,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def _extract_code(text: str, fallback_prefix: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    if fence:
        return fence.group(1)
    return fallback_prefix + text


def task_humaneval(run: Runner, n: int | None) -> dict:
    rows = _sample(_table("humaneval"), n)
    ok = 0
    for i, r in enumerate(rows):
        prompt = (
            "次の関数を完成させてください。完成した関数全体を 1 つの Python コード"
            "ブロックで出力し、説明は付けないでください。\n\n```python\n"
            + r["prompt"]
            + "```"
        )
        text = run.greedy(prompt, max_tokens=640)
        code = _extract_code(text, r["prompt"])
        program = f"{code}\n\n{r['test']}\n\ncheck({r['entry_point']})\n"
        hit = _run_humaneval_case(program)
        ok += hit
        if (i + 1) % 20 == 0:
            print(f"  [humaneval] {i + 1}/{len(rows)} pass@1={ok / (i + 1):.3f}")
    return {"n": len(rows), "pass@1": ok / len(rows)}


_LETTERS = ("A", "B", "C", "D")


def task_mmlu(run: Runner, n: int | None) -> dict:
    rows = _sample(_table("mmlu"), n)
    ok = 0
    per_subject: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        opts = "\n".join(f"{L}. {c}" for L, c in zip(_LETTERS, r["choices"]))
        prompt = (
            f"{r['question']}\n\n{opts}\n\n"
            "正しい選択肢の記号 (A/B/C/D) だけを答えてください。"
        )
        lps = run.choice_logprobs(prompt, _LETTERS)
        pred = int(max(range(4), key=lambda k: lps[k]))
        hit = int(pred == int(r["answer"]))
        ok += hit
        per_subject.setdefault(r["subject"], []).append(hit)
        if (i + 1) % 100 == 0:
            print(f"  [mmlu] {i + 1}/{len(rows)} 正解率={ok / (i + 1):.3f}")
    subjects = {k: sum(v) / len(v) for k, v in sorted(per_subject.items())}
    return {"n": len(rows), "accuracy": ok / len(rows), "per_subject": subjects}


TASKS = {"gsm8k": task_gsm8k, "humaneval": task_humaneval, "mmlu": task_mmlu}
DEFAULT_N = {"gsm8k": 200, "humaneval": 164, "mmlu": 800}


# ---------------------------------------------------------------- 実行


def cmd_run(args):
    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in names:
        if t not in TASKS:
            raise SystemExit(f"未知の課題 {t}。選べるのは {', '.join(TASKS)}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.endpoint:
        if not args.served_model:
            raise SystemExit("--endpoint には --served-model も要る")
        print(f"接続先: {args.endpoint} ({args.served_model})")
        run = HttpRunner(args.endpoint, args.served_model,
                         os.environ.get("TASK_EVAL_API_KEY"))
    else:
        print(f"モデル読み込み: {args.model}")
        t0 = time.time()
        run = Runner(args.model, args.ngram)
        print(f"  {time.time() - t0:.0f} 秒")

    result = {
        "tag": args.tag,
        "ngram_sidecar": args.ngram,
        "model": args.model,
        "endpoint": args.endpoint,
        "served_model": args.served_model,
        "at": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }
    for t in names:
        n = args.limit if args.limit else DEFAULT_N[t]
        print(f"[{t}] n={n}")
        t0 = time.time()
        result["tasks"][t] = TASKS[t](run, n)
        result["tasks"][t]["seconds"] = round(time.time() - t0, 1)
        print(f"  -> {json.dumps({k: v for k, v in result['tasks'][t].items() if k != 'per_subject'}, ensure_ascii=False)}")

    out = RESULTS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def cmd_report(args):
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        t = d["tasks"]
        rows.append(
            (
                d["tag"],
                t.get("gsm8k", {}).get("accuracy"),
                t.get("humaneval", {}).get("pass@1"),
                t.get("mmlu", {}).get("accuracy"),
            )
        )
    print(f"{'tag':14s} {'GSM8K':>8s} {'HumanEval':>10s} {'MMLU':>8s}")
    for tag, g, h, m in rows:
        f = lambda v: f"{v:8.3f}" if isinstance(v, float) else " " * 8  # noqa: E731
        print(f"{tag:14s} {f(g)} {f(h)[:10]:>10s} {f(m)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("run")
    p.add_argument("--model", default="", help="in-process で測るときのパス")
    p.add_argument("--endpoint", default=None,
                   help="OpenAI 互換の口 (例 http://127.0.0.1:11234)。"
                        "指定すると --model ではなくこちら越しに測る")
    p.add_argument("--served-model", default=None,
                   help="--endpoint 側でのモデル名")
    p.add_argument("--tag", required=True)
    p.add_argument("--ngram", default=None, help="n-gram サイドカー")
    p.add_argument("--tasks", default="gsm8k,humaneval,mmlu")
    p.add_argument("--limit", type=int, default=0, help="0 なら課題ごとの既定値")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
