"""OpenAI 互換の口を持つエンジンを、同じ課題・同じ手順で測る。

片方ずつ起動して測り、同じ JSON に足していく。ストリームで TTFT と
デコード速度を分けて取る (壁時計から。サーバーの自己申告は使わない)。

  python vs_engine.py <ラベル> <base_url> <model>

揃えるもの: プロンプト、max_tokens、temperature=0、反復数、停止条件
揃わないもの: 重み (それぞれの推奨構成)。報告に明記する。
"""
import json, statistics, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, "/Users/ht/dev/fastmlx")
from bench.eval_prompts import PROMPTS  # noqa: E402

TOKENS = 128
REPS = 3            # 暖機 1 + 計測 2
OUT = Path(__file__).with_name("vs_engine.json")

FILLER = (
    "分散システムでは、ノード間の通信が失敗しうるという前提のもとで設計を行う。"
    "リーダー選出、ログ複製、スナップショット、メンバーシップ変更のそれぞれに、"
    "独立した失敗モードがある。"
)


def ask(url, model, text, tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": tokens, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", body,
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n = None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    total = time.perf_counter() - t0
    dec = total - (ttft or total)
    return dict(ttft=ttft or total, n=n,
                decode_tok_s=(n - 1) / dec if dec > 0 and n > 1 else 0.0)


def main():
    label, url, model = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"=== {label} ({model}) ===", flush=True)

    tasks = [(k, p.text, TOKENS) for k, p in PROMPTS.items()]
    body = FILLER
    while len(body) < 24000:
        body += FILLER
    tasks.append(("LONG_prefill", body + "\n\n上を3行で要約してください。", 32))

    res = {}
    for key, text, tokens in tasks:
        runs = []
        for rep in range(REPS):
            try:
                r = ask(url, model, text, tokens)
            except Exception as e:                      # noqa: BLE001
                print(f"  {key:22s} 失敗 {type(e).__name__}: {e}", flush=True)
                break
            if rep:                                     # 1 回目は暖機で捨てる
                runs.append(r)
        if not runs:
            continue
        res[key] = dict(
            ttft=statistics.median(r["ttft"] for r in runs),
            decode_tok_s=statistics.median(r["decode_tok_s"] for r in runs),
            n=runs[0]["n"])
        print(f"  {key:22s} TTFT={res[key]['ttft']:6.2f}s "
              f"decode={res[key]['decode_tok_s']:6.2f} tok/s n={res[key]['n']}",
              flush=True)

    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text())
    out[label] = res
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    short = [v["decode_tok_s"] for k, v in res.items() if k != "LONG_prefill"]
    if short:
        print(f"\ndecode 中央値 {statistics.median(short):.2f} tok/s "
              f"({min(short):.2f}〜{max(short):.2f}, {len(short)} 課題)")
    if "LONG_prefill" in res:
        print(f"長文 TTFT {res['LONG_prefill']['ttft']:.2f}s")
    print("DONE")


if __name__ == "__main__":
    main()
