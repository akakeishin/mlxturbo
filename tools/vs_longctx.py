"""文脈長を振って decode 速度を測る。エンジン間で同条件。

  python vs_longctx.py <ラベル> <base_url> <model>

各長さで 3 回投げ、1 回目 (cold prefill) を捨てて 2 回の中央値。
prefix cache が効く前提なので、測っているのは純粋なデコード速度。
"""
import json, statistics, sys, time, urllib.request
from pathlib import Path

TOKENS = 128
LENGTHS = [1000, 4000, 16000, 48000]
OUT = Path(__file__).with_name("vs_longctx.json")
FILLER = ("分散システムでは、ノード間の通信が失敗しうるという前提のもとで設計を行う。"
          "リーダー選出、ログ複製、スナップショット、メンバーシップ変更のそれぞれに、"
          "独立した失敗モードがある。")


def ask(url, model, text):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": text}],
                       "max_tokens": TOKENS, "temperature": 0.0,
                       "stream": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body,
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
    return (ttft or total), ((n - 1) / dec if dec > 0 and n > 1 else 0.0), n


def main():
    label, url, model = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"=== {label} ===", flush=True)
    res = {}
    for L in LENGTHS:
        body = FILLER
        while len(body) < L * 1.6:
            body += FILLER
        text = body[:int(L * 1.6)] + "\n\n上の文章について、要点を詳しく説明してください。"
        runs = []
        for rep in range(3):
            try:
                t, d, n = ask(url, model, text)
            except Exception as e:                       # noqa: BLE001
                print(f"  {L:6d} 失敗 {type(e).__name__}: {e}", flush=True)
                break
            print(f"  {L:6d} rep{rep} TTFT={t:7.2f}s decode={d:6.2f} tok/s n={n}",
                  flush=True)
            if rep:
                runs.append((t, d))
        if runs:
            res[L] = dict(ttft=statistics.median(r[0] for r in runs),
                          decode=statistics.median(r[1] for r in runs))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out[label] = res
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\n=== 中央値 (2-3 回目) ===")
    for L, v in res.items():
        print(f"  {L:6d} TTFT={v['ttft']:7.2f}s decode={v['decode']:6.2f} tok/s")
    print("DONE")


if __name__ == "__main__":
    main()
