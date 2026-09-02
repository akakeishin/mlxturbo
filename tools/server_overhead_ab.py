"""HTTP 経路 (uvicorn + executor + ThinkingRouter/SegmentAssembler + queue +
SSE) の費用と、エンジン直叩きの decode 速度差を、同一プロセス内で
mlxturbo.server を起動して切り分ける道具。

作法: プロンプトごとに A(HTTP)→B(直叩き)→B→A を `--rounds` 回、直列で測る
(直叩きと HTTP を同時に走らせない)。プロセス最初の 1 本 (SHORT を HTTP で
8 トークン) は温めとして捨てる。判定は複数プロンプトの平均でだけ行うこと —
単一プロンプトは chunk 境界の丸めで挙動が変わる (CLAUDE.md の計測の作法)。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("bench", "tools"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vs_mlx_serve import SHORT, model_id, stream_once, wait_ready  # noqa: E402
from decode_ab import SHORT_PROMPTS  # noqa: E402


def _start_server_inprocess(model: str, ngram: str | None, port: int):
    """``mlxturbo.server.main()`` を daemon スレッドで起動し、STATE の載った
    モジュールを返す (``/v1/models`` が 200 を返すまで待つ)。server.py 自体は
    一切変更しない。

    確認済みの注意点 (uvicorn 0.42.0, .venv 実測):

    - シグナル: uvicorn の ``Server.capture_signals()`` は
      ``threading.current_thread() is not threading.main_thread()`` のとき
      ``signal.signal`` を一切呼ばずに抜ける。daemon スレッドで
      ``server_obj.run()`` を呼んでも "signal only works in main thread" で
      は落ちない — server.py の ``_install_graceful_shutdown`` が上書きする
      ``handle_exit`` もその分岐の中でしか使われないので実害なし。
    - executor: ``main()`` は起動時に ``STATE.executor``
      (``ThreadPoolExecutor(max_workers=1)``) を作り、モデルロードも以降の
      generate もそのスレッド 1 本に固定する (重み/KV キャッシュがロードした
      スレッドに紐づくため)。直叩き側 (B) はこのツールのスレッドから
      ``STATE.runner.generate()`` を直に呼ばない — server.py 自身のコメントが
      明言する通り "There is no Stream(gpu, N) in current thread" で落ちるので、
      ``STATE.executor.submit(...).result()`` で同じスレッドに投げる。
    - STATE の初期化順: ``main()`` は ``_load()`` を
      ``executor.submit().result()`` で同期的に待ってから ``STATE`` を構築し、
      その後で uvicorn を起動する。つまり ``/v1/models`` が 200 を返す時点で
      STATE は必ず組み上がっている (``wait_ready()`` の完了を "STATE 使用可"
      の合図にしてよい)。
    """

    import mlxturbo.server as server_mod

    argv = [
        "mlxturbo.server", "--model", os.path.expanduser(model),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if ngram:
        argv += ["--ngram", os.path.expanduser(ngram)]
    sys.argv = argv

    boot_error: dict = {}

    def _run():
        try:
            server_mod.main()
        except BaseException as exc:  # noqa: BLE001  起動失敗を wait_ready 側に伝える
            boot_error["exc"] = exc

    thread = threading.Thread(target=_run, name="mlxturbo-server-inproc", daemon=True)
    thread.start()
    if not wait_ready(port, timeout=900.0):
        if "exc" in boot_error:
            raise RuntimeError(f"サーバー起動に失敗: {boot_error['exc']!r}") from boot_error["exc"]
        raise RuntimeError("サーバーが時間内に起動しなかった (/v1/models が 200 を返さない)")
    return server_mod


def _direct_generate(server_mod, prompt: str, n_tokens: int) -> dict:
    """B (直叩き)。``_apply_template`` で HTTP 側と同じ ids を作り、
    ``STATE.executor`` 上で ``STATE.runner.generate()`` を呼ぶ。

    ``client_tps`` は executor への submit/result 往復を含む壁時計、
    ``engine_tps``/``tok_per_step`` は generate() 自身が返す res の値
    (``decode_tps``/``tokens_per_step``) — HTTP 側 (A) の client/engine 分離と
    対になる形にしてある。
    """

    STATE = server_mod.STATE
    ids = server_mod._apply_template([{"role": "user", "content": prompt}], False)

    def _call():
        t0 = time.perf_counter()
        res = STATE.runner.generate(
            ids, max_tokens=n_tokens, temp=0.0, eos_ids=STATE.eos_ids,
            on_tokens=None, session=None,
        )
        return res, time.perf_counter() - t0

    res, wall = STATE.executor.submit(_call).result()
    tokens = res.get("tokens") or []
    n = len(tokens)
    decode_wall = wall - res.get("ttft_s", 0.0)
    client_tps = (n - 1) / decode_wall if decode_wall > 0 and n > 1 else 0.0
    text = STATE.tokenizer.decode(tokens)
    return dict(
        path="B", client_tps=client_tps, engine_tps=res.get("decode_tps", 0.0),
        tok_per_step=res.get("tokens_per_step", 0.0), ttft=res.get("ttft_s", 0.0),
        n_tokens=n, text=text,
    )


def _http_generate(
    port: int, model_name: str, prompt: str, n_tokens: int, captured: dict,
) -> dict:
    """A (HTTP)。``bench/vs_mlx_serve.py`` の ``stream_once`` と同じ数え方
    (最初のチャンクまでを TTFT、以降 (n-1)/経過秒) で client 側 tok/s を出す。
    engine 側の値は ``captured`` (``_log_gen_stats`` の monkeypatch) から拾う —
    呼び出し前に空であることを呼び出し元が保証する。
    """

    ttft, dec_s, n, text = stream_once(
        port, [{"role": "user", "content": prompt}], n_tokens, model_name,
        extra_body={"reasoning_effort": "none"},
    )
    client_tps = (n - 1) / dec_s if dec_s > 0 and n > 1 else 0.0
    res = captured.get("res") or {}
    return dict(
        path="A", client_tps=client_tps, engine_tps=res.get("decode_tps", 0.0),
        tok_per_step=res.get("tokens_per_step", 0.0), ttft=ttft,
        n_tokens=n, text=text,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", default="bench/results/server-overhead-ab.json")
    args = ap.parse_args()

    prompts = [SHORT] + list(SHORT_PROMPTS)

    server_mod = _start_server_inprocess(args.model, args.ngram, args.port)

    captured: dict = {}
    orig_log_gen_stats = server_mod._log_gen_stats

    def _capture(res):
        captured["res"] = res
        orig_log_gen_stats(res)

    server_mod._log_gen_stats = _capture

    model_name = model_id(args.port)

    # 温め: プロセス最初の 1 本は捨てる (測定プロンプトとは別物の SHORT、8 トークン)。
    stream_once(
        args.port, [{"role": "user", "content": SHORT}], 8, model_name,
        extra_body={"reasoning_effort": "none"},
    )
    captured.clear()

    rows: list[dict] = []
    for prompt in prompts:
        for _round in range(args.rounds):
            for path in ("A", "B", "B", "A"):
                if path == "A":
                    captured.clear()
                    row = _http_generate(args.port, model_name, prompt, args.tokens, captured)
                else:
                    row = _direct_generate(server_mod, prompt, args.tokens)
                row["prompt"] = prompt
                rows.append(row)
                print(
                    f"  [{row['path']}] {prompt[:24]!r:<28} "
                    f"client={row['client_tps']:6.2f}tok/s "
                    f"engine={row['engine_tps']:6.2f}tok/s "
                    f"tok/step={row['tok_per_step']:5.2f} "
                    f"n={row['n_tokens']:4d} text={row['text'][:60]!r}"
                )

    print()
    for path in ("A", "B"):
        sub = [r for r in rows if r["path"] == path]
        if not sub:
            continue
        print(
            f"[{path}] 平均 ({len(sub)} 本): "
            f"client={statistics.mean(r['client_tps'] for r in sub):.2f}tok/s "
            f"engine={statistics.mean(r['engine_tps'] for r in sub):.2f}tok/s "
            f"tok/step={statistics.mean(r['tok_per_step'] for r in sub):.2f}"
        )

    print("\nA/B のテキスト一致 (同じ ids・貪欲なら同一のはず):")
    for prompt in prompts:
        a_texts = {r["text"] for r in rows if r["prompt"] == prompt and r["path"] == "A"}
        b_texts = {r["text"] for r in rows if r["prompt"] == prompt and r["path"] == "B"}
        if a_texts == b_texts and len(a_texts) == 1:
            print(f"  一致: {prompt[:24]!r}")
        else:
            print(f"  *** 不一致 (計測無効) *** {prompt[:24]!r}: A={a_texts} B={b_texts}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                dict(rows=rows, tokens=args.tokens, rounds=args.rounds, port=args.port),
                f, ensure_ascii=False, indent=2,
            )
        print(f"\n書き出し: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
