"""出荷経路のデコード速度を 1 つの数字に合成する。

公表値は「融合 HC カーネル + ビット配分 (v-fast6) + n-gram の並列 pread」を
**同時に有効にした** ときの逐次デコード速度でなければならない。3 つは個別に
測ってあるが (docs/STATUS.md)、合成した数字が無かった。

経路を外すと数字が出荷経路を表さなくなる:

- ``mlxturbo.runner.build_runner`` を通す。Flash-Next (qwen4_exp) は SpecEngine
  の契約を満たさず ``FallbackRunner`` (非投機) に落ちるが、**融合 HC カーネルを
  有効化しているのは build_runner** で、SpecEngine 経路には効かない。つまり
  Flash-Next が実際に通る唯一の道が build_runner なので、モデルを直接叩く
  ループ (tools/bench_fused.py) は融合の有無こそ比べられても出荷経路ではない。
- HTTP は挟まない。既存の 32.67ms 系列と比較できなくなる。
- moe_route (実測 +0.34ms の純損) と rms_norm_gated (空振り) は有効にしない。
  build_runner の既定がそうなっているので、ここでは何もしない。

pread と mmap の差は **同じプロセス・同じロード済みモデルの中で交互に**取る。
別プロセス・別時刻の比較は時間帯で 20-30% 動く (docs/KERNEL-HANDOFF-HC.md)。

**同じプロンプトを測り直してはいけない。**greedy なら同じトークン列が出るので、
毎回まったく同じ n-gram 行 (64 トークンで 1024 行 = 100KB) しか触らない。2 回目
以降はページキャッシュに全部載っていて、ディスクを一度も引かない測定になる。
n-gram は 32GB の表から毎トークン 16 行をランダムに引く部品なので、これを
温めてしまうと pread の効果 (フォールトの直列化を消す) が丸ごと消える。
なので 1 回の計測につき未使用のプロンプトを 1 本ずつ使い、行を毎回冷たいまま
にする。比較のため温まった条件も並べて測る。

    tools/biglock.sh uv run python tools/bench_shipping.py \
        --model ~/models/qwen38fn-mlx-v-fast6 \
        --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 温まった行を測らないための使い捨てプロンプト。1 計測 = 1 本、使い回さない。
# 長さを揃えてあるのは、プロンプト長の差が attention のコストに乗って
# ms/token をずらすため
PROMPTS = [
    "分散システムにおける合意アルゴリズムについて、Paxos と Raft の違いを含めて詳しく説明してください。",
    "データベースのトランザクション分離レベルについて、実際の異常現象を挙げて詳しく説明してください。",
    "ガベージコレクションの世代別仮説について、その根拠と例外を挙げて詳しく説明してください。",
    "TCP の輻輳制御について、Reno と BBR の設計思想の違いを含めて詳しく説明してください。",
    "分散ハッシュテーブルの設計について、Chord と Kademlia を比較しながら詳しく説明してください。",
    "CPU のキャッシュコヒーレンシについて、MESI プロトコルの動作を追いながら詳しく説明してください。",
    "型システムにおける部分型付けについて、変性の規則を具体例とともに詳しく説明してください。",
    "暗号学的ハッシュ関数の設計について、Merkle-Damgard 構成の弱点を含めて詳しく説明してください。",
    "オペレーティングシステムの仮想記憶について、ページ置換方式を比較しながら詳しく説明してください。",
    "コンパイラの最適化について、レジスタ割り付けをグラフ彩色として解く方法を詳しく説明してください。",
    "機械学習における正則化について、L1 と L2 が解に与える違いを幾何的に詳しく説明してください。",
    "並行プログラミングのメモリモデルについて、acquire と release の意味を詳しく説明してください。",
    "ファイルシステムのジャーナリングについて、メタデータのみ記録する場合の危険を詳しく説明してください。",
    "情報理論における符号化定理について、雑音のある通信路の容量と絡めて詳しく説明してください。",
    "グラフアルゴリズムの最短経路について、Dijkstra が負辺で破綻する理由を詳しく説明してください。",
    "分散ストレージの整合性モデルについて、線形化可能性と逐次一貫性の差を詳しく説明してください。",
    "浮動小数点演算の誤差について、桁落ちが起きる条件と回避策を具体例で詳しく説明してください。",
    "データ構造の償却計算量について、動的配列とスプレー木を例に取って詳しく説明してください。",
    "ネットワークの経路制御について、BGP の経路選択規則と不安定性を詳しく説明してください。",
    "プロセススケジューリングについて、CFS が仮想実行時間を使う理由を詳しく説明してください。",
    "公開鍵基盤の信頼モデルについて、証明書の失効が難しい理由を含めて詳しく説明してください。",
    "データベースの索引構造について、B+ 木と LSM 木の書き込み特性を比較して詳しく説明してください。",
    "形式検証におけるモデル検査について、状態爆発への対処法を挙げながら詳しく説明してください。",
    "画像認識の畳み込み演算について、受容野の広がり方を層ごとに追って詳しく説明してください。",
    "分散トレーシングの設計について、サンプリング方式が観測性に与える影響を詳しく説明してください。",
    "メモリ確保器の設計について、断片化を抑えるサイズ分類の考え方を詳しく説明してください。",
    "自然言語処理の注意機構について、位置符号化の選択が外挿に効く理由を詳しく説明してください。",
    "リアルタイム系のスケジューリングについて、優先度逆転とその継承を詳しく説明してください。",
    "確率的データ構造について、Bloom filter と Count-Min sketch の誤り方を詳しく説明してください。",
    "分散システムの障害検知について、故障検知器の完全性と正確性の両立を詳しく説明してください。",
    "量子計算の誤り訂正について、表面符号がしきい値を持つ理由を含めて詳しく説明してください。",
    "ソフトウェアの依存関係解決について、版の選択を充足可能性問題として解く方法を詳しく説明してください。",
    "ストリーム処理の時刻管理について、ウォーターマークが遅延データを扱う仕組みを詳しく説明してください。",
    "ハードウェアの分岐予測について、TAGE 予測器が履歴長を混ぜる理由を詳しく説明してください。",
    "分散合意の実装について、Raft のログ整合性検査が果たす役割を詳しく説明してください。",
    "データ圧縮の原理について、算術符号が Huffman を上回る条件を詳しく説明してください。",
]


def summarize(ms: list[float]) -> dict:
    """中央値と IQR。平均ではなく中央値で見る (外れ値が 1 本で動くため)。"""

    med = statistics.median(ms)
    if len(ms) >= 4:
        q1, _, q3 = statistics.quantiles(ms, n=4)
    else:
        q1, q3 = min(ms), max(ms)
    return {
        "n": len(ms),
        "median_ms": med,
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "median_tps": 1000.0 / med,
        "samples_ms": ms,
    }


def swap_backend(model, stream) -> None:
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        ple.ple_embedding.ngram_embedding = stream


def installed_stream(model):
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is not None:
            return ple.ple_embedding.ngram_embedding
    raise RuntimeError("PLE 層が無い")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", required=True)
    ap.add_argument("--reps", type=int, default=9, help="合成値を取る回数")
    ap.add_argument("--tokens", type=int, default=64, help="1 回あたりの生成トークン数")
    ap.add_argument("--ab-blocks", type=int, default=7, help="pread/mmap 交互の往復数")
    ap.add_argument("--ab-tokens", type=int, default=32)
    ap.add_argument("--out", default=str(REPO_ROOT / "bench/results/composed-shipping.json"))
    args = ap.parse_args()

    # qwen4_exp の NGRAM_ON_DISK は import 時に評価される。読み込みより前に立てる
    os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx

    from mlxturbo._mlx_compat import mlx_lm_load
    from mlxturbo.ngram_stream import StreamNGram, warn_if_not_installed
    from mlxturbo.runner import FallbackRunner, build_runner

    t0 = time.perf_counter()
    model, tokenizer, config = mlx_lm_load(args.model, return_config=True)
    from mlxturbo.ngram_stream import install

    install(model, args.ngram)
    load_s = time.perf_counter() - t0
    print(f"[bench] loaded in {load_s:.1f}s: {args.model}", flush=True)

    # cli.py:94-101 と同じ組み立て。--no-mtp は Flash-Next では MTP 重みが
    # そもそも無いためで、経路の選択 (SpecEngine か否か) には影響しない
    runner_args = Namespace(
        model=args.model,
        original=args.model,
        mtp_bits=4,
        no_mtp=True,
        no_fused=False,
    )
    runner = build_runner(model, tokenizer, config, runner_args)
    route = type(runner).__name__
    speculative = not isinstance(runner, FallbackRunner)
    print(f"[bench] runner={route}  投機={'あり' if speculative else 'なし'}", flush=True)
    if speculative:
        raise SystemExit(
            "この計測は非投機経路の数字を出すためのもの。SpecRunner に落ちたので中止する"
        )
    if not warn_if_not_installed(model):
        raise SystemExit("n-gram がサイドカーに差し替わっていない。中止する")

    stream = installed_stream(model)
    eos_ids = set(tokenizer.eos_token_ids)
    encoded = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )
        for p in PROMPTS
    ]
    used = 0  # 使い捨てプロンプトのカーソル。使い回さない

    def decode(prompt_ids, n_tokens: int) -> tuple[float, int]:
        res = runner.generate(
            prompt_ids,
            max_tokens=n_tokens,
            temp=0.0,
            eos_ids=eos_ids,
            on_tokens=None,
            session=None,
        )
        tps = res["decode_tps"]
        return (1000.0 / tps if tps > 0 else float("nan")), len(res["tokens"])

    def next_prompt():
        nonlocal used
        if used >= len(encoded):
            raise SystemExit(
                f"未使用のプロンプトが尽きた ({len(encoded)} 本)。使い回すと "
                "n-gram の行が温まって計測が壊れるので、PROMPTS を足すこと"
            )
        used += 1
        return encoded[used - 1]

    print("[bench] 温め (モデル本体。n-gram の行は使い捨て側で冷たいまま)", flush=True)
    decode(encoded[-1], 16)  # 温めは末尾を使い、計測用のプロンプトを消費しない

    # --------------------------------------------------- 合成値 (冷たい行)
    print("\n=== 合成 (融合 HC + v-fast6 + pread、build_runner 経路、行は冷たい) ===",
          flush=True)
    composed: list[float] = []
    token_counts: list[int] = []
    for r in range(args.reps):
        ms, ntok = decode(next_prompt(), args.tokens)
        composed.append(ms)
        token_counts.append(ntok)
        print(f"  rep {r + 1:2d}  {ms:6.2f} ms/token ({1000 / ms:5.2f} tok/s)  {ntok} tok",
              flush=True)
    comp = summarize(composed)
    print(f"\n  中央値 {comp['median_ms']:.2f} ms/token = {comp['median_tps']:.2f} tok/s"
          f"  IQR {comp['q1_ms']:.2f}-{comp['q3_ms']:.2f}", flush=True)

    # --------------------------------------------------- 合成値 (温まった行)
    # 同じプロンプトを測り直した場合。触る行が毎回同じでページキャッシュに載る
    # ので、n-gram の読み出しが実質ゼロになる。公表値には使えないが、なぜ
    # 使えないかを数字で残しておく
    print("\n=== 参考: 同じプロンプトを測り直した場合 (行が温まる) ===", flush=True)
    warm_ids = next_prompt()
    decode(warm_ids, args.tokens)  # 1 本目で行を温める
    warm: list[float] = []
    for r in range(args.reps):
        ms, _ = decode(warm_ids, args.tokens)
        warm.append(ms)
    warm_sum = summarize(warm)
    print(f"  中央値 {warm_sum['median_ms']:.2f} ms/token = "
          f"{warm_sum['median_tps']:.2f} tok/s  IQR "
          f"{warm_sum['q1_ms']:.2f}-{warm_sum['q3_ms']:.2f}", flush=True)

    # --------------------------------------------------- pread vs mmap (同一プロセス内)
    print("\n=== pread / mmap 交互 (同じロード済みモデル内、1 計測 1 プロンプト) ===",
          flush=True)
    mmap_stream = StreamNGram(Path(args.ngram), backend="mmap")
    streams = {"pread": stream, "mmap": mmap_stream}
    ab: dict[str, list[float]] = {"pread": [], "mmap": []}
    order = ["pread", "mmap"]
    for b in range(args.ab_blocks):
        # ブロックごとに先後を入れ替える。ドリフトと順序効果を両条件に均等に配る。
        # プロンプトは条件ごとに使い捨てにする: 同じものを両 backend に回すと、
        # 後から走った方が前の走行で温まった行を引くことになって差が消える
        seq = order if b % 2 == 0 else list(reversed(order))
        for name in seq:
            swap_backend(model, streams[name])
            ms, _ = decode(next_prompt(), args.ab_tokens)
            ab[name].append(ms)
            print(f"  block {b:2d}  {name:6s} {ms:6.2f} ms/token", flush=True)
    swap_backend(model, stream)

    ab_sum = {k: summarize(v) for k, v in ab.items()}
    delta = ab_sum["pread"]["median_ms"] - ab_sum["mmap"]["median_ms"]
    for name in ("pread", "mmap"):
        s = ab_sum[name]
        print(f"  {name:6s} 中央値 {s['median_ms']:6.2f} ms/token"
              f"  IQR {s['q1_ms']:.2f}-{s['q3_ms']:.2f}  (n={s['n']})", flush=True)
    print(f"  pread - mmap = {delta:+.2f} ms/token"
          f"  ({ab_sum['mmap']['median_ms'] / ab_sum['pread']['median_ms']:.3f}x)", flush=True)

    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        rev = None

    payload = {
        "what": "v-fast6 + 融合 HC カーネル + n-gram 並列 pread を合成した"
                "逐次デコード速度 (非投機)",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_rev": rev,
        "speculative": False,
        "speculative_note": "投機は一切入っていない。README の「自己投機 1.5-2.2 倍」は "
                            "Qwen3.8-27B レーンの数字で、この値には適用できない",
        "conditions": {
            "route": f"mlxturbo.runner.build_runner -> {route} (非投機、HTTP を挟まない)",
            "model_path": str(Path(args.model).expanduser()),
            "ngram_sidecar": str(Path(args.ngram).expanduser()),
            "fused_hyper_connection": True,
            "fused_moe_route": False,
            "fused_rms_norm_gated": False,
            "ngram_backend": stream.backend,
            "ngram_threads": getattr(stream, "n_threads", None),
            "ngram_bits": stream.bits,
            "env": {
                "FASTMLX_NGRAM_DISK": os.environ.get("FASTMLX_NGRAM_DISK"),
                "FASTMLX_NGRAM_BACKEND": os.environ.get("FASTMLX_NGRAM_BACKEND"),
                "FASTMLX_NGRAM_THREADS": os.environ.get("FASTMLX_NGRAM_THREADS"),
            },
            "temp": 0.0,
            "max_tokens_per_rep": args.tokens,
            "tokens_generated_per_rep": token_counts,
            "prompt_policy": "1 計測につき未使用のプロンプトを 1 本。使い回すと "
                             "n-gram の行がページキャッシュに載って計測が壊れる",
            "prompts_used": used,
            "prompts_available": len(PROMPTS),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "load_s": load_s,
        },
        "composed": comp,
        "composed_warm_rows": {
            "note": "同じプロンプトを測り直した場合。触る n-gram 行が毎回同じで "
                    "ページキャッシュに載るため、表の読み出しが実質ゼロになる。"
                    "公表値には使えない",
            **warm_sum,
        },
        "ngram_backend_ab": {
            "note": "同じプロセス・同じロード済みモデル内でブロックごとに交互に振り、"
                    "ブロックの先後も入れ替えた。プロンプトは 1 計測 1 本の使い捨て。"
                    "別プロセス比較は使えない",
            "tokens_per_block": args.ab_tokens,
            "pread": ab_sum["pread"],
            "mmap": ab_sum["mmap"],
            "pread_minus_mmap_ms": delta,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[bench] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
