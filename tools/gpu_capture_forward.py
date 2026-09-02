"""S=1 decode forward 1 回を Metal GPU capture (``.gputrace``) に落とす。

## 2026-09-03 追記: この手は使えない (実測、68GB)

実モデル (~/models/ddalcu-mlxlm、90GB 級 + ngram 32GB 常駐) で ``--ctx 0
--width 1`` を実行したところ、35 分経っても `capture 終了` まで到達せず、
`bench/results/gputrace/` が **68GB** に膨らんだところで親セッションが GPU の
待ち行列を塞いでいると判断して強制終了・削除した。**forward 1 回でもこの
サイズになる時点で実用にならない。** 原因と代替案の要点:

- `MTLCaptureManager` のデバイススコープ capture は、キャプチャ区間で実行した
  コマンドが触れたリソースだけでなく、**その時点でデバイスに常駐している
  リソース全部** (使った/使っていない両方) をバンドルへ書き出す。小さい合成
  op (512x512 行列積 1 回) で試した capture にも `device-resources-*` /
  `unused-device-resources-*` という名前のファイルが両方あった (実測、後者は
  対象が少ない合成テストではほぼ空。本番ではここが 90GB 級の重み全部を巻き
  込む側)。Xcode の GPU フレームデバッガが「このフレームで使わなかったバッファ
  も含めて全リソースを閲覧できる」ようにするための仕様と考えられる。
- `mx.metal.start_capture(path: str) -> None` はパス 1 個しか取らない
  (`.venv/lib/python3.13/site-packages/mlx/core/metal.pyi`)。バッファを
  絞る・除外するオプションは無い。MLX 側にもそれを制御する引数は無い。
- 対象を絞る一般的な Metal API は `MTLCaptureDescriptor.captureObject` (device
  丸ごとではなく特定の `MTLCommandQueue` や自前の `MTLCaptureScope` に絞る) だが、
  これは「どのコマンドを録るか」を絞るだけで、「常駐リソースを丸ごと書き出す」
  という一覧化の仕様そのものは変わらない。加えて MLX の Python API はこの
  descriptor も command queue/encoder も露出していない (C++ 内部専用) ので、
  そもそも Python 側からは選べない。
- 次善の代替 (未検証、GPU 不使用の範囲で判断): 合成モデル (小さい層数・小さい
  hidden dim、`tools/vendor_fingerprint.py` と同じ発想) で同じ capture をやれば、
  常駐リソースが小さいのでバンドルも小さく収まり、**カーネル名・dispatch 回数
  の構造**は見えるはず (絶対時間は形状が違うので使えない)。試すならこちら。
- Metal のカウンタ API (`MTLCounterSampleBuffer` をエンコーダに付けて
  dispatch ごとの GPU タイムスタンプを取る、軽量でバッファ内容は記録しない)
  が本来の筋だが、MLX の Python API は command buffer/encoder を渡さないので
  今のところ届かない (MLX の C++ 側に手を入れないと使えない)。
- Instruments の Metal System Trace (xctrace) は既に別の理由で失敗済み
  (`docs/research/SESSION-2026-09-02-CATCHUP.md` の「xctrace は今回は使えなかった」節)。

このファイル自体は記録として残すが、**実モデルに対しては呼ばないこと。**
合成モデル向けに書き直すか、別の道具 (カーネル名だけならソース読解、時間なら
壁時計ベースの ablate/A-B) に切り替えるのが現実的。

## 背景 (捨てた案の元の狙い、参考)

xctrace (Metal System Trace) は MLX (python) の GPU 区間をほぼ拾えなかった
(`docs/research/SESSION-2026-09-02-CATCHUP.md` の「xctrace は今回は使えなかった」節、
`tools/gpu_trace_kernels.py`)。相談役の提案 (レーン 11 の 9): xctrace の代わりに、
MLX が自前で呼べる Metal のフレームキャプチャ API
(`mx.metal.start_capture(path)` / `mx.metal.stop_capture()`、内部は
`MTLCaptureManager`) で直接 `.gputrace` を取る。外部プロセスの attach/launch
に頼らないので、xctrace で問題になった「python (MLX) の Compute 区間がほぼ
出ない」という取りこぼしが原理的に起きない。

## `.gputrace` の中身についてわかったこと (実測、2026-09-03)

`.gputrace` は Xcode の GPU デバッガ用のバンドル (ディレクトリ)。中身を見ると:

- `metadata` だけが読める Apple binary plist (`plutil -p` で見える) だが、
  中身はキャプチャセッションのメタ情報 (uuid・API 種別など) だけで、
  カーネル名や時間は無い。
- `capture` / `unsorted-capture` / `store0` / `index` / `device-resources-*`
  はどれも `file` コマンドで "data" としか判定できない非公開バイナリ形式
  (Xcode の GPUTools*.framework 群が内部で使う形式で、公開ドキュメントは無い)。

`xcrun xctrace export --input <bundle> --toc` を実バンドルに向けると
**即座に `Export failed: Document Missing Template Error` (終了コード 10)**
で失敗することを確認済み -- `.gputrace` は xctrace が扱う `.trace` 形式とは
別物で、xctrace では読めない。Xcode の Toolchain 内 (`xcrun --find` で探せる
範囲) にも `metal-capture-*` のような専用 CLI は無い (`metal` (コンパイラ) と
`metal-package-builder` だけ)。**カーネル名別の時間を CLI で読む公式の方法は
現状ここには無い。** Xcode で `.gputrace` を開いて GPU デバッガの
"Performance" / "GPU" ペインを見るのが唯一の経路 -- このツールはそこまでを
用意し、(b) として xctrace 経由の読み出しを毎回試すだけ試して (将来の Xcode
で変わるかもしれない)、ダメなら (a) の Xcode 手動確認だけを案内する。

## 仕様どおりの使い方 (実モデルには使わないこと -- 上の追記参照)

    tools/biglock.sh .venv/bin/python tools/gpu_capture_forward.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctx 0 --width 1

`tools/verify_width_cost.py` の `build_runner`/`build_prompt_ids`/`build_pair`
と `tools/decode_ab.py` の `prefill_once` をそのまま流用する (写しを作ると
挙動がずれるため)。prefill を 1 回済ませたあと、S=`--width` の verify 幅の
トークン列を組み、`spec_flash.capture(model)` の下で `model(pair, cache=caches)`
を 1 回呼ぶだけを capture 区間にする -- これは ms/token を測る道具ではない
(`decode_ab.py`/`verify_width_cost.py` のような「最初の 3 回を捨てる」
「複数回の中央値を取る」計測作法はここでは適用しない。捨てるのは JIT/kernel
コンパイルの一過性コストで、それ自体もカーネル構成を知る材料になるため
そのまま残す)。

出力は `bench/results/gputrace/<tag>.gputrace` (既定 tag は `ctx{ctx}-w{width}`)。
`mx.metal.start_capture` は出力パスが既に存在すると失敗する (実測エラー:
"couldn't be saved ... because a file with the same name already exists")
ので、既定では既存ファイルがあれば止まる。上書きしたければ `--force`。
"""

from __future__ import annotations

import os

# MTL_CAPTURE_ENABLED は Metal (mlx) の import 前でなければ効かない。
# 呼び出し元に環境変数を仕込ませず、このプロセス自身がここで一番先に立てる。
os.environ.setdefault("MTL_CAPTURE_ENABLED", "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# verify_width_cost.py / decode_ab.py と同じ組み立て・prefill ヘルパーを
# そのまま使う。どちらのモジュールも mlx の import はトップレベルでは
# 行わない (関数内で遅延 import) ので、ここで import しても
# MTL_CAPTURE_ENABLED の設定順は崩れない。
from verify_width_cost import build_pair, build_prompt_ids, build_runner  # noqa: E402
from decode_ab import prefill_once  # noqa: E402

OUT_DIR = REPO_ROOT / "bench" / "results" / "gputrace"


def _dir_size_bytes(path: Path) -> int:
    if not path.is_dir():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def _human(nbytes: int) -> str:
    v = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}TB"


def try_xctrace_export(gputrace_path: Path) -> tuple[bool, str]:
    """(b): xctrace で `.gputrace` の中を読めるか試す。

    実測 (2026-09-03、このツール自身の合成テストで確認): `.gputrace` は
    xctrace の `.trace` 形式とは別物で、
    `xcrun xctrace export --input <bundle> --toc` は
    "Export failed: Document Missing Template Error" (終了コード 10) で
    即座に失敗する。それでも将来の Xcode で変わるかもしれないので、
    ここで実際に毎回試し、失敗したら理由をそのまま返して (a) だけに倒す。
    """
    cmd = ["xcrun", "xctrace", "export", "--input", str(gputrace_path), "--toc"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return False, "xcrun/xctrace が見つからない (Xcode Command Line Tools 未導入?)"
    except subprocess.TimeoutExpired:
        return False, "xctrace export がタイムアウトした (60s)"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, f"xctrace export 失敗 (終了コード {proc.returncode}): {detail}"
    return True, (proc.stdout or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument("--mtp", default=None, help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=0, help="既定 0 = 短プロンプト")
    ap.add_argument("--width", type=int, default=1, help="verify forward の幅 S")
    ap.add_argument("--tag", default=None, help="出力ファイル名 (既定 ctx{ctx}-w{width})")
    ap.add_argument("--out", default=None, help=f"既定 {OUT_DIR}/<tag>.gputrace")
    ap.add_argument("--force", action="store_true", help="既存の .gputrace を消して録り直す")
    args = ap.parse_args()

    import mlx.core as mx

    if not (hasattr(mx.metal, "start_capture") and hasattr(mx.metal, "stop_capture")):
        print(
            "mx.metal.start_capture / stop_capture が無い (このビルドの mlx では "
            f"GPU capture 未対応、mlx version={getattr(mx, '__version__', '?')})。"
            "止める。"
        )
        return 1

    tag = args.tag or f"ctx{args.ctx}-w{args.width}"
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else OUT_DIR / f"{tag}.gputrace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not args.force:
            print(f"{out_path} が既にある (--force で消して録り直す)")
            return 1
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    print(f"device_info: {mx.device_info()}")
    print(
        f"MTL_CAPTURE_ENABLED={os.environ.get('MTL_CAPTURE_ENABLED')}  "
        f"model={args.model}  ngram={args.ngram}  ctx={args.ctx}  width={args.width}"
    )

    eng, model, tok, eos_ids = build_runner(args)
    ids = build_prompt_ids(tok, args.ctx)

    from mlxturbo.spec_flash import capture

    t0 = time.perf_counter()
    caches, _snap, resume, _first = prefill_once(eng, ids, eos_ids)
    print(f"prefill 完了 (n={ids.shape[1]}, {time.perf_counter() - t0:.1f}s)")

    pair, _cur = build_pair(eng, resume, args.width)
    print(f"pair.shape={tuple(pair.shape)} (--width {args.width})")

    print(f"capture 開始 -> {out_path}")
    try:
        mx.metal.start_capture(str(out_path))
    except Exception as e:
        print(f"start_capture 失敗: {e!r}")
        print(
            "原因の見当: MTL_CAPTURE_ENABLED の扱い (import 前に立っているか)、"
            "出力パスの既存ファイル、Metal のバージョン・権限のいずれか。"
        )
        return 1

    try:
        with capture(model) as _cap:
            lg = model(pair, cache=caches)
        mx.eval(lg)
    finally:
        mx.metal.stop_capture()
    print("capture 終了 (forward 1 回)")

    if not out_path.exists():
        print(f"{out_path} が生成されなかった (capture が実際には走らなかった可能性)")
        return 1

    size_bytes = _dir_size_bytes(out_path)
    print(f"\n.gputrace -> {out_path}  ({_human(size_bytes)}, {size_bytes} bytes)")

    print(
        "\n(a) Xcode で開く手順:\n"
        f"    open '{out_path}'\n"
        "  Finder / open コマンドで .gputrace をダブルクリックすると Xcode の "
        "GPU デバッガが開く。フレーム一覧 (唯一の 1 フレーム) を選び、右側の "
        "\"GPU\" / \"Performance\" ペインでカーネル (shader) 別の実行時間・"
        "回数が見える。"
    )

    print("\n(b) xctrace で .gputrace のカーネル別時間を CLI に落とせるか試す...")
    ok, detail = try_xctrace_export(out_path)
    if ok:
        print("  成功: xctrace が .gputrace を読めた (docstring の実測と食い違う。中身を確認すること):")
        print(f"  {detail[:2000]}")
    else:
        print(f"  不可: {detail}")
        print("  (b) は無し。(a) の Xcode 手動確認だけが経路 (このファイルの docstring 参照)。")

    sys.stdout.flush()
    sys.stderr.flush()
    # verify_width_cost.py と同じ理由: interpreter shutdown 待ちでプロセスが
    # Metal のメモリ (91GB 級モデル) を握ったまま残った実測があるので、
    # 結果を書き終えたら即 _exit で落とす。
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
