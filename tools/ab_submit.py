"""GPU の仕事を常駐 worker (`tools/ab_daemon.py`) の 1 本の列に投げる。

## 使い方 (3 通り)

    # (a) A/B: `tools/biglock.sh .venv/bin/python tools/decode_ab.py ...` の置き換え
    .venv/bin/python tools/ab_submit.py -- --knob null --model ~/models/ddalcu-mlxlm

    # (b) 何でも: worker が生きていれば列に乗せ、居なければ従来どおり
    #     (`tools/biglock.sh` がこの形で自分を呼ぶ)
    .venv/bin/python tools/ab_submit.py --from-biglock --prio 1 -- <コマンド一式>

    # (c) 世話
    .venv/bin/python tools/ab_submit.py --status
    .venv/bin/python tools/ab_submit.py --stop

`--` の後ろは (a) では `tools/decode_ab.py` の引数そのまま、(b) では実行する
コマンドそのまま。どちらもログは標準出力に流れ、終了コードを引き継ぐので、
呼ぶ側から見た挙動は今までと同じ。違うのは **98GB の読み直しが 1 回で済む**
ことだけ。

## ジョブの振り分け (`--from-biglock`)

| コマンド | ジョブ種 | 走り方 |
|---|---|---|
| `tools/decode_ab.py ...` | `decode_ab` | worker の中で in-process |
| `TOOL_JOBS` の道具 | `tool` | worker の中で in-process (`run_with_model(argv, bundle)`) |
| micro / verify_ / smoke_ / 連鎖ツール | `exec` | worker が GPU の番だけ渡して subprocess |
| それ以外 (self_snapshot、mlx-serve など) | -- | 終了コード 64 を返し、biglock が従来どおり自分でロックを取る |

worker に載せられない A/B (`--round-trace` / `--draft-trace` /
`--knob oracle-draft` / `--knob ngram-layout`)、worker が別のモデルを
抱えている場合も 64 (biglock 側が worker に「降りろ」を伝えてから走る)。

段 (prio) の既定は `tools/biglock.sh` と同じ規則: 祖先に `run_chain*.sh` が
いれば 0、それ以外 1、`BIGLOCK_PRIO` で明示。worker の列も段の高い順 →
先着順で回す。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import ab_daemon as AD  # noqa: E402  (sys.path を通した後でないと引けない)

PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

# tail 中に worker が降りた (`--yield-after-s`) ことを表す内部の印。
# 終了コードとしては返らない (起こし直して待ちを続ける)。
WORKER_GONE = -1
# 「worker には載らないので biglock が自分でやれ」。`--from-biglock` 専用。
NOT_ROUTABLE = 64

# in-process で走らせる道具 (`run_with_model(argv, bundle)` を持つもの)。
# ここに名前があっても、その関数が無ければ worker が err に理由を書く。
TOOL_JOBS = ("longctx_quality.py", "moe_split.py", "quant_eval.py")
# モデルを読まない道具。worker はモデルを抱えたまま GPU の番だけ渡す。
# `tools/biglock.sh` が段 2 を自動で振るのと同じ集合。
EXEC_RE = re.compile(
    r"(_micro\.py|/micro_[^/]*\.py|/verify_[^/]*\.py|/smoke_[^/]*\.py"
    r"|kernel_chain_cost\.py)$")


def default_prio(argv: list[str]) -> int:
    """`tools/biglock.sh` と同じ規則で段を決める。

    micro の段 2 は decode_ab のジョブには当たらない (あちらの自動判定は
    `*_micro.py` などのコマンド名で決まる)。ここでは 0 (親の列) か 1。
    """
    env = os.environ.get("BIGLOCK_PRIO")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    pid = os.getppid()
    for _ in range(24):
        if pid <= 1:
            break
        try:
            out = subprocess.run(["ps", "-o", "ppid=,command=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            break
        line = out.strip()
        if not line:
            break
        ppid_s, _, cmd = line.partition(" ")
        if "run_chain" in cmd and ".sh" in cmd:
            return 0
        try:
            pid = int(ppid_s)
        except ValueError:
            break
    return 1


def daemon_record() -> dict | None:
    try:
        rec = json.loads(AD.PID_FILE.read_text())
    except Exception:
        return None
    pid = int(rec.get("pid", -1))
    if pid <= 0 or not AD._alive(pid):
        return None
    return rec


def stop_daemon(timeout_s: float = 7200.0, verbose: bool = True) -> bool:
    rec = daemon_record()
    if rec is None:
        return True
    AD.STOP_FILE.write_text("stop\n")
    if verbose:
        print(f"[ab_submit] worker (pid={rec['pid']}) の停止を待つ", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not AD._alive(int(rec["pid"])):
            AD.STOP_FILE.unlink(missing_ok=True)
            return True
        time.sleep(1.0)
    AD.STOP_FILE.unlink(missing_ok=True)
    return False


def start_daemon(model, ngram=None, mtp=None, mtp_bits=4,
                 verbose: bool = True) -> dict | None:
    cmd = [PYTHON, str(REPO_ROOT / "tools" / "ab_daemon.py"),
           "--model", model, "--mtp-bits", str(mtp_bits)]
    if ngram:
        cmd += ["--ngram", ngram]
    if mtp:
        cmd += ["--mtp", mtp]
    log = Path(os.environ.get("MLXTURBO_AB_DAEMON_LOG",
                              str(AD.TMP / "mlxturbo-ab-daemon.log")))
    if verbose:
        print(f"[ab_submit] worker が居ないので起こす (ログ: {log})", flush=True)
    with open(log, "a", buffering=1) as fh:
        fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} 起動 ====\n")
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=str(REPO_ROOT))
    for _ in range(120):
        rec = daemon_record()
        if rec is not None:
            return rec
        time.sleep(0.5)
    return None


def run_fallback(argv: list[str]) -> int:
    """従来どおり `tools/biglock.sh` 経由の別プロセスで流す (decode_ab 専用)。"""
    if daemon_record() is not None:
        print("[ab_submit] このジョブは worker に載せられない。"
              "98GB を 2 本載せないよう、先に worker を止める", flush=True)
        if not stop_daemon():
            print("[ab_submit] worker が止まらない。中止する", file=sys.stderr)
            return 4
    cmd = [str(REPO_ROOT / "tools" / "biglock.sh"), PYTHON,
           str(REPO_ROOT / "tools" / "decode_ab.py")] + argv
    print(f"[ab_submit] 別プロセスで流す: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def tail_until_done(job_path: Path, log_path: Path) -> int:
    """ジョブファイルが .done / .err に変わるまでログを流し続ける。"""
    done = job_path.with_suffix(".done")
    err = job_path.with_suffix(".err")
    pos = 0
    while True:
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        if done.exists() or err.exists():
            time.sleep(0.2)  # 最後の書き込みを取りこぼさない
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    rest = fh.read()
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
            path = done if done.exists() else err
            try:
                rec = json.loads(path.read_text())
            except Exception:
                rec = {}
            code = int(rec.get("code", 0 if path is done else 1))
            if rec.get("traceback"):
                sys.stderr.write(rec["traceback"] + "\n")
            if rec.get("reason"):
                sys.stderr.write(str(rec["reason"]) + "\n")
            return code
        if not job_path.exists():
            sys.stderr.write("[ab_submit] ジョブファイルが消えた\n")
            return 5
        if daemon_record() is None:
            # 常駐 worker は「他人が biglock を待っている」と降りる
            # (`--yield-after-s`)。ジョブはキューに残っているので、
            # 起こし直して待ちを続ける。
            return WORKER_GONE
        time.sleep(0.4)


def enqueue(spec: dict) -> tuple[Path, Path]:
    """ジョブを置く。**自分の `MLXTURBO_*` / `FASTMLX_*` を全部載せる。**

    `MLXTURBO_QSA_TAIL=query tools/biglock.sh ... decode_ab.py --knob ...` の
    ように env で構成を変える A/B があるので、載せないと worker が起動時の
    構成で黙って測ることになる。worker 側は起動時の env と突き合わせ、
    読み込み時に効くものが違えば**その env で自分を作り直してから**走る
    (`tools/ab_daemon.py` の `RUNTIME_ENV`)。
    """
    AD.QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.time_ns()}-{os.getpid()}"
    job_path = AD.QUEUE_DIR / f"{stamp}.json"
    log_path = AD.QUEUE_DIR / f"{stamp}.log"
    log_path.write_text("")
    spec = dict(spec, log=str(log_path), submitted=time.time(), cwd=os.getcwd(),
                env=AD.ab_env())
    tmp = job_path.with_suffix(".staging")
    tmp.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    tmp.replace(job_path)  # 半端な JSON を worker に読ませない
    return job_path, log_path


def submit_and_wait(spec: dict, restart: dict | None) -> int:
    """ジョブを置いて、終わるまでログを流す。

    ``restart`` があれば、待っている間に worker が降りたとき同じ引数で
    起こし直す (`start_daemon` の kwargs)。None なら降りた時点で諦める。
    """
    job_path, log_path = enqueue(spec)
    print(f"[ab_submit] 投入 {job_path.name} 段={spec['prio']} "
          f"{spec.get('type', 'decode_ab')}", flush=True)
    while True:
        code = tail_until_done(job_path, log_path)
        if code != WORKER_GONE:
            return code
        print("[ab_submit] worker が降りた。起こし直して待ちを続ける", flush=True)
        if restart is None or start_daemon(**restart) is None:
            sys.stderr.write("[ab_submit] worker を起こせなかった。"
                             f"ジョブは {job_path} に残してある\n")
            return 6


# ---- `--from-biglock`: 任意のコマンドをジョブ種に振り分ける ----------------

def classify(cmd: list[str]) -> dict | None:
    """コマンドを見てジョブの中身を返す。載らないものは None。"""
    for i, a in enumerate(cmd):
        if a.endswith("tools/decode_ab.py") or Path(a).name == "decode_ab.py":
            return dict(type="decode_ab", argv=cmd[i + 1:])
    for i, a in enumerate(cmd):
        if Path(a).name in TOOL_JOBS:
            rel = a
            try:
                rel = str(Path(a).resolve().relative_to(REPO_ROOT))
            except (ValueError, OSError):
                pass
            return dict(type="tool", path=rel, argv=cmd[i + 1:])
    for a in cmd:
        if EXEC_RE.search("/" + a.lstrip("/")):
            return dict(type="exec", cmd=cmd)
    return None


def from_biglock(cmd: list[str], prio: int) -> int:
    """`tools/biglock.sh` から呼ばれる入口。

    載せられれば worker の列に投げて終了コードを返す。載せられなければ
    `NOT_ROUTABLE` を返し、biglock が従来どおり自分でロックを取る
    (そのとき biglock は worker に「降りろ」を伝えてから走る)。
    """
    rec = daemon_record()
    if rec is None:
        return NOT_ROUTABLE  # worker が居ないなら従来どおり
    spec = classify(cmd)
    if spec is None:
        return NOT_ROUTABLE
    if spec["type"] == "decode_ab":
        import decode_ab

        try:
            job_args, knob_names = decode_ab.parse_args(spec["argv"])
        except SystemExit as e:
            return int(e.code or 2)
        if AD.job_reject_reason(job_args, knob_names):
            return NOT_ROUTABLE
        if _model_mismatch(rec, job_args):
            return NOT_ROUTABLE
    elif spec["type"] == "tool" and AD.tool_reject_reason(spec["path"]):
        # 道具側が `run_with_model` をまだ持っていない。従来の別プロセスへ。
        return NOT_ROUTABLE
    spec["prio"] = prio
    return submit_and_wait(spec, restart=None)


def _model_mismatch(rec: dict, job_args) -> str | None:
    class _Have:
        model = rec.get("model")
        ngram = rec.get("ngram")
        mtp = rec.get("mtp")
        mtp_bits = rec.get("mtp_bits")

    return AD.bundle_matches(job_args, _Have)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prio", type=int, default=None,
                    help="段 (0=親の列 / 1=既定 / 2=micro)。既定は biglock.sh と同じ規則")
    ap.add_argument("--stop", action="store_true", help="worker を止めて終わる")
    ap.add_argument("--status", action="store_true", help="worker とキューを見て終わる")
    ap.add_argument("--no-start", action="store_true",
                    help="worker が居なくても起こさない (別プロセスに落とす)")
    ap.add_argument("--from-biglock", action="store_true",
                    help="`--` の後ろをコマンドとして振り分ける (biglock.sh 専用)。"
                         f"載らないときは終了コード {NOT_ROUTABLE} を返す")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="`--` の後ろに decode_ab の引数 (既定) か"
                         " コマンド一式 (--from-biglock)")
    args = ap.parse_args()

    if args.stop:
        ok = stop_daemon()
        print("止めた" if ok else "止まらなかった")
        return 0 if ok else 1
    if args.status:
        rec = daemon_record()
        print(json.dumps(rec, ensure_ascii=False, indent=1) if rec else "worker は居ない")
        AD.QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        pending = sorted(p.name for p in AD.QUEUE_DIR.glob("*.json"))
        print(f"待ち {len(pending)}: {', '.join(pending) or '(なし)'}")
        return 0

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        ap.error("`--` の後ろに引数を渡すこと")
    prio = args.prio if args.prio is not None else default_prio(rest)

    if args.from_biglock:
        return from_biglock(rest, prio)

    # 既定: decode_ab の引数として受ける
    import decode_ab

    try:
        job_args, knob_names = decode_ab.parse_args(rest)
    except SystemExit as e:
        return int(e.code or 2)

    reason = AD.job_reject_reason(job_args, knob_names)
    if reason:
        print(f"[ab_submit] worker には載せられない: {reason}", flush=True)
        return run_fallback(rest)

    restart = dict(model=job_args.model, ngram=job_args.ngram,
                   mtp=job_args.mtp, mtp_bits=job_args.mtp_bits)
    rec = daemon_record()
    if rec is None:
        if args.no_start:
            return run_fallback(rest)
        rec = start_daemon(**restart)
        if rec is None:
            print("[ab_submit] worker を起こせなかった", file=sys.stderr)
            return run_fallback(rest)
    else:
        mismatch = _model_mismatch(rec, job_args)
        if mismatch:
            print(f"[ab_submit] worker は別のモデルを抱えている: {mismatch}",
                  flush=True)
            return run_fallback(rest)

    return submit_and_wait(
        dict(type="decode_ab", argv=rest, prio=prio, out=job_args.out),
        restart=None if args.no_start else restart)


if __name__ == "__main__":
    # ab_daemon._reexec_absolute と同じ理由 (あちらの docstring 参照):
    # 相対パスのコマンドラインは biglock.sh の「ロックを持たない相手」検出に
    # 一致してしまい、**古い写しを読み進めている待ち手を止める。**
    if not os.path.isabs(sys.argv[0]):
        os.execv(sys.executable,
                 [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
    raise SystemExit(main())
