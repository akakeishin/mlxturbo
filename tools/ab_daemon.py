"""98GB のモデルを 1 回だけ読んで常駐し、A/B のジョブを次々に受ける worker。

## なぜ

`tools/biglock.sh .venv/bin/python tools/decode_ab.py ...` は 1 本ごとに
`~/models/ddalcu-mlxlm` (91GB) と n-gram サイドカーを SSD から読み直す。
実測で 1 回 3 分、biglock のメモリ待ちを足すと所要時間の 3〜4 割が読み直し。
**同じモデルを何度も測るのだから、読むのは 1 回でいい。**

## 使い方

    # 投入 (これだけ覚えればよい。daemon は要るときに勝手に立つ)
    .venv/bin/python tools/ab_submit.py -- --knob null --only short \
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram

    # `tools/biglock.sh ...` はそのままでよい。worker が居れば勝手に列に乗る
    tools/biglock.sh .venv/bin/python tools/decode_ab.py --knob null ...

    # 手で起こす / 落とす
    nohup .venv/bin/python tools/ab_daemon.py \
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \
        > /tmp/ab_daemon.log 2>&1 &
    .venv/bin/python tools/ab_submit.py --stop

## ジョブの種類

キューのファイルは JSON。`type` (既定 `decode_ab`)、`prio` (0/1/2)、`log`
(標準出力の行き先) は共通。

| `type` | 中身 | 走り方 |
|---|---|---|
| `decode_ab` | `argv` = decode_ab の引数 | `decode_ab.run_with_model(argv, bundle)` を in-process |
| `tool` | `path` = 道具の .py、`argv` | その道具の `run_with_model(argv, bundle)` を in-process (規約は `tools/ab_bundle.py`) |
| `exec` | `cmd` = コマンドの配列、`cwd` | subprocess。**worker はモデルを抱えたまま GPU の番だけ渡す** (micro / verify / smoke / 連鎖ツール) |

`decode_ab` と `tool` は同じ 98GB の上で走るので、ジョブの後に状態を戻す
(下記)。`exec` は別プロセスなので戻すものが無く、落ちても worker は無傷。

## 守っていること (CLAUDE.md の計測の作法)

- **GPU の排他は biglock と同じ規約。**ジョブを実行する間だけ
  `${TMPDIR}/mlxturbo-bigmodel.lock` に自分の pid を noclobber で書き、
  `${TMPDIR}/mlxturbo-biglock-prio/` に札を出して上の段に譲る。ジョブと
  ジョブの間はロックを放すので、別プロセスの micro が割り込める。
  **ジョブ実行時にメモリ待ちはしない** (モデルは既に載っている)。起動時の
  読み込みだけは 100GB の空きを待つ (98GB を 2 本載せると必ず落ちる)。
- **モデルを読む段 0/1 が来たら降りる。**worker が居座ると、worker を通れ
  ない仕事 (self_snapshot、mlx-serve など) が biglock のメモリ待ち 10 分を
  空振りしたうえでスラッシングする。biglock はロックを取った直後に
  `${TMPDIR}/mlxturbo-ab-daemon.stop` を置き、この pid が消えるまで待つ。
  worker はジョブ実行中ならそれを終えてから、待機中なら即座に 98GB を返す
  (`os._exit` で返す --- shutdown 待ちで握ったまま残った実測がある)。
  次に投入があれば worker は自分で立ち上がり直す。
- **最初のジョブだけ冷たいのを作らない。**読み込み直後に null の A/B を
  1 本捨てる (`burn_in`、10 秒級)。実測で、空焼き無しの 1 本目は各ケースの
  1 行目 (= A) が 8〜9% 遅く、回文順では相殺できない。
- **ジョブごとに状態を戻す。**decode_ab の knob は「baseline を貼り直して
  終わる」が、その baseline は多くの knob で **B = 旧実装 / off** であって
  本番の既定ではない (`mtp-append` の baseline B は `_MTP_CACHE_APPEND=False`、
  `gather-tile` の baseline -1 は gather 自体が切れたまま、`gdn-metal` の
  baseline B は Metal カーネルが切れたまま)。1 プロセス 1 ジョブなら害が
  無いが、常駐すると**次のジョブが別のモデルを測ることになる。**そこで
  ジョブの後で (1) 環境変数 (2) 差し替え点の属性 (3)
  `enable_default_fusions` の再適用、の 3 段で本番既定に戻す。
- **コードが変わったら自分を作り直す。**エージェントが `mlxturbo/` や
  `tools/decode_ab.py` を編集した直後に A/B を投げてくる。ジョブを始める
  前に mtime を起動時と比べ、変わっていたら `os.execv` で自分を入れ替える
  (pid は変わらないのでキューも pid ファイルもそのまま)。古いコードで
  測るより 3 分払って読み直すほうが正しい。

## 載せられないジョブ

投入側 (`tools/ab_submit.py`) が同じ判定を持っていて、該当したら従来どおり
`tools/biglock.sh` の別プロセスに落とす。理由は `UNSAFE_KNOBS` と
`LOAD_TIME_FLAGS` のコメントに書いてある。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

TMP = Path(os.environ.get("TMPDIR", "/tmp"))
QUEUE_DIR = Path(os.environ.get("MLXTURBO_AB_QUEUE", str(TMP / "mlxturbo-ab-queue")))
PID_FILE = Path(os.environ.get("MLXTURBO_AB_PID", str(TMP / "mlxturbo-ab-daemon.pid")))
# 「降りろ」の合図。`tools/biglock.sh` の段 0/1 (モデルを読む仕事) がロックを
# 取った直後にここへ触り、pid が消えるまで待ってからメモリ待ちに進む。
STOP_FILE = PID_FILE.with_suffix(".stop")

# biglock.sh と同じ場所・同じ規約 (あちらの zsh と読み書きし合う)
LOCK_FILE = TMP / "mlxturbo-bigmodel.lock"
PRIO_DIR = TMP / "mlxturbo-biglock-prio"

# ---- daemon に載せられないもの -------------------------------------------
#
# knob: ジョブが終わっても状態が戻らないもの。**戻せないものを載せると、
# 次のジョブが黙って別の構成を測る。**
UNSAFE_KNOBS = {
    # setup で `FlashSpecEngine._draft_chain` をクラスごと差し替え、
    # `apply` は `eng._depth_adapt = False` まで書く。どちらも元に戻す口が
    # 無い (`_knob_oracle_draft` の末尾)。
    "oracle-draft",
    # setup で `ngram_stream.StreamNGram`/`RamNGram` をキャッシュ付きの
    # サブクラスに差し替え、B 側の 32GB (RamNGram) をプロセスが死ぬまで
    # RAM に抱える。98GB の常駐と足すと 128GB に収まらない。
    "ngram-layout",
}
# ---- 環境変数 ------------------------------------------------------------
#
# 投入側 (`tools/ab_submit.py`) は自分の `MLXTURBO_*` / `FASTMLX_*` を全部
# ジョブに載せる。worker は起動時の env と突き合わせ、**違えばその env で
# 自分を作り直してから** ジョブを走らせる (3 分の読み直しを払う)。
# これが無いと `MLXTURBO_QSA_TAIL=query tools/biglock.sh ... decode_ab.py` の
# ような「env で構成を変える A/B」が、黙って worker の起動時の構成で走る。
ENV_PREFIXES = ("MLXTURBO_", "FASTMLX_")
# worker 自身の配管。モデルの挙動に関係しないので突き合わせから外す
# (これを見ていると、キューの場所を変えただけで読み直しになる)。
ENV_IGNORE = {
    "MLXTURBO_AB_QUEUE", "MLXTURBO_AB_PID", "MLXTURBO_AB_DAEMON_LOG",
    "MLXTURBO_MIN_FREE_GB",
}
# **実行中に切り替えられると確かめた変数だけ**をここに置く。ここに無いものは
# 全部「読み込み時に効く」扱いで作り直す --- 判断を誤ったときの代金が
# 非対称だから (作り直しは 3 分、取りこぼしは *嘘の数字*)。
#
# 根拠 (2026-09-03 に読んだ場所):
#   PIPELINE            spec_flash.py:2380  generate_stream の中で毎ラウンド
#   PREFILL_ATTN_MIN_KV _vendor/qwen4_exp.py:1224  attention の forward の中
#   PHASE_TIMERS        spec_flash.py:2359  generate_stream の中
#   DEPTH_CAP / _COST   spec_flash.py:346 / :374  controller が毎ラウンド読む
#   DEPTH_BETA / _EXPLORE / _MARGIN
#                       spec_flash.py:239 / :270 / :299  DepthController.__init__
#                       -- worker はジョブの頭で controller を作り直す
#                       (`reset_engine`) ので、**env を当ててから reset する
#                       限り**効く。順番を入れ替えないこと。
# 逆に `fused.enable_*` が読むもの (MLXTURBO_HC / _SORT_MIN / _MOE_* /
# _GDN_* / _SDPA_* / _QMM_* / _WIDE* / _GATHER_* / _PREFILL_ATTN / _FAST_*)
# は**融合を当てる瞬間にしか読まれない**。読み込み済みの worker に後から
# env だけ置いても何も起きないので、runtime 側には入れない。
# モジュール先頭で読まれるもの (spec_flash の _STAGE_EVERY / PRIME_WINDOW /
# _PREFILL_* / _MTP_CACHE_APPEND / _DRAFT_PRESYNC / MLXTURBO_DEPTH_ADAPT、
# qsa_tail の MODE / TIEBREAK、_vendor の NGRAM_ON_DISK / _SORT_MIN、
# kernels/gated_delta_blocked の MIN_T / BLOCK / SUB_BLOCK) も同じ。
RUNTIME_ENV = {
    "MLXTURBO_PIPELINE",
    "MLXTURBO_PREFILL_ATTN_MIN_KV",
    "MLXTURBO_PHASE_TIMERS",
    "MLXTURBO_DEPTH_CAP",
    "MLXTURBO_DEPTH_COST",
    "MLXTURBO_DEPTH_BETA",
    "MLXTURBO_DEPTH_EXPLORE",
    "MLXTURBO_DEPTH_MARGIN",
}
# 同じジョブで 2 回続けて作り直そうとしたら止める (読み直しの無限ループ避け)。
# 値はジョブファイル名。MLXTURBO_/FASTMLX_ で始めないこと -- 突き合わせに
# 混ざって、それ自体が「env が違う」理由になってしまう。
ENV_JOB_MARK = "AB_DAEMON_ENV_JOB"

# フラグ: 読み込み時にしか読まれない環境変数を立てるもの。
# `spec_flash._ROUND_TRACE` / `_DRAFT_TRACE` は import 時の 1 回だけ評価
# されるので、読み込み済みの daemon に後から効かせられない。
LOAD_TIME_FLAGS = ("round_trace", "draft_trace")


# ---- biglock (tools/biglock.sh と同じ規約を Python で) --------------------

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _free_gb() -> int:
    """空き + 非活性 + 投機 (回収可能) を GB で。圧縮済みは使用中なので数えない
    (biglock.sh の awk と同じ式)。"""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return 1 << 30
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            pages += int(m.group(1))
    return int(pages * 16384 / 1073741824)


_OTHER_RE = re.compile(r"\.venv/bin/python3? (tools|bench)/")
# 待ち合わせている側を「ロック無しで走っている python」と数えないこと
# (biglock.sh の同じ場所のコメント参照。4 本並べて 47 分空転した前例)。
_OTHER_SKIP = ("biglock.sh", "ab_daemon.py", "ab_submit.py")


def _lockless_other(self_pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == self_pid or not _OTHER_RE.search(cmd):
            continue
        if any(s in cmd for s in _OTHER_SKIP):
            continue
        return pid
    return None


def foreign_biglock_user(self_pid: int) -> int | None:
    """自分以外に「98GB を読む仕事」の待ち手・持ち手が居るか。

    常駐している間は空きメモリが 100GB を切るので、`tools/biglock.sh` を
    素で使う側 (別のエージェント、親の列) は段 0/1 のメモリ待ち 10 分に
    毎回引っ掛かる。**居座って他人の計測を熱と圧で汚すのは、この道具が
    避けるべき筆頭。**キューが空のままここが真であり続けたら降りる
    (`--yield-after-s`)。投入があればすぐ起き直る (ab_submit が起こす)。
    """
    try:
        if LOCK_FILE.exists():
            owner = int((LOCK_FILE.read_text().splitlines() or ["0"])[0])
            if owner and owner != self_pid and _alive(owner):
                return owner
    except Exception:
        pass
    if PRIO_DIR.is_dir():
        for t in PRIO_DIR.iterdir():
            try:
                other = int(t.name)
            except ValueError:
                continue
            if other != self_pid and _alive(other):
                return other
    return None


def _live_higher(prio: int, self_pid: int, ticket: Path) -> bool:
    """自分より先に通すべき札 (自分以外) が生きているか。死んだ札は掃除する。

    `tools/biglock.sh` の `live_higher` と同じ規則: **上の段**、または
    **同じ段で自分の札より古い札**。同じ段を先着順にしないと、poll の短い
    新しい待ち手が古い待ち手を追い越し続ける (2 時間待ちの前例)。
    daemon の札はジョブを取り出した時点で出すので (`biglock_acquire` の
    呼び出し口)、札の mtime がそのまま並び順になる。
    """
    if not PRIO_DIR.is_dir():
        return False
    try:
        mine = ticket.stat().st_mtime_ns
    except OSError:
        return False
    for t in PRIO_DIR.iterdir():
        try:
            other = int(t.name)
        except ValueError:
            continue
        if other == self_pid:
            continue
        if _alive(other):
            try:
                lv = int((t.read_text().strip() or "1"))
            except Exception:
                lv = 1
            if lv > prio:
                return True
            if lv == prio:
                try:
                    if t.stat().st_mtime_ns < mine:
                        return True
                except OSError:
                    pass
        else:
            t.unlink(missing_ok=True)
    return False


class StopRequested(Exception):
    """待っている最中に「降りろ」の合図が来た。

    `tools/biglock.sh` の段 0/1 は、ロックを取った直後にこの合図を置いて
    **daemon の pid が消えるまで待つ。**待ち行列の中で気づかずに待ち続けると
    あちらが 1 時間空転するので、待ちループの毎周で見る。
    """


def biglock_acquire(prio: int, mem_need_gb: int = 0, log=print) -> None:
    """`tools/biglock.sh` と同じ手順でロックを取る。

    ``mem_need_gb`` が 0 ならメモリ待ちをしない (ジョブ実行時。モデルは
    既に載っている)。起動時の読み込みだけ 100 を渡す。

    待っている最中に停止の合図が来たら `StopRequested` を投げる。
    """
    pid = os.getpid()
    PRIO_DIR.mkdir(parents=True, exist_ok=True)
    ticket = PRIO_DIR / str(pid)
    ticket.write_text(f"{prio}\n")
    poll = {2: 3, 1: 5}.get(prio, 15)
    waited = 0
    while True:
        if STOP_FILE.exists():
            ticket.unlink(missing_ok=True)
            raise StopRequested()
        if LOCK_FILE.exists():
            try:
                owner = int((LOCK_FILE.read_text().splitlines() or ["0"])[0])
            except Exception:
                owner = 0
            if not owner or not _alive(owner):
                log(f"ab_daemon: 死んだロック (pid={owner}) を掃除する")
                LOCK_FILE.unlink(missing_ok=True)
            else:
                if waited == 0:
                    log(f"ab_daemon: pid={owner} がロックを持っている。待つ")
                time.sleep(poll)
                waited += poll
                continue
        other = _lockless_other(pid)
        if other is not None:
            if waited == 0:
                log(f"ab_daemon: ロック無しの pid={other} が走っている。待つ")
            time.sleep(15)
            waited += 15
            continue
        if _live_higher(prio, pid, ticket):
            if waited == 0:
                log(f"ab_daemon: 段 {prio}。先に並んでいる待ち手を通す")
            time.sleep(poll)
            waited += poll
            continue
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            time.sleep(5)
            waited += 5
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{pid}\n")
        ticket.unlink(missing_ok=True)
        break
    if waited:
        log(f"ab_daemon: {waited}s 待った。開始する")
    if mem_need_gb <= 0:
        return
    mem_waited = 0
    while mem_waited < 600:
        if STOP_FILE.exists():
            raise StopRequested()
        free = _free_gb()
        if free >= mem_need_gb:
            break
        if mem_waited == 0:
            log(f"ab_daemon: 回収可能メモリ {free}GB < {mem_need_gb}GB。空くまで待つ")
        time.sleep(15)
        mem_waited += 15
    else:
        log(f"ab_daemon: 警告 -- 10 分待ってもメモリが空かない ({_free_gb()}GB)")


def biglock_release() -> None:
    pid = os.getpid()
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip().startswith(str(pid)):
            LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    (PRIO_DIR / str(pid)).unlink(missing_ok=True)


# ---- コードの鮮度 ---------------------------------------------------------

def code_fingerprint() -> dict:
    """`mlxturbo/**/*.py` と `tools/decode_ab.py` と自分自身の mtime。

    エージェントが編集した直後に A/B を投げてくるので、**古いコードで
    測らないため**の見張り。値が変わったら `os.execv` で作り直す。
    """
    files: list[Path] = sorted(REPO_ROOT.glob("mlxturbo/**/*.py"))
    files += [REPO_ROOT / "tools" / "decode_ab.py",
              REPO_ROOT / "tools" / "ab_bundle.py", Path(__file__).resolve()]
    # `tool` ジョブで in-process に走る道具も同じ扱い。**読み込み済みの
    # モデルの上で走る以上、古い写しで測るのは decode_ab と同じ罪。**
    # 判定は「run_with_model を持つか」で、テキストを見るだけ (import しない)。
    files += [t for t in sorted([*REPO_ROOT.glob("tools/*.py"),
                                 *REPO_ROOT.glob("bench/*.py")])
              if t.name not in ("decode_ab.py", "ab_bundle.py", "ab_daemon.py")
              and tool_has_entry(t)]
    fp = {}
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        fp[str(p)] = (st.st_mtime_ns, st.st_size)
    return fp


def fingerprint_diff(old: dict, new: dict) -> list[str]:
    keys = set(old) | set(new)
    return sorted(k for k in keys if old.get(k) != new.get(k))


# ---- ジョブとジョブの間で本番既定に戻す ----------------------------------
#
# decode_ab の knob は「baseline を貼り直して終わる」だけで、その baseline は
# 本番の既定とは限らない (モジュール docstring 参照)。常駐する以上、ここで
# 戻し切らないと次のジョブが黙って別の構成を測る。

def _patch_points(bundle):
    """knob が差し替えうる (obj, attr) の一覧。

    `tools/decode_ab.py` の `_knob_*` が `apply()` の中で代入している先を
    そのまま列挙してある (`fused._ORIG_*` の帳簿は**入れない** --- あれを
    識別子ごと戻すと `enable_*` の再入ガードが「もう当ててある」と誤認して
    パッチが当たらないまま素通りする)。
    """
    import mlx.core as mx

    import mlxturbo.fused as F
    import mlxturbo.ngram_stream as NS
    import mlxturbo.qsa_tail as QT
    import mlxturbo.spec_flash as SF
    from mlxturbo.arch import qwen4_arch

    Q = qwen4_arch()
    pts = [
        (mx, "argpartition"),
        (SF, "_STAGE_EVERY"), (SF, "_PREFILL_GROUP"), (SF, "_PREFILL_PIPELINE"),
        (SF, "_PREFILL_FOLD_TAIL"), (SF, "_PREFILL_TAIL_IN_GROUP"),
        (SF, "_MTP_CACHE_APPEND"), (SF, "_DRAFT_PRESYNC"), (SF, "PRIME_WINDOW"),
        (SF, "MTP_DEPTH"),
        (SF.FlashSpecEngine, "_draft_chain"),
        (SF.FlashSpecEngine, "_prime_accepted_gap"),
        (F, "_MOE_DISPATCH_SORT_MIN"),
        (NS, "StreamNGram"), (NS, "RamNGram"),
        (Q, "gated_delta_update"),
        (Q.Attention, "__call__"), (Q.Attention, "_final_mask"),
        (Q.Attention, "_sdpa_split_width"),
        (Q.QSAIndexer, "__call__"), (Q.QSAIndexer, "_pooled_and_top"),
        (Q._IndexerCache, "update"),
        (Q.SparseMoeBlock, "__call__"),
        (Q.GatedDeltaNet, "_gdn_metal"),
        (QT, "MODE"),
    ]
    try:
        from mlxturbo.kernels import gdn_blocked_metal as gbm

        pts.append((gbm, "gated_delta_blocked_seq"))
    except Exception:
        pass
    eng = bundle.engine
    if eng is not None:
        for attr in ("depth", "depth_ctx_limit", "_depth_adapt", "_rerank",
                     "depth_trace_prompt_id"):
            pts.append((eng, attr))
    text_args = getattr(bundle.model.args, "text", None)
    if text_args is not None and hasattr(text_args, "indexer_budget"):
        pts.append((text_args, "indexer_budget"))
    for stream in bundle.ngram_streams:
        for attr in ("prefetch_enabled", "batch_min_rows"):
            if hasattr(stream, attr):
                pts.append((stream, attr))
    return pts


_MISSING = object()


def snapshot_state(bundle) -> list:
    return [(obj, attr, getattr(obj, attr, _MISSING))
            for obj, attr in _patch_points(bundle)]


def restore_state(bundle, snap: list, log=print) -> list[str]:
    """ジョブが触った差し替え点を戻し、本番の融合を貼り直す。

    順番が大事: 属性を戻してから `enable_default_fusions` を呼ぶ。逆にすると
    「knob の `disable_*` が外した融合」を貼り直した直後に、古い属性で
    上書きしてしまう。
    """
    changed = []
    for obj, attr, val in snap:
        cur = getattr(obj, attr, _MISSING)
        if cur is val:
            continue
        name = f"{getattr(obj, '__name__', type(obj).__name__)}.{attr}"
        changed.append(name)
        if val is _MISSING:
            try:
                delattr(obj, attr)
            except Exception:
                pass
        else:
            setattr(obj, attr, val)
    # 融合そのもの (fused.enable_*/disable_* が持つ帳簿) は属性の識別子では
    # 戻せない。**本番の起動と同じ呼び出しをもう一度通す。**enable_* は全部
    # 再入ガード付きなので二重には掛からない (mlxturbo/fused.py)。
    import io
    from contextlib import redirect_stdout

    from mlxturbo.runner import enable_default_fusions

    buf = io.StringIO()
    with redirect_stdout(buf):
        enable_default_fusions(bundle.model, log_prefix="[ab_daemon]")
    return changed


def reset_engine(bundle) -> None:
    """ジョブの頭で、文脈と学習が前のジョブから漏れないようにする。"""
    import mlx.core as mx

    from mlxturbo.kernels import _fire
    from mlxturbo.spec_flash import DepthController

    eng = bundle.engine
    if eng is None:
        mx.clear_cache()
        return
    # DepthController はエンジンの生存期間中ずっと学習を持ち回る設計
    # (spec_flash.__init__ のコメント)。プロセスを分けていた頃は 1 ジョブで
    # 捨てていたので、**ジョブごとに作り直さないと 2 本目が有利になる。**
    if getattr(eng, "_depth_adapt", False):
        eng._depth_controller = DepthController(ctx_limit=eng.depth_ctx_limit)
    eng.depth_trace_prompt_id = None
    eng._trace_top2 = None
    eng._trace_margins = None
    # `decode_ab --draft-topk` の trace スロット。属性なので (load-time env と
    # 違い) worker に載る -- 前のジョブから漏れると、次のジョブが黙って段ごと
    # に粗ヘッドを余分に引くことになる。
    eng._topk_k = 0
    eng._topk_records = None
    eng._trace_topk = None
    eng._trace_topk_true = None
    _fire.reset()
    mx.clear_cache()


# ---- キュー ---------------------------------------------------------------

def read_jobs() -> list[tuple[int, float, Path, dict]]:
    jobs = []
    for p in sorted(QUEUE_DIR.glob("*.json")):
        try:
            spec = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(spec, dict) or not job_shape_ok(spec):
            continue
        prio = int(spec.get("prio", 1))
        try:
            submitted = float(spec.get("submitted") or p.stat().st_mtime)
        except OSError:
            continue
        jobs.append((prio, submitted, p, spec))
    # 段の高い順 -> 古い順
    jobs.sort(key=lambda j: (-j[0], j[1], j[2].name))
    return jobs


JOB_TYPES = ("decode_ab", "tool", "exec")


def job_shape_ok(spec: dict) -> bool:
    """キューに置かれた JSON がジョブの形をしているか (種類ごと)。"""
    kind = spec.get("type") or "decode_ab"
    if kind == "decode_ab":
        return isinstance(spec.get("argv"), list)
    if kind == "tool":
        return isinstance(spec.get("path"), str) and isinstance(
            spec.get("argv", []), list)
    if kind == "exec":
        return isinstance(spec.get("cmd"), list) and bool(spec["cmd"])
    return False


def finish(path: Path, ok: bool, payload: dict) -> None:
    dest = path.with_suffix(".done" if ok else ".err")
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    tmp.replace(dest)
    path.unlink(missing_ok=True)


def job_reject_reason(args, knob_names) -> str | None:
    for flag in LOAD_TIME_FLAGS:
        if getattr(args, flag, False):
            return (f"--{flag.replace('_', '-')} は読み込み時の環境変数"
                    " (spec_flash の import 時に 1 回だけ読まれる) を立てるので"
                    " 常駐 worker には載せられない")
    bad = [k for k in knob_names if k in UNSAFE_KNOBS]
    if bad:
        return f"knob {','.join(bad)} はジョブ後に状態を戻せないので載せられない"
    return None


# ---- `tool` ジョブ: 読み込み済みの一式を渡して他の道具を in-process で走らせる

def tool_path(rel: str) -> Path:
    """ジョブが指す道具のパス。相対ならリポジトリ root からたどる。"""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)


def tool_has_entry(path: Path) -> bool:
    """`def run_with_model(` をソースに持つか (import せずにテキストで見る)。

    import は重いうえ副作用がありうるので、**呼べるかどうかの一次判定は
    読むだけで済ませる。**実際に呼ぶ直前に `call_tool` が改めて確かめる。
    """
    try:
        return "def run_with_model(" in path.read_text(errors="replace")
    except OSError:
        return False


def tool_reject_reason(rel: str) -> str | None:
    path = tool_path(rel)
    if not path.is_file():
        return f"{path} が無い"
    if not tool_has_entry(path):
        return (f"{path.name} は `tool` ジョブに未対応"
                " (`def run_with_model(argv, bundle) -> int` を生やすこと。"
                " 規約は tools/ab_bundle.py の docstring)")
    return None


def call_tool(rel: str, argv: list[str], bundle) -> int:
    """道具をモジュールとして読み込み、`run_with_model(argv, bundle)` を呼ぶ。

    毎回読み込み直す (`importlib` のキャッシュを使わない) --- エージェントが
    道具を編集した直後に投げてくるため。モデルを読み直す `code_fingerprint`
    の再起動と違い、こちらは数 ms で済むので毎回でよい。
    """
    import importlib.util

    path = tool_path(rel)
    name = f"_ab_tool_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} を読み込めない")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        fn = getattr(mod, "run_with_model", None)
        if fn is None:
            raise AttributeError(
                f"{path.name} に run_with_model(argv, bundle) が無い")
        rc = fn(list(argv), bundle)
    finally:
        sys.modules.pop(name, None)
    return int(rc or 0)


def ab_env(environ=None) -> dict:
    """突き合わせの対象になる環境変数だけを抜き出す。"""
    src = os.environ if environ is None else environ
    return {k: v for k, v in src.items()
            if k.startswith(ENV_PREFIXES) and k not in ENV_IGNORE}


def env_delta(job_env: dict, launch_env: dict) -> tuple[set, set]:
    """(違うキー全部, そのうち作り直しが要るキー) を返す。

    判定はキー集合と値の一致だけ。片方にしか無いキーも「違う」。
    """
    diff = {k for k in set(job_env) | set(launch_env)
            if job_env.get(k) != launch_env.get(k)}
    return diff, diff - RUNTIME_ENV


def bundle_matches(args, load_args) -> str | None:
    def norm(p):
        return os.path.realpath(os.path.expanduser(p)) if p else None

    for attr in ("model", "ngram", "mtp"):
        want, have = norm(getattr(args, attr)), norm(getattr(load_args, attr))
        if want != have:
            return f"{attr} が違う (daemon={have} job={want})"
    if args.mtp_bits != load_args.mtp_bits:
        return f"mtp-bits が違う (daemon={load_args.mtp_bits} job={args.mtp_bits})"
    return None


# ---- 本体 -----------------------------------------------------------------

def burn_in(decode_ab, load_args, bundle, log=print) -> None:
    """読み込み直後に null の A/B を 1 本捨てる。

    **最初のジョブだけが冷たいのを消すため。**実測 (2026-09-03、null knob
    を 2 本続けて流した): 読み込み直後の 1 本目は、decode_ab 自身の温めを
    済ませてもなお**各ケースの 1 行目 (回文順の先頭 = A) が 8〜9% 遅い**
    (40.5 / 36.7 / 37.2 / 36.8 ms/round)。2 本目は段差が消える
    (36.9 / 36.8 / 36.9 / 36.9)。回文順は線形のドリフトしか相殺できず、
    位置 1 の段差は必ず A に乗る。

    プロセスを分けていた頃は全部のジョブが「1 本目」だったので、この段差は
    全条件に等しく乗っていた。常駐すると 1 本目だけが冷たくなり、**ジョブに
    よって雑音の形が違う**という新しい歪みが生まれる。ここで 1 本焼いて
    全部のジョブを「2 本目以降」に揃える。

    捨てるので出力は読まない (ログにも残さない)。32 トークン x 短文脈 3 本で
    10 秒級。読み込みの biglock を持ったまま走らせる。
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr

    argv = ["--knob", "null", "--only", "short", "--tokens", "32",
            "--model", load_args.model]
    if load_args.ngram:
        argv += ["--ngram", load_args.ngram]
    t0 = time.perf_counter()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            decode_ab.run_with_model(argv, bundle)
    except Exception:
        log("空焼きに失敗した (続行):\n" + traceback.format_exc())
        return
    log(f"空焼き完了 ({time.perf_counter() - t0:.1f}s)。"
        "最初のジョブも 2 本目以降と同じ温度で始まる")


def _cleanup_files() -> None:
    """降りるときに、自分が置いたものだけ片付ける。

    停止の合図は**自分で消す** --- `tools/biglock.sh` は「pid が消えたか」
    しか見ないので、残すと次に立つ daemon が読み込む前に降りてしまう。
    """
    STOP_FILE.unlink(missing_ok=True)
    biglock_release()
    try:
        rec = json.loads(PID_FILE.read_text())
        if int(rec.get("pid", -1)) == os.getpid():
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--idle-exit-s", type=float, default=900.0,
                    help="この秒数ジョブが来なければ終了する (0 = 終了しない)。"
                         "98GB を抱えたまま放置しないための保険")
    ap.add_argument("--yield-after-s", type=float, default=60.0,
                    help="キューが空で、かつ他人が biglock を待って (または持って) "
                         "いる状態がこの秒数続いたら降りる。常駐が他人の計測の"
                         "メモリ待ちを 10 分にしてしまうのを避ける (0 = 降りない)")
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--no-burn-in", action="store_true",
                    help="読み込み直後の空焼きをしない。既定では 32 トークンの "
                         "null A/B を 1 本捨てて、**最初のジョブが冷たい**のを"
                         "防ぐ (この関数の docstring 参照)")
    daemon_args = ap.parse_args()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)

    import decode_ab

    load_argv = ["--knob", "null", "--model", daemon_args.model]
    if daemon_args.ngram:
        load_argv += ["--ngram", daemon_args.ngram]
    if daemon_args.mtp:
        load_argv += ["--mtp", daemon_args.mtp]
    load_argv += ["--mtp-bits", str(daemon_args.mtp_bits)]
    load_args, _ = decode_ab.parse_args(load_argv)

    fp0 = code_fingerprint()
    # **読み込みの前に**控えること。`ab_bundle.load_bundle` が
    # `FASTMLX_NGRAM_DISK=1` を立てるので、読み込み後の env と投入側の env は
    # 必ず食い違い、毎ジョブ読み直しの無限ループになる。
    launch_env = ab_env()
    PID_FILE.write_text(json.dumps({
        "pid": os.getpid(),
        "model": os.path.realpath(os.path.expanduser(daemon_args.model)),
        "ngram": (os.path.realpath(os.path.expanduser(daemon_args.ngram))
                  if daemon_args.ngram else None),
        "mtp": (os.path.realpath(os.path.expanduser(daemon_args.mtp))
                if daemon_args.mtp else None),
        "mtp_bits": daemon_args.mtp_bits,
        "queue": str(QUEUE_DIR),
        "started": time.time(),
        "state": "loading",
    }, ensure_ascii=False, indent=1))

    def log(msg):
        print(f"[ab_daemon {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # 読み込みの間だけは 98GB を 2 本載せないための排他が要る (ジョブの
    # 実行時と違い、ここは本当に 91GB を SSD から読む)。
    log(f"モデルを読む: {daemon_args.model}")
    t0 = time.perf_counter()
    try:
        biglock_acquire(int(os.environ.get("BIGLOCK_PRIO", "1")),
                        mem_need_gb=100, log=log)
        try:
            bundle = decode_ab.load_bundle(load_args)
            if not daemon_args.no_burn_in:
                burn_in(decode_ab, load_args, bundle, log)
        finally:
            biglock_release()
    except StopRequested:
        log("読み込みの待ちの最中に停止の合図が来た。何も読まずに降りる")
        _cleanup_files()
        return 0
    log(f"読み込み完了 ({time.perf_counter() - t0:.1f}s)。キュー: {QUEUE_DIR}")

    base_env = dict(os.environ)
    base_state = snapshot_state(bundle)

    def refresh_pid(state: str, extra: dict | None = None):
        try:
            rec = json.loads(PID_FILE.read_text())
        except Exception:
            return
        rec["state"] = state
        rec.update(extra or {})
        PID_FILE.write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    refresh_pid("idle")
    idle_since = time.time()
    foreign_since = None
    rc = 0
    try:
        while True:
            if STOP_FILE.exists():
                STOP_FILE.unlink(missing_ok=True)
                log("停止の合図を受けた。終了する")
                break
            jobs = read_jobs()
            if not jobs:
                if (daemon_args.idle_exit_s > 0
                        and time.time() - idle_since > daemon_args.idle_exit_s):
                    log("暇なので終了する (98GB を返す)")
                    break
                if daemon_args.yield_after_s > 0:
                    other = foreign_biglock_user(os.getpid())
                    if other is None:
                        foreign_since = None
                    else:
                        foreign_since = foreign_since or time.time()
                        if time.time() - foreign_since > daemon_args.yield_after_s:
                            log(f"他人 (pid={other}) が biglock を待っている。"
                                "98GB を返して降りる")
                            break
                time.sleep(daemon_args.poll_s)
                continue
            idle_since = time.time()
            foreign_since = None

            # **ジョブを始める前にコードの鮮度を見る。**編集直後の A/B を
            # 古いコードで測るくらいなら、3 分払って読み直すほうが正しい。
            diff = fingerprint_diff(fp0, code_fingerprint())
            if diff:
                log(f"コードが変わった ({len(diff)} ファイル: "
                    f"{', '.join(Path(d).name for d in diff[:4])}"
                    f"{' ...' if len(diff) > 4 else ''})。自分を作り直す")
                refresh_pid("restarting")
                biglock_release()
                sys.stdout.flush()
                sys.stderr.flush()
                os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)]
                         + sys.argv[1:])

            prio, _submitted, path, spec = jobs[0]
            kind = spec.get("type") or "decode_ab"
            argv = [str(a) for a in spec.get("argv", [])]
            log_path = Path(spec.get("log") or (path.with_suffix(".log")))
            started = time.time()

            # ---- 受けられるか (モデルを読む前に弾けるものは投入側でも見て
            # いるが、キューに直接置かれた場合もあるのでここでも見る)
            label, reason = kind, None
            if kind == "decode_ab":
                try:
                    job_args, knob_names = decode_ab.parse_args(argv)
                except SystemExit as e:
                    log(f"{path.name}: 引数が不正 ({argv})")
                    log_path.write_text(f"decode_ab の引数が不正: {argv}\n")
                    finish(path, False, dict(code=int(e.code or 2),
                                             error="parse_args", argv=argv))
                    continue
                label = f"decode_ab knob={','.join(knob_names)}"
                reason = (job_reject_reason(job_args, knob_names)
                          or bundle.mismatch(job_args.model, job_args.ngram,
                                             job_args.mtp, job_args.mtp_bits))
            elif kind == "tool":
                label = f"tool {Path(spec['path']).name}"
                reason = tool_reject_reason(spec["path"])
            elif kind == "exec":
                label = "exec " + " ".join(spec["cmd"][:3])
            else:
                reason = f"知らないジョブ種 {kind!r} (使えるのは {JOB_TYPES})"
            if reason:
                log(f"{path.name}: 載せられない ({reason})")
                log_path.write_text(f"常駐 worker には載せられない: {reason}\n")
                finish(path, False, dict(code=3, error="unsupported",
                                         reason=reason, argv=argv))
                continue

            # ---- 環境変数の突き合わせ。読み込み時に効くものが違えば、
            # **その env で自分を作り直してから**測る (3 分は払う価値がある)。
            job_env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()
                       if str(k).startswith(ENV_PREFIXES)
                       and str(k) not in ENV_IGNORE}
            diff, need_reload = env_delta(job_env, launch_env)
            if kind == "exec":
                # 別プロセスなので env はそのまま渡せばよい。**読み直しは要らない**
                # (micro 1 本のために 98GB を読み直すのは割に合わない)。
                need_reload = set()
            if need_reload:
                if os.environ.get(ENV_JOB_MARK) == path.name:
                    # 作り直したのにまだ食い違う。**2 回目は諦める** ---
                    # 読み直しの無限ループのほうが害が大きい。
                    reason = ("env を当てて作り直したのに一致しない: "
                              + ", ".join(sorted(need_reload)))
                    log(f"{path.name}: {reason}")
                    log_path.write_text(reason + "\n")
                    finish(path, False, dict(code=3, error="env", reason=reason,
                                             argv=argv))
                    continue
                log(f"{path.name}: 読み込み時に効く env が違う "
                    f"({', '.join(sorted(need_reload))})。その env で作り直す")
                refresh_pid("restarting", {"job": path.name})
                new_env = {k: v for k, v in os.environ.items()
                           if not (k.startswith(ENV_PREFIXES)
                                   and k not in ENV_IGNORE)}
                new_env.update(job_env)
                new_env[ENV_JOB_MARK] = path.name
                biglock_release()
                sys.stdout.flush()
                sys.stderr.flush()
                os.execve(sys.executable,
                          [sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
                          new_env)

            log(f"開始 {path.name} 段={prio} {label}")
            refresh_pid("running", {"job": path.name})
            try:
                biglock_acquire(prio, mem_need_gb=0, log=log)
            except StopRequested:
                # ロック待ちの列に並んでいる間に「モデルを読む段 0/1」が来た。
                # ジョブはキューに残したまま降りる (ab_submit が立て直す)。
                log("ロック待ちの最中に停止の合図が来た。ジョブは残して降りる")
                refresh_pid("stopping", {"job": None})
                break
            code, err = 0, None
            out_stream = open(log_path, "w", buffering=1, encoding="utf-8")
            try:
                if kind == "exec":
                    # モデルを読まない道具 (micro / verify / smoke / 連鎖)。
                    # **worker はモデルを抱えたまま GPU の番だけ渡す。**
                    # 別プロセスなので bundle は触られず、状態も戻さなくてよい。
                    child_env = {k: v for k, v in os.environ.items()
                                 if not (k.startswith(ENV_PREFIXES)
                                         and k not in ENV_IGNORE)}
                    child_env.update(job_env)
                    child_env.pop(ENV_JOB_MARK, None)
                    code = subprocess.call(
                        [str(c) for c in spec["cmd"]],
                        stdout=out_stream, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, env=child_env,
                        cwd=spec.get("cwd") or str(REPO_ROOT))
                else:
                    if diff:
                        # ここに来るのは RUNTIME_ENV だけの差。**reset_engine
                        # より先に当てる** -- DepthController は reset で
                        # 作り直され、そのとき BETA/EXPLORE/MARGIN を読む。
                        for k in diff:
                            if k in job_env:
                                os.environ[k] = job_env[k]
                            else:
                                os.environ.pop(k, None)
                        log(f"env を当てた (実行中に効くぶんだけ): "
                            f"{', '.join(sorted(diff))}")
                    old_out, old_err = sys.stdout, sys.stderr
                    try:
                        reset_engine(bundle)
                        sys.stdout = sys.stderr = out_stream
                        if kind == "decode_ab":
                            code = decode_ab.run_with_model(argv, bundle)
                        else:
                            code = call_tool(spec["path"], argv, bundle)
                    finally:
                        sys.stdout, sys.stderr = old_out, old_err
            except BaseException:  # noqa: BLE001  Metal のエラーも含めて拾う
                err = traceback.format_exc()
                code = 1
                out_stream.write("\n" + err)
            finally:
                out_stream.close()
                # 状態を戻すのは**ロックを持ったまま**。融合の貼り直しは
                # モデルの属性を触るので、別プロセスが GPU を使い始める前に
                # 済ませる。exec は別プロセスなので戻すものが無い。
                try:
                    if kind != "exec":
                        os.environ.clear()
                        os.environ.update(base_env)
                        changed = restore_state(bundle, base_state, log=log)
                        reset_engine(bundle)
                        if changed:
                            log(f"戻した差し替え点 {len(changed)}: "
                                f"{', '.join(changed[:6])}"
                                f"{' ...' if len(changed) > 6 else ''}")
                except Exception:
                    log("状態を戻せなかった。安全のため終了する\n"
                        + traceback.format_exc())
                    err = err or traceback.format_exc()
                    code = code or 1
                    biglock_release()
                    finish(path, False, dict(code=code, error="restore",
                                             traceback=err, argv=argv))
                    rc = 1
                    break
                biglock_release()
            dt = time.time() - started
            finish(path, code == 0 and err is None,
                   dict(code=code, seconds=dt, type=kind, argv=argv,
                        **({"cmd": spec["cmd"]} if kind == "exec" else {}),
                        **({"traceback": err} if err else {})))
            refresh_pid("idle", {"job": None})
            log(f"完了 {path.name} code={code} {dt:.1f}s")
            if err is not None and kind != "exec":
                # 計測の途中で落ちた後の状態は信用できない (Metal の
                # コンテキストが壊れている可能性がある)。**次のジョブを
                # 汚さないよう、ここで降りる。**投入側が立て直す。
                # exec は別プロセスなので、落ちても worker は無傷。
                log("ジョブが落ちた。状態が信用できないので終了する")
                rc = 1
                break
            idle_since = time.time()
    except KeyboardInterrupt:
        log("中断された")
        rc = 130
    finally:
        _cleanup_files()
    return rc


def _reexec_absolute() -> None:
    """自分のコマンドラインを絶対パスにしてから走り直す。

    `tools/biglock.sh` の「ロックを持たない相手」検出は
    `pgrep -f "\\.venv/bin/python3? (tools|bench)/"` で、**相対パスで起動
    した自分がこれに一致する。**新しい biglock.sh は名前で外しているが、
    **既に走っている待ち手は古い写しを読み進めているので書き換えでは
    直せない** (だから mv で入れ替える規約になっている)。絶対パスで
    起動し直せば、古い写しから見ても一致しなくなる。

    実際にこれで 3 本の待ち手が止まった (2026-09-03、ab_daemon の初回)。
    """
    if os.path.isabs(sys.argv[0]):
        return
    os.execv(sys.executable,
             [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


if __name__ == "__main__":
    _reexec_absolute()
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # `tools/decode_ab.py` の末尾と同じ理由: interpreter shutdown を待つ間
    # 98GB を握ったまま残られた実測がある。**降りると言った以上は即座に
    # 返す** --- biglock の段 0/1 はこの pid が消えるのを待っている。
    os._exit(_rc)
