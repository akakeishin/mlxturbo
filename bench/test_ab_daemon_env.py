"""常駐 worker の env 突き合わせ (`tools/ab_daemon.py` の `ab_env` / `env_delta`)。

的は 1 つ: **worker が自分で立てるキー (`SELF_SET_ENV`) で読み直しを起こさない
こと。**`FASTMLX_NGRAM_DISK=1` は `--ngram` 付きの読み込みで worker 自身が
立てるので、素のシェルから投げたジョブとは永久に一致せず、ジョブごとに 98GB を
読み直していた (2026-09-04)。モデルを読まない (数 ms で終わる)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import ab_daemon as AD


def test_ab_env_drops_self_set_keys():
    env = {"FASTMLX_NGRAM_DISK": "1", "MLXTURBO_QSA_TAIL": "query",
           "PATH": "/usr/bin"}
    assert AD.ab_env(env) == {"MLXTURBO_QSA_TAIL": "query"}


def test_ab_env_still_drops_plumbing_and_keeps_prefixed():
    env = {"MLXTURBO_AB_QUEUE": "/tmp/q", "MLXTURBO_MIN_FREE_GB": "8",
           "MLXTURBO_DEPTH_ADAPT": "0"}
    assert AD.ab_env(env) == {"MLXTURBO_DEPTH_ADAPT": "0"}


def test_env_delta_ignores_self_set_key_in_either_direction():
    # worker だけが持っている (`os.execv` / Popen の継承で読み込み後の env を
    # 引き継いだ場合)
    job = AD.ab_env({})
    launch = AD.ab_env({"FASTMLX_NGRAM_DISK": "1"})
    assert AD.env_delta(job, launch) == (set(), set())
    # ジョブだけが持っている (回避策で投入時に明示した場合)
    assert AD.env_delta(launch, job) == (set(), set())


def test_env_delta_still_reloads_on_a_load_time_key():
    job = AD.ab_env({"MLXTURBO_DEPTH_ADAPT": "0", "FASTMLX_NGRAM_DISK": "1"})
    launch = AD.ab_env({})
    diff, need_reload = AD.env_delta(job, launch)
    assert diff == {"MLXTURBO_DEPTH_ADAPT"}
    assert need_reload == {"MLXTURBO_DEPTH_ADAPT"}


def test_env_delta_runtime_key_differs_without_reload():
    job = AD.ab_env({"MLXTURBO_PIPELINE": "1"})
    launch = AD.ab_env({})
    diff, need_reload = AD.env_delta(job, launch)
    assert diff == {"MLXTURBO_PIPELINE"}
    assert need_reload == set()


def test_mem_need_matches_biglock():
    """起動時の読み込みで待つ量は `tools/biglock.sh` の段 0/1 と同じ値。"""
    src = (Path(__file__).resolve().parent.parent / "tools" / "biglock.sh").read_text()
    assert f'MEM_NEED_GB="${{MLXTURBO_MIN_FREE_GB:-{AD.MEM_NEED_GB}}}"' in src
