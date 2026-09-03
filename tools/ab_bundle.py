"""読み込み済みのモデル一式 (`AbBundle`) と、それを作る `load_bundle`。

98GB のモデルを SSD から読むのは 1 回 3 分かかる。**同じモデルを何度も測る
道具のあいだで、その 1 回を共有するための入れ物**がこれ。常駐 worker
(`tools/ab_daemon.py`) が起動時に 1 つ作り、ジョブごとに道具へ渡す。

## 道具側の約束 (`tool` ジョブ)

worker に `{"type": "tool", "path": "tools/<name>.py", "argv": [...]}` を
投げると、worker はそのファイルをモジュールとして読み込み

    def run_with_model(argv: list[str], bundle: AbBundle) -> int

を呼ぶ。**この 1 つの関数さえ生やせば、その道具は 98GB の読み直し無しで
worker の列に乗る。**規約は 4 つ:

1. `argv` は自分の CLI の引数そのまま (`sys.argv[1:]` 相当)。自前の
   argparse をそのまま使ってよい。モデルのパスを指す引数は
   `bundle.model_path` と食い違っていないか確かめること (worker は
   1 つのモデルしか抱えていない)。
2. 返り値は終了コード (0 = 成功)。例外はそのまま上げてよい (worker が
   ジョブの `.err` に記録する)。
3. **`bundle` の中身は借り物。**差し替え (monkeypatch)、`fused.enable_*` /
   `disable_*`、engine の属性の書き換えをしたら、**終わりに戻すこと。**
   worker 側にも戻しの網はあるが (`ab_daemon.restore_state`)、あれは
   `tools/decode_ab.py` の knob が触る場所を列挙した保険であって、
   道具が触る場所まで知らない。
4. `os._exit()` を呼ばない。プロセスは worker のもので、落とすと 98GB の
   読み直しになる。

## フィールド

| 名前 | 意味 |
|---|---|
| `model` | `mlx_lm.load` が返したモデル。**出荷経路の融合 (`enable_default_fusions`) と wired limit を当て済み。** |
| `tokenizer` | `mlx_lm.load` が返した tokenizer (`TokenizerWrapper`) |
| `config` | `<model_path>/config.json` を読んだ dict (無ければ空 dict) |
| `mtp` | `mtp_flash.load_flash_mtp` が返した MTP head。`load_mtp=False` なら None |
| `engine` | `spec_flash.FlashSpecEngine(model, mtp)`。`mtp` が None なら None |
| `eos_ids` | `tuple[int, ...]`。tokenizer が持っていなければ空タプル |
| `model_path` | 展開・realpath 済みのモデルのパス (str) |
| `ngram_path` | 同上、n-gram サイドカー。`--ngram` 無しなら None |
| `mtp_path` | 同上、MTP の safetensors。既定は `<model_path>/mtp.safetensors` |
| `mtp_bits` | MTP の量子化ビット数 (0 なら量子化しない) |
| `ngram_streams` | PLE 層に配られている n-gram の実体 (重複を除いた list)。`--ngram` 無しなら空 |
| `loaded_at` | 読み終えた時刻 (`time.time()`) |

`model` と `tokenizer` は `mlx_lm.load` の返り値そのままなので、既存の道具は
`model, tok = load(path)` を `bundle.model, bundle.tokenizer` に置き換える
だけで動く。**融合が当たっているぶん、素の `load()` とは挙動が違う**
(それが本番の状態なので、計測はこちらが正しい)。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _abspath(p: str | None) -> str | None:
    return os.path.realpath(os.path.expanduser(p)) if p else None


@dataclass
class AbBundle:
    """読み込み済みの一式。フィールドの意味はモジュール docstring の表。"""

    model: Any
    tokenizer: Any
    config: dict
    mtp: Any
    engine: Any
    eos_ids: tuple[int, ...]
    model_path: str
    ngram_path: str | None
    mtp_path: str | None
    mtp_bits: int
    ngram_streams: list = field(default_factory=list)
    loaded_at: float = 0.0

    def identity(self) -> tuple:
        """「同じモデルを抱えているか」の判定に使う組。"""
        return (self.model_path, self.ngram_path, self.mtp_path, self.mtp_bits)

    def mismatch(self, model_path=None, ngram_path=None, mtp_path=None,
                 mtp_bits=None) -> str | None:
        """指定と食い違っていれば理由を、合っていれば None を返す。

        `model_path` と `ngram_path` は**未指定も突き合わせる** --- 「n-gram
        無しで測るつもりのジョブ」を n-gram 入りの worker に載せたら、
        黙って別の構成を測ることになる。`mtp_path` だけは未指定 (None) を
        「既定 = `<model>/mtp.safetensors`」の意味で読み飛ばす (worker 側は
        その既定を解決済みのパスで持っているため)。
        """
        want_model = _abspath(model_path) if model_path else self.model_path
        if want_model != self.model_path:
            return f"model が違う (worker={self.model_path} job={want_model})"
        want_ngram = _abspath(ngram_path)
        if want_ngram != self.ngram_path:
            return f"ngram が違う (worker={self.ngram_path} job={want_ngram})"
        if mtp_path is not None and _abspath(mtp_path) != self.mtp_path:
            return f"mtp が違う (worker={self.mtp_path} job={_abspath(mtp_path)})"
        if mtp_bits is not None and mtp_bits != self.mtp_bits:
            return f"mtp-bits が違う (worker={self.mtp_bits} job={mtp_bits})"
        return None

    # 旧 dict 形式との橋 (`bundle["eng"]` で書かれた呼び出しを壊さない)。
    # 新しく書くコードは属性で引くこと。
    _ALIASES = {"eng": "engine", "tok": "tokenizer"}

    def __getitem__(self, key: str):
        return getattr(self, self._ALIASES.get(key, key))


def load_bundle(model_path: str, ngram_path: str | None = None,
                mtp_path: str | None = None, mtp_bits: int = 4,
                load_mtp: bool = True, log_prefix: str = "[ab_bundle]") -> AbBundle:
    """98GB のモデル / n-gram / MTP を読み、`AbBundle` にして返す。

    **読み込み時にしか読まれない環境変数はここより前に立てること。**
    `FASTMLX_NGRAM_DISK` はこの関数が `ngram_path` から立てる
    (`mlxturbo/_vendor/qwen4_exp.py` の import 時に評価される) が、
    `MLXTURBO_ROUND_TRACE` / `MLXTURBO_DRAFT_TRACE` は `spec_flash` の
    import 時なので、**呼ぶ側の責任**。

    出荷経路と同じ状態にして返す:

    - `enable_default_fusions` (env で切り替わる融合の既定はここが唯一の
      出どころ。以前 decode_ab が `enable_hyper_connection_kernel()` だけを
      呼んでいて、gather のソートが入らないまま測っていた)
    - `set_wired_limit_default` (engine を直叩きする道具は mlx_lm の
      `wired_limit()` を通らない。wire しないと macOS がページを退避・圧縮
      でき、読み出しが劣化する)
    """
    model_path = _abspath(model_path)
    ngram_path = _abspath(ngram_path)
    if ngram_path:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401  (arch の登録を先に通す)
    from mlxturbo import mtp_flash, spec_flash
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    model, tok = load(model_path)
    if ngram_path:
        from mlxturbo.ngram_stream import install

        install(model, ngram_path)

    enable_default_fusions(model, log_prefix=log_prefix)
    set_wired_limit_default(log_prefix=log_prefix)

    mtp = engine = None
    resolved_mtp = _abspath(mtp_path or os.path.join(model_path, "mtp.safetensors"))
    if load_mtp:
        q = {"group_size": 64, "bits": mtp_bits} if mtp_bits else None
        mtp = mtp_flash.load_flash_mtp(resolved_mtp, model.args.text, quantize=q)
        mx.eval(mtp.parameters())
        engine = spec_flash.FlashSpecEngine(model, mtp)

    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    cfg_path = Path(model_path) / "config.json"
    try:
        config = json.loads(cfg_path.read_text())
    except Exception:
        config = {}

    return AbBundle(
        model=model,
        tokenizer=tok,
        config=config,
        mtp=mtp,
        engine=engine,
        eos_ids=tuple(eos) if eos else (),
        model_path=model_path,
        ngram_path=ngram_path,
        mtp_path=resolved_mtp,
        mtp_bits=mtp_bits,
        ngram_streams=ngram_streams(model),
        loaded_at=time.time(),
    )


def ngram_streams(model) -> list:
    """PLE 層に配られている n-gram の実体を重複無しで集める。

    `mlxturbo.ngram_stream.install` は PLE 層全部に同じ 1 インスタンスを
    配るので普通は 1 個だが、そう決め打ちにはしない。
    """
    out, seen = [], set()
    for layer in getattr(getattr(model, "model", None), "layers", []):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
        if emb is not None and id(emb) not in seen:
            seen.add(id(emb))
            out.append(emb)
    return out
