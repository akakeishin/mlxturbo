"""長文脈ベンチのプロンプト素材。

繰り返し文字列で長さを作ってはいけない -- n-gram と MTP が当てすぎて受理率が
嘘になる。リポジトリ内の実文 (docs の Markdown + 自前のソース) を並べて池を
作り、互いに重ならない窓を切る。日本語の散文と英語のコードが混ざるので、
コーディング支援の実利用に近い分布になる。

既定では凍結ファイル `bench/textpool-frozen.txt` を読む (無ければ動的に池を
組む)。セッション中に docs を編集すると、動的な池は同じトークン位置の窓の
中身が変わってしまい、走行をまたぐ比較 (別プロセスの A/B、小ベンチどうし)
の prompt が別物になる (2026-09-03 に実際に発生)。凍結ファイルはこれを防ぐ。
パスは環境変数 `FASTMLX_TEXTPOOL` で差し替えられる。

池の大きさは凍結時点で約 5.1MB。100k の窓を 3 本切っても足りる。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FROZEN_PATH = REPO_ROOT / "bench" / "textpool-frozen.txt"


def _frozen_path() -> Path:
    override = os.environ.get("FASTMLX_TEXTPOOL")
    return Path(override) if override else DEFAULT_FROZEN_PATH


def _build_text_pool() -> str:
    """docs + 自前ソースを連結して池を動的に組む (凍結ファイルが無いときの経路)。"""
    files = (
        sorted(REPO_ROOT.glob("docs/**/*.md"))
        + [REPO_ROOT / "README.md"]
        + sorted(REPO_ROOT.glob("mlxturbo/*.py"))
        + sorted(REPO_ROOT.glob("mlxturbo/kernels/*.py"))
        + sorted(REPO_ROOT.glob("mlxturbo/_vendor/*.py"))
        + sorted(REPO_ROOT.glob("tools/*.py"))
        + sorted(REPO_ROOT.glob("bench/*.py"))
    )
    return "\n\n".join(f.read_text(errors="ignore") for f in files if f.exists())


def text_pool() -> str:
    path = _frozen_path()
    if path.exists():
        print(f"[_bench_text] 凍結プールを使用: {path}", file=sys.stderr)
        return path.read_text(encoding="utf-8", errors="ignore")
    return _build_text_pool()


def long_prompts(
    tok, ctx: int, questions: list[str], offset_tokens: int = 0
) -> list[str]:
    """互いに重ならない窓を切って、末尾に質問を付けたプロンプトを返す。

    `offset_tokens` を足すと、池の先頭からその分だけずらした位置から窓を
    切り出す (呼び出しをまたいで重ならない窓を作るのに使う)。

    足りなければ ValueError。**足りないまま繰り返しで埋めない** (受理率が
    嘘になる)。
    """
    ids = tok.encode(text_pool())
    win = max(ctx - 200, 16)  # 質問文とテンプレートのぶんを空ける
    need = offset_tokens + win * len(questions)
    if need > len(ids):
        raise ValueError(
            f"素材が足りない (必要 {need} tok [offset {offset_tokens} 込み], "
            f"手元 {len(ids)} tok)。窓を減らすか ctx を下げること"
        )
    out = []
    for i, q in enumerate(questions):
        lo = offset_tokens + i * win
        body = tok.decode(ids[lo : lo + win])
        out.append(f"{body}\n\n---\n\n{q}")
    return out


def _freeze() -> None:
    """`bench/textpool-frozen.txt` を今の動的な池の中身で作り直す。

    **走行をまたぐ比較の基準を取り直すとき以外はやらない。** 凍結ファイルを
    作り直すと、既存の凍結ファイルを前提にした窓の中身が変わり、それより前
    に取った A/B やベンチ結果と比較できなくなる。
    """
    path = _frozen_path()
    text = _build_text_pool()
    path.write_text(text, encoding="utf-8")
    n = len(text.encode("utf-8"))
    print(f"[_bench_text] 凍結し直した: {path} ({n:,} bytes)", file=sys.stderr)


if __name__ == "__main__":
    if "--freeze" in sys.argv[1:]:
        _freeze()
    else:
        print(__doc__, file=sys.stderr)
        print("使い方: python -m tools._bench_text --freeze", file=sys.stderr)
        sys.exit(1)
