"""長文脈ベンチのプロンプト素材。

繰り返し文字列で長さを作ってはいけない -- n-gram と MTP が当てすぎて受理率が
嘘になる。リポジトリ内の実文 (docs の Markdown + 自前のソース) を並べて池を
作り、互いに重ならない窓を切る。日本語の散文と英語のコードが混ざるので、
コーディング支援の実利用に近い分布になる。

池の大きさは約 2.2MB = 145 万トークン相当。100k の窓を 3 本切っても足りる。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def text_pool() -> str:
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


def long_prompts(tok, ctx: int, questions: list[str]) -> list[str]:
    """互いに重ならない窓を切って、末尾に質問を付けたプロンプトを返す。

    足りなければ ValueError。**足りないまま繰り返しで埋めない** (受理率が
    嘘になる)。
    """
    ids = tok.encode(text_pool())
    win = max(ctx - 200, 16)  # 質問文とテンプレートのぶんを空ける
    need = win * len(questions)
    if need > len(ids):
        raise ValueError(
            f"素材が足りない (必要 {need} tok, 手元 {len(ids)} tok)。"
            "窓を減らすか ctx を下げること"
        )
    out = []
    for i, q in enumerate(questions):
        body = tok.decode(ids[i * win : (i + 1) * win])
        out.append(f"{body}\n\n---\n\n{q}")
    return out
