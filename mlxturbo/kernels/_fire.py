"""カーネルが実際に発火したかを数える。

適格判定 (`eligible`) は条件を外すと黙って False を返すので、有効化した
つもりで一度も走っていない、という空振りが起きる。2026-09-01 に GDN 前処理で
実際に起きた: `spec_flash.capture()` が `GatedDeltaNet.__call__` ごと差し替える
ため、投機の検証フォワードで融合が一度も呼ばれていなかった。runner は
「有効」と表示するが、それは `enable` したときに出るだけで発火の証拠ではない。

A/B の前後で `snapshot()` を取れば、効果ゼロが「カーネルが遅い」なのか
「届いていない」なのかを区別できる。
"""

from __future__ import annotations

from collections import Counter

_COUNTS: Counter[str] = Counter()


def bump(name: str, n: int = 1) -> None:
    """`name` のカーネルが発火した回数を足す。"""
    _COUNTS[name] += n


def snapshot() -> dict[str, int]:
    """現在の発火回数を返す。"""
    return dict(_COUNTS)


def reset() -> None:
    """数え直す。A/B の条件ごとの頭で呼ぶ。"""
    _COUNTS.clear()
