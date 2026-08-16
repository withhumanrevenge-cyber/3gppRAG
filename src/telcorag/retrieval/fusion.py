"""Reciprocal rank fusion.

BM25 scores and cosine similarities are not comparable quantities -- BM25 is
unbounded and corpus-dependent, cosine is bounded and model-dependent -- so
combining them by weighted sum requires per-corpus normalisation that silently
drifts as the corpus changes. RRF discards the magnitudes and fuses on rank
position only, which is why it stays stable when specs are added or the
embedding backend is swapped.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            fused[int(doc_id)] += weight / (k + rank + 1)
    return dict(fused)


def top_n(fused: dict[int, float], n: int) -> list[tuple[int, float]]:
    return sorted(fused.items(), key=lambda kv: -kv[1])[:n]
