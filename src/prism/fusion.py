"""Deterministic score fusion of candidate lists."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .retrievers import Candidate


def rrf(lists: list[list[Candidate]], weights: list[float] | None = None, k: int = 60) -> dict[str, float]:
    weights = weights or [1.0] * len(lists)
    out: dict[str, float] = defaultdict(float)
    for w, cands in zip(weights, lists):
        for c in cands:
            out[c.snippet_id] += w / (k + c.rank)
    return dict(out)


def zscore(lists: list[list[Candidate]], weights: list[float] | None = None) -> dict[str, float]:
    """Weighted sum of per-list z-normalised scores; a doc missing from a list gets that list's min."""
    weights = weights or [1.0] * len(lists)
    normed = []
    for cands in lists:
        s = np.array([c.score for c in cands], dtype=float)
        mu, sd = s.mean(), s.std() or 1.0
        normed.append(({c.snippet_id: (c.score - mu) / sd for c in cands}, (s.min() - mu) / sd))
    ids = {c.snippet_id for cands in lists for c in cands}
    return {d: sum(w * m.get(d, floor) for w, (m, floor) in zip(weights, normed)) for d in ids}
