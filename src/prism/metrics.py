"""Ranking metrics, matching MTEB's definitions.

NDCG@k uses pytrec_eval (as MTEB does). MRR@k = mean over queries of 1/rank of
the first relevant doc if it is within the top k, else 0.
"""

from __future__ import annotations

import numpy as np
import pytrec_eval

Run = dict[str, dict[str, float]]  # qid -> {doc_id: score}
Qrels = dict[str, dict[str, int]]

KS = (10, 50, 100, 200)


def gold_ranks(run: Run, qrels: Qrels) -> dict[str, int | None]:
    """1-based rank of the best-ranked relevant doc per query (None if not retrieved)."""
    out: dict[str, int | None] = {}
    for q, rel in qrels.items():
        ranked = sorted(run.get(q, {}).items(), key=lambda x: (-x[1], x[0]))
        out[q] = next((i + 1 for i, (d, _) in enumerate(ranked) if rel.get(d, 0) > 0), None)
    return out


def per_query_ndcg10(run: Run, qrels: Qrels) -> dict[str, float]:
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    res = ev.evaluate({q: run.get(q, {}) for q in qrels})
    return {q: res[q]["ndcg_cut_10"] for q in qrels}


def aggregate(run: Run, qrels: Qrels) -> dict[str, float]:
    ranks = gold_ranks(run, qrels)
    ndcg = per_query_ndcg10(run, qrels)
    r = np.array([x if x is not None else np.inf for x in ranks.values()], dtype=float)
    out = {
        "n_queries": len(qrels),
        "ndcg_at_10": float(np.mean(list(ndcg.values()))),
        "mrr_at_10": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0))),
        "mrr_at_100": float(np.mean(np.where(r <= 100, 1.0 / r, 0.0))),
    }
    for k in (1, 5) + KS:
        out[f"recall_at_{k}"] = float(np.mean(r <= k))
    return out
