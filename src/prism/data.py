"""CoIR AppsRetrieval loader for development.

Loads the same pinned revision the official MTEB task uses. Dev work uses the
*train* queries against the full 8,765-doc corpus (the same corpus the test
split is scored against). Test qrels are only read by the frozen final run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

from datasets import load_dataset

REPO = "CoIR-Retrieval/apps"
REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"  # == mteb AppsRetrieval pin


@dataclass(frozen=True)
class Split:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]


@lru_cache(maxsize=None)
def _raw():
    corpus = load_dataset(REPO, "corpus", revision=REVISION, split="corpus")
    queries = load_dataset(REPO, "queries", revision=REVISION, split="queries")
    qrels = load_dataset(REPO, "default", revision=REVISION)
    return corpus, queries, qrels


def load_split(split: str = "train") -> Split:
    if split not in ("train", "test"):
        raise ValueError(split)
    corpus, queries, qrels_ds = _raw()
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds[split]:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qtext = {str(r["_id"]): r["text"] for r in queries}
    qids = sorted(qrels, key=_numeric_key)
    return Split(
        doc_ids=[str(i) for i in corpus["_id"]],
        doc_texts=list(corpus["text"]),
        query_ids=qids,
        query_texts=[qtext[q] for q in qids],
        qrels=qrels,
    )


def _numeric_key(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)


def dev_folds(query_ids: list[str], n_folds: int = 5) -> list[list[str]]:
    """Deterministic folds by hashing query id (stable across runs/machines)."""
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for q in query_ids:
        h = int(hashlib.sha256(q.encode()).hexdigest(), 16)
        folds[h % n_folds].append(q)
    return folds
