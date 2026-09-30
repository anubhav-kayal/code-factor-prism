"""retrieve(query, snapshot_id, k): search restricted to one snapshot's snippets,
plus history search across all snapshots (evolutionary retrieval)."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..cache import EmbeddingStore
from ..fusion import rrf
from ..retrievers import BM25Retriever, Candidate
from .store import SnapshotStore

EmbedQuery = Callable[[str], np.ndarray]


@dataclass
class Hit:
    snippet_id: str  # occ_key
    rank: int
    score: float
    path: str
    symbol: str
    start_line: int
    end_line: int
    content_key: str
    code: str


def _doc_text(o: dict) -> str:
    # BM25 sees path + symbol too; the dense side embeds code only (content-addressed)
    return f"{o['path']} {o['symbol']}\n{o['code']}"


class SnapshotSearcher:
    def __init__(self, store: SnapshotStore, emb_store: EmbeddingStore, embed_query: EmbedQuery,
                 embed_docs: Callable[[list[str]], np.ndarray]) -> None:
        self.store, self.emb, self.embed_query, self.embed_docs = store, emb_store, embed_query, embed_docs
        self._cache: dict[str, tuple] = {}

    def _prepare(self, snapshot_id: str):
        if snapshot_id not in self._cache:
            occs = self.store.occurrences(snapshot_id)
            ids = [o["occ_key"] for o in occs]
            vecs, _ = self.emb.get_or_compute([o["code"] for o in occs], self.embed_docs)
            bm25 = BM25Retriever()
            bm25.index(ids, [_doc_text(o) for o in occs])
            self._cache[snapshot_id] = (occs, ids, vecs, bm25)
        return self._cache[snapshot_id]

    def retrieve(self, query: str, snapshot_id: str, k: int = 10, depth: int = 100) -> list[Hit]:
        occs, ids, vecs, bm25 = self._prepare(snapshot_id)
        if not ids:
            return []
        q = self.embed_query(query)
        sims = vecs @ q
        order = np.argsort(-sims, kind="stable")[:depth]
        dense = [Candidate(ids[i], "dense", r + 1, float(sims[i])) for r, i in enumerate(order)]
        lexical = bm25.search([query], min(depth, len(ids)))[0]
        fused = rrf([dense, lexical])
        by_id = {o["occ_key"]: o for o in occs}
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [
            Hit(sid, r + 1, s, by_id[sid]["path"], by_id[sid]["symbol"], by_id[sid]["start_line"],
                by_id[sid]["end_line"], by_id[sid]["content_key"], by_id[sid]["code"])
            for r, (sid, s) in enumerate(ranked)
        ]


def search_history(store: SnapshotStore, emb_store: EmbeddingStore, repo: str, query: str,
                   embed_query: EmbedQuery, embed_docs, k: int = 10,
                   commit_range: tuple[int, int] | None = None) -> list[dict]:
    """Search every version of every snippet.

    Distinct content_keys are ranked (a snippet unchanged across 50 commits is one
    result, annotated with the commits it lives in). For each hit we attach the
    diff against the nearest other version of the same path::symbol, which is what
    distinguishes near-identical implementations from different commits.
    `commit_range` optionally restricts to snapshot seq numbers [lo, hi].
    """
    rows = store.history(repo)
    if commit_range:
        lo, hi = commit_range
        rows = [r for r in rows if lo <= r["seq"] <= hi]
    by_content: dict[str, dict] = {}
    for r in rows:
        e = by_content.setdefault(r["content_key"], {**r, "commits": [], "seqs": []})
        e["commits"].append(r["commit"])
        e["seqs"].append(r["seq"])
    entries = list(by_content.values())
    if not entries:
        return []
    vecs, _ = emb_store.get_or_compute([e["code"] for e in entries], embed_docs)
    q = embed_query(query)
    sims = vecs @ q
    dense = [Candidate(entries[i]["content_key"], "dense", r + 1, float(sims[i]))
             for r, i in enumerate(np.argsort(-sims, kind="stable")[:200])]
    bm25 = BM25Retriever()
    bm25.index([e["content_key"] for e in entries], [_doc_text(e) for e in entries])
    lexical = bm25.search([query], min(200, len(entries)))[0]
    fused = sorted(rrf([dense, lexical]).items(), key=lambda x: (-x[1], x[0]))[:k]

    versions: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        versions.setdefault((e["path"], e["symbol"]), []).append(e)
    out = []
    for rank, (ck, score) in enumerate(fused, 1):
        e = by_content[ck]
        siblings = sorted((v for v in versions[(e["path"], e["symbol"])] if v["content_key"] != ck),
                          key=lambda v: abs(min(v["seqs"]) - min(e["seqs"])))
        diff = None
        if siblings:
            other = siblings[0]
            older, newer = (other, e) if min(other["seqs"]) < min(e["seqs"]) else (e, other)
            diff = "".join(difflib.unified_diff(
                older["code"].splitlines(True), newer["code"].splitlines(True),
                fromfile=f"{older['path']}@{older['commits'][0][:8]}",
                tofile=f"{newer['path']}@{newer['commits'][0][:8]}", n=1))
        out.append({
            "rank": rank, "score": score, "content_key": ck, "path": e["path"], "symbol": e["symbol"],
            "first_commit": e["commits"][0], "last_commit": e["commits"][-1],
            "n_commits": len(e["commits"]), "n_versions_of_symbol": len(versions[(e["path"], e["symbol"])]),
            "code": e["code"], "diff_vs_nearest_version": diff,
        })
    return out
