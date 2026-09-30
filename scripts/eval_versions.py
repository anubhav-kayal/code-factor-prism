"""Evaluate version-aware retrieval on a labelled repo query set.

    python scripts/eval_versions.py eval/express_version_queries.json

Requires the repo to be indexed first (scripts/index_repo.py). For each label
the gold item is `symbol`'s content at `commit`. Two modes are scored:

  history   search across all indexed commits (one result per distinct content).
            rank of the gold version; "version-correct" means the gold version is
            ranked above every other version of the same symbol.
  snapshot  search restricted to the labelled commit's snapshot.
            rank of the symbol (only one version exists in a snapshot).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index_repo import make_encoder  # noqa: E402
from prism.cache import EmbeddingStore  # noqa: E402
from prism.versioning.indexer import git  # noqa: E402
from prism.versioning.search import SnapshotSearcher, search_history  # noqa: E402
from prism.versioning.store import SnapshotStore  # noqa: E402


def main(label_path: str, model: str = "e5-base-v2", k: int = 50) -> None:
    spec = json.loads(Path(label_path).read_text())
    name = spec["repo"]
    repo = ROOT / "data" / "repos" / name
    idx = ROOT / "cache" / "repos" / name
    recipe, embed_docs, embed_query = make_encoder(model)
    store = SnapshotStore(idx / "index.sqlite")
    emb = EmbeddingStore(recipe, root=idx / "emb")
    searcher = SnapshotSearcher(store, emb, embed_query, embed_docs)
    embed_query("warm up")

    rows = []
    for lab in spec["queries"]:
        commit = git(repo, "rev-parse", lab["commit"]).strip()
        sid = store.snapshot_id_for(name, commit)
        if sid is None:
            sys.exit(f"{lab['id']}: commit {commit[:10]} not indexed")
        gold = [o for o in store.occurrences(sid) if o["symbol"] == lab["symbol"]]
        if len(gold) != 1:
            sys.exit(f"{lab['id']}: expected one {lab['symbol']} at {commit[:10]}, found {len(gold)}")
        gold_ck = gold[0]["content_key"]

        t = time.perf_counter()
        hist = search_history(store, emb, name, lab["query"], embed_query, embed_docs, k=k)
        hist_ms = (time.perf_counter() - t) * 1000
        h_rank = next((h["rank"] for h in hist if h["content_key"] == gold_ck), None)
        other_ranks = [h["rank"] for h in hist if h["symbol"] == lab["symbol"] and h["content_key"] != gold_ck]
        n_versions = next((h["n_versions_of_symbol"] for h in hist if h["content_key"] == gold_ck), None)

        t = time.perf_counter()
        snap = searcher.retrieve(lab["query"], sid, k=k)
        snap_ms = (time.perf_counter() - t) * 1000
        s_rank = next((h.rank for h in snap if h.symbol == lab["symbol"]), None)

        rows.append({
            "id": lab["id"], "symbol": lab["symbol"], "commit": commit[:10],
            "history_rank": h_rank,
            "version_correct": h_rank is not None and all(h_rank < r for r in other_ranks),
            "n_versions_of_symbol": n_versions,
            "snapshot_rank": s_rank,
            "history_ms": round(hist_ms, 1), "snapshot_ms": round(snap_ms, 1),
        })

    def mrr(ranks):
        return float(np.mean([1 / r if r else 0 for r in ranks]))

    h = [r["history_rank"] for r in rows]
    s = [r["snapshot_rank"] for r in rows]
    summary = {
        "labels": len(rows), "model": recipe["model"],
        "history": {"mrr": round(mrr(h), 4), "hit@1": sum(r == 1 for r in h), "hit@5": sum(bool(r and r <= 5) for r in h),
                    "version_correct": sum(r["version_correct"] for r in rows),
                    "p50_ms": float(np.median([r["history_ms"] for r in rows]))},
        "snapshot": {"mrr": round(mrr(s), 4), "hit@1": sum(r == 1 for r in s), "hit@5": sum(bool(r and r <= 5) for r in s),
                     "p50_ms": float(np.median([r["snapshot_ms"] for r in rows]))},
    }
    out = ROOT / "reports" / f"versions-{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_query.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in rows:
        print(f"{r['id']:<18} hist#{r['history_rank']!s:<4} version_ok={r['version_correct']!s:<5} "
              f"({r['n_versions_of_symbol']} versions)  snap#{r['snapshot_rank']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
