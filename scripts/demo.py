"""Live retrieval demo (ranked results + latency).

AppsRetrieval corpus:
    python scripts/demo.py apps --config configs/dev/e5_bm25_rrf.json "count pairs whose sum is divisible by k"
    python scripts/demo.py apps --config ... --qid q1234          # a train query, shows where the gold lands

Repository (after scripts/index_repo.py):
    python scripts/demo.py repo express --commit HEAD "set a response header"
    python scripts/demo.py repo express --history "check if request is fresh"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def _head(code: str, n: int = 6) -> str:
    lines = code.strip("\n").splitlines()
    return "\n".join("      " + l for l in lines[:n]) + ("\n      ..." if len(lines) > n else "")


def apps(args) -> None:
    from prism import data
    from prism.evaluate import _fuse, _search, build_retriever

    cfg = json.loads(Path(args.config).read_text())
    split = data.load_split("train")
    text_of = dict(zip(split.doc_ids, split.doc_texts))
    retrievers = [build_retriever(s) for s in cfg["retrievers"]]
    for r in retrievers:
        r.index(split.doc_ids, split.doc_texts)
    gold = None
    if args.qid:
        query = dict(zip(split.query_ids, split.query_texts))[args.qid]
        gold = next(iter(split.qrels[args.qid]))
    else:
        query = args.query
    for r in retrievers:  # warm-up (model load, first-call overhead)
        _search(r, ["warm up"], 5)
    t = time.perf_counter()
    lists = [_search(r, [query], cfg.get("depth", 200))[0] for r in retrievers]
    ranked = sorted(_fuse(lists, cfg.get("fusion")).items(), key=lambda x: (-x[1], x[0]))
    ms = (time.perf_counter() - t) * 1000
    print(f"query ({len(query.split())} words): {query[:300]!r}{'...' if len(query) > 300 else ''}")
    print(f"retrieval latency: {ms:.1f} ms over {len(split.doc_ids):,} snippets (CPU)\n")
    for i, (d, s) in enumerate(ranked[: args.k], 1):
        mark = "  <-- gold" if d == gold else ""
        srcs = ", ".join(f"{r.name}#{next((c.rank for c in l if c.snippet_id == d), '-')}"
                         for r, l in zip(retrievers, lists))
        print(f"{i:>2}. {d:<6} score={s:.4f}  [{srcs}]{mark}\n{_head(text_of[d])}")
    if gold:
        pos = next((i for i, (d, _) in enumerate(ranked, 1) if d == gold), None)
        print(f"\ngold {gold} rank: {pos}")


def repo(args) -> None:
    from prism.cache import EmbeddingStore
    from prism.versioning.indexer import git
    from prism.versioning.search import SnapshotSearcher, search_history
    from prism.versioning.store import SnapshotStore

    sys.path.insert(0, str(ROOT / "scripts"))
    from index_repo import make_encoder

    idx_dir = ROOT / "cache" / "repos" / args.name
    recipe, embed_docs, embed_query = make_encoder(args.model)
    store = SnapshotStore(idx_dir / "index.sqlite")
    emb = EmbeddingStore(recipe, root=idx_dir / "emb")
    embed_query("warm up")
    t = time.perf_counter()
    if args.history:
        res = search_history(store, emb, args.name, args.query, embed_query, embed_docs, k=args.k)
        ms = (time.perf_counter() - t) * 1000
        print(f"history search over all indexed commits: {ms:.1f} ms\n")
        for h in res:
            print(f"{h['rank']:>2}. {h['path']}::{h['symbol']}  present in {h['n_commits']} commits "
                  f"({h['first_commit'][:8]}..{h['last_commit'][:8]}), {h['n_versions_of_symbol']} versions\n"
                  f"{_head(h['code'])}")
            if h["diff_vs_nearest_version"]:
                print("    diff vs nearest other version:\n" + _head(h["diff_vs_nearest_version"], 14))
        return
    commit = git(ROOT / "data" / "repos" / args.name, "rev-parse", args.commit).strip()
    sid = store.snapshot_id_for(args.name, commit)
    if sid is None:
        sys.exit(f"commit {commit[:10]} not indexed; run scripts/index_repo.py first")
    searcher = SnapshotSearcher(store, emb, embed_query, embed_docs)
    searcher.retrieve("warm up", sid, k=1)
    t = time.perf_counter()
    hits = searcher.retrieve(args.query, sid, k=args.k)
    ms = (time.perf_counter() - t) * 1000
    print(f"snapshot {sid}: {ms:.1f} ms\n")
    for h in hits:
        print(f"{h.rank:>2}. {h.path}:{h.start_line}-{h.end_line} {h.symbol}  score={h.score:.4f}\n{_head(h.code)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("apps")
    a.add_argument("query", nargs="?")
    a.add_argument("--config", default="configs/dev/e5_bm25_rrf.json")
    a.add_argument("--qid")
    a.add_argument("-k", type=int, default=10)
    r = sub.add_parser("repo")
    r.add_argument("name")
    r.add_argument("query")
    r.add_argument("--commit", default="HEAD")
    r.add_argument("--history", action="store_true")
    r.add_argument("--model", default="e5-base-v2")
    r.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    apps(args) if args.mode == "apps" else repo(args)


if __name__ == "__main__":
    main()
