"""Index the last N first-parent commits of a repo, incrementally, and verify.

    python scripts/index_repo.py data/repos/express --include lib/ --last 30 --verify

Reports per-commit incremental cost, a cold full rebuild of the newest commit
(fresh embedding cache) for comparison, and (with --verify) checks that the
incremental snapshot equals a full rebuild at every commit, for index contents
and for retrieval results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.cache import EmbeddingStore  # noqa: E402
from prism.versioning.indexer import CommitIndexer, git  # noqa: E402
from prism.versioning.search import SnapshotSearcher  # noqa: E402
from prism.versioning.store import SnapshotStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VERIFY_QUERIES = [
    "parse the query string of a request",
    "set a response header",
    "render a view template",
    "send a JSON response",
    "match a route path with parameters",
]


def make_encoder(model_key: str):
    from prism.mteb_encoder import MODEL_SPECS
    from sentence_transformers import SentenceTransformer

    spec = MODEL_SPECS[model_key]
    m = SentenceTransformer(spec.name, revision=spec.revision, device="cpu")
    m.max_seq_length = spec.max_seq_length
    recipe = {"model": spec.name, "revision": spec.revision, "prompt": spec.doc_prompt,
              "max_seq_length": spec.max_seq_length, "view": "repo-code"}

    def embed_docs(texts):
        return m.encode(texts, prompt=spec.doc_prompt, batch_size=16, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)

    def embed_query(q):
        return m.encode([q], prompt=spec.query_prompt, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)[0]

    return recipe, embed_docs, embed_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--last", type=int, default=30)
    ap.add_argument("--model", default="e5-base-v2")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    name = repo.name
    out = ROOT / "reports" / f"repo-{name}"
    out.mkdir(parents=True, exist_ok=True)
    idx_dir = ROOT / "cache" / "repos" / name
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    idx_dir.mkdir(parents=True)

    recipe, embed_docs, embed_query = make_encoder(args.model)
    store = SnapshotStore(idx_dir / "index.sqlite")
    emb = EmbeddingStore(recipe, root=idx_dir / "emb")
    ix = CommitIndexer(store, emb, embed_docs, name, include=tuple(args.include))

    commits = git(repo, "rev-list", "--first-parent", f"-n{args.last}", "HEAD").split()[::-1]
    per_commit = []
    for c in commits:
        sid = ix.index_commit(repo, c)
        st = next(s for s in store.snapshots(name) if s["snapshot_id"] == sid)["stats"]
        per_commit.append({"commit": c, **st})
        print(c[:10], json.dumps(st))

    # cold full rebuild of the newest commit with an empty embedding cache
    cold_emb = EmbeddingStore(recipe, root=idx_dir / "emb-cold")
    cold_store = SnapshotStore(idx_dir / "cold.sqlite")
    t = time.perf_counter()
    CommitIndexer(cold_store, cold_emb, embed_docs, name, include=tuple(args.include)).index_commit(
        repo, commits[-1], mode="full")
    cold_full_s = time.perf_counter() - t

    incr = per_commit[1:]  # first commit is necessarily a full build
    summary = {
        "repo": name, "include": args.include, "model": recipe["model"], "commits": len(commits),
        "snippets_at_head": per_commit[-1]["snippets"],
        "cold_full_rebuild_head_s": round(cold_full_s, 2),
        "incremental_total_s": {"mean": round(sum(p["total_s"] for p in incr) / max(1, len(incr)), 3),
                                "max": max((p["total_s"] for p in incr), default=0)},
        "incremental_new_embeddings_mean": round(sum(p["embedded_new_content"] for p in incr) / max(1, len(incr)), 2),
        "distinct_contents_embedded": len(emb),
        "total_occurrence_versions": sum(p["snippets"] for p in per_commit),
    }

    if args.verify:
        searcher = SnapshotSearcher(store, emb, embed_query, embed_docs)
        mismatches = []
        for c in commits:
            inc = store.snapshot_id_for(name, c, "incremental")
            full = ix.index_commit(repo, c, mode="full")
            a = {o["occ_key"] for o in store.occurrences(inc)}
            b = {o["occ_key"] for o in store.occurrences(full)}
            if a != b:
                mismatches.append({"commit": c, "kind": "contents", "only_inc": len(a - b), "only_full": len(b - a)})
                continue
            for q in VERIFY_QUERIES:
                ra = [(h.snippet_id, round(h.score, 9)) for h in searcher.retrieve(q, inc, k=10)]
                rb = [(h.snippet_id, round(h.score, 9)) for h in searcher.retrieve(q, full, k=10)]
                if ra != rb:
                    mismatches.append({"commit": c, "kind": "results", "query": q})
        summary["verify"] = {"commits_checked": len(commits), "queries": len(VERIFY_QUERIES),
                             "mismatches": mismatches}

    (out / "per_commit.json").write_text(json.dumps(per_commit, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
