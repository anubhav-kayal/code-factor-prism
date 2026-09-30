"""Heuristic failure buckets for a dev run's misses (gold not in top 10).

    python scripts/analyze_misses.py dev-e5-bm25-rrf

Buckets are heuristics for triage, not ground truth; confirm on a manual sample.
A query can fall in several buckets.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
_IDENT = re.compile(r"`[^`]+`|\b[a-z]+_[a-z_]+\b|\b[a-z]+[A-Z][A-Za-z]+\b|\bdef \w+|\bclass \w+")


def main(run_id: str) -> None:
    rows = [json.loads(l) for l in (REPORTS / run_id / "per_query.jsonl").open()]
    qlen_p75 = np.percentile([r["query_words"] for r in rows], 75)
    misses = [r for r in rows if not r["hit@10"]]
    buckets: Counter = Counter()
    examples: dict[str, list[str]] = {}

    def add(b: str, r: dict) -> None:
        buckets[b] += 1
        examples.setdefault(b, []).append(r["qid"])

    from prism import data  # noqa: E402

    qtext = dict(zip(*[(s := data.load_split("train")).query_ids, s.query_texts]))
    for r in misses:
        src_ranks = {k: v["rank"] for k, v in r["sources"].items()}
        if not r["hit@100"]:
            add("candidate: absent from fused top100", r)
        else:
            add("ranking: in top100, ranked >10", r)
        if any(v is not None and v <= 10 for v in src_ranks.values()):
            add("fusion loss: some source had it in top10", r)
        if r.get("query_truncated"):
            add("query truncated by encoder", r)
        if r.get("gold_truncated"):
            add("gold code truncated by encoder", r)
        if r["query_words"] > qlen_p75:
            add("long query (>p75 words)", r)
        if _IDENT.search(qtext[r["qid"]]):
            add("query has identifiers/starter code", r)
        if (r.get("gold_words") or 0) < 15:
            add("very short gold snippet (<15 words)", r)

    print(f"{run_id}: {len(misses)} misses of {len(rows)} queries")
    for b, n in buckets.most_common():
        print(f"  {n:>5}  {100 * n / len(misses):5.1f}%  {b}   e.g. {examples[b][:4]}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main(sys.argv[1])
