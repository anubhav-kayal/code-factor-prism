"""Paired comparison of two dev runs: per-fold deltas and per-query wins/regressions.

    python scripts/compare.py dev-bm25 dev-bm25-raw-core-rrf
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism import data  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def load(run_id: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in (REPORTS / run_id / "per_query.jsonl").open()]
    return {r["qid"]: r for r in rows}


def rr10(r: dict) -> float:
    return 1.0 / r["rank"] if r["rank"] and r["rank"] <= 10 else 0.0


def main(base_id: str, new_id: str) -> None:
    base, new = load(base_id), load(new_id)
    qids = sorted(set(base) & set(new))
    folds = data.dev_folds(qids)
    print(f"{base_id} -> {new_id}   ({len(qids)} paired queries)")
    print(f"{'fold':>4} {'n':>5} {'dNDCG@10':>10} {'dMRR@10':>9}")
    d_ndcg_f, d_mrr_f = [], []
    for i, f in enumerate(folds):
        dn = np.mean([new[q]["ndcg10"] - base[q]["ndcg10"] for q in f])
        dm = np.mean([rr10(new[q]) - rr10(base[q]) for q in f])
        d_ndcg_f.append(dn)
        d_mrr_f.append(dm)
        print(f"{i:>4} {len(f):>5} {dn:>+10.4f} {dm:>+9.4f}")
    print(f"mean dNDCG@10 {np.mean(d_ndcg_f):+.4f} (fold sd {np.std(d_ndcg_f, ddof=1):.4f}); "
          f"folds improved {sum(d > 0 for d in d_ndcg_f)}/{len(folds)}")

    moved_in = [q for q in qids if new[q]["hit@10"] and not base[q]["hit@10"]]
    moved_out = [q for q in qids if base[q]["hit@10"] and not new[q]["hit@10"]]
    better = [q for q in qids if new[q]["ndcg10"] > base[q]["ndcg10"]]
    worse = [q for q in qids if new[q]["ndcg10"] < base[q]["ndcg10"]]
    print(f"per-query NDCG@10: {len(better)} better, {len(worse)} worse; "
          f"entered top10 {len(moved_in)}, left top10 {len(moved_out)}")
    print("sample regressions (qid base_rank -> new_rank):",
          [(q, base[q]["rank"], new[q]["rank"]) for q in moved_out[:8]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
