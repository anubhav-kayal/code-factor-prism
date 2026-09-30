# PRISM Theme 1 — CPU-first code retrieval

Given a natural-language query and a collection of code snippets, return the
snippets ranked by relevance. Retrieval only: no generation or explanation.

Primary target: CoIR **AppsRetrieval** test split (NDCG@10, MRR@10), scored
through the official `mteb.evaluate` path on CPU.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dataset, model and library revisions are pinned in `requirements.txt` and
`configs/pins.yaml`. The dataset revision equals the one pinned by
`mteb==2.21.10`'s `AppsRetrieval` task.

## Official run (produces the submission JSON)

```bash
.venv/bin/python scripts/run_official.py --model e5-base-v2
# -> results/e5-base-v2/appsretrieval_results.json   (upload this to the GitHub Release)
#    results/e5-base-v2/run_profile.json             (wall time, peak RSS, threads, versions)
#    results/e5-base-v2/AppsRetrieval_predictions.json (raw ranked lists)
```

`--smoke` runs the same pipeline on a 50-query / 200-doc subset as a sanity check.
The encoder is `src/prism/mteb_encoder.py::PrePostPipelineEncoder`, an MTEB
`AbsEncoder` exactly as in the organisers' recipe.

## Development (train queries only)

```bash
.venv/bin/python scripts/run_dev.py configs/dev/bm25.json
.venv/bin/python -m pytest -q tests
```

Dev runs use the 5,000 **train** queries against the full 8,765-doc corpus and
write `reports/<run_id>/{aggregate.json, per_query.jsonl}`. Test labels are only
read by the frozen final run. `tests/test_metrics_match_mteb.py` checks that the
dev metrics reproduce MTEB's numbers from the same ranked lists.

## Layout

| Path | Role |
|---|---|
| `src/prism/mteb_encoder.py` | Official-path encoder (pre/post-processing hooks) |
| `src/prism/data.py` | Pinned loader, deterministic dev folds |
| `src/prism/metrics.py` | NDCG@10 (pytrec_eval), MRR@k, Recall@k, gold ranks |
| `src/prism/retrievers.py` | BM25 (identifier-aware tokens), dense exact search |
| `src/prism/fusion.py` | RRF, weighted z-score |
| `src/prism/cache.py` | Content-addressed embedding cache (also the repo-track content key) |
| `src/prism/evaluate.py` | `evaluate(config)` → aggregate + per-query error report |

## Results

Only measured numbers appear here. See `submissions/`, `reports/` and `results/`.

### Official AppsRetrieval test split (via `mteb.evaluate`, CPU)

| Submission | NDCG@10 | MRR@10 | Recall@100 | Wall time | Peak RSS |
|---|---|---|---|---|---|
| e5-base-v2 encoder (`submissions/e5-base-v2`) | 0.1152 | 0.0988 | 0.3434 | 20.6 min (4 threads, Apple M3) | 1.9 GB |

### Dev findings (train queries; no test labels used)

* **Train ≠ test.** 97% of test queries are contest-style (an `Input` section,
  stdin/stdout) with no starter code; only 26% of train queries are. BM25 on all
  train queries scores NDCG@10 0.349 but only 0.101 on the contest-style subset.
  Dev decisions therefore use the **test-like subset** (1,311 train queries,
  `query_filter: "stdio"`), selected from query text only.
* On that subset, fusing BM25 over the raw query and a deterministic "core" view
  (examples/notes stripped) gives NDCG@10 0.101 → 0.108, Recall@100 0.275 → 0.296.
* `jina-code-embeddings-0.5b` was not run in full: on this 8 GB CPU machine it
  had used >17 CPU-minutes on a 250-text smoke subset without finishing.

## Version track (P1 / Bonus)

`src/prism/versioning/` indexes JavaScript repositories per commit:
function/method snippets (tree-sitter) → content key `sha256(code)` →
occurrence (path, symbol, span) → snapshot membership. `index_commit` re-parses
only files changed since the parent snapshot and embeds only unseen content.
`tests/test_versioning.py` checks incremental == full rebuild (contents and
search results) over add/modify/rename/delete commits, snapshot isolation, and
that history search returns both versions of a changed function with the
distinguishing diff. This track is evaluated separately from AppsRetrieval.

Measured on express `lib/` (130 first-parent commits, 76 functions at HEAD,
e5-base-v2, CPU): incremental update mean 0.07 s vs 6.3 s cold full rebuild;
incremental == full rebuild at all 130 commits (contents and top-10 results for
5 probe queries). See `measurements/repo-express/`.

Version-aware retrieval on 16 author-labelled queries (`eval/express_version_queries.json`;
each describes one specific version of a function, written from that commit's diff):

| Mode | MRR | Hit@1 | Hit@5 | Gold version above other versions |
|---|---|---|---|---|
| History (all commits) | 0.865 | 12/16 | 16/16 | 13/16 |
| Snapshot (labelled commit) | 0.969 | 15/16 | 16/16 | — |

Caveat: the set is small and written by us, and many queries contain the
distinguishing token (e.g. `trimEnd`), so it checks the mechanism rather than
generalisation. Failures are the 4-version `res.send` and a one-line `app.render` change.

```bash
.venv/bin/python scripts/index_repo.py data/repos/express --include lib/ --last 130 --verify
.venv/bin/python scripts/eval_versions.py eval/express_version_queries.json
.venv/bin/python scripts/demo.py repo express --history "check whether the request is fresh"
```
