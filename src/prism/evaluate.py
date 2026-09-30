"""evaluate(config) -> aggregate metrics + per-query error report.

Config (dict):
    run_id:      name for reports/<run_id>/
    split:       "train" (dev, default) or "test" (final frozen run only)
    folds:       optional list of fold indices to restrict dev queries
    retrievers:  [{"type": "bm25", ...}, {"type": "dense", "model": "e5-base-v2", ...}]
    fusion:      null | {"method": "rrf", "k": 60, "weights": [...]} | {"method": "zscore", ...}
    depth:       candidates per retriever (default 200)
    latency_probe: number of single queries to time end-to-end (default 50)
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from . import data, fusion, metrics
from .retrievers import BM25Retriever, Candidate, DenseRetriever

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def build_retriever(spec: dict):
    if spec["type"] == "bm25":
        return BM25Retriever(k1=spec.get("k1", 1.2), b=spec.get("b", 0.75))
    if spec["type"] == "dense":
        return DenseRetriever(spec["model"], spec.get("max_seq_length"), spec.get("batch_size", 32))
    raise ValueError(spec["type"])


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def _fuse(lists: list[list[Candidate]], cfg: dict | None) -> dict[str, float]:
    if not cfg:
        return {c.snippet_id: c.score for c in lists[0]}
    if cfg["method"] == "rrf":
        return fusion.rrf(lists, cfg.get("weights"), cfg.get("k", 60))
    if cfg["method"] == "zscore":
        return fusion.zscore(lists, cfg.get("weights"))
    raise ValueError(cfg["method"])


def evaluate(config: dict) -> dict:
    split = data.load_split(config.get("split", "train"))
    qids, qtexts = split.query_ids, split.query_texts
    if config.get("folds") is not None:
        folds = data.dev_folds(split.query_ids)
        keep = {q for f in config["folds"] for q in folds[f]}
        pairs = [(q, t) for q, t in zip(qids, qtexts) if q in keep]
        qids, qtexts = [p[0] for p in pairs], [p[1] for p in pairs]
    qrels = {q: split.qrels[q] for q in qids}
    depth = config.get("depth", 200)

    retrievers = [build_retriever(s) for s in config["retrievers"]]
    profile: dict = {"index_s": {}, "search_s": {}}
    per_source: list[list[list[Candidate]]] = []
    for r in retrievers:
        t = time.perf_counter()
        r.index(split.doc_ids, split.doc_texts)
        profile["index_s"][r.name] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        per_source.append(r.search(qtexts, depth))
        profile["search_s"][r.name] = round(time.perf_counter() - t, 2)

    run = {q: _fuse([src[i] for src in per_source], config.get("fusion")) for i, q in enumerate(qids)}
    agg = metrics.aggregate(run, qrels)
    source_aggs = {
        r.name: metrics.aggregate(
            {q: {c.snippet_id: c.score for c in per_source[j][i]} for i, q in enumerate(qids)}, qrels
        )
        for j, r in enumerate(retrievers)
    }

    # ---- per-query report
    fused_ranks = metrics.gold_ranks(run, qrels)
    ndcg = metrics.per_query_ndcg10(run, qrels)
    doc_len = {d: len(t.split()) for d, t in zip(split.doc_ids, split.doc_texts)}
    tok_limits = _token_lengths(retrievers, qtexts, qrels, qids, split)
    rows = []
    for i, q in enumerate(qids):
        gold = next(iter(qrels[q]))
        ranked = sorted(run[q].items(), key=lambda x: (-x[1], x[0]))
        src = {}
        for j, r in enumerate(retrievers):
            hit = next((c for c in per_source[j][i] if c.snippet_id == gold), None)
            src[r.name] = {"rank": hit.rank if hit else None, "score": hit.score if hit else None,
                           "top1_score": per_source[j][i][0].score if per_source[j][i] else None}
        gr = fused_ranks[q]
        rows.append({
            "qid": q,
            "gold": gold,
            "rank": gr,
            "ndcg10": round(ndcg[q], 4),
            **{f"hit@{k}": gr is not None and gr <= k for k in metrics.KS},
            "top5": [d for d, _ in ranked[:5]],
            "sources": src,
            "query_words": len(qtexts[i].split()),
            "gold_words": doc_len.get(gold),
            **tok_limits.get(q, {}),
        })

    # ---- bottleneck split over misses (gold not in top 10)
    misses = [r for r in rows if not r["hit@10"]]
    bottleneck = {
        "n_miss_at_10": len(misses),
        "absent_from_top100": sum(1 for r in misses if not r["hit@100"]),
        "in_top100_ranked_low": sum(1 for r in misses if r["hit@100"]),
    }

    profile["latency_ms"] = _latency_probe(retrievers, qtexts, config)
    profile["peak_rss_mb"] = round(_peak_rss_mb(), 1)

    out_dir = REPORTS / config["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"config": config, "aggregate": agg, "per_source": source_aggs,
              "bottleneck": bottleneck, "profile": profile}
    (out_dir / "aggregate.json").write_text(json.dumps(result, indent=2))
    with open(out_dir / "per_query.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {**result, "per_query": rows}


def _token_lengths(retrievers, qtexts, qrels, qids, split) -> dict[str, dict]:
    """Token counts + truncation flags for dense retrievers (query and gold doc)."""
    dense = [r for r in retrievers if isinstance(r, DenseRetriever)]
    if not dense:
        return {}
    r = dense[0]
    tok = r.model.tokenizer
    text_of = dict(zip(split.doc_ids, split.doc_texts))
    golds = [next(iter(qrels[q])) for q in qids]
    qlen = [len(x) for x in tok(qtexts, add_special_tokens=False)["input_ids"]]
    dlen = [len(x) for x in tok([text_of[g] for g in golds], add_special_tokens=False)["input_ids"]]
    lim = r.max_seq_length
    return {q: {"query_tokens": a, "gold_tokens": b, "query_truncated": a > lim, "gold_truncated": b > lim}
            for q, a, b in zip(qids, qlen, dlen)}


def _latency_probe(retrievers, qtexts, config) -> dict:
    """Time single-query end-to-end retrieval (uncached encoding), after warm-up."""
    n = config.get("latency_probe", 50)
    if n <= 0:
        return {}
    sample = qtexts[:: max(1, len(qtexts) // n)][:n]
    times = []
    for i, q in enumerate(sample):
        t = time.perf_counter()
        lists = []
        for r in retrievers:
            if isinstance(r, DenseRetriever):
                v = r.model.encode([q], prompt=r.spec.query_prompt, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
                lists.append(r.search_vecs(v, config.get("depth", 200))[0])
            else:
                lists.append(r.search([q], config.get("depth", 200))[0])
        _fuse(lists, config.get("fusion"))
        if i > 0:  # first query is warm-up
            times.append((time.perf_counter() - t) * 1000)
    return {"n": len(times), "p50": round(float(np.percentile(times, 50)), 1),
            "p95": round(float(np.percentile(times, 95)), 1)}
