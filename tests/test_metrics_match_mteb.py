"""Our metrics must reproduce MTEB's scores from the same ranked lists."""

import json
from pathlib import Path

import pytest

from prism import metrics

SMOKE = Path(__file__).resolve().parents[1] / "results" / "e5-base-v2-smoke"


@pytest.mark.skipif(not (SMOKE / "AppsRetrieval_predictions.json").exists(), reason="run smoke first")
def test_matches_mteb_smoke():
    run = json.loads((SMOKE / "AppsRetrieval_predictions.json").read_text())["default"]["test"]
    official = json.loads((SMOKE / "appsretrieval_results.json").read_text())["scores"]["test"][0]
    import mteb

    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    rel = task.dataset["default"]["test"]["relevant_docs"]
    qrels = {q: {d: int(s) for d, s in rel[q].items()} for q in run}
    ours = metrics.aggregate(run, qrels)
    assert ours["ndcg_at_10"] == pytest.approx(official["ndcg_at_10"], abs=1e-4)
    assert ours["mrr_at_10"] == pytest.approx(official["mrr_at_10"], abs=1e-4)
    assert ours["recall_at_100"] == pytest.approx(official["recall_at_100"], abs=1e-4)
