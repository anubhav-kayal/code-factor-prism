"""Official AppsRetrieval run, following the organisers' recipe.

    python scripts/run_official.py --model jina-code-0.5b
    python scripts/run_official.py --model e5-base-v2 --smoke   # tiny subset sanity run
    python scripts/run_official.py --config configs/dev/e5_bm25_rrf.json   # hybrid via SearchProtocol
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mteb  # noqa: E402

from prism.mteb_encoder import PipelineConfig, PrePostPipelineEncoder  # noqa: E402
from prism.mteb_search import PrismHybridSearch  # noqa: E402


def peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, KiB on Linux
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def _json_default(o):
    if isinstance(o, datetime):
        return o.timestamp()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jina-code-0.5b")
    ap.add_argument("--config", default=None, help="dev config JSON -> run as MTEB SearchProtocol")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true", help="evaluate on a small subsample")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text()) if args.config else None
    run_name = cfg["run_id"].removeprefix("dev-") if cfg else args.model
    out = Path(args.out) / (run_name + ("-smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.threads:
        torch.set_num_threads(args.threads)
    if cfg:
        model = PrismHybridSearch(cfg, name=f"prism/{run_name}")
    else:
        model = PrePostPipelineEncoder(
            PipelineConfig(model_key=args.model, max_seq_length=args.max_seq_length, num_threads=args.threads)
        )
    load_s = time.perf_counter() - t0

    task = mteb.get_task("AppsRetrieval")
    if args.smoke:
        task.load_data()
        split = task.dataset["default"]["test"]
        qids = list(split["relevant_docs"])[:50]
        keep_docs = {d for q in qids for d in split["relevant_docs"][q]}
        corpus = split["corpus"]
        extra = [i for i in corpus["id"][:500] if i not in keep_docs][: 200 - len(keep_docs)]
        keep_docs |= set(extra)
        split["corpus"] = corpus.filter(lambda r: r["id"] in keep_docs)
        split["queries"] = split["queries"].filter(lambda r: r["id"] in set(qids))
        split["relevant_docs"] = {q: split["relevant_docs"][q] for q in qids}

    t1 = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,
        overwrite_strategy="always",
        prediction_folder=out,  # raw ranked lists, for CSV export and error analysis
    )
    eval_s = time.perf_counter() - t1

    task_result = list(result.task_results)[0]
    with open(out / "appsretrieval_results.json", "w") as f:
        # Organisers' recipe is json.dump(task_result.to_dict()); the dict holds a
        # datetime, so serialise it the way TaskResult.to_disk does.
        json.dump(task_result.to_dict(), f, indent=2, default=_json_default)

    scores = task_result.to_dict()["scores"]["test"][0]
    profile = {
        "model": cfg if cfg else model.spec.name,
        "model_revision": None if cfg else model.spec.revision,
        "max_seq_length": None if cfg else model.model.max_seq_length,
        "smoke": args.smoke,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "machine": f"{platform.machine()} {platform.processor()} {platform.platform()}",
        "model_load_s": round(load_s, 2),
        "evaluate_wall_s": round(eval_s, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "versions": {"mteb": mteb.__version__, "torch": torch.__version__},
        "ndcg_at_10": scores.get("ndcg_at_10"),
        "mrr_at_10": scores.get("mrr_at_10"),
        "recall_at_100": scores.get("recall_at_100"),
    }
    with open(out / "run_profile.json", "w") as f:
        json.dump(profile, f, indent=2)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
