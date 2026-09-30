# Runbook: jina-code experiments on a faster machine

Goal: measure whether `jinaai/jina-code-embeddings-0.5b` beats our current
submission on CoIR AppsRetrieval, **on CPU**, and send back the result files.

Current official test scores to beat (NDCG@10 / MRR@10):

| Run | NDCG@10 | MRR@10 |
|---|---|---|
| e5-base-v2 (released) | 0.1152 | 0.0988 |
| e5-base-v2 + BM25 hybrid (kept on branch `hybrid-submission`, not on `main`) | 0.1212 | 0.1025 |

Rules for every run: **CPU only** (`--device cpu`, the default), do not edit
code or configs mid-run, and don't touch the test split outside
`scripts/run_official.py`.

## 0. Setup (≈5 min)

```bash
git clone https://github.com/anubhav-kayal/code-factor-prism.git
cd code-factor-prism
python3.12 -m venv .venv            # Python 3.12 required
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests  # expect: all passed (the MTEB-parity test may be skipped)
```

Record your machine specs. Paste the output of one of these into your reply:

```bash
# macOS
sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize
# Linux
lscpu | head -20; free -g
```

Close browsers/IDEs before the long runs; memory pressure slows them badly.

## 1. Smoke test: timing estimate (≈2–10 min)

```bash
.venv/bin/python scripts/run_official.py --model jina-code-0.5b --max-seq-length 512 --smoke
```

This runs the full pipeline on a fixed 50-query / 200-doc subset and prints
`evaluate_wall_s`. The full test run encodes about **50× more text** (8,765
docs + 3,765 queries), so **full time ≈ 50 × smoke time**. Tell us this
number before continuing if it's over ~3 hours.

## 2. Official test run: jina-code, 512 tokens (main job)

```bash
.venv/bin/python scripts/run_official.py --model jina-code-0.5b --max-seq-length 512 \
    > jina512.log 2>&1
```

Output: `results/jina-code-0.5b-len512/`
- `appsretrieval_results.json`  ← the submission-format file
- `run_profile.json`            ← time, peak memory, threads, versions
- `AppsRetrieval_predictions.json` (large; not needed unless we ask)

To use more CPU cores, add `--threads N` (N = physical core count). Put the
N you used in your reply.

## 3. Optional: 1024 tokens (only if step 2 took < 1.5 h)

```bash
.venv/bin/python scripts/run_official.py --model jina-code-0.5b > jina1024.log 2>&1
```

Output: `results/jina-code-0.5b/`. This tests whether the longer context helps
with long problem statements. It costs about 2× the time of step 2.

## 4. Optional: dev runs, jina alone vs jina + BM25 (no test labels)

These evaluate on the 1,311 contest-style **train** queries. That's how we
decide whether jina + BM25 fusion helps, without looking at the test split.

```bash
.venv/bin/python scripts/run_dev.py configs/dev/stdio_jina_len512.json         > dev_jina.log 2>&1
.venv/bin/python scripts/run_dev.py configs/dev/stdio_jina_bm25_rrf_len512.json > dev_jina_bm25.log 2>&1
```

The first run embeds the corpus once (cached in `cache/`), so the second is
much faster. Outputs are under `reports/dev-stdio-jina-*`.

## 5. What to send back

Zip these and share them (Drive/Slack). **Don't push to `main`.**

```
results/jina-code-0.5b-len512/appsretrieval_results.json
results/jina-code-0.5b-len512/run_profile.json
results/jina-code-0.5b/…            (if step 3 was run)
reports/dev-stdio-jina-*/aggregate.json   (if step 4 was run)
jina512.log  (last 50 lines are enough)
+ machine specs from step 0
```

If you prefer git: `git checkout -b jina-results`, add only the files
above, then push that branch.

## Troubleshooting

- **Killed / out of memory:** retry with `--batch-size 16`. If it still dies, use `--max-seq-length 384` and say so.
- **Very slow download:** set `HF_TOKEN` (a free Hugging Face token) in the environment.
- **Any other error:** send the last 50 lines of the log. Don't change the code to work around it.
