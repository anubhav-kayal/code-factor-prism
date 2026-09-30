"""Run a dev evaluation from a JSON config: python scripts/run_dev.py configs/dev/bm25.json"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.evaluate import evaluate  # noqa: E402

cfg = json.loads(Path(sys.argv[1]).read_text())
for kv in sys.argv[2:]:  # overrides like latency_probe=0
    k, v = kv.split("=", 1)
    cfg[k] = json.loads(v)
res = evaluate(cfg)
print(json.dumps({k: res[k] for k in ("aggregate", "per_source", "bottleneck", "profile")}, indent=2))
