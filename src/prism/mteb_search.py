"""Official-path hybrid: an MTEB SearchProtocol running the same retriever/fusion
config as `evaluate(config)`, so a dev config can be scored by `mteb.evaluate`
unchanged.
"""

from __future__ import annotations

from typing import Any

from mteb.models.model_meta import ModelMeta

from .evaluate import _fuse, _search, build_retriever


class PrismHybridSearch:
    def __init__(self, config: dict, name: str = "prism/hybrid") -> None:
        self.config = config
        self.retrievers = [build_retriever(s) for s in config["retrievers"]]
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={"name": name, "model_type": ["hybrid"], "open_weights": True}
        )

    def index(self, corpus, *, task_metadata, hf_split: str, hf_subset: str,
              encode_kwargs: dict[str, Any], num_proc: int | None = None) -> None:
        doc_ids = [str(i) for i in corpus["id"]]
        doc_texts = list(corpus["text"])
        for r in self.retrievers:
            r.index(doc_ids, doc_texts)

    def search(self, queries, *, task_metadata, hf_split: str, hf_subset: str, top_k: int,
               encode_kwargs: dict[str, Any], top_ranked=None, num_proc: int | None = None):
        qids = [str(i) for i in queries["id"]]
        qtexts = list(queries["text"])
        depth = max(top_k, self.config.get("depth", 200))
        per_source = [_search(r, qtexts, depth) for r in self.retrievers]
        out = {}
        for i, q in enumerate(qids):
            fused = _fuse([src[i] for src in per_source], self.config.get("fusion"))
            ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]
            out[q] = dict(ranked)
        return out
