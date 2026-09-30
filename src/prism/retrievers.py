"""Candidate retrievers over a fixed document collection.

Every retriever exposes
    index(doc_ids, doc_texts)
    search(query_texts, k) -> list[list[Candidate]]
so fusion and evaluation don't care where candidates came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    snippet_id: str
    source: str
    rank: int  # 1-based
    score: float


# --------------------------------------------------------------------------- BM25

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_STOP = frozenset(
    "the a an of to in and or is are be for on with that this it as by at from you your "
    "input output print return if else while def int str".split()
)


def code_tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: keep the full identifier plus its snake/camel parts."""
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        parts = [p.lower() for piece in tok.split("_") for p in _CAMEL.split(piece) if p]
        if low not in _STOP:
            out.append(low)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in _STOP and len(p) > 1)
    return out


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        import bm25s

        self._bm25s = bm25s
        self.k1, self.b = k1, b

    def index(self, doc_ids: list[str], doc_texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.model = self._bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index([code_tokenize(t) for t in doc_texts], show_progress=False)

    def search(self, query_texts: list[str], k: int) -> list[list[Candidate]]:
        toks = [code_tokenize(q) or ["_empty_"] for q in query_texts]
        idx, scores = self.model.retrieve(toks, k=min(k, len(self.doc_ids)), show_progress=False)
        return [
            [
                Candidate(self.doc_ids[int(d)], self.name, r + 1, float(s))
                for r, (d, s) in enumerate(zip(row_i, row_s))
            ]
            for row_i, row_s in zip(idx, scores)
        ]


# -------------------------------------------------------------------------- Dense


class DenseRetriever:
    """Exact cosine search over cached, L2-normalised embeddings."""

    def __init__(self, model_key: str, max_seq_length: int | None = None, batch_size: int = 32) -> None:
        from .mteb_encoder import MODEL_SPECS

        self.spec = MODEL_SPECS[model_key]
        self.name = model_key
        self.max_seq_length = max_seq_length or self.spec.max_seq_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(self.spec.name, revision=self.spec.revision, device="cpu")
            m.max_seq_length = self.max_seq_length
            self._model = m
        return self._model

    def _store(self, role: str, view: str):
        from .cache import EmbeddingStore

        prompt = self.spec.query_prompt if role == "query" else self.spec.doc_prompt
        return EmbeddingStore(
            {
                "model": self.spec.name,
                "revision": self.spec.revision,
                "prompt": prompt,
                "max_seq_length": self.max_seq_length,
                "view": view,
            }
        )

    def _encoder(self, prompt: str):
        def enc(texts: list[str]) -> np.ndarray:
            return self.model.encode(
                texts,
                prompt=prompt,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

        return enc

    def embed(self, texts: list[str], role: str, view: str = "raw") -> np.ndarray:
        prompt = self.spec.query_prompt if role == "query" else self.spec.doc_prompt
        vecs, _ = self._store(role, view).get_or_compute(texts, self._encoder(prompt))
        return vecs

    def index(self, doc_ids: list[str], doc_texts: list[str], view: str = "raw") -> None:
        self.doc_ids = doc_ids
        self.doc_vecs = self.embed(doc_texts, "document", view)

    def search(self, query_texts: list[str], k: int, view: str = "raw") -> list[list[Candidate]]:
        q = self.embed(query_texts, "query", view)
        return self.search_vecs(q, k)

    def search_vecs(self, q: np.ndarray, k: int) -> list[list[Candidate]]:
        sims = q @ self.doc_vecs.T
        k = min(k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out = []
        for row, cand in zip(sims, top):
            order = cand[np.argsort(-row[cand], kind="stable")]
            out.append(
                [Candidate(self.doc_ids[i], self.name, r + 1, float(row[i])) for r, i in enumerate(order)]
            )
        return out
