"""Official-path encoder: an MTEB AbsEncoder wrapping a sentence-transformers model.

Pre/post-processing hooks live here so that anything we submit is exactly what
`mteb.evaluate` scores. Stage 0 uses identity hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str
    query_prompt: str
    doc_prompt: str
    max_seq_length: int
    n_parameters: int
    license: str
    embed_dim: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "jina-code-0.5b": ModelSpec(
        name="jinaai/jina-code-embeddings-0.5b",
        revision="4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
        query_prompt="Find the most relevant code snippet given the following query:\n",
        doc_prompt="Candidate code snippet:\n",
        max_seq_length=1024,
        n_parameters=494_000_000,
        license="cc-by-nc-4.0",
        embed_dim=896,
    ),
    "e5-base-v2": ModelSpec(
        name="intfloat/e5-base-v2",
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        query_prompt="query: ",
        doc_prompt="passage: ",
        max_seq_length=512,
        n_parameters=109_000_000,
        license="mit",
        embed_dim=768,
    ),
}


def identity(texts: list[str]) -> list[str]:
    return texts


@dataclass
class PipelineConfig:
    model_key: str = "jina-code-0.5b"
    max_seq_length: int | None = None  # None -> spec default
    num_threads: int | None = None
    query_pre: Callable[[list[str]], list[str]] = identity
    doc_pre: Callable[[list[str]], list[str]] = identity
    tag: str = "stage0"
    extra: dict[str, Any] = field(default_factory=dict)


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        spec = MODEL_SPECS[self.config.model_key]
        self.spec = spec
        if self.config.num_threads:
            torch.set_num_threads(self.config.num_threads)
        self.model = SentenceTransformer(spec.name, revision=spec.revision, device="cpu")
        self.model.max_seq_length = self.config.max_seq_length or spec.max_seq_length
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": f"prism/{self.config.model_key}-{self.config.tag}",
                "revision": spec.revision,
                "n_parameters": spec.n_parameters,
                "max_tokens": self.model.max_seq_length,
                "embed_dim": spec.embed_dim,
                "license": spec.license,
                "open_weights": True,
                "similarity_fn_name": ScoringFunction.COSINE,
                "framework": ["Sentence Transformers", "PyTorch"],
                "adapted_from": spec.name,
            }
        )

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        if prompt_type == PromptType.query:
            texts, prompt = self.config.query_pre(texts), self.spec.query_prompt
        else:
            texts, prompt = self.config.doc_pre(texts), self.spec.doc_prompt
        return self.model.encode(
            texts,
            prompt=prompt,
            batch_size=kwargs.get("batch_size", 32),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=kwargs.get("show_progress_bar", True),
        )
