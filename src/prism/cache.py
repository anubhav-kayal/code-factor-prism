"""Content-addressed embedding cache.

An embedding is keyed by (recipe, content_key):
  recipe      = model + revision + prompt + max_seq_length + view name
  content_key = sha256 of the exact text fed to the encoder (before the prompt)

The same content_key is the repository track's snippet content key, so an
unchanged snippet is never re-embedded across commits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache" / "emb"


def content_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recipe_id(**parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class EmbeddingStore:
    def __init__(self, recipe: dict, root: Path = CACHE_ROOT) -> None:
        self.recipe = recipe
        self.dir = root / recipe_id(**recipe)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True))
        self._keys: list[str] = []
        self._vecs: np.ndarray | None = None
        if (self.dir / "keys.json").exists():
            self._keys = json.loads((self.dir / "keys.json").read_text())
            self._vecs = np.load(self.dir / "vecs.npy")
        self._index = {k: i for i, k in enumerate(self._keys)}

    def __len__(self) -> int:
        return len(self._keys)

    def get_or_compute(
        self, texts: list[str], encode: Callable[[list[str]], np.ndarray]
    ) -> tuple[np.ndarray, int]:
        """Return embeddings for `texts` (row-aligned) and how many were newly computed."""
        keys = [content_key(t) for t in texts]
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in self._index and k not in missing:
                missing[k] = t
        if missing:
            new = np.asarray(encode(list(missing.values())), dtype=np.float32)
            start = len(self._keys)
            self._keys.extend(missing)
            self._index.update({k: start + i for i, k in enumerate(missing)})
            self._vecs = new if self._vecs is None else np.vstack([self._vecs, new])
            self._save()
        assert self._vecs is not None
        return self._vecs[[self._index[k] for k in keys]], len(missing)

    def _save(self) -> None:
        np.save(self.dir / "vecs.npy", self._vecs)
        (self.dir / "keys.json").write_text(json.dumps(self._keys))
