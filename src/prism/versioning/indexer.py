"""index_commit(repo, commit) -> snapshot_id

Incremental: take the parent snapshot, re-extract only files changed by
`git diff parent..commit`, embed only content keys never seen before, publish.
Full: extract every JS file at the commit (used to verify incremental == full).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..cache import EmbeddingStore
from .extract import JS_EXTENSIONS, extract_js
from .store import SnapshotStore

Embed = Callable[[list[str]], np.ndarray]


def git(repo: str | Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _is_js(path: str) -> bool:
    return path.endswith(JS_EXTENSIONS) and "node_modules/" not in path


def _first_parent(repo, commit: str) -> str | None:
    parents = git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
    return parents[0] if parents else None


def _changed_files(repo, parent: str, commit: str) -> tuple[set[str], set[str]]:
    """(paths to (re)extract at `commit`, paths whose old occurrences must be dropped)."""
    to_extract, to_drop = set(), set()
    for line in git(repo, "diff", "--name-status", "-M", parent, commit).splitlines():
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") or status.startswith("C"):
            old, new = parts[1], parts[2]
            if status.startswith("R"):
                to_drop.add(old)
            to_extract.add(new)
        elif status.startswith("D"):
            to_drop.add(parts[1])
        else:  # A, M, T
            to_extract.add(parts[1])
            to_drop.add(parts[1])
    return {p for p in to_extract if _is_js(p)}, {p for p in to_drop if _is_js(p)}


def _extract_at(repo, commit: str, paths: set[str]):
    snippets = []
    for p in sorted(paths):
        src = git(repo, "show", f"{commit}:{p}")
        snippets.extend(extract_js(p, src))
    return snippets


class CommitIndexer:
    def __init__(self, store: SnapshotStore, emb_store: EmbeddingStore, embed: Embed, repo_name: str) -> None:
        self.store, self.emb, self.embed, self.repo_name = store, emb_store, embed, repo_name

    def _sid(self, commit: str, mode: str) -> str:
        return f"{self.repo_name}@{commit[:12]}:{mode}"

    def index_commit(self, repo: str | Path, commit: str, mode: str = "incremental") -> str:
        commit = git(repo, "rev-parse", commit).strip()
        t0 = time.perf_counter()
        parent = _first_parent(repo, commit)
        parent_sid = self.store.snapshot_id_for(self.repo_name, parent, "incremental") if parent else None

        if mode == "incremental" and parent_sid:
            to_extract, to_drop = _changed_files(repo, parent, commit)
            kept = [o["occ_key"] for o in self.store.occurrences(parent_sid) if o["path"] not in to_drop]
            new_snips = _extract_at(repo, commit, to_extract)
            files_parsed = len(to_extract)
        else:
            all_files = {p for p in git(repo, "ls-tree", "-r", "--name-only", commit).splitlines() if _is_js(p)}
            kept, new_snips = [], _extract_at(repo, commit, all_files)
            files_parsed = len(all_files)
        t_extract = time.perf_counter() - t0

        new_occs = self.store.add_snippets(new_snips)
        occ_keys = sorted(set(kept) | set(new_occs))

        before = len(self.emb)
        t1 = time.perf_counter()
        codes = [o["code"] for o in self._occ_rows(occ_keys)]
        _, n_embedded = self.emb.get_or_compute(codes, self.embed)
        t_embed = time.perf_counter() - t1
        self.store.db.commit()

        sid = self._sid(commit, mode)
        stats = {
            "files_parsed": files_parsed,
            "snippets": len(occ_keys),
            "reused_occurrences": len(kept),
            "new_occurrences": len(set(new_occs) - set(kept)),
            "embedded_new_content": n_embedded,
            "embedding_cache_size_before": before,
            "extract_s": round(t_extract, 3),
            "embed_s": round(t_embed, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        }
        self.store.publish(sid, self.repo_name, commit, parent, mode, occ_keys, stats)
        return sid

    def _occ_rows(self, occ_keys: list[str]) -> list[dict]:
        q = ",".join("?" * len(occ_keys))
        rows = self.store.db.execute(
            f"SELECT o.occ_key, c.code FROM occurrence o JOIN content c ON c.content_key = o.content_key "
            f"WHERE o.occ_key IN ({q}) ORDER BY o.occ_key",
            occ_keys,
        ).fetchall() if occ_keys else []
        return [{"occ_key": a, "code": b} for a, b in rows]
