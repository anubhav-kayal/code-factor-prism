"""SQLite store for commit-aware snippet indexing.

content     one row per distinct snippet text   (content_key = sha256(code))
occurrence  where a content appears             (occ_key = hash(path, symbol, span, content_key))
snapshot    one row per indexed commit
membership  which occurrences a snapshot contains

Occurrence keys are commit-independent so unchanged occurrences are shared
between snapshots; the commit an occurrence belongs to is given by membership.
Embeddings live in the content-addressed EmbeddingStore, keyed by content_key.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from ..cache import content_key
from .extract import Snippet

SCHEMA = """
CREATE TABLE IF NOT EXISTS content (
    content_key TEXT PRIMARY KEY,
    code        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS occurrence (
    occ_key     TEXT PRIMARY KEY,
    content_key TEXT NOT NULL REFERENCES content(content_key),
    path        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    callees     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot (
    snapshot_id TEXT PRIMARY KEY,
    repo        TEXT NOT NULL,
    commit_sha  TEXT NOT NULL,
    parent_sha  TEXT,
    seq         INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    stats       TEXT NOT NULL,
    UNIQUE (repo, commit_sha, mode)
);
CREATE TABLE IF NOT EXISTS membership (
    snapshot_id TEXT NOT NULL REFERENCES snapshot(snapshot_id),
    occ_key     TEXT NOT NULL REFERENCES occurrence(occ_key),
    PRIMARY KEY (snapshot_id, occ_key)
);
CREATE INDEX IF NOT EXISTS occ_path ON occurrence(path);
"""


def occ_key(s: Snippet, ckey: str) -> str:
    blob = f"{s.path}\0{s.symbol}\0{s.start_line}\0{s.end_line}\0{ckey}"
    return hashlib.sha256(blob.encode()).hexdigest()


class SnapshotStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db = sqlite3.connect(str(db_path))
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------------- writes
    def add_snippets(self, snippets: list[Snippet]) -> list[str]:
        occs = []
        for s in snippets:
            ck = content_key(s.code)
            ok = occ_key(s, ck)
            self.db.execute("INSERT OR IGNORE INTO content VALUES (?, ?)", (ck, s.code))
            self.db.execute(
                "INSERT OR IGNORE INTO occurrence VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ok, ck, s.path, s.symbol, s.start_line, s.end_line, json.dumps(list(s.callees))),
            )
            occs.append(ok)
        return occs

    def publish(self, snapshot_id: str, repo: str, commit: str, parent: str | None,
                mode: str, occ_keys: list[str], stats: dict) -> None:
        seq = self.db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM snapshot").fetchone()[0]
        with self.db:
            self.db.execute("DELETE FROM membership WHERE snapshot_id = ?", (snapshot_id,))
            self.db.execute("DELETE FROM snapshot WHERE snapshot_id = ?", (snapshot_id,))
            self.db.execute(
                "INSERT INTO snapshot VALUES (?, ?, ?, ?, ?, ?, ?)",
                (snapshot_id, repo, commit, parent, seq, mode, json.dumps(stats)),
            )
            self.db.executemany(
                "INSERT OR IGNORE INTO membership VALUES (?, ?)", [(snapshot_id, o) for o in occ_keys]
            )

    # ----------------------------------------------------------------- reads
    def snapshot_id_for(self, repo: str, commit: str, mode: str = "incremental") -> str | None:
        row = self.db.execute(
            "SELECT snapshot_id FROM snapshot WHERE repo = ? AND commit_sha = ? AND mode = ?",
            (repo, commit, mode),
        ).fetchone()
        return row[0] if row else None

    def occurrences(self, snapshot_id: str, paths: set[str] | None = None) -> list[dict]:
        rows = self.db.execute(
            """SELECT o.occ_key, o.content_key, o.path, o.symbol, o.start_line, o.end_line, o.callees, c.code
               FROM membership m JOIN occurrence o ON o.occ_key = m.occ_key
               JOIN content c ON c.content_key = o.content_key
               WHERE m.snapshot_id = ? ORDER BY o.path, o.start_line, o.symbol""",
            (snapshot_id,),
        ).fetchall()
        cols = ("occ_key", "content_key", "path", "symbol", "start_line", "end_line", "callees", "code")
        out = [dict(zip(cols, r)) for r in rows]
        return [r for r in out if paths is None or r["path"] in paths]

    def snapshots(self, repo: str, mode: str = "incremental") -> list[dict]:
        rows = self.db.execute(
            "SELECT snapshot_id, commit_sha, parent_sha, seq, stats FROM snapshot "
            "WHERE repo = ? AND mode = ? ORDER BY seq",
            (repo, mode),
        ).fetchall()
        return [dict(snapshot_id=a, commit=b, parent=c, seq=d, stats=json.loads(e)) for a, b, c, d, e in rows]

    def history(self, repo: str, mode: str = "incremental") -> list[dict]:
        """Every (snapshot, occurrence) pair — the corpus for evolutionary search."""
        rows = self.db.execute(
            """SELECT s.commit_sha, s.seq, o.occ_key, o.content_key, o.path, o.symbol,
                      o.start_line, o.end_line, c.code
               FROM snapshot s JOIN membership m ON m.snapshot_id = s.snapshot_id
               JOIN occurrence o ON o.occ_key = m.occ_key
               JOIN content c ON c.content_key = o.content_key
               WHERE s.repo = ? AND s.mode = ? ORDER BY s.seq""",
            (repo, mode),
        ).fetchall()
        cols = ("commit", "seq", "occ_key", "content_key", "path", "symbol", "start_line", "end_line", "code")
        return [dict(zip(cols, r)) for r in rows]
