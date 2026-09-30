"""Incremental commit indexing must give the same search results as a full rebuild."""

import hashlib
import subprocess

import numpy as np
import pytest

from prism.cache import EmbeddingStore
from prism.retrievers import code_tokenize
from prism.versioning.extract import extract_js
from prism.versioning.indexer import CommitIndexer
from prism.versioning.search import SnapshotSearcher, search_history
from prism.versioning.store import SnapshotStore

DIM = 256


def hash_embed(texts):
    """Deterministic bag-of-hashed-tokens embedding: fast, no model download."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in code_tokenize(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


def embed_query(q):
    return hash_embed([q])[0]


def sh(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def commit(repo, files: dict, msg: str, delete=(), rename=()):
    for old, new in rename:
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        sh(repo, "mv", old, new)
    for p in delete:
        sh(repo, "rm", "-q", p)
    for p, src in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(src)
        sh(repo, "add", p)
    sh(repo, "commit", "-q", "-m", msg)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


AUTH_V1 = """function checkPassword(user, pw) {
  return user.password === pw;
}
function login(user, pw) {
  if (!checkPassword(user, pw)) throw new Error('bad');
  return createSession(user);
}
"""
AUTH_V2 = """const bcrypt = require('bcrypt');
function checkPassword(user, pw) {
  return bcrypt.compareSync(pw, user.hash);
}
function login(user, pw) {
  if (!checkPassword(user, pw)) throw new Error('bad');
  return createSession(user);
}
"""
UTIL = """const normalize = (str) => str.trim().toLowerCase();
class Cache {
  get(key) { return this.map.get(key); }
  set(key, value) { this.map.set(key, value); }
}
"""


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    sh(r, "init", "-q")
    sh(r, "config", "user.email", "t@t")
    sh(r, "config", "user.name", "t")
    c = [
        commit(r, {"src/auth.js": AUTH_V1, "src/util.js": UTIL, "README.md": "x"}, "initial"),
        commit(r, {"src/auth.js": AUTH_V2}, "auth refactor: bcrypt"),                     # modify
        commit(r, {"src/session.js": "function createSession(u) { return {u}; }\n"}, "add session"),  # add
        commit(r, {}, "move util", rename=[("src/util.js", "src/lib/util.js")]),          # rename
        commit(r, {}, "drop session", delete=["src/session.js"]),                         # delete
    ]
    return r, c


def _indexers(tmp_path):
    store = SnapshotStore(tmp_path / "idx.sqlite")
    emb = EmbeddingStore({"model": "hash256"}, root=tmp_path / "emb")
    return store, emb, CommitIndexer(store, emb, hash_embed, "fixture")


QUERIES = ["check password hash", "create a session for user", "normalize string", "cache get key", "login"]


def test_incremental_equals_full(repo, tmp_path):
    r, commits = repo
    store, emb, ix = _indexers(tmp_path)
    searcher = SnapshotSearcher(store, emb, embed_query, hash_embed)
    for c in commits:
        inc = ix.index_commit(r, c, mode="incremental")
        full = ix.index_commit(r, c, mode="full")
        inc_occ = {o["occ_key"] for o in store.occurrences(inc)}
        full_occ = {o["occ_key"] for o in store.occurrences(full)}
        assert inc_occ == full_occ, c
        for q in QUERIES:
            a = [(h.snippet_id, round(h.score, 9)) for h in searcher.retrieve(q, inc, k=10)]
            b = [(h.snippet_id, round(h.score, 9)) for h in searcher.retrieve(q, full, k=10)]
            assert a == b, (c, q)


def test_snapshot_isolation_and_reuse(repo, tmp_path):
    r, commits = repo
    store, emb, ix = _indexers(tmp_path)
    sids = [ix.index_commit(r, c) for c in commits]
    paths = [{o["path"] for o in store.occurrences(s)} for s in sids]
    assert "src/session.js" in paths[2] and "src/session.js" not in paths[4]
    assert "src/lib/util.js" in paths[3] and "src/util.js" not in paths[3]

    stats = {s["commit"]: s["stats"] for s in store.snapshots("fixture")}
    # auth.js modify: only checkPassword's text changed -> one new embedding
    assert stats[commits[1]]["embedded_new_content"] == 1
    # rename: code unchanged -> nothing re-embedded
    assert stats[commits[3]]["embedded_new_content"] == 0
    # delete: nothing parsed, nothing embedded
    assert stats[commits[4]]["files_parsed"] == 0

    searcher = SnapshotSearcher(store, emb, embed_query, hash_embed)
    hits = searcher.retrieve("createSession", sids[4], k=20)
    assert all(h.path != "src/session.js" for h in hits)


def test_history_distinguishes_versions(repo, tmp_path):
    r, commits = repo
    store, emb, ix = _indexers(tmp_path)
    for c in commits:
        ix.index_commit(r, c)
    res = search_history(store, emb, "fixture", "checkPassword compare password", embed_query, hash_embed, k=5)
    cp = [h for h in res if h["symbol"] == "checkPassword"]
    assert len(cp) == 2  # both versions retrievable, as distinct results
    assert {h["first_commit"] for h in cp} == {commits[0], commits[1]}
    assert all(h["diff_vs_nearest_version"] and "bcrypt" in h["diff_vs_nearest_version"] for h in cp)


def test_extract_assignment_style():
    src = """var res = module.exports = { __proto__: http.ServerResponse.prototype };
res.send = function send(body) { return this.end(body); };
exports.etag = createETagGenerator({ weak: false });
exports.compile = (x) => x + 1;
module.exports = function createApplication() { return app; };
var handlers = { handle: function (req) { return req; } };
defineGetter(req, 'protocol', function protocol(){ return this.socket.encrypted ? 'https' : 'http'; });
defineGetter(req, 'fresh', function(){ return fresh(this.headers); });
"""
    names = [s.symbol for s in extract_js("lib/response.js", src)]
    assert names == ["res.send", "exports.compile", "module.exports", "handle", "req.protocol", "req.fresh"]


def test_extract_symbols():
    names = [s.symbol for s in extract_js("u.js", UTIL)]
    assert names == ["normalize", "Cache.get", "Cache.set"]
    login = [s for s in extract_js("a.js", AUTH_V1) if s.symbol == "login"][0]
    assert "checkPassword" in login.callees and "createSession" in login.callees
