"""Tests for the SQLite semantic cache (nlp/cache/semantic_cache.py).

The embedding model is stubbed out: these tests are about cache mechanics
(threshold behaviour, vector round-tripping, isolation), not about the
quality of all-MiniLM-L6-v2.
"""
import numpy as np
import pytest

from nlp.cache import semantic_cache as sc_mod


class FakeEmbeddings:
    """Deterministic, cheap stand-in for HuggingFaceEmbeddings.

    Embeds a query into a bag-of-characters-counts vector so that strings
    sharing many characters land close together -- enough structure to exercise
    the cosine threshold without downloading a transformer.
    """

    def __init__(self, dim=64):
        self.dim = dim

    def _vec(self, text: str):
        v = np.zeros(self.dim, dtype=np.float32)
        for ch in text.lower().encode("utf-8", "ignore"):
            v[ch % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_query(self, text: str):
        return self._vec(text)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sc_mod, "HuggingFaceEmbeddings", lambda **kw: FakeEmbeddings())
    return sc_mod.SemanticCache(db_path=str(tmp_path / "cache.db"))


def test_set_then_get_exact_query(cache):
    cache.set("What is the minimum car weight?", "768 kg")
    assert cache.get("What is the minimum car weight?") == "768 kg"


def test_get_returns_none_for_unrelated_query(cache):
    cache.set("What is the minimum car weight?", "768 kg")
    # 'zzzz' shares no characters with the stored query -> orthogonal vectors
    assert cache.get("zzzz") is None


def test_threshold_is_respected(cache):
    cache.set("minimum car weight", "768 kg")
    # force an unreachable threshold: everything is a miss
    assert cache.get("minimum car weight", threshold=1.01) is None


def test_near_duplicate_hits_with_default_threshold(cache):
    cache.set("What is the 2026 minimum car weight in kilograms",
              "768 kg")
    hit = cache.get("What is the 2026 minimum car weight in kg")
    assert hit == "768 kg", "paraphrase of the same question should hit"


def test_vector_roundtrip_is_normalised(cache):
    cache.set("alpha beta", "ok")
    row = cache.conn.execute(
        "SELECT query_vector FROM cache WHERE query = ?", ("alpha beta",)
    ).fetchone()
    stored = np.frombuffer(row[0], dtype=np.float32)
    assert stored.shape[0] == 64
    assert np.isclose(np.linalg.norm(stored), 1.0, atol=1e-5)


def test_cache_creates_parent_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sc_mod, "HuggingFaceEmbeddings", lambda **kw: FakeEmbeddings())
    nested = tmp_path / "deep" / "nested" / "c.db"
    c = sc_mod.SemanticCache(db_path=str(nested))
    assert nested.parent.exists()


def test_empty_cache_returns_none(cache):
    assert cache.get("anything at all") is None


def test_multiple_entries_best_match_wins(cache):
    cache.set("zzzz yyyy xxxx", "wrong")
    cache.set("minimum car weight", "right")
    assert cache.get("minimum car weight") == "right"
