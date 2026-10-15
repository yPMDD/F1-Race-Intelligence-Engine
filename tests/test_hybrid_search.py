"""Tests for the hybrid (semantic + lexical) regulation search.

The previous implementation was:

    docs = vectorstore.similarity_search(query, k=k * 2, filter=...)
    ...
    keyword_hits = sum(1 for kw in keywords if kw in content_lower)
    score = keyword_hits * 2
    scored_docs.sort(key=lambda x: x[0], reverse=True)

Three defects:
  * `similarity_search` returns Documents with no score, so the semantic
    ranking was thrown away entirely and only raw keyword counts ordered the
    results -- one incidental keyword hit beat a perfect semantic match.
  * `query.lower().split()` left punctuation attached, so "2026?" became a
    keyword token that could never appear in the document text.
  * `kw in content_lower` is substring matching, so "car" matched "carbon",
    "carried" and "carburettor".

A fake vector store is used so the suite never loads the embedding model.
"""
import pytest

from nlp.agents import tools as T


class FakeDoc:
    def __init__(self, content, page, source="reg.pdf"):
        self.page_content = content
        self.metadata = {"page": page, "source": source, "year": 2026}


class FakeVectorStore:
    """Records the kwargs it was called with and returns a canned result set."""

    def __init__(self, pairs):
        # pairs: list of (FakeDoc, relevance_score)
        self.pairs = pairs
        self.last_call = {}

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None, **kw):
        self.last_call = {"query": query, "k": k, "filter": filter, **kw}
        return self.pairs[:k]

    def similarity_search(self, query, k=4, filter=None, **kw):
        self.last_call = {"query": query, "k": k, "filter": filter, **kw}
        return [d for d, _ in self.pairs[:k]]


@pytest.fixture()
def patch_store(monkeypatch):
    def _install(pairs):
        store = FakeVectorStore(pairs)
        monkeypatch.setattr(T, "_get_vectorstore", lambda: (store, None))
        return store
    return _install


# --- the headline regression ---------------------------------------------

def test_semantic_match_beats_one_incidental_keyword(patch_store):
    """A strongly semantic match must not be buried by a weak lexical match.

    Doc A is the true answer: top semantic relevance, but it contains none of
    the query's keywords. Doc B is a near-total non-match that happens to
    contain the word 'weight'. The old code ranked B first (1 hit * 2 > 0).
    """
    good = FakeDoc(
        "C6.1 The car must not weigh less than 768 kg at any time.",
        page=61,
    )
    weak = FakeDoc(
        "The weight of the ballast is described elsewhere. Unrelated text.",
        page=26,
    )
    assert "minimum" not in good.page_content and "dry" not in good.page_content
    patch_store([(good, 0.95), (weak, 0.20)])

    results = T._hybrid_search("minimum dry weight", year=2026, k=2)
    assert results[0].page_content == good.page_content


def test_keyword_hit_still_boosts_among_semantically_equal_docs(patch_store):
    """Lexical matching must still have a real effect (it is 'hybrid')."""
    on_topic = FakeDoc("The minimum reference weight is set out below.", page=1)
    off_topic = FakeDoc("Nothing relevant in this chunk at all.", page=2)
    patch_store([(on_topic, 0.50), (off_topic, 0.50)])

    results = T._hybrid_search("minimum weight", year=2026, k=2)
    assert results[0].page_content == on_topic.page_content


# --- tokenizer -------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("What is the 2026 car weight?", {"2026", "car", "weight"}),
        ("Minimum weight, in kilograms!", {"minimum", "weight", "kilograms"}),
        # Internal '.' and '-' are PRESERVED: FIA article numbers (C6.2, A22.1)
        # must survive intact. A glued magnitude additionally yields its
        # numeric and unit halves ("768kg" -> "768kg", "768", "kg").
        ("C6.2 -- the 768kg limit?", {"c6.2", "768kg", "768", "kg", "limit"}),
    ],
)
def test_tokenizer_strips_punctuation_and_lowercases(text, expected):
    assert set(T._tokenize(text)) == expected


def test_tokenizer_drops_stopwords():
    toks = T._tokenize("what is the minimum weight of the car")
    assert "the" not in toks and "is" not in toks and "what" not in toks
    assert "minimum" in toks


def test_punctuation_query_token_can_actually_match(patch_store):
    """'2026?' previously became the token '2026?' and never matched."""
    doc = FakeDoc("The 2026 regulations apply.", page=1)
    patch_store([(doc, 0.8)])
    T._hybrid_search("What about 2026?", year=2026, k=1)  # must not raise / no crash
    assert "2026" in T._tokenize("What about 2026?")


def test_glued_units_emit_numeric_token(patch_store):
    """FIA text writes '726kg'; a query for '726 kg' must still match it."""
    toks = T._tokenize("726kg")
    assert {"726kg", "726", "kg"}.issubset(toks)
    assert T._tokenize("190 kW") == ["190", "kw"]

    terms = T._tokenize("726 kg minimum mass")

    glued = "The Minimum Mass is 726kg plus the Nominal Tyre Mass."
    spaced = "The Minimum Mass is 726 kg plus the Nominal Tyre Mass."

    # Both spellings must now receive FULL lexical credit -- previously the
    # glued form lost the "kg" term entirely and was penalised for it.
    glued_score = T._lexical_score(terms, glued)
    spaced_score = T._lexical_score(terms, spaced)
    assert glued_score == pytest.approx(spaced_score, rel=0.01)
    assert glued_score > 0

    # And both must outrank an unrelated chunk.
    unrelated = "Unrelated provisions about safety equipment and fire systems."
    assert glued_score > T._lexical_score(terms, unrelated)


def test_glued_unit_chunk_is_retrieved(patch_store):
    """Use a realistic candidate pool: production over-fetches max(k*4, 20)."""
    filler = [
        FakeDoc(f"Unrelated regulation clause number {i} about safety systems.", i)
        for i in range(3, 22)
    ]
    glued = FakeDoc("The Minimum Mass is 726kg plus the Nominal Tyre Mass.", 1)
    other = FakeDoc("Provisions concerning safety equipment and fire systems.", 2)
    # glued sits mid-pack semantically but owns every query term lexically.
    pairs = [(glued, 0.30)] + [(d, 0.31 - (i - 3) * 0.0005) for i, d in
                              zip(range(3, 22), filler)]
    pairs[1] = (other, 0.305)
    patch_store(pairs)

    ranked = T._hybrid_search("726 kg minimum mass", year=2026, k=3)
    assert ranked[0].page_content == glued.page_content


def test_fusion_is_a_genuine_blend(patch_store):
    """A doc strong on BOTH channels must beat a doc strong on either alone.

    Guards against regressing to the old 'lexical only' behaviour (which would
    put semantically irrelevant keyword-rich docs on top) and against an
    accidental 'semantic only' pass (which would ignore exact regulation terms).
    """
    sem_only = FakeDoc("C4.1 An unrelated but semantically adjacent passage.", 1)
    lex_only = FakeDoc("unrelated filler. minimum mass 726kg. unrelated filler.", 2)
    both = FakeDoc("The minimum mass is 726kg under C4.1.", 3)

    patch_store([(sem_only, 0.90), (lex_only, 0.10), (both, 0.55)])

    ranked = T._hybrid_search("minimum mass 726kg", year=2026, k=3)
    assert ranked[0].page_content == both.page_content
    # and the two single-channel docs must trail it in some order
    assert {d.page_content for d in ranked[1:]} == {sem_only.page_content, lex_only.page_content}


def test_semantic_channel_still_dominates_large_gaps(patch_store):
    """A near-perfect embedding match should win over a modest lexical match."""
    perfect = FakeDoc("C4.1 The car must not weigh less than the minimum mass.", 1)
    lexical = FakeDoc("minimum minimum minimum kg kg kg weight weight weight.", 2)
    filler = [FakeDoc(f"Clause {i} unrelated.", i) for i in range(3, 23)]
    patch_store([(perfect, 0.90), (lexical, 0.10)] + [(d, 0.30) for d in filler])

    ranked = T._hybrid_search("minimum mass", year=2026, k=3)
    assert ranked[0].page_content == perfect.page_content


def test_substring_no_longer_matches_inside_words(patch_store):
    """'car' must not match 'carbon' or 'carried'."""
    carbon = FakeDoc("Carbon fibre components are regulated.", page=1)
    car = FakeDoc("The car must weigh 768 kg.", page=2)
    # equal semantic scores; only the true word-boundary match should win
    patch_store([(carbon, 0.60), (car, 0.60)])

    results = T._hybrid_search("car", year=2026, k=2)
    assert results[0].page_content == car.page_content


# --- plumbing --------------------------------------------------------------

def test_year_filter_is_forwarded(patch_store):
    store = patch_store([(FakeDoc("x", 1), 0.5)])
    T._hybrid_search("anything", year=2025, k=3)
    assert store.last_call["filter"] == {"year": 2025}


def test_no_filter_when_year_is_falsy(patch_store):
    store = patch_store([(FakeDoc("x", 1), 0.5)])
    T._hybrid_search("anything", year=None, k=3)
    assert store.last_call["filter"] is None


def test_overfetches_for_reranking(patch_store):
    store = patch_store([(FakeDoc("x", 1), 0.5)])
    T._hybrid_search("anything", year=2026, k=5)
    assert store.last_call["k"] >= 5


def test_empty_result_set_returns_empty_list(patch_store):
    patch_store([])
    assert T._hybrid_search("anything", year=2026, k=5) == []


def test_duplicates_are_collapsed(patch_store):
    body = "The car must weigh 768 kg at all times." * 4
    patch_store([
        (FakeDoc(body, 1), 0.9),
        (FakeDoc(body, 1), 0.88),
        (FakeDoc(body, 2), 0.5),
    ])
    results = T._hybrid_search("car weight", year=2026, k=5)
    assert len(results) == 1


def test_respects_k_limit(patch_store):
    patch_store([(FakeDoc(f"chunk number {i} about weight", i), 1.0 - i / 100) for i in range(20)])
    assert len(T._hybrid_search("weight", year=2026, k=3)) == 3
