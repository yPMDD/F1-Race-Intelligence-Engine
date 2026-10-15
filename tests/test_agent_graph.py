"""Tests for nlp/agents/graph.py — LLM wiring and the sync bridge.

Two regressions are covered here:

1. The Groq model id. It used to be llama-3.1-8b-instant, which Groq gates to
   Enterprise rate limits and which therefore fails on the Developer plan.
2. run_f1_agent(). It used asyncio.get_event_loop(), which raises RuntimeError
   on 3.12+ when the calling thread has no loop; the bare except swallowed it
   and returned "Streaming error. Please retry." -- which then got written into
   the predictions table.
"""
import asyncio
import threading

import pytest

from nlp.agents import graph as graph_mod


def test_default_model_is_qwen_not_retired_llama():
    assert graph_mod.get_model_name() == "qwen/qwen3.8-27b"
    assert "llama-3.1-8b-instant" not in graph_mod.get_model_name()


def test_model_name_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    assert graph_mod.get_model_name() == "some/other-model"


def test_max_tokens_is_capped_below_the_free_tier_otpm_budget():
    """Regression: every agent call 429'd until max_tokens was capped.

    Groq's free tier allows 1000 output tokens/minute, while
    qwen/qwen3.8-27b defaults to a 16384 max completion. Sending the model
    default makes the provider reject the request outright with
    "Request too large ... on output tokens per minute (OTPM): Limit 1000".
    """
    assert graph_mod.DEFAULT_MAX_TOKENS <= 1000
    assert graph_mod.get_max_tokens() <= 1000


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MAX_TOKENS", "4096")
    assert graph_mod.get_max_tokens() == 4096


def test_max_tokens_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("GROQ_MAX_TOKENS", "not-a-number")
    assert graph_mod.get_max_tokens() == graph_mod.DEFAULT_MAX_TOKENS


def test_get_llm_raises_helpfully_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    graph_mod.reset_llm_cache()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        graph_mod.get_llm()


def test_module_imports_without_api_key(monkeypatch):
    """Importing the module must not require a key (routes /tests need it)."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    graph_mod.reset_llm_cache()
    import importlib
    importlib.reload(graph_mod)
    assert hasattr(graph_mod, "run_f1_agent_stream")


def test_tools_are_bound():
    names = {t.name for t in graph_mod.tools}
    assert names == {
        "query_f1_regulations",
        "get_race_telemetry_summary",
        "get_track_history",
        "get_ml_prediction",
    }


# --- run_f1_agent sync bridge ---------------------------------------------

def test_run_f1_agent_returns_full_text(monkeypatch):
    async def fake_stream(q):
        for tok in ["Hello", " ", "world"]:
            yield tok

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", fake_stream)
    assert graph_mod.run_f1_agent("anything") == "Hello world"


def test_run_f1_agent_works_with_no_event_loop_current(monkeypatch):
    """The exact condition that used to raise RuntimeError.

    On 3.12+ `asyncio.get_event_loop()` raises if the thread has no current
    loop. We clear the policy's current loop first to reproduce it.
    """
    async def fake_stream(q):
        yield "ok"

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", fake_stream)

    # Ensure this thread definitely has no loop bound.
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass

    with pytest.raises(RuntimeError):
        asyncio.get_event_loop()

    assert graph_mod.run_f1_agent("anything") == "ok"


def test_run_f1_agent_works_when_called_from_inside_a_running_loop(monkeypatch):
    """Callers that are already async must not deadlock."""
    async def fake_stream(q):
        yield "from-async"

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", fake_stream)

    async def caller():
        return graph_mod.run_f1_agent("anything")

    assert asyncio.run(caller()) == "from-async"


def test_run_f1_agent_propagates_errors_instead_of_swallowing(monkeypatch):
    """Must raise, not return the 'Streaming error' placeholder string."""
    async def boom(q):
        yield "partial"
        raise ValueError("groq exploded")

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", boom)
    with pytest.raises(RuntimeError, match="groq exploded"):
        graph_mod.run_f1_agent("anything")


def test_run_f1_agent_does_not_leak_threads():
    before = threading.active_count()

    async def fake_stream(q):
        yield "x"

    original = graph_mod.run_f1_agent_stream
    graph_mod.run_f1_agent_stream = fake_stream
    try:
        for _ in range(5):
            graph_mod.run_f1_agent("q")
    finally:
        graph_mod.run_f1_agent_stream = original

    assert threading.active_count() <= before + 1
