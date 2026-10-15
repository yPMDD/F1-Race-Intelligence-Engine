"""Tests for the /agent/chat SSE endpoint.

Regression covered: the cache-hit replay used `cached_answer.split(" ")`,
which converted every newline into a space. Cached predictions therefore lost
their markdown table structure, and the dashboard's `parseMarkdownTable`
(splits on newline) rendered the answer as a wall of pipe characters.
"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers.agent import tokenize_preserving_whitespace, replay_cached


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# --- tokenizer -------------------------------------------------------------

def test_newlines_are_preserved():
    text = "line one\nline two\n\nline four"
    assert "".join(tokenize_preserving_whitespace(text)) == text


def test_markdown_table_survives_tokenisation():
    table = (
        "| Pos | Driver | Team |\n"
        "| --- | ------ | ---- |\n"
        "| 1 | VER | Red Bull Racing |\n"
    )
    chunks = tokenize_preserving_whitespace(table)
    assert "".join(chunks) == table
    # The dashboard requires >= 2 newline-terminated lines to detect a table.
    assert "".join(chunks).count("\n") == table.count("\n")


def test_tokens_keep_trailing_space():
    chunks = tokenize_preserving_whitespace("alpha beta gamma")
    assert chunks == ["alpha ", "beta ", "gamma"]


def test_collapses_are_preserved():
    chunks = tokenize_preserving_whitespace("a  b")
    assert "".join(chunks) == "a  b"


def test_empty_string():
    assert tokenize_preserving_whitespace("") == []


# --- SSE framing -----------------------------------------------------------

def _frames(body: str):
    return [
        line[len("data: "):]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_cached_replay_preserves_newlines():
    table = "| Pos | Driver |\n| --- | ------ |\n| 1 | VER |\n"
    frames = _frames(asyncio.run(_collect(replay_cached(table))))
    assert frames[-1] == "[DONE]"
    payload = "".join(json.loads(f)["token"] for f in frames[:-1])
    assert payload == table
    assert "\n" in payload, "newline lost on cache replay"


async def _collect(agen):
    return "".join([chunk async for chunk in agen])


# --- endpoint --------------------------------------------------------------

def test_chat_returns_sse_content_type(client, monkeypatch):
    import nlp.agents.graph as graph_mod

    async def fake_stream(prompt):
        yield "Hello"
        yield " world"

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", fake_stream)

    r = client.post("/agent/chat", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")


def test_chat_streams_tokens_then_done(client, monkeypatch):
    import nlp.agents.graph as graph_mod

    async def fake_stream(prompt):
        for t in ["VER", " wins", "."]:
            yield t

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", fake_stream)

    r = client.post("/agent/chat", json={"prompt": "who wins"})
    frames = _frames(r.text)
    assert frames[-1] == "[DONE]"
    tokens = [json.loads(f)["token"] for f in frames[:-1]]
    assert "".join(tokens) == "VER wins."


def test_chat_surfaces_agent_errors(client, monkeypatch):
    import nlp.agents.graph as graph_mod

    async def boom(prompt):
        raise RuntimeError("model unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", boom)

    r = client.post("/agent/chat", json={"prompt": "x"})
    frames = _frames(r.text)
    assert frames[-1] == "[DONE]"
    assert "model unavailable" in frames[0]


def test_chat_serves_from_cache_on_second_identical_call(client, monkeypatch):
    import nlp.agents.graph as graph_mod

    calls = []

    async def fake_stream(prompt):
        calls.append(prompt)
        yield "cached body\nsecond line"

    monkeypatch.setattr(graph_mod, "run_f1_agent_stream", fake_stream)

    first = client.post("/agent/chat", json={"prompt": "unique question about regs"})
    assert len(calls) == 1

    second = client.post("/agent/chat", json={"prompt": "unique question about regs"})
    assert len(calls) == 1, "second identical query should hit the cache"

    frames = _frames(second.text)
    payload = "".join(json.loads(f)["token"] for f in frames[:-1])
    assert payload == "cached body\nsecond line"


def test_chat_rejects_empty_prompt(client):
    """Pydantic still accepts an empty string, so guard explicitly."""
    r = client.post("/agent/chat", json={"prompt": ""})
    # Either a 422 from validation we add, or a streamed error; must never be
    # a 500 with a traceback.
    assert r.status_code in (200, 422)
    assert "Traceback" not in r.text
