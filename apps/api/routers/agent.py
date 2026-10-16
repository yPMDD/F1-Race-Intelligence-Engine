import asyncio
import json
import logging
import random
import re

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


class AgentQuery(BaseModel):
    prompt: str


class AgentResponse(BaseModel):
    answer: str


# Emitted on cache hits so a replayed answer doesn't land instantly and look
# canned. Purely cosmetic; both bounds are seconds.
CACHED_TOTAL_DELAY_RANGE = (0.8, 1.5)
CACHED_PER_WORD_DELAY_RANGE = (0.02, 0.045)


def tokenize_preserving_whitespace(text: str) -> list:
    """Split into word-ish chunks that keep their trailing whitespace.

    The previous implementation used `text.split(" ")` and re-joined with a
    single space. That silently converted every newline into a space, so a
    cached prediction lost its markdown table structure and the dashboard's
    table parser rendered nothing but a wall of pipe characters.
    """
    return re.findall(r"\S+\s*|\s+", text)


async def replay_cached(answer: str):
    """Stream a cached answer back with a humanised cadence."""
    await asyncio.sleep(random.uniform(*CACHED_TOTAL_DELAY_RANGE))
    for chunk in tokenize_preserving_whitespace(answer):
        yield f"data: {json.dumps({'token': chunk})}\n\n"
        await asyncio.sleep(random.uniform(*CACHED_PER_WORD_DELAY_RANGE))
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat_with_agent(query: AgentQuery):
    """
    Interact with the F1 Intelligence Agent with real-time streaming.
    """
    # Imported lazily: pulling in the agent graph drags in LangGraph, the Groq
    # client and the tool bindings, none of which the other routes need.
    from nlp.cache.semantic_cache import cache_instance
    from nlp.agents.graph import run_f1_agent_stream

    logger.info("Agent chat request: %s", query.prompt)

    # 1. Semantic cache.
    cached_answer = cache_instance.get(query.prompt)
    if cached_answer:
        logger.info("Semantic cache hit for %r", query.prompt)
        return StreamingResponse(
            replay_cached(cached_answer), media_type="text/event-stream"
        )

    # 2. Run the agent with real streaming.
    async def stream_agent():
        full_response = ""
        try:
            async for token in run_f1_agent_stream(query.prompt):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            if full_response:
                cache_instance.set(query.prompt, full_response)

            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Agent execution failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_agent(), media_type="text/event-stream")
