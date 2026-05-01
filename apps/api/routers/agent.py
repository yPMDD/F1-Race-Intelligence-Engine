from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from nlp.agents.graph import run_f1_agent
import logging
import json
import asyncio
import random

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)

class AgentQuery(BaseModel):
    prompt: str

class AgentResponse(BaseModel):
    answer: str

@router.post("/chat")
async def chat_with_agent(query: AgentQuery):
    """
    Interact with the F1 Intelligence Agent with real-time streaming.
    """
    from nlp.cache.semantic_cache import cache_instance
    from nlp.agents.graph import run_f1_agent_stream

    logger.info(f"Agent chat request: {query.prompt}")

    # 1. Check Semantic Cache first
    cached_answer = cache_instance.get(query.prompt)
    if cached_answer:
        logger.info("Semantic Cache Hit!")
        async def stream_cached():
            # Brief randomized 'thinking' pause (0.8 - 1.5s) before first token
            # so cached answers don't feel instant/robotic
            await asyncio.sleep(random.uniform(0.8, 1.5))
            words = cached_answer.split(" ")
            for word in words:
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                # Small per-word delay (20-45ms) for a natural typing feel
                await asyncio.sleep(random.uniform(0.02, 0.045))
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_cached(), media_type="text/event-stream")

    # 2. Run Agent with real streaming
    async def stream_agent():
        full_response = ""
        try:
            async for token in run_f1_agent_stream(query.prompt):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Cache the result after full generation
            if full_response:
                cache_instance.set(query.prompt, full_response)
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_agent(), media_type="text/event-stream")
