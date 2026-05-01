from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from nlp.agents.graph import run_f1_agent
import logging

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

@router.post("/chat", response_model=AgentResponse)
async def chat_with_agent(query: AgentQuery):
    """
    Interact with the F1 Intelligence Agent. 
    The agent uses LangGraph to coordinate between telemetry data and FIA regulations.
    """
    logger.info(f"Agent chat request: {query.prompt}")
    try:
        # Note: Since run_f1_agent uses invoke (synchronous), 
        # in a real high-load scenario we'd use aync ainvoke.
        answer = run_f1_agent(query.prompt)
        return AgentResponse(answer=answer)
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")
