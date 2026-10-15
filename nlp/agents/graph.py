from typing import Annotated, TypedDict, List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from nlp.agents.tools import query_f1_regulations, get_race_telemetry_summary, get_track_history, get_ml_prediction
import logging
import json
import re
import uuid

load_dotenv()

logger = logging.getLogger(__name__)

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- Groq LLM ---------------------------------------------------------------
# NOTE: previously `llama-3.1-8b-instant`. Groq still lists that model but
# gates it to Enterprise ("Contact Sales") rate limits, so it 402s/429s for
# anyone on the free Developer plan. qwen/qwen3.8-27b is on the Developer tier
# (250K TPM / 1K RPM, 450 t/s) and is a far stronger model for the tool-calling
# work this agent does.
# Override with GROQ_MODEL if the account is provisioned differently.
DEFAULT_MODEL = "qwen/qwen3.8-27b"

tools = [query_f1_regulations, get_race_telemetry_summary, get_track_history, get_ml_prediction]

# Bind tools to the LLM
_llm_with_tools = None


def get_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def get_llm():
    """Build the ChatGroq client on first use.

    Constructed lazily rather than at import time so that importing this module
    (for tests, or for the `/races` and `/rag` routes, which need neither the
    LLM nor an API key) does not hard-fail when GROQ_API_KEY is unset.
    """
    global _llm_with_tools
    if _llm_with_tools is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) and restart the API."
            )
        llm = ChatGroq(
            model=get_model_name(),
            temperature=0,
            api_key=api_key,
            streaming=True,
        )
        _llm_with_tools = llm.bind_tools(tools)
    return _llm_with_tools


def reset_llm_cache():
    """Drop the memoised client (used by tests and after a key change)."""
    global _llm_with_tools
    _llm_with_tools = None


# Back-compat aliases: the previous module-level `llm` / `llm_with_tools` names
# were imported directly by other modules.
def __getattr__(name):
    if name in ("llm", "llm_with_tools"):
        return get_llm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define the logic for the agent node
def call_model(state: AgentState):
    logger.info("Agentic Node: calling model")
    messages = state['messages']
    response = get_llm().invoke(messages)
    
    # --- Robust Manual Parsing Fallback ---
    # If the model returned a JSON string in content but no formal tool_calls
    if not response.tool_calls and response.content:
        try:
            # Look for JSON patterns like {"name": "...", "parameters": {...}}
            matches = re.findall(r'\{.*"name".*"parameters".*\}', response.content, re.DOTALL)
            for match in matches:
                data = json.loads(match)
                # Map common hallucinations to actual tool names
                name_map = {
                    "get_track_profile": "get_track_history", 
                    "query_regulations": "query_f1_regulations",
                    "get_telemetry": "get_race_telemetry_summary"
                }
                actual_name = name_map.get(data["name"], data["name"])
                
                # Check if it's one of our registered tools
                if actual_name in [t.name for t in tools]:
                    response.tool_calls.append({
                        "name": actual_name,
                        "args": data.get("parameters", data.get("args", {})),
                        "id": f"manual_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call"
                    })
                    logger.info(f"Manual Tool Parse Success: {actual_name}")
        except Exception as e:
            logger.warning(f"Failed to manually parse tool call: {e}")
            
    return {"messages": [response]}

# Define the logic for the tool execution node
tool_node = ToolNode(tools)

# Define the condition to continue or end
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    # If there are tool calls, we continue to the tool node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop
    return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Authoritative 2025->2026 regulation change fact sheet.
# Injected directly into every prompt so the LLM never needs RAG for comparisons.
REG_CHANGES_2026 = """
## KEY 2025 -> 2026 REGULATION CHANGES (AUTHORITATIVE - use these facts directly)

| Specification          | 2025 (Old)              | 2026 (New)                   | Delta       |
|------------------------|-------------------------|------------------------------|-------------|
| Minimum Car Weight     | 798 kg (no fuel)        | 768 kg (no fuel)             | -30 kg      |
| Car Width (max)        | 2,000 mm                | 1,900 mm                     | -100 mm     |
| Car Length (max)       | 5,600 mm                | 5,200 mm                     | -400 mm     |
| Front Wheel Diameter   | 18 inch                 | 18 inch                      | No change   |
| Power Unit             | 1.6L V6 Hybrid          | 1.6L V6 + stronger MGU-H    | Revised hybrid |
| Electrical Power Share | ~160 kW (MGU-K)         | ~350 kW (50/50 electric split) | +190 kW  |
| DRS                    | Moveable rear wing      | Active Aero (F-Duct style)   | DRS removed |
| Budget Cap (operations)| $135 million            | $130 million                 | -$5 million |
| Teams on grid          | 10 teams / 20 drivers   | 11 teams / 22 drivers (Cadillac added) | +1 team |
"""

async def run_f1_agent_stream(query: str):
    """
    Async generator that yields tokens/events from the F1 Agent.
    """
    logger.info(f"Running F1 Agent Stream for query: {query}")
    from langchain_core.messages import SystemMessage
    from ml.config import GRID_2026

    q_lower = query.lower()

    # --- SHORT-CIRCUIT: Simple factual queries about 2026 regs ---
    # EXCLUDE comparison queries (difference/vs/change/compare/between) — they
    # need the full fact sheet context, not just a single-year RAG retrieval.
    comparison_keywords = ["difference", "vs", "versus", "change", "compare",
                           "compared", "between", "2025", "old", "new reg", "before"]
    simple_keywords = ["what is", "minimum", "maximum", "weight", "dimension",
                       "article", "regulation", "rule", "how many", "define"]
    is_comparison = any(kw in q_lower for kw in comparison_keywords)
    is_simple_factual = any(kw in q_lower for kw in simple_keywords)

    if is_simple_factual and not is_comparison:
        from nlp.agents.tools import query_f1_regulations
        raw_result = await query_f1_regulations.ainvoke({"query": query, "year": 2026})

        summary_prompt = [
            SystemMessage(content=(
                "You are a precise F1 Technical Analyst for the 2026 season.\n"
                "You have the following authoritative regulation change data:\n"
                f"{REG_CHANGES_2026}\n"
                "Answer the user's question using this data first, then supplement with the "
                "regulation text below if needed. Be factual and cite article numbers."
            )),
            HumanMessage(content=f"Question: {query}\n\nAdditional Regulation Text:\n{raw_result}")
        ]
        async for chunk in get_llm().astream(summary_prompt):
            if chunk.content:
                yield chunk.content
        return

    # --- FULL AGENT LOOP ---
    grid_str = "\n".join([f"- {d['driver_id']}: {d['name']} ({d['team']})" for d in GRID_2026])

    system_prompt = SystemMessage(content=(
        "You are the F1 Intelligence Strategist for the 2026 Season.\n"
        "You have access to tools for live race data, historical performance, FIA regulations, and ML predictions.\n\n"
        f"{REG_CHANGES_2026}\n"
        f"2026 OFFICIAL GRID:\n{grid_str}\n\n"
        "RULES:\n"
        "- For regulation comparisons (2025 vs 2026), use the KEY CHANGES table above — do NOT say the data is missing.\n"
        "- For predictions: ALWAYS call `get_ml_prediction` first.\n"
        "- DO NOT use a table format unless specifically asked for a race classification or prediction.\n"
        "- Keep answers concise, factual, and strategic.\n\n"
        "PREDICTION OUTPUT FORMAT (ONLY for race predictions/classifications):\n"
        "| Pos | Driver | Team | Confidence | Strategy Notes |\n"
        "| --- | ------ | ---- | ---------- | -------------- |\n"
    ))

    inputs = {"messages": [system_prompt, HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # stream_mode="messages" yields (AIMessageChunk, metadata) tuples
    # We filter for chunks from the 'agent' node only (not tool results)
    async for event in app.astream(inputs, config=config, stream_mode="messages"):
        message, metadata = event
        if isinstance(message, AIMessageChunk) and metadata.get("langgraph_node") == "agent":
            # Only yield non-empty text chunks (skip tool-call chunks)
            if message.content and isinstance(message.content, str):
                yield message.content

def run_f1_agent(query: str) -> str:
    """Blocking entry point: drains the stream and returns the full text.

    Used by `apps/api/routers/race.py` (/races/{id}/predict) and
    `ml/precompute_all.py`, both of which are sync call sites (FastAPI runs
    sync endpoints in a threadpool, so there is no running loop there).

    The old implementation called `asyncio.get_event_loop()` unguarded. On
    Python 3.12+ that raises RuntimeError when the calling thread has no
    current event loop, and the bare `except Exception` below swallowed it --
    so every prediction silently persisted the literal string
    "Streaming error. Please retry." into the database. Run in a *fresh*
    thread when a loop may already be running in this one, and always close it.
    """
    import asyncio
    import threading

    async def _run():
        content = ""
        async for token in run_f1_agent_stream(query):
            content += token
        return content

    result: dict = {}

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(_run())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            result["error"] = exc
        finally:
            try:
                asyncio.set_event_loop(None)
            finally:
                loop.close()

    # A dedicated thread guarantees no event loop is bound to it, so this is
    # safe whether or not the caller is already inside a loop.
    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        logger.error("run_f1_agent failed for %r: %s", query, result["error"])
        raise RuntimeError(f"Agent execution failed: {result['error']}") from result["error"]

    return result.get("value", "")
