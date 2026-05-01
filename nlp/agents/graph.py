from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from nlp.agents.tools import query_f1_regulations, get_race_telemetry_summary, get_track_history, get_ml_prediction
import logging
import json
import re
import uuid

logger = logging.getLogger(__name__)

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# Initialize the Ollama LLM
# Explicitly set base_url to avoid IPv6/IPv4 resolution issues on Windows
# streaming=True enables real-time token emission for astream calls
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://127.0.0.1:11434",
    streaming=True
)

# Bind tools to the LLM
tools = [query_f1_regulations, get_race_telemetry_summary, get_track_history, get_ml_prediction]
llm_with_tools = llm.bind_tools(tools)

# Define the logic for the agent node
def call_model(state: AgentState):
    logger.info("Agentic Node: calling model")
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    
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
        async for chunk in llm.astream(summary_prompt):
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

def run_f1_agent(query: str):
    """Legacy synchronous entry point (falls back to a simple loop)"""
    import asyncio
    
    async def _run():
        content = ""
        async for token in run_f1_agent_stream(query):
            content += token
        return content
    
    try:
        # Check if an event loop is already running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # This is tricky in FastAPI, but for now we assume run_f1_agent 
            # is used in non-async contexts or we refactor callers.
            return "Error: Use run_f1_agent_stream in async contexts."
        return loop.run_until_complete(_run())
    except Exception:
        # Fallback for complex environments
        return "Streaming error. Please retry."
