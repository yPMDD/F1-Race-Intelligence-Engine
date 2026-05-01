from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://127.0.0.1:11434"
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

def run_f1_agent(query: str):
    """
    Entry point to run the F1 Intelligence Agent.
    """
    logger.info(f"Running F1 Agent for query: {query}")
    from langchain_core.messages import SystemMessage
    from ml.config import GRID_2026
    grid_str = "\n".join([f"- {d['driver_id']}: {d['name']} ({d['team']})" for d in GRID_2026])

    system_prompt = SystemMessage(content=(
        "You are the F1 Intelligence Strategist (2026 Season). "
        f"GRID:\n{grid_str}\n\n"
        "TASKS:\n"
        "1. INFO: Answer regs/history concisely using tools.\n"
        "2. PREDICT: Call `get_ml_prediction` + `query_f1_regulations`. Output Top 10 Table ONLY.\n\n"
        "RESPONSE FORMAT (PREDICTION ONLY):\n"
        "# 🤖 AI STRATEGIC FORECAST\n> [!IMPORTANT]\n> ML + LLM Hybrid Prediction.\n"
        "| Pos | Driver | Team | Confidence (%) | Strategy Notes |\n"
        "| --- | --- | --- | --- | --- |\n\n"
        "RULES: No retired drivers. No old teams. Use tool data only. Be brief."
    ))
    inputs = {"messages": [system_prompt, HumanMessage(content=query)]}
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph
    result = app.invoke(inputs, config=config)
    
    # Extract the final answer
    final_message = result["messages"][-1]
    return final_message.content if final_message.content else "AI strategist was unable to formulate a final result. Please check telemetry data."
