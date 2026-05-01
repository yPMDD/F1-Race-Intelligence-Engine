from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from nlp.agents.tools import query_f1_regulations, get_race_telemetry_summary, get_track_history
import logging

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
tools = [query_f1_regulations, get_race_telemetry_summary, get_track_history]
llm_with_tools = llm.bind_tools(tools)

# Define the logic for the agent node
def call_model(state: AgentState):
    logger.info("Agentic Node: calling model")
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
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
    system_prompt = SystemMessage(content=(
        "You are the F1 Intelligence Engine Strategist. You have access to real telemetry and FIA regulations for 2024, 2025, and 2026. "
        "Your goal is to provide deep, comparative insights. "
        "GUIDELINES: "
        "1. To compare rules across years, YOU MUST CALL 'query_f1_regulations' MULTIPLE TIMES (once for each year you are comparing). "
        "2. Do NOT assume rules are the same. For example, DRS is replaced by Active Aero in 2026. "
        "3. To analyze performance, use 'get_race_telemetry_summary' for specific races (e.g., year=2024, race_name='Bahrain'). "
        "4. If a user asks about 'Bahrain telemetry', look up the data first. "
        "5. Correct physical impossibilities (e.g., car weight 72kg -> 726kg)."
    ))
    inputs = {"messages": [system_prompt, HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "f1_session"}}
    
    # Run the graph
    result = app.invoke(inputs, config=config)
    
    # Extract the final answer
    return result["messages"][-1].content
