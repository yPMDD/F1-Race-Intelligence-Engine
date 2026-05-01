from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from nlp.agents.tools import query_2026_regulations, get_race_telemetry_summary
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
tools = [query_2026_regulations, get_race_telemetry_summary]
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
        "You are the F1 Race Strategist AI, an expert in 2026 FIA Technical and Sporting Regulations. "
        "When answering questions, prioritize accuracy and technical precision. "
        "CRITICAL: PDF parsing can sometimes introduce errors like spaces in numbers (e.g., '72 6kg' instead of '726kg'). "
        "Use your domain knowledge to perform sanity checks: an F1 car should weigh around 700-800kg, not 70kg. "
        "If you encounter a value that seems physically impossible, correct it based on context or mention the discrepancy."
    ))
    inputs = {"messages": [system_prompt, HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "f1_session"}}
    
    # Run the graph
    result = app.invoke(inputs, config=config)
    
    # Extract the final answer
    return result["messages"][-1].content
