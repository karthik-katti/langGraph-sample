# agent.py
import os
import datetime
from typing import TypedDict, Annotated, List, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage


# 1. Define the custom tool
@tool
def get_current_time() -> str:
    """Returns the current date and time as a string."""
    return str(datetime.datetime.now())


# 2. Define the LLM and bind the tool
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_current_time]
llm_with_tools = llm.bind_tools(tools)


# 3. Define the state of the graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. Define the nodes for the graph
def call_llm(state: AgentState):
    """Invokes the LLM to determine the next action."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def call_tool(state: AgentState):
    """Executes the tool call and returns the result."""
    last_message = state["messages"][-1]
    tool_outputs = []

    # Process each tool call that the LLM requested
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]

        if tool_name == "get_current_time":
            # Invoke the custom tool
            output = get_current_time.invoke({})
            tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

    return {"messages": tool_outputs}


# 5. Define the conditional logic for the agent
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # Check if the LLM invoked a tool call
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"


# 6. Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool", call_tool)

workflow.add_edge(START, "llm")
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {"continue": "tool", "end": END}
)
workflow.add_edge("tool", "llm")

# 7. Compile the graph
agent_executor = workflow.compile()
