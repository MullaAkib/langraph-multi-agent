import asyncio
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langgraph.prebuilt import ToolNode

@tool
async def my_tool(fact: str, config: RunnableConfig):
    """My tool"""
    print("CONFIG KEYS:", config.keys())
    print("CONFIGURABLE:", config.get("configurable"))
    return "done"

async def test():
    tools = [my_tool]
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tools")
    store = InMemoryStore()
    graph = builder.compile(store=store)
    
    # We invoke the tools node directly with a tool call
    await graph.ainvoke({"messages": [{"role": "ai", "content": "", "tool_calls": [{"name": "my_tool", "args": {"fact": "x"}, "id": "1"}]}]}, {"configurable": {"thread_id": "1", "user_id": "2"}})

asyncio.run(test())
