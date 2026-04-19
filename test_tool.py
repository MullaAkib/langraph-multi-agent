import asyncio
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langgraph.store.base import BaseStore

@tool
async def my_tool(fact: str, store: Annotated[BaseStore, InjectedToolArg()]):
    """My tool"""
    return "done"

print(my_tool.args_schema.model_fields)
