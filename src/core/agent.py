from contextlib import asynccontextmanager
from typing import AsyncGenerator, Tuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from typing import Annotated
import uuid
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.tools import tool, InjectedToolArg
from langgraph.prebuilt import ToolNode, tools_condition, InjectedStore
import os

from src.core.config import settings

# LangGraph's postgres savers require a psycopg connection pool.
# We convert our asyncpg URI to a standard postgres URI for psycopg.
conn_info = settings.SQLALCHEMY_DATABASE_URI.replace("postgresql+asyncpg://", "postgresql://")

# Global connection pool
pool = AsyncConnectionPool(
    conninfo=conn_info,
    max_size=20,
    open=False, # Wait to open until we actually need it
)

# LLM setup (you'll need to add OPENAI_API_KEY to your .env)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

@tool
async def save_memory(
    fact: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """Save a fact about the user to long-term memory. Use this when the user tells you something important about themselves."""
    user_id = config["configurable"].get("user_id")
    namespace = ("memories", str(user_id))
    memory_id = str(uuid.uuid4())
    
    await store.aput(namespace, memory_id, {"fact": fact})
    return f"Successfully remembered: {fact}"

tools = [save_memory]
llm_with_tools = llm.bind_tools(tools)

async def chatbot_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # 1. Extract the user_id from the runtime config
    user_id = config["configurable"].get("user_id")
    
    # 2. Query Long-Term Memory from the Store
    namespace = ("memories", str(user_id))
    memories = await store.asearch(namespace)
    
    # Format memories into a string
    facts = "\n".join([mem.value["fact"] for mem in memories]) if memories else "No specific facts known yet."
    
    # 3. Create a System prompt injecting the Long-Term Memory
    system_msg = SystemMessage(
        content=f"You are a helpful assistant. You have access to long-term memory about the user.\n\nHere is what you know about the user:\n{facts}\n\nIf the user tells you something important about themselves or their preferences, use the save_memory tool to save it."
    )
    
    # 4. Call the LLM with the System Prompt + Short-Term Memory (state["messages"])
    messages_to_send = [system_msg] + state["messages"]
    response = await llm_with_tools.ainvoke(messages_to_send)
    
    return {"messages": [response]}

# Build the Graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

@asynccontextmanager
async def get_agent_savers() -> AsyncGenerator[Tuple[AsyncPostgresSaver, AsyncPostgresStore], None]:
    """Provide the async postgres savers."""
    await pool.open()
    
    # Use context managers to ensure connections are handled safely
    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(pool)
    
    try:
        yield checkpointer, store
    finally:
        # We don't close the pool here so it can be reused, but you might want to 
        # close it on app shutdown
        pass

async def setup_agent_db():
    """Setup the LangGraph database tables. Run this once on startup."""
    await pool.open()
    async with pool.connection() as conn:
        # We need a connection with autocommit for CREATE INDEX CONCURRENTLY
        await conn.set_autocommit(True)
        checkpointer = AsyncPostgresSaver(conn)
        store = AsyncPostgresStore(conn)
        await checkpointer.setup()
        await store.setup()


