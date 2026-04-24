from contextlib import asynccontextmanager
from typing import AsyncGenerator, Tuple, Annotated, Literal
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import uuid
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedStore
from pydantic import BaseModel

from src.core.config import settings

# --- Connections ---
conn_info = settings.SQLALCHEMY_DATABASE_URI.replace("postgresql+asyncpg://", "postgresql://")
pool = AsyncConnectionPool(conninfo=conn_info, max_size=20, open=False)

print("connection string",conn_info)

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

# --- State Definition ---
class AgentState(MessagesState):
    # The supervisor tells us who goes next
    next: str

# --- Tools ---
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

@tool
def search_web(query: str) -> str:
    """Search the live internet for facts, news, and data to back up arguments. Always use this when making claims about real-world facts."""
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

advocate_tools = [save_memory]
researcher_tools = [search_web]


# --- Nodes ---

# 1. Supervisor Node
class Route(BaseModel):
    next: Literal["FINISH", "DevilsAdvocate", "WebResearcher"]

async def supervisor_node(state: AgentState):
    system_prompt = (
        "You are a supervisor managing a conversation between the User, a Devil's Advocate, and a Web Researcher.\n"
        "1. The Devil's Advocate playfully challenges the user's opinions and interacts directly with the user.\n"
        "2. The Web Researcher searches the internet for facts to support or refute arguments.\n"
        "Based on the conversation, decide who should act next.\n"
        "- If the user just spoke, ALWAYS route to 'DevilsAdvocate'.\n"
        "- If the Devil's Advocate asked the Web Researcher to look something up, route to 'WebResearcher'.\n"
        "- If the Web Researcher just provided facts, route back to 'DevilsAdvocate' so it can formulate a response to the user.\n"
        "- If the Devil's Advocate just responded to the user and no research is needed, route to 'FINISH'."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Force the LLM to output the routing decision
    supervisor_chain = llm.with_structured_output(Route)
    response = await supervisor_chain.ainvoke(messages)
    
    return {"next": response.next}


# 2. Devil's Advocate Node
async def advocate_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"].get("user_id")
    namespace = ("memories", str(user_id))
    memories = await store.asearch(namespace)
    facts = "\n".join([mem.value["fact"] for mem in memories]) if memories else "No specific facts known yet."
    
    system_msg = SystemMessage(
        content=f"You are a 'Devil's Advocate' assistant. Your goal is to playfully challenge the user's opinions, point out flaws in their arguments, and force them to defend their reasoning.\n\n"
                f"You have access to long-term memory about the user. Here is what you know about them:\n{facts}\n\n"
                f"If the user tells you something important about themselves or their preferences, use the `save_memory` tool to save it.\n"
                f"If you need real-world data to back up your counter-arguments, do NOT guess. Write a message directed to the 'Web Researcher' asking it to search for the specific facts you need."
    )
    
    messages_to_send = [system_msg] + state["messages"]
    response = await llm.bind_tools(advocate_tools).ainvoke(messages_to_send)
    response.name = "DevilsAdvocate"
    
    return {"messages": [response]}


# 3. Web Researcher Node
async def researcher_node(state: AgentState):
    system_msg = SystemMessage(
        content="You are a Web Researcher. Your goal is to search the live internet for facts, news, and data to back up arguments.\n"
                "Use the `search_web` tool to find the information. Summarize the findings clearly.\n"
                "Do NOT address the user directly. Address your findings to the Devil's Advocate so they can use it in their argument."
    )
    messages_to_send = [system_msg] + state["messages"]
    response = await llm.bind_tools(researcher_tools).ainvoke(messages_to_send)
    response.name = "WebResearcher"
    
    return {"messages": [response]}


# --- Graph Building ---

# Routers for Tool Nodes
def advocate_tools_condition(state: AgentState) -> Literal["advocate_tools", "Supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "advocate_tools"
    return "Supervisor"

def researcher_tools_condition(state: AgentState) -> Literal["researcher_tools", "Supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "researcher_tools"
    return "Supervisor"


builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("Supervisor", supervisor_node)
builder.add_node("DevilsAdvocate", advocate_node)
builder.add_node("advocate_tools", ToolNode(advocate_tools))
builder.add_node("WebResearcher", researcher_node)
builder.add_node("researcher_tools", ToolNode(researcher_tools))

# Add Edges
builder.add_edge(START, "Supervisor")

# Supervisor routing
builder.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        "DevilsAdvocate": "DevilsAdvocate",
        "WebResearcher": "WebResearcher",
        "FINISH": END
    }
)

# Agent routing (If they use tools, go to tools, else go back to Supervisor)
builder.add_conditional_edges("DevilsAdvocate", advocate_tools_condition)
builder.add_edge("advocate_tools", "DevilsAdvocate")

builder.add_conditional_edges("WebResearcher", researcher_tools_condition)
builder.add_edge("researcher_tools", "WebResearcher")


# --- Infrastructure ---

@asynccontextmanager
async def get_agent_savers() -> AsyncGenerator[Tuple[AsyncPostgresSaver, AsyncPostgresStore], None]:
    """Provide the async postgres savers."""
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(pool)
    try:
        yield checkpointer, store
    finally:
        pass

async def setup_agent_db():
    """Setup the LangGraph database tables. Run this once on startup."""
    await pool.open()
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        checkpointer = AsyncPostgresSaver(conn)
        store = AsyncPostgresStore(conn)
        await checkpointer.setup()
        await store.setup()
