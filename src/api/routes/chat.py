import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Any, List
from langchain_core.messages import HumanMessage, AIMessage

from src.api.deps import CurrentUser
from src.core.agent import get_agent_savers, builder

router = APIRouter()

class ThreadResponse(BaseModel):
    thread_id: str
    message: str

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class MemoryInput(BaseModel):
    fact: str

@router.post("/", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(current_user: CurrentUser) -> Any:
    """
    Generate a new thread ID for a new chat session.
    LangGraph implicitly creates the thread in the database upon the first message.
    This endpoint gives the client a clean UUID to use for a new session.
    """
    new_thread_id = str(uuid.uuid4())
    return {
        "thread_id": new_thread_id,
        "message": "Use this thread_id for future messages in this session."
    }

@router.post("/{thread_id}/message", response_model=ChatResponse)
async def send_message(
    thread_id: str,
    chat_input: ChatMessage,
    current_user: CurrentUser
) -> Any:
    """
    Send a message to the LangGraph agent for a specific thread.
    Short-term memory (chat history) is automatically loaded from postgres via thread_id.
    Long-term memory (facts) is automatically queried via user_id.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,             # Short-term memory key
            "user_id": str(current_user.id)     # Long-term memory key
        }
    }
    
    # We acquire the connections and compile the agent per-request
    async with get_agent_savers() as (checkpointer, store):
        agent = builder.compile(
            checkpointer=checkpointer,
            store=store
        )
        
        # Invoke the agent asynchronously
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=chat_input.message)]},
            config
        )
        
        # Extract the last message (the AI's response)
        last_message = response["messages"][-1]
        
        return {"reply": last_message.content}

@router.post("/memory", status_code=status.HTTP_201_CREATED)
async def add_long_term_memory(
    memory_input: MemoryInput,
    current_user: CurrentUser
) -> Any:
    """
    Explicitly add a fact to the user's long term memory.
    In a fully autonomous setup, the agent would do this via a Tool call.
    """
    import uuid
    memory_id = str(uuid.uuid4())
    namespace = ("memories", str(current_user.id))
    
    async with get_agent_savers() as (_, store):
        await store.aput(namespace, memory_id, {"fact": memory_input.fact})
        
    return {"status": "success", "memory_id": memory_id, "fact": memory_input.fact}

@router.get("/memory")
async def get_long_term_memories(current_user: CurrentUser) -> Any:
    """
    Get all facts currently stored in the user's long-term memory.
    """
    namespace = ("memories", str(current_user.id))
    
    async with get_agent_savers() as (_, store):
        memories = await store.asearch(namespace)
        
    return [
        {"id": mem.key, "fact": mem.value["fact"], "updated_at": mem.updated_at}
        for mem in memories
    ]
