# Multi-Agent Supervisor Architecture in LangGraph

This document explains the architecture implemented in `src/core/agent.py` to create a multi-agent system.

When a single agent has too many tools or instructions, it can get confused. The solution is to create highly specialized "worker" agents and one smart "supervisor" agent that manages them.

## 1. The Multi-Agent `AgentState`
The LangGraph state was updated to include a `next` string field.
```python
class AgentState(MessagesState):
    # The supervisor tells us who goes next
    next: str
```
This field tells LangGraph exactly which agent needs to take control of the conversation at any given time.

## 2. The `supervisor_node` (The Boss)
The Supervisor Agent does *not* talk to the user. Instead, it reads the conversation and uses OpenAI's Structured Outputs (`with_structured_output`) to output exactly one of three commands:
*   `"DevilsAdvocate"`
*   `"WebResearcher"`
*   `"FINISH"`

It makes these routing decisions based on the conversation history.

## 3. The `advocate_node` (The Devil's Advocate)
This is the main conversational agent. Its responsibilities are focused:
*   It handles all direct conversation with the user.
*   It has the `save_memory` tool to manage long-term PostgreSQL memory.
*   It does *not* have the web search tool. If it needs facts, it writes an internal message directed at the Web Researcher and lets the Supervisor route it.

## 4. The `researcher_node` (The Web Researcher)
This is a background worker agent.
*   It has the `search_web` tool (using DuckDuckGo).
*   Its only job is to find facts when the Devil's Advocate asks for them, summarize them, and pass them back so the Devil's Advocate can use them in its argument. It does not address the user directly.

## 5. Complex Graph Routing
The `StateGraph` flow dynamically routes between agents:
1. The user sends a message.
2. The **Supervisor** reads it and routes to the **Devil's Advocate**.
3. The **Devil's Advocate** decides it needs to verify a fact and outputs a message asking the Web Researcher.
4. The **Supervisor** reads this request and routes to the **Web Researcher**.
5. The **Web Researcher** uses its tools to find the facts and puts them in the chat.
6. The **Supervisor** reads the facts and routes back to the **Devil's Advocate**.
7. The **Devil's Advocate** uses the facts to argue with the user.
8. The **Supervisor** sees the user got a reply, and routes to **FINISH**.

## Summary
In LangGraph, a Multi-Agent system is just a state machine where the nodes are different LLM prompts, and the edges are controlled by a "Boss" LLM deciding which "Worker" LLM needs to take the keyboard next. The memory (Checkpointer and Store) works exactly the same way, sharing the context across all the agents seamlessly!
