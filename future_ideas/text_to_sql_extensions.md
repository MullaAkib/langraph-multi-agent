# Future Ideas for LangGraph Agent

## 1. Single Table Query Extension (Simple Tool)
To query a specific table (e.g., people's names and ages):
- Create a SQLAlchemy model and Alembic migration for the table.
- Create a new LangChain `@tool` function (e.g., `get_people_by_age(max_age: int)`).
- The tool executes a read-only SQLAlchemy query on the specific table.
- Add the tool to the agent's `tools` list. The LLM will autonomously decide when to call it based on the user's prompt.

## 2. Advanced Text-to-SQL (Dynamic Multi-Table Queries)
If the user needs to ask arbitrary questions across multiple tables without hardcoding specific tools for each query:
- **Approach A (Custom Graph):**
  - Schema Reader: Inject the database schema (tables, columns) into the system prompt.
  - Query Generator Node: The LLM generates a raw SQL string.
  - SQL Validator Node: Python code intercepts the SQL, ensures it is read-only (no DROP/UPDATE), and executes it. If it fails, loop back to the generator with the error.
  - Answer Generator Node: Pass the raw DB rows back to the LLM to format into a natural language response.
- **Approach B (LangChain Toolkit):**
  - Use `SQLDatabaseToolkit` from `langchain-community`.
  - This provides pre-built tools (`sql_db_list_tables`, `sql_db_schema`, `sql_db_query`) that the agent can use to autonomously explore the database and execute safe queries.

## 3. Autonomous Memory Saving
(Already implemented in `src/core/agent.py`)
- Give the agent an injected `save_memory` tool connected to the Postgres `Store`.
- The agent autonomously decides what facts are worth saving across sessions.