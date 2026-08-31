# Infonet AI Router v0.5 Architecture

## Parent LangGraph

```text
START
  -> route
  -> validate_route
  -> announce_route
  -> conditional dispatch
       |- direct
       |- gist-regulations
       |- web-search
       `- research-paper
  -> finalize
  -> END
```

Every `auto` request goes through the local semantic router. Explicitly selecting a mode bypasses semantic classification but still executes through the same parent graph. Grant and Website remain disabled metadata only.

## Direct subgraph

`START -> answer -> END`. One LLM call through LiteLLM.

## GIST Regulations subgraph

`START -> retrieve -> answer -> END`. The query is embedded through LiteLLM, searched against the read-only Jireumgil FAISS index, and answered from numbered retrieved passages. This differs from the blueprint's PostgreSQL/pgvector description because the current product requirement is to reuse the supplied legacy vectorstore.

## Web Search subgraph

`START -> search -> answer -> END`.

1. `WebSearchService` runs `DDGS.text()` with the DuckDuckGo backend in a worker thread.
2. A workflow timeout bounds the blocking search call.
3. Results are normalized, URL-validated, deduplicated, and limited.
4. The answer node receives only numbered titles, URLs, and snippets marked as untrusted evidence.
5. The LLM must cite claims as `[1]`, `[2]`, etc.
6. The backend appends a deterministic Markdown source list and returns structured source metadata.

The service does not fetch result pages. Open WebUI's native web-search feature is disabled to avoid a second, competing search path.

## Research Paper Drafting subgraph

```text
START -> orchestrator
             |
             +-> content_agent ---
             `-> structure_agent --+-> draft -> validator -> final -> END
```

This existing sequential/parallel LLM workflow is treated as covering the blueprint's research-drafter route; no Hermes dependency is added.

## Model routing

The local router classifies workflow + difficulty. Application policy maps difficulty to LiteLLM aliases and enforces specialist floors. LiteLLM handles provider/deployment mapping, credentials, budgets, rate limits, retries/fallbacks, and accounting.

## UI modes

```text
auto
direct
gist-regulations
web-search
research-paper
```

`DEFAULT_MODELS=auto`. File uploads and Open WebUI native web search remain disabled; the routed `web-search` workflow is the only live-search path.
