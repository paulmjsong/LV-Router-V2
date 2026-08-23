# Infonet AI Router v0.4 Architecture

## Parent LangGraph

The control plane is one compiled parent graph with isolated workflow subgraphs:

```text
START
  -> route
  -> validate_route
  -> announce_route
  -> conditional dispatch
       |- direct
       |- gist-regulations
       `- research-paper
  -> finalize
  -> END
```

`grant` and `website` exist only as disabled workflow metadata so the UI can communicate “coming soon”; they are not graph nodes and the local router cannot select them.

## Direct subgraph

`START -> answer -> END`. One LLM call through LiteLLM.

## GIST Regulations subgraph

`START -> retrieve -> answer -> END`.

`retrieve`:
1. embeds the query through LiteLLM alias `embedding`;
2. performs top-k FAISS search over `jireumgil_index/index.faiss`;
3. resolves FAISS IDs through the supplied `index.pkl` docstore mapping;
4. formats source snippets.

`answer` calls the selected answer tier and must ground the response in those snippets.

The supplied vectorstore is read-only. No user upload pipeline exists.

## Research Paper Drafting subgraph

```text
START -> orchestrator
             |             | +-> content_agent ---             `---> structure_agent --+-> draft -> validator -> final -> END
```

This is a minimal multi-agent composition:
- orchestrator = planning agent;
- content_agent and structure_agent = parallel specialist subagents;
- draft = synthesis agent;
- validator = independent validation agent;
- final = correction/finalization agent.

Each node is an independent LLM call and receives a stage-specific system prompt. It is deliberately not a literature-search or experiment-validation system yet.

## Model routing

The local routing model classifies workflow + difficulty. Application policy maps difficulty to LiteLLM aliases and enforces specialist floors. LiteLLM handles the concrete provider/deployment, credentials, budgets, rate limits, retries/fallbacks, and accounting.

## UI modes

Selectable provider models exposed to Open WebUI:

```text
auto
direct
gist-regulations
research-paper
```

`DEFAULT_MODELS=auto`. File and web attachments are disabled for normal users, and the backend rejects requests carrying files. Grant/Website are rendered as grey “coming soon” status in an Open WebUI banner because provider-backed model lists do not have a reliable portable disabled-model state.
