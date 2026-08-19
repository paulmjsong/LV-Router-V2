# Architecture

```text
Open WebUI
  accounts / chat / workflow selector
          │ OpenAI-compatible API
          ▼
FastAPI control plane
  signed user identity / RBAC / collection permissions
          │
          ├─ lab-auto
          │     ▼
          │  local-router via LiteLLM
          │     ├─ answer -> final response
          │     └─ delegate -> workflow + tier
          │
          ├─ direct LangGraph
          ├─ domain RAG LangGraph
          ├─ paper LangGraph
          ├─ grant LangGraph
          └─ website LangGraph + interrupt/approval
                    │
          ┌─────────┴───────────┐
          ▼                     ▼
PostgreSQL + pgvector       MinIO/S3
state / ACL / chunks        original files
          │
          ▼
LiteLLM Proxy — exclusive model gateway
   ├─ local-router -> vLLM or Ollama
   ├─ local-fast   -> vLLM or Ollama
   ├─ cloud-small  -> configured provider deployment(s)
   ├─ cloud-large  -> configured provider deployment(s)
   └─ embedding    -> configured embedding deployment(s)
```

## Routing contract

`lab-auto` performs one local call through the `local-router` LiteLLM alias. The structured result is one of:

```text
answer:
  final local answer

delegate:
  workflow
  recommended model tier
  whether document retrieval is required
```

A locally answered request stops immediately. A delegated general request runs exactly one downstream direct-inference node. Specialist workflows may have multiple stages because that is the requested workflow, not a routing artifact.

No query-length or keyword complexity score is used. Model selection is based on the local semantic decision, explicit user quality selection, and fixed stage policy. LiteLLM chooses deployments within each alias.

## Deterministic controls

The following are never delegated to an LLM:

- explicit workflow selection;
- workflow role authorization;
- collection visibility and SQL permission filtering;
- LiteLLM virtual-key budgets and model access;
- repository path allow-listing;
- human approval before a GitHub pull request;
- rejection of malformed router output.

## RAG

Documents are stored in object storage and indexed in PostgreSQL. Retrieval fuses pgvector cosine ranking and PostgreSQL full-text ranking with reciprocal-rank fusion. Access rules are applied in SQL before candidates are ranked.

When `use_documents=true` and no collection IDs are specified, RAG searches all collections accessible to the current user. This supports `lab-rag` and automatic document routing from Open WebUI without exposing inaccessible collections.

## Open WebUI integration

The backend exposes:

```text
GET  /v1/models
POST /v1/chat/completions
```

The model IDs represent workflows, not provider models. Open WebUI forwards a signed identity JWT and chat ID. The chat ID is converted to a stable conversation UUID, while LangGraph checkpoints remain run-scoped to avoid state collisions between different workflow graphs.
