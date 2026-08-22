# Infonet AI Router Architecture

## Control plane

`backend/app/runtime.py` compiles one parent graph from `backend/app/workflows/parent.py`. The parent graph owns route selection and policy validation and mounts six independently compiled workflow subgraphs as nodes.

```text
Open WebUI
  → FastAPI OpenAI adapter
  → Parent LangGraph
      route
      validate_route
      announce_route
      conditional workflow subgraph
      finalize
  → LiteLLM aliases
  → Ollama/vLLM/cloud deployments
```

### Parent graph state

The parent graph receives request identity, user roles, selected quality, PDF attachment context, collection IDs, and a stable conversation message history maintained by the PostgreSQL LangGraph checkpointer. It records the selected workflow, logical model tier, route confidence, fallback status, sources, answer, and model-call events.

### Subgraph contracts

- `chat`: one direct answer node.
- `pdf`: retrieve PDF evidence, then answer; missing evidence terminates without a model call.
- `regulations`: retrieve only from the reserved `gist-regulations` collection, then answer.
- `paper`: one placeholder draft node.
- `grant`: one placeholder draft node.
- `website`: one non-mutating placeholder proposal node.

The placeholder workflows intentionally do not claim project management, multi-agent collaboration, repository mutation, approval, or publication.

## Routing policy

The local router model classifies semantic workflow and difficulty (`simple`, `standard`, or
`advanced`). Python maps difficulty to the logical LiteLLM tier and enforces:

- allowed workflow IDs and roles;
- PDF/regulations document requirements;
- a `cloud-small` floor for automatically selected specialist workflows;
- a minimum confidence for accepting `chat/simple` as `local-fast`;
- visible fallback behavior;
- explicit `fast` and `high` quality overrides.

This removes the prior failure mode where malformed or disallowed router output silently became `direct + local-fast`.

## LiteLLM boundary

Application code selects a logical alias:

```text
local-router | local-fast | cloud-small | cloud-large | embedding
```

LiteLLM handles the concrete deployment, provider translation, keys, budgets, rate limits, load balancing, retries/fallbacks, and usage accounting. No workflow imports provider-specific model clients.

## Document architecture

Original files are stored in MinIO/S3. Parsed chunks and embeddings are stored in PostgreSQL/pgvector. Hybrid retrieval fuses vector and PostgreSQL full-text ranks.

The GIST regulations corpus is a reserved system collection. The old local FAISS index is not a runtime dependency and is not deserialized.
