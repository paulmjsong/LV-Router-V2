# Infonet AI Router

A multi-user research-lab AI service built around **Open WebUI + FastAPI + one parent LangGraph + LiteLLM**.

## Active modes

| UI mode | What it does |
|---|---|
| `auto` | Local router selects an active workflow and difficulty. **Default.** |
| `direct` | One general-purpose inference path. |
| `gist-regulations` | Straightforward RAG over the supplied GIST/Jireumgil FAISS vectorstore. |
| `research-paper` | Basic multi-agent drafting workflow. |

`grant` and `website` are intentionally **not selectable**. Open WebUI shows them as grey “coming soon” status in a non-dismissible banner until those workflows are designed.

User file uploads are disabled in this version. The backend also rejects file-bearing requests.

## Architecture

```text
Open WebUI
   |
   v
FastAPI OpenAI-compatible adapter
   |
   v
Parent LangGraph
   route -> validate -> announce -> conditional subgraph -> finalize
                     |         |                    |
                     |         |                    +-- Research Paper multi-agent
                     |         +-- GIST Regulations RAG
                     +-- Direct
   |
   v
LiteLLM aliases
   local-router | local-fast | cloud-small | cloud-large | embedding
   |
   +--> Ollama / vLLM / cloud providers
```

The parent graph owns workflow selection and mounts three isolated compiled subgraphs. LiteLLM is still the only model/embedding gateway.

## Research Paper Drafting workflow

This is intentionally a **basic** multi-agent system, not a finished autonomous paper platform:

```text
orchestrator
   |   | +--> content subagent ----   +----> structure subagent ---+--> drafter --> validator --> finalizer
```

- **Orchestrator**: converts the request into a small structured plan.
- **Content subagent**: identifies substantive claims, technical content, and missing evidence.
- **Structure subagent**: proposes academic organization, argument order, and transitions.
- **Drafter**: synthesizes the plan and both specialist outputs.
- **Validator**: checks requested coverage, unsupported claims, fabricated results/citations, and major logical/clarity issues.
- **Finalizer**: revises according to validation and returns the final user-facing draft.

The agents are separate LangGraph nodes with separate model calls. They do not pretend to search literature or validate experiments unless that evidence is actually provided in conversation context.

## GIST Regulations workflow

This version uses the supplied legacy vectorstore directly:

```text
index.faiss + index.pkl
      |
query -> LiteLLM embedding alias
      |
FAISS top-k search
      |
retrieved regulation passages
      |
answer model with [SOURCE n] citations
```

The supplied index contains 266 mapped chunks and uses a **1536-dimensional embedding space**. It was created by the original Jireumgil setup using `text-embedding-3-small`, so the query embedding must stay compatible:

```dotenv
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
GIST_REGULATIONS_VECTOR_DIMENSIONS=1536
```

Set `EMBEDDING_API_KEY` (or configure the same embedding deployment in LiteLLM). If you later change embedding models, rebuild the FAISS index rather than querying the old index with incompatible vectors.

For safety, the backend does **not** call `FAISS.load_local(...allow_dangerous_deserialization=True)`. It loads the FAISS binary normally and uses a restricted unpickler that allows only the two LangChain classes present in the supplied `index.pkl`.

## Auto routing and the old local-fast collapse

The local router returns only:

```json
{"workflow":"direct","difficulty":"standard","confidence":0.91}
```

Difficulty maps deterministically:

```text
simple   -> local-fast
standard -> cloud-small
advanced -> cloud-large
```

Additional policy:

- `gist-regulations` and `research-paper` can never resolve to `local-fast` in balanced auto mode.
- `direct/simple` is accepted as `local-fast` only above `LOCAL_FAST_MIN_CONFIDENCE`.
- malformed, disallowed, or low-confidence router output uses a **visible cloud-small fallback**, never silent local-fast.
- the route line displayed before an answer includes workflow, difficulty, confidence, and fallback status.

## Configure

```powershell
Copy-Item .env.example .env
python scripts/generate_secrets.py
```

Copy generated secrets into `.env`. For the supplied GIST FAISS index, configure the compatible embedding endpoint/key.

For Windows CPU development:

```dotenv
LOCAL_BACKEND=ollama
OLLAMA_ROUTER_MODEL_ID=qwen3:0.6b
OLLAMA_MODEL_ID=qwen3:4b-instruct-2507-q4_K_M
```

## Start

```powershell
docker compose --profile ollama down --remove-orphans
docker compose --profile ollama up -d --build --force-recreate
```

Open:

```text
http://localhost:3000
```

Open WebUI defaults to `auto`.

Useful checks:

```powershell
docker compose ps
docker compose logs -f ollama-init ollama litellm backend open-webui
curl.exe http://localhost:8000/api/health
curl.exe http://localhost:4000/health/readiness
```

## Validate

```powershell
python -X utf8 scripts/validate_static.py
python -m compileall -q backend/app backend/tests infra/litellm scripts
```

Full runtime tests require installing the backend development dependencies:

```powershell
cd backend
python -m pip install -e ".[dev]"
pytest
```

## Deliberate omissions in this version

- no PDF/user-file workflow;
- no document upload/indexing UI;
- no generic pgvector RAG workflow;
- no active Grant workflow;
- no active Website workflow;
- no claim that the research-paper workflow performs autonomous literature search, citation verification, or experiment validation.
