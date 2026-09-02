# Infonet AI Router

A multi-user research-lab AI service built around **Open WebUI + FastAPI + one parent LangGraph + LiteLLM**.

## Active modes

| UI mode | What it does |
|---|---|
| `auto` | Local router selects an active workflow and difficulty. **Default.** |
| `direct` | One general-purpose inference path. |
| `web-search` | Live DuckDuckGo search followed by a cited answer from search snippets. |
| `pdf-document`  | Evidence-grounded Q&A over PDFs uploaded in the current chat. |
| `gist-regulations` | Straightforward RAG over the supplied GIST/Jireumgil FAISS vectorstore. |
| `research-paper`  | Basic multi-agent drafting workflow. |

`grant` and `website` are intentionally **not selectable**. Open WebUI shows them as grey “coming soon” status in a non-dismissible banner until those workflows are designed.

PDF uploads are enabled through Open WebUI. Other file extensions remain disabled, and uploaded evidence is handled only by the isolated `pdf-document` workflow.

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
                         |         |                    +-- Uploaded PDF Q&A
                     |         +-- Web Search (DDGS)
                     |         +-- GIST Regulations RAG
                     +-- Direct
   |
   v
LiteLLM aliases
   local-router | local-fast | cloud-small | cloud-large | embedding
   |
   +--> Ollama / vLLM / cloud providers
```

The parent graph owns workflow selection and mounts five isolated compiled subgraphs. LiteLLM is still the only model/embedding gateway.

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

## Uploaded PDF workflow

PDF ingestion remains in Open WebUI instead of being duplicated inside FastAPI:

    upload PDF -> Open WebUI extraction/chunking/embedding/retrieval
               -> <context><source> evidence in the provider request
               -> backend context validation and attachment-gated routing
               -> isolated pdf-document child graph
               -> evidence-only answer with inline [source-id] citations

In `auto`, an attachment signal selects `pdf-document` before the semantic router is called. Selecting `pdf-document` explicitly also works. Selecting any other workflow explicitly keeps that workflow isolated and does not expose uploaded PDF evidence to it.

Open WebUI accepts only `.pdf` files here and uses LiteLLM's existing `embedding` alias. Retrieval-query generation is disabled, and any remaining Open WebUI internal task is forced to Direct on an isolated conversation ID with file metadata removed. The PDF prompt also filters stale query-generation artifacts from earlier contaminated chat state.

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

## Web Search workflow

```text
query
  -> DDGS text search (DuckDuckGo backend)
  -> normalize and deduplicate up to five title/URL/snippet results
  -> inject numbered snippets as untrusted evidence
  -> answer model through LiteLLM
  -> inline [n] citations + deterministic source links
```

The workflow does not fetch arbitrary result pages; it answers from the returned search snippets only. Search is bounded by `WEB_SEARCH_TIMEOUT_SECONDS`, and Open WebUI's separate native web-search feature is disabled so there is only one web-search execution path. This route requires outbound HTTPS access from the backend container.

## Auto routing and the old local-fast collapse

The local router returns only:

```json
{"workflow":"direct","difficulty":"standard"}
```

Difficulty maps deterministically:

```text
simple   -> local-fast
standard -> cloud-small
advanced -> cloud-large
```

Additional policy:

- `gist-regulations`, `web-search`, and `research-paper` cannot resolve to `local-fast` in balanced Auto mode.
- malformed or disallowed router output uses a **visible cloud-small fallback**, never silent local-fast.
- the route line displayed before an answer includes workflow, difficulty, and fallback status.
- an uploaded PDF deterministically selects `pdf-document` in `auto`, without a router-model call;
- explicit modes bypass semantic routing and remain isolated from attached PDF evidence;
- the semantic router runs for other `auto` requests.

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

## Initial suggested prompts

The landing page deterministically shows five actual questions in this order: Auto, Direct, Web Search, PDF Document, and GIST Regulations. Research Paper is intentionally omitted. Attach a PDF before clicking the PDF question.

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

- no persistent shared user-document library or backend-owned upload API;
- no non-PDF upload workflow;
- no generic pgvector RAG workflow;
- no active Grant workflow;
- no active Website workflow;
- no claim that the research-paper workflow performs autonomous literature search, citation verification, or experiment validation.


## GIST regulation citations and PDF links

GIST regulation evidence is retrieved from the supplied FAISS index. Each
retrieved source is mapped to a controlled Markdown link for the corresponding
PDF. Open WebUI serves those files under `/static/gist-regulations/`; when page
metadata exists, the link includes `#page=N` so compatible browser PDF viewers
open near the retrieved page. The answer prompt requires inline links and a
`📌 References` section, and the backend appends a deterministic reference list
if the model omits one.

## Output-length policy

User-facing Direct, Web Search, GIST Regulations, Research Paper, and Uploaded PDF generation
no longer has an application-level `max_tokens` ceiling. Small structured-control
calls such as the paper orchestrator and validator remain bounded. Providers and
models still enforce their own context/output limits.
