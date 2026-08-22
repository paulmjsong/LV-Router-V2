# Infonet AI Router

A multi-user research-lab AI platform with:

- **Open WebUI** for accounts and chat.
- **One parent LangGraph** for route selection, policy validation, and subgraph dispatch.
- **Isolated workflow subgraphs** for chat, PDF Q&A, GIST regulations, and three placeholders.
- **LiteLLM Proxy** as the only LLM and embedding gateway.
- **Ollama or vLLM** for local inference.
- **PostgreSQL + pgvector** for checkpoints, documents, and hybrid retrieval.
- **MinIO/S3** for original uploaded files.

## Workflow IDs

Open WebUI exposes one automatic selector plus six workflow selections:

| ID | Purpose |
|---|---|
| `auto` | Local semantic router selects a workflow and logical model tier. |
| `chat` | General direct inference. |
| `pdf` | Answer from Open WebUI PDF File Context or indexed PDF collections. |
| `regulations` | Answer only from the reserved GIST regulations corpus. |
| `paper` | Deliberately minimal paper-assistant placeholder. |
| `grant` | Deliberately minimal grant-assistant placeholder. |
| `website` | Deliberately non-mutating website-assistant placeholder. |

The paper, grant, and website entries are not represented as finished multi-agent systems. Each currently contains one clearly labeled placeholder model stage.

## Graph structure

```text
Parent LangGraph
  START
    → route
    → validate_route
    → announce_route
    → conditional dispatch
        ├─ chat subgraph
        ├─ pdf subgraph
        ├─ regulations subgraph
        ├─ paper placeholder subgraph
        ├─ grant placeholder subgraph
        └─ website placeholder subgraph
    → finalize
    → END
```

Routing, model-tier validation, and the selected workflow now live in the same LangGraph trace and PostgreSQL checkpoint thread. The stable thread is derived from the user and Open WebUI conversation ID. This replaces the previous design that selected one of several independent graphs in `WorkflowRuntime`.

## Model-tier policy

The local router emits a compact semantic classification:

```json
{
  "workflow": "chat",
  "difficulty": "simple",
  "confidence": 0.92
}
```

Python then maps `simple`, `standard`, and `advanced` to `local-fast`, `cloud-small`,
and `cloud-large`. This is deliberately easier for the small router model than asking it to reason
about provider aliases directly. The application validates the decision before dispatch:

- malformed, disallowed, or low-confidence routing uses a **visible `cloud-small` fallback**;
- automatic/BALANCED specialist routing has a `cloud-small` floor;
- balanced `local-fast` is accepted only for `chat/simple` above `LOCAL_FAST_MIN_CONFIDENCE`;
- an explicit `fast` or `high` quality override selects `local-fast` or `cloud-large`.

The UI displays the selected workflow, difficulty, confidence, fallback status, logical alias, and served model. LiteLLM maps the logical alias to a concrete deployment and enforces provider policy, keys, budgets, limits, retries, and accounting.

## RAG design

### PDF Q&A

Two input modes are supported:

1. **Open WebUI File Context**: attach a PDF and keep File Context enabled. Open WebUI extracts/retrieves file context and injects it into the OpenAI-compatible request; the backend separates that evidence from the user's question.
2. **Indexed PDF collection**: upload PDFs through `/api/documents/upload`, pass the collection IDs in request metadata, and use `pdf` or `auto`. Reserved system collections are excluded from this path.

Indexed PDF retrieval is restricted to PDF MIME types and uses:

```text
query embedding through LiteLLM
+ pgvector cosine retrieval
+ PostgreSQL full-text retrieval
+ reciprocal-rank fusion
```

### GIST regulations

At startup, the backend creates a reserved public collection identified by:

```text
gist-regulations
```

Only administrators can modify it. The `regulations` subgraph retrieves exclusively from this collection, so it cannot silently mix ordinary lab files into an institutional-policy answer.

The old Jireumgil FAISS directory is **not loaded**. That serialized index is tied to its original embedding setup, is local-process state, and previously required dangerous deserialization. Re-index the original regulation source PDFs/DOCX/TXT files into pgvector instead. See [docs/JIREUMGIL_MIGRATION.md](docs/JIREUMGIL_MIGRATION.md).

## Configure

```bash
cp .env.example .env
python scripts/generate_secrets.py
```

Copy the generated values into `.env`. At minimum configure:

```dotenv
LITELLM_MASTER_KEY=sk-...
LITELLM_SALT_KEY=sk-...
BACKEND_LITELLM_KEY=sk-...
OPENWEBUI_BACKEND_KEY=sk-...
OPENWEBUI_IDENTITY_JWT_SECRET=...
OPENWEBUI_SECRET_KEY=...
LAB_ADMIN_API_KEY=sk-...
```

For local CPU development:

```dotenv
LOCAL_BACKEND=ollama
OLLAMA_ROUTER_MODEL_ID=qwen3:0.6b
OLLAMA_MODEL_ID=qwen3:4b-instruct-2507-q4_K_M
```

For a lab GPU server:

```dotenv
LOCAL_BACKEND=vllm
LOCAL_MODEL_ID=Qwen/Qwen3-8B
```

Cloud and embedding aliases require their configured provider credentials. Existing installations may keep their old PostgreSQL/MinIO usernames and database names; those are storage credentials, not product branding.

## Start

### Windows or CPU development

```powershell
docker compose --profile ollama up -d --build
```

### NVIDIA GPU / vLLM

```bash
docker compose --profile gpu up -d --build
```

Open:

```text
http://localhost:3000
```

Open WebUI defaults to `auto`. Notes, Calendar, Automations, and Open WebUI's own
sub-agent feature are disabled so they cannot add unrelated UI surfaces or hidden model calls to this
platform. Workspace administration should remain restricted to administrators.

### Branding boundary

The configured product name is `Infonet AI Router`. Open WebUI may still identify its upstream
software on the unauthenticated login surface. This repository does not disguise or patch third-party
branding; complete white-labeling requires a permitted Open WebUI fork/license or a custom frontend.

Useful checks:

```bash
docker compose ps
docker compose logs -f ollama-init ollama litellm backend open-webui
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:4000/health/readiness
```

## Index PDF documents

Create a collection:

```bash
docker compose exec backend python -m app.admin_cli create-collection \
  --name "Project PDFs" \
  --description "Shared PDF collection" \
  --visibility team
```

Upload PDF files using the returned collection ID:

```bash
docker compose exec backend python -m app.admin_cli upload \
  --collection-id <collection-uuid> \
  /imports/paper.pdf
```

## Index GIST regulations

Place the original regulation source files under `imports/gist-regulations/`, then run:

```bash
docker compose exec backend python -m app.admin_cli upload-regulations \
  /imports/gist-regulations
```

The CLI scans supported files in the directory recursively.

Do not copy `index.faiss` or `index.pkl` into the new service. The source documents must be embedded again through the configured LiteLLM `embedding` alias.

## Upgrade notes from the previous build

- Existing PostgreSQL and MinIO usernames may remain unchanged. They are storage credentials, not UI branding.
- Old prefixed Open WebUI model IDs are removed. Recreate the Open WebUI container, start a new chat, and select `auto`.
- Existing `.env` secrets must be preserved. Add the new routing/GIST variables rather than replacing the file.
- The previous one-file website approval flow is intentionally removed while `website` is a placeholder. Old interrupted website runs cannot be resumed by this version.
- The old Jireumgil FAISS files are not loaded; re-index the reviewed source documents as described below.

## Validation

```bash
python -m compileall backend/app backend/tests infra/litellm scripts
python scripts/validate_static.py

cd backend
python -m pip install -e '.[dev]'
pytest
```

A complete integration test still requires Docker, PostgreSQL, MinIO, LiteLLM, and a functioning local/cloud model configuration.
