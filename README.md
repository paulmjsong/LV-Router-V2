# SaeGyeol Lab AI

A multi-user lab AI platform built around:

- **Open WebUI** for accounts, conversations, and workflow selection.
- **FastAPI + LangGraph** for task routing and specialist workflows.
- **LiteLLM Proxy** as the only LLM/embedding gateway.
- **vLLM or Ollama** for the local answer-or-delegate model.
- **PostgreSQL + pgvector** for persistent state and hybrid RAG.
- **MinIO/S3** for original documents.

## What changed from the earlier prototype

The old path was:

```text
cloud router LLM -> simple/complex answer LLM
```

The new automatic path is:

```text
local model through LiteLLM
   ├─ answers locally -> return that answer; no second model call
   └─ delegates -> LangGraph workflow -> one or more workflow model calls through LiteLLM
```

The local model returns a structured decision containing:

```text
action, workflow, model_tier, use_documents, confidence, reason, answer
```

It selects from `direct`, `domain_rag`, `paper`, `grant`, and `website`. It also recommends one of `local-fast`, `cloud-small`, or `cloud-large`. There is no query-length or keyword complexity score.

Hard rules remain outside the model: role permissions, explicit workflow selection, document access, budgets, and approval before repository changes.

## Deployment location

Run this on a **lab-controlled Linux server**. For a multi-user lab deployment, use an NVIDIA GPU and the vLLM profile. The Ollama profile is provided for CPU-only development and functional testing, not for serving twenty simultaneous users.

## 1. Configure

```bash
cp .env.example .env
python scripts/generate_secrets.py
```

Copy the generated values into `.env`, then set:

```dotenv
CLOUD_SMALL_MODEL=...
CLOUD_SMALL_API_KEY=...
CLOUD_LARGE_MODEL=...
CLOUD_LARGE_API_KEY=...
EMBEDDING_MODEL=...
EMBEDDING_API_KEY=...
EMBEDDING_DIMENSIONS=1536
```

`EMBEDDING_DIMENSIONS` must match the configured embedding model before documents are indexed.

For initial setup, `BACKEND_LITELLM_KEY` may equal `LITELLM_MASTER_KEY`. Before lab use, replace it with a restricted LiteLLM virtual key.

## 2. Start

### NVIDIA GPU / vLLM — recommended

Prerequisites: Docker Engine, Docker Compose, an NVIDIA driver, and NVIDIA Container Toolkit.

Set:

```dotenv
LOCAL_BACKEND=vllm
LOCAL_MODEL_ID=Qwen/Qwen3-8B
```

Run:

```bash
docker compose --profile gpu up -d --build
```

### CPU-only development / Ollama

Set:

```dotenv
LOCAL_BACKEND=ollama
OLLAMA_MODEL_ID=qwen3:4b
```

Run:

```bash
docker compose --profile ollama up -d --build
```

Open:

```text
http://<server-ip>:3000
```

Useful checks:

```bash
docker compose ps
docker compose logs -f open-webui backend litellm
curl http://127.0.0.1:8000/api/health
```

## 3. Use the interface

Open WebUI exposes these workflow models:

| Model | Behavior |
|---|---|
| `lab-auto` | Local model answers or delegates automatically |
| `lab-direct` | Explicit direct inference |
| `lab-rag` | Hybrid RAG over accessible collections |
| `lab-paper` | Paper workflow |
| `lab-grant` | Grant workflow |
| `lab-website` | Website proposal with approval |

Use `lab-auto` as the default. Explicit workflow models bypass automatic workflow selection.

Open WebUI title/tag generation is disabled in `docker-compose.yml`, so a visible user request does not trigger hidden background LLM calls. Re-enable those features only after assigning them a separate local-only task endpoint.

For a website proposal requiring approval, reply with:

```text
/approve <run-id>
```

or:

```text
/reject <run-id> <reason>
```

Approval creates a branch and pull request; it does not push directly to the protected branch.

## 4. Add RAG documents

Place files under the project’s `imports/` directory. Supported formats are PDF, DOCX, TXT, Markdown, HTML, and JSON.

Create a team collection:

```bash
docker compose exec backend python -m app.admin_cli create-collection \
  --name "Lab Knowledge" \
  --description "Shared papers and institutional documents" \
  --visibility team
```

Copy the returned collection ID, then upload files:

```bash
docker compose exec backend python -m app.admin_cli upload \
  --collection-id <collection-uuid> \
  /imports/document1.pdf /imports/document2.docx
```

List collections:

```bash
docker compose exec backend python -m app.admin_cli list-collections
```

The indexing path is:

```text
file -> MinIO -> parser -> chunks -> LiteLLM embedding alias -> pgvector
```

The query path is:

```text
query -> embedding through LiteLLM -> vector + PostgreSQL full-text search -> RRF -> workflow answer
```

## Model-routing boundaries

### Application/local router

The local model decides:

- return a final local answer; or
- delegate to a workflow;
- recommend `local-fast`, `cloud-small`, or `cloud-large`;
- state whether documents are needed.

### LangGraph

LangGraph executes the selected specialist workflow and persists checkpoints in PostgreSQL.

### LiteLLM

LiteLLM does not decide whether a query is a paper, grant, RAG, or website task. It receives a stable alias and handles:

- provider translation;
- virtual keys and model access;
- team/user budgets;
- deployment load balancing;
- retries and configured fallbacks;
- usage and cost logs.

By default, `local-router` has **no cloud fallback**. A failed local router does not silently create a paid routing call. Set `ALLOW_REMOTE_ROUTER_FALLBACK=true` only if that trade-off is intentional.

## Authentication

Open WebUI forwards a signed user-identity JWT to the backend. The backend verifies both:

1. the Open WebUI backend connection key; and
2. the signed identity token.

Open WebUI `admin` maps to lab roles `member, editor, admin`; a normal Open WebUI user maps to `member`. Replace this simple mapping with institutional groups/OIDC claims before broader deployment if different teams or permissions are required.

After lab accounts are provisioned, set:

```dotenv
OPENWEBUI_ENABLE_SIGNUP=false
```

and restart Open WebUI.

## Tests and static validation

```bash
cd backend
python -m pip install -e '.[dev]'
pytest

cd ..
python -m compileall backend/app backend/tests infra/litellm scripts
python scripts/validate_static.py
```

## Important limits

- The OpenAI-compatible backend currently buffers each workflow result and then emits it to Open WebUI; graph nodes do not yet stream tokens individually.
- Open WebUI is the chat interface. Document collection administration is intentionally handled through the included CLI in this version.
- The local router is prompted and schema-validated, but it is still a model. Evaluate routing accuracy on labeled lab requests before relying on it for cost-critical policy.
- Sensitive collections need an explicit policy stating which model tiers may receive their retrieved text. Collection visibility alone does not implement data-residency policy.
