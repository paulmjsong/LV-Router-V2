# Infonet AI Router

Infonet AI Router is a multi-user research-lab AI platform that routes each request to the appropriate workflow and sends every LLM/embedding call through LiteLLM.

## Current stack

- **Open WebUI** — login, conversations, and workflow selection
- **FastAPI** — OpenAI-compatible backend API
- **LangGraph** — parent routing graph and workflow subgraphs
- **LiteLLM Proxy** — the only LLM/embedding gateway
- **Ollama or vLLM** — local model serving
- **PostgreSQL** — application state and LangGraph checkpoints
- **Redis** — LiteLLM coordination
- **FAISS** — read-only GIST regulations vector store

## Current workflows

| Mode | Purpose |
|---|---|
| `auto` | Default. Local routing LLM selects workflow and difficulty/model tier. |
| `direct` | General-purpose inference. |
| `gist-regulations` | RAG over the supplied GIST regulations FAISS index. |
| `research-paper` | Basic multi-agent research-paper drafting workflow. |
| Grant | Coming soon; not selectable. |
| Website | Coming soon; not selectable. |

The research-paper workflow currently uses:

```text
Orchestrator
   ├── Content agent
   └── Structure agent
          ↓
        Drafter
          ↓
       Validator
          ↓
       Finalizer
```

All model calls go through LiteLLM.

---

# 1. Important project paths

```text
backend/app/
  main.py                FastAPI application
  openai_compat.py       Open WebUI/OpenAI-compatible adapter
  routing.py             Auto workflow + difficulty routing
  llm.py                 LiteLLM gateway client
  gist_regulations.py    FAISS-based GIST regulations retrieval
  workflows/
    parent.py            Parent LangGraph
    builders.py          Workflow subgraphs/nodes
    prompts.py           Workflow prompts
    state.py             Graph state

infra/litellm/
  render_config.py       Generates LiteLLM config from .env
  ollama_startup_gate.py Waits for/preloads Ollama models

jireumgil_index/
  index.faiss
  index.pkl

docker-compose.yml
.env
```

---

# 2. One-time Linux/NVIDIA prerequisites

Install:

1. Docker Engine
2. Docker Compose v2
3. NVIDIA driver
4. NVIDIA Container Toolkit
5. Git

Check the host GPU:

```bash
nvidia-smi
```

Check Docker GPU access independently of this project:

```bash
sudo docker run --rm --gpus all \
  nvidia/cuda:12.9.0-base-ubuntu22.04 \
  nvidia-smi
```

If that fails:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then repeat the CUDA-container test.

---

# 3. Configure Docker Ollama for GPU

For Docker Compose 2.30+, the simplest configuration is:

```yaml
ollama:
  image: ${OLLAMA_IMAGE:-ollama/ollama:latest}
  profiles: ["ollama"]
  restart: unless-stopped
  environment:
    OLLAMA_KEEP_ALIVE: "-1"
    OLLAMA_MAX_LOADED_MODELS: "2"
  volumes:
    - ollama-data:/root/.ollama
  gpus: all
```

Alternatively:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

After changing GPU settings, **recreate** Ollama; a restart is not enough:

```bash
sudo docker compose --profile ollama rm -sf ollama ollama-init
sudo docker compose --profile ollama up -d --force-recreate ollama
```

Confirm Docker attached a GPU:

```bash
sudo docker inspect <project>-ollama-1 \
  --format '{{json .HostConfig.DeviceRequests}}'
```

The result must not be `null`.

After loading a model:

```bash
sudo docker compose exec ollama ollama ps
```

The `PROCESSOR` column should show GPU use instead of `100% CPU`.

> `nvidia-smi` may not exist inside the Ollama image. Use the CUDA-container test and `ollama ps` instead.

---

# 4. Configure `.env`

For a new deployment:

```bash
cp .env.example .env
python3 scripts/generate_secrets.py
```

Copy the generated secrets into `.env`.

Minimum important settings:

```dotenv
APP_NAME=Infonet AI Router

LITELLM_MASTER_KEY=sk-...
LITELLM_SALT_KEY=sk-...
BACKEND_LITELLM_KEY=sk-...

OPENWEBUI_BACKEND_KEY=sk-...
OPENWEBUI_IDENTITY_JWT_SECRET=...
OPENWEBUI_SECRET_KEY=...
LAB_ADMIN_API_KEY=sk-...

LOCAL_BACKEND=ollama
OLLAMA_API_BASE=http://ollama:11434

# Recommended on a GPU server
OLLAMA_ROUTER_MODEL_ID=qwen3:1.7b
OLLAMA_MODEL_ID=qwen3:4b-instruct-2507-q4_K_M

CLOUD_SMALL_MODEL=openai/gpt-5-mini
CLOUD_SMALL_API_KEY=...

CLOUD_LARGE_MODEL=openai/gpt-5
CLOUD_LARGE_API_KEY=...

# Required for the existing GIST FAISS index
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=...
EMBEDDING_DIMENSIONS=1536
GIST_REGULATIONS_VECTOR_DIMENSIONS=1536
```

The supplied GIST index was built in a 1536-dimensional embedding space. Do not change the embedding model/dimension unless you rebuild the index.

For initial testing, `BACKEND_LITELLM_KEY` may equal `LITELLM_MASTER_KEY`. Before multi-user use, replace it with a restricted LiteLLM virtual key.

---

# 5. First-time startup

Run from the repository root.

## 5.1 Validate Compose

```bash
sudo docker compose --profile ollama config --quiet
```

No output means the configuration parsed successfully.

## 5.2 Start Ollama

```bash
sudo docker compose --profile ollama up -d ollama
```

## 5.3 Download configured models

```bash
sudo docker compose --profile ollama up ollama-init
```

Wait for:

```text
Ollama models are downloaded.
```

Confirm:

```bash
sudo docker compose exec ollama ollama list
```

## 5.4 Start the full stack

```bash
sudo docker compose --profile ollama up -d --build
```

Check:

```bash
sudo docker compose --profile ollama ps -a
```

Expected steady state:

```text
postgres       healthy
litellm-db     healthy
redis          running
ollama         running
ollama-init    exited (0)
litellm        healthy
backend        healthy
open-webui     healthy/running
```

`ollama-init` exiting with code 0 is normal.

---

# 6. Normal daily startup

Once the models and volumes exist:

```bash
cd /path/to/LV-Router-V2
sudo docker compose --profile ollama up -d
```

Then:

```bash
sudo docker compose ps
```

You do not need `--build` every time.

---

# 7. Access the site

From the server:

```text
http://localhost:3000
```

From another computer on the same network:

```text
http://<SERVER-IP>:3000
```

Find the Linux server IP:

```bash
hostname -I
```

Open WebUI manages accounts and sessions.

After pilot accounts are created, disable open signup:

```dotenv
OPENWEBUI_ENABLE_SIGNUP=false
```

Then:

```bash
sudo docker compose up -d --force-recreate open-webui
```

---

# 8. Health checks

All containers:

```bash
sudo docker compose ps
```

LiteLLM, if port 4000 is published:

```bash
curl -i http://localhost:4000/health/readiness
```

If it is not published:

```bash
sudo docker compose exec litellm \
  python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:4000/health/readiness').status)"
```

Backend:

```bash
curl http://127.0.0.1:8000/api/health
```

---

# 9. Verify which models are really being used

First check what Compose resolves from `.env`:

```bash
sudo docker compose config \
  | grep -E 'OLLAMA_ROUTER_MODEL_ID|OLLAMA_MODEL_ID'
```

Then check LiteLLM's rendered aliases:

```bash
sudo docker compose exec litellm \
  grep -A4 "model_name: local-router" \
  /tmp/litellm-config.yaml

sudo docker compose exec litellm \
  grep -A4 "model_name: local-fast" \
  /tmp/litellm-config.yaml
```

Installed Ollama models:

```bash
sudo docker compose exec ollama ollama list
```

Currently loaded models:

```bash
sudo docker compose exec ollama ollama ps
```

An old model in `ollama list` is only persisted in the Ollama volume; it is not necessarily in use.

---

# 10. After changing `.env`

Editing `.env` does not modify existing containers.

If the change affects model/provider settings:

```bash
sudo docker compose rm -sf litellm
sudo docker compose --profile ollama up -d --force-recreate litellm backend
```

Then re-check `/tmp/litellm-config.yaml`.

If only backend code/config changed:

```bash
sudo docker compose up -d --build --force-recreate backend
```

If only Open WebUI settings changed:

```bash
sudo docker compose up -d --force-recreate open-webui
```

---

# 11. After pulling new code

```bash
git pull
sudo docker compose --profile ollama config --quiet
sudo docker compose --profile ollama up -d --build --force-recreate
```

Then:

```bash
sudo docker compose ps
```

Do not use `docker compose down -v` unless you intentionally want to delete persistent data.

---

# 12. Stop / restart

Stop while preserving data:

```bash
sudo docker compose --profile ollama down
```

Restart running containers:

```bash
sudo docker compose --profile ollama restart
```

Full clean recreation while preserving volumes:

```bash
sudo docker compose --profile ollama down --remove-orphans
sudo docker compose --profile ollama up -d --build --force-recreate
```

---

# 13. Logs

All relevant services:

```bash
sudo docker compose logs -f \
  ollama ollama-init litellm backend open-webui
```

Recent LiteLLM logs:

```bash
sudo docker compose logs --since=10m --no-color litellm
```

Recent Ollama logs:

```bash
sudo docker compose logs --since=10m --no-color ollama
```

Recent backend logs:

```bash
sudo docker compose logs --since=10m --no-color backend
```

---

# 14. Common problems

## Port 11434 already in use

The containers talk to Ollama internally through:

```text
http://ollama:11434
```

If the host does not need direct Ollama access, remove this from the Ollama service:

```yaml
ports:
  - "127.0.0.1:11434:11434"
```

## Port 5432 already in use

PostgreSQL does not need to be published to the host for normal operation. Remove its `ports:` mapping. The backend still uses:

```text
postgres:5432
```

## LiteLLM is unhealthy

```bash
sudo docker compose logs --no-color --tail=300 \
  litellm ollama-init ollama
```

Common causes:

- local model still downloading
- Ollama cannot load the configured model
- `.env` changed but LiteLLM was not recreated
- cloud-provider environment/config error
- Ollama request timeout

Test LiteLLM config rendering directly:

```bash
sudo docker compose run --rm --no-deps \
  --entrypoint python \
  litellm \
  /config/render_config.py
```

## Ollama is extremely slow

Check:

```bash
sudo docker compose exec ollama ollama ps
```

If it shows `100% CPU`, GPU acceleration is not active.

Check device request:

```bash
sudo docker inspect <project>-ollama-1 \
  --format '{{json .HostConfig.DeviceRequests}}'
```

If it returns `null`, recreate the container after fixing GPU configuration.

Test Docker GPU access:

```bash
sudo docker run --rm --gpus all \
  nvidia/cuda:12.9.0-base-ubuntu22.04 \
  nvidia-smi
```

## Router stalls for 240 seconds

Inspect Ollama:

```bash
sudo docker compose logs --since=15m --no-color ollama | tail -n 300
```

If generation speed is extremely low, the issue is local inference performance, not LangGraph routing.

## `.env` says one model but LiteLLM uses another

Check for shell overrides:

```bash
env | grep OLLAMA_ROUTER_MODEL_ID
```

Shell environment variables override `.env`.

If needed:

```bash
unset OLLAMA_ROUTER_MODEL_ID
```

Then recreate LiteLLM.

---

# 15. GIST Regulations RAG

The GIST workflow uses the supplied read-only FAISS store:

```text
jireumgil_index/index.faiss
jireumgil_index/index.pkl
```

Query path:

```text
user query
→ embedding through LiteLLM
→ FAISS nearest-neighbor retrieval
→ top regulation passages
→ answer model through LiteLLM
→ cited response
```

No user file upload is required.

Keep:

```dotenv
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
GIST_REGULATIONS_VECTOR_DIMENSIONS=1536
```

unless the FAISS index is rebuilt.

---

# 16. LiteLLM's role

Application routing and infrastructure routing are separate.

LangGraph decides:

```text
Auto
→ Direct / GIST Regulations / Research Paper
→ difficulty
→ logical tier
```

The backend calls LiteLLM aliases:

```text
local-router
local-fast
cloud-small
cloud-large
embedding
```

LiteLLM handles:

- provider abstraction
- provider credentials
- virtual keys
- team/user budgets
- model access restrictions
- retries/fallbacks
- rate limits
- usage/cost accounting
- mapping aliases to actual deployments

No workflow should call OpenAI, Ollama, Anthropic, etc. directly.

---

# 17. Pilot deployment checklist

Before giving the system to lab users:

- [ ] NVIDIA GPU works from Docker
- [ ] Ollama shows GPU usage
- [ ] LiteLLM is healthy
- [ ] Backend is healthy
- [ ] Open WebUI is healthy
- [ ] `auto` routing is tested on a labeled request set
- [ ] `direct` works with expected local/cloud tiers
- [ ] GIST Regulations returns source-grounded answers
- [ ] Research Paper Drafting completes
- [ ] Cloud API keys are configured
- [ ] LiteLLM virtual team/user keys are provisioned
- [ ] Budget/RPM/TPM limits are verified
- [ ] Public Open WebUI signup is disabled
- [ ] Basic load test is completed
- [ ] Logs and model spend are reviewed

---

# 18. Quick command reference

Start:

```bash
sudo docker compose --profile ollama up -d
```

Start after code changes:

```bash
sudo docker compose --profile ollama up -d --build --force-recreate
```

Stop:

```bash
sudo docker compose --profile ollama down
```

Status:

```bash
sudo docker compose ps
```

Logs:

```bash
sudo docker compose logs -f ollama litellm backend open-webui
```

UI:

```text
http://<server-ip>:3000
```

Local models:

```bash
sudo docker compose exec ollama ollama list
sudo docker compose exec ollama ollama ps
```

LiteLLM config:

```bash
sudo docker compose exec litellm cat /tmp/litellm-config.yaml
```

Backend health:

```bash
curl http://127.0.0.1:8000/api/health
```

---

# 19. Persistent-data warning

These Docker volumes contain persistent state:

```text
postgres-data
litellm-postgres-data
redis-data
ollama-data
open-webui-data
```

Normal `docker compose down` preserves them.

This command deletes volumes and data:

```bash
docker compose down -v
```

Do not use `-v` unless deletion is intentional.
