# Validation Record

Validated in this delivery environment:

- Python syntax compilation for backend, tests, LiteLLM helpers, and scripts: **PASS**
- Parent graph/subgraph static invariants: **PASS**
- Simple Open WebUI workflow IDs and Infonet branding audit: **PASS**
- Legacy Gradio/provider-client/FAISS runtime audit: **PASS**
- Docker Compose YAML parsing: **PASS**
- Git whitespace/error check (`git diff --check`): **PASS**
- GIST reserved collection and administrator ingestion path: **included and statically verified**
- Obsolete website approval/resume interfaces after placeholder simplification: **removed and statically verified**
- Router normalization/fallback/tier tests and local streaming-limit tests: **13 passed**
- Parent-dispatch, Open WebUI PDF-context, and RAG-isolation tests: **included but not executed here**

Not executable in this environment:

- Full Docker Compose startup: no Docker daemon is available.
- Full pytest suite: LangGraph, LangChain Core, and OpenAI runtime dependencies are not installed and package download is unavailable in this environment.
- Live calls to Ollama/vLLM, LiteLLM, cloud providers, MinIO, PostgreSQL, or Open WebUI.

Run on the target machine:

```bash
python -m compileall backend/app backend/tests infra/litellm scripts
python scripts/validate_static.py
cd backend
python -m pip install -e '.[dev]'
pytest
```

Then run the Docker smoke checks described in `README.md`.
