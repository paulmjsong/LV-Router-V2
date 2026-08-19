# Validation Record

Validated in the delivery environment:

- Python compilation: PASS
- Docker Compose YAML parsing: PASS
- LiteLLM config rendering for vLLM: PASS
- LiteLLM config rendering for Ollama: PASS
- Local semantic-router unit tests: 3/3 PASS
- Static routing invariants: PASS
- Provider-bypass audit: PASS
- Legacy Gradio/FAISS audit: PASS

Not executed here:

- Full Docker Compose startup: Docker is unavailable in this environment.
- LangGraph graph-integration test: the required packages cannot be downloaded from this environment.
- Live calls to vLLM/Ollama, LiteLLM, cloud providers, PostgreSQL, MinIO, or Open WebUI.

Run `make test`, `make audit`, and the deployment smoke checks in `README.md` on the target server.
