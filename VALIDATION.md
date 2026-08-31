# Validation

Static validation for v0.5 checks:
- five selectable Open WebUI modes: Auto, Direct, GIST Regulations, Web Search, and Research Paper;
- Auto is the default mode;
- Grant/Website are not selectable or routable;
- user file uploads and Open WebUI native web search are disabled;
- one parent LangGraph mounts Direct, GIST Regulations, Web Search, and Research Paper subgraphs;
- Web Search uses the `ddgs` DuckDuckGo backend, a bounded timeout, snippet-only evidence, inline citations, and deterministic source links;
- paper subgraph contains orchestrator, two specialist subagents, drafter, validator, and finalizer;
- GIST Regulations uses the supplied FAISS pair and a restricted pickle loader;
- no runtime import of the removed generic upload/RAG modules;
- invalid router output uses a visible non-local fallback rather than silently collapsing to local-fast.

Run:

```bash
python -X utf8 scripts/validate_static.py
python -m compileall -q backend/app backend/tests infra/litellm scripts
```

Full runtime tests require backend development dependencies:

```bash
cd backend
python -m pip install -e ".[dev]"
pytest -q
```
