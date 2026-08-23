# Validation

Static validation for v0.4 checks:
- four selectable Open WebUI modes only;
- Auto is the default mode;
- Grant/Website are not selectable or routable;
- user file/web uploads are disabled in the UI adapter;
- one parent LangGraph mounts only Direct, GIST Regulations, and Research Paper subgraphs;
- paper subgraph contains orchestrator, two specialist subagents, drafter, validator, and finalizer;
- GIST Regulations uses the supplied FAISS pair and a restricted pickle loader;
- no runtime import of the removed generic upload/RAG modules;
- router fallbacks do not collapse to local-fast.

Run:

```powershell
python -X utf8 scripts/validate_static.py
python -m compileall -q backend/app backend/tests infra/litellm scripts
```
