# Migrating the Jireumgil GIST Regulations Corpus

## Decision

The legacy FAISS vector store is not used by Infonet AI Router.

Reasons:

1. It was local process state and unsuitable for a multi-user service.
2. Its vectors depend on the original embedding model and dimensions.
3. Its metadata was loaded through pickle-backed dangerous deserialization.
4. It had no central access control, document status, audit metadata, or shared retrieval layer.
5. Maintaining two retrieval stacks would make behavior and evaluation inconsistent.

## Required migration

Use the original regulation source files, not `index.faiss` or `index.pkl`.

1. Copy PDFs, DOCX, TXT, Markdown, HTML, or JSON files into `imports/gist-regulations/`.
2. Configure a working LiteLLM `embedding` alias and matching `EMBEDDING_DIMENSIONS`.
3. Start the stack.
4. Run:

```bash
docker compose exec backend python -m app.admin_cli upload-regulations \
  /imports/gist-regulations
```

The CLI scans supported files recursively. The backend parses, chunks, embeds, and writes the documents into the reserved collection whose `system_key` is `gist-regulations`.

## If only the FAISS files remain

Do not enable `allow_dangerous_deserialization` in the production service. Recover the original source documents or perform a separate offline, explicitly trusted extraction/migration outside this runtime, inspect the recovered text, and then ingest the reviewed files through the normal endpoint.
