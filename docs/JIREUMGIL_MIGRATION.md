# Jireumgil / GIST Regulations Vectorstore

This release **does not migrate** the original Jireumgil FAISS store to pgvector. It uses the supplied pair directly:

```text
jireumgil_index/index.faiss
jireumgil_index/index.pkl
```

The index is treated as immutable application data. Query embeddings must remain compatible with the embedding model used to build the index. The supplied store is 1536-dimensional and the original code used `text-embedding-3-small`.

If the GIST regulation corpus changes substantially or you change embedding models, rebuild the vectorstore from the source regulation documents and replace both files together.

The pickle loader is restricted to the LangChain `InMemoryDocstore` and `Document` classes present in this supplied file; unrestricted deserialization is not enabled.
