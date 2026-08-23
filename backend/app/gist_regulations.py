from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway
from .schemas import SourceCitation


class _RestrictedUnpickler(pickle.Unpickler):
    """Load only the two LangChain classes present in the supplied Jireumgil pickle."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("langchain_community.docstore.in_memory", "InMemoryDocstore"):
            from langchain_community.docstore.in_memory import InMemoryDocstore

            return InMemoryDocstore
        if (module, name) == ("langchain_core.documents.base", "Document"):
            from langchain_core.documents.base import Document

            return Document
        raise pickle.UnpicklingError(f"Forbidden global in Jireumgil index: {module}.{name}")


def _safe_load_pickle(path: Path) -> tuple[Any, dict[int, str]]:
    with path.open("rb") as handle:
        value = _RestrictedUnpickler(handle).load()
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("index.pkl must contain (docstore, index_to_docstore_id)")
    docstore, mapping = value
    if not isinstance(mapping, dict) or not all(isinstance(k, int) for k in mapping):
        raise ValueError("index.pkl contains an invalid FAISS document mapping")
    return docstore, mapping


class GISTRegulationsRetriever:
    """Read the supplied legacy FAISS store and perform a simple top-k semantic lookup."""

    def __init__(self, settings: Settings, llm: LLMGateway) -> None:
        self.settings = settings
        self.llm = llm
        self.index = None
        self.docstore = None
        self.index_to_docstore_id: dict[int, str] = {}

    async def initialize(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required for the GIST regulations workflow") from exc

        index_dir = self.settings.gist_regulations_index_dir
        index_path = index_dir / "index.faiss"
        pickle_path = index_dir / "index.pkl"
        if not index_path.is_file() or not pickle_path.is_file():
            raise FileNotFoundError(
                f"GIST regulation vectorstore is missing from {index_dir}. "
                "Expected index.faiss and index.pkl."
            )
        index = faiss.read_index(str(index_path))
        docstore, mapping = _safe_load_pickle(pickle_path)
        if int(index.d) != self.settings.gist_regulations_vector_dimensions:
            raise ValueError(
                "GIST FAISS embedding dimension mismatch: "
                f"index={index.d}, configured={self.settings.gist_regulations_vector_dimensions}"
            )
        if int(index.ntotal) != len(mapping):
            raise ValueError(
                f"GIST FAISS mapping mismatch: index has {index.ntotal} vectors, mapping has {len(mapping)}"
            )
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = mapping

    async def retrieve(
        self,
        *,
        query: str,
        user: UserContext,
        run_id: str,
    ) -> list[SourceCitation]:
        if self.index is None or self.docstore is None:
            raise RuntimeError("GIST regulation vectorstore is not initialized")
        vectors = await self.llm.embed_texts(
            user=user,
            texts=[query],
            run_id=run_id,
            stage="gist_regulations_query_embedding",
        )
        vector = np.asarray(vectors[0], dtype="float32")
        if vector.shape != (int(self.index.d),):
            raise ValueError(
                f"GIST query embedding dimension mismatch: got {vector.shape}, expected {(int(self.index.d),)}"
            )
        return await asyncio.to_thread(self._search, vector)

    def _search(self, vector: np.ndarray) -> list[SourceCitation]:
        import faiss

        distances, indices = self.index.search(
            vector.reshape(1, -1),
            self.settings.gist_regulations_top_k,
        )
        results: list[SourceCitation] = []
        for distance, faiss_id in zip(distances[0].tolist(), indices[0].tolist()):
            if faiss_id < 0:
                continue
            doc_id = self.index_to_docstore_id.get(int(faiss_id))
            if not doc_id:
                continue
            document = self.docstore.search(doc_id)
            if isinstance(document, str):
                continue
            metadata = dict(getattr(document, "metadata", {}) or {})
            source = str(metadata.get("source") or "GIST regulation")
            page_raw = metadata.get("page")
            page = int(page_raw) + 1 if isinstance(page_raw, int) else None
            text = str(getattr(document, "page_content", "")).strip()
            if not text:
                continue
            if getattr(self.index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
                score = 1.0 / (1.0 + max(float(distance), 0.0))
            else:
                score = float(distance)
            results.append(
                SourceCitation(
                    chunk_id=int(faiss_id),
                    document_id=uuid5(NAMESPACE_URL, f"gist-regulation:{doc_id}"),
                    title=re.split(r"[\\/]", source)[-1] or "GIST regulation",
                    page=page,
                    score=score,
                    excerpt=text[:1600],
                )
            )
        return results

    def context_from_sources(self, sources: list[SourceCitation]) -> str:
        blocks: list[str] = []
        used = 0
        for rank, source in enumerate(sources, start=1):
            page = f" p.{source.page}" if source.page is not None else ""
            block = f"[SOURCE {rank}] {source.title}{page}\n{source.excerpt}"
            if used + len(block) > self.settings.gist_regulations_context_chars:
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)
