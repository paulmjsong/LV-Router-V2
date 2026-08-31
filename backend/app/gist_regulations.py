from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway
from .schemas import SourceCitation


_PROVISION_RE = re.compile(
    r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)
_FR_CODE_RE = re.compile(r"FR\d+", re.IGNORECASE)
_REFERENCES_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?📌\s*(?:\*\*)?"
    r"(?:references|참조\s*조항|참고\s*문헌)(?:\*\*)?\s*$"
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Load only the two LangChain classes present in the supplied index pickle."""

    def find_class(self, module: str, name: str):
        if (module, name) == (
            "langchain_community.docstore.in_memory",
            "InMemoryDocstore",
        ):
            from langchain_community.docstore.in_memory import InMemoryDocstore

            return InMemoryDocstore
        if (module, name) == ("langchain_core.documents.base", "Document"):
            from langchain_core.documents.base import Document

            return Document
        raise pickle.UnpicklingError(
            f"Forbidden global in Jireumgil index: {module}.{name}"
        )


def _safe_load_pickle(path: Path) -> tuple[Any, dict[int, str]]:
    with path.open("rb") as handle:
        value = _RestrictedUnpickler(handle).load()
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("index.pkl must contain (docstore, index_to_docstore_id)")
    docstore, mapping = value
    if not isinstance(mapping, dict) or not all(isinstance(key, int) for key in mapping):
        raise ValueError("index.pkl contains an invalid FAISS document mapping")
    return docstore, mapping


class GISTRegulationsRetriever:
    """Retrieve GIST rules and attach controlled links to the actual PDF files."""

    def __init__(self, settings: Settings, llm: LLMGateway) -> None:
        self.settings = settings
        self.llm = llm
        self.index = None
        self.docstore = None
        self.index_to_docstore_id: dict[int, str] = {}
        self._pdf_by_key: dict[str, Path] = {}

    async def initialize(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required for the GIST regulations workflow"
            ) from exc

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
                f"index={index.d}, "
                f"configured={self.settings.gist_regulations_vector_dimensions}"
            )
        if int(index.ntotal) != len(mapping):
            raise ValueError(
                f"GIST FAISS mapping mismatch: index has {index.ntotal} vectors, "
                f"mapping has {len(mapping)}"
            )

        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = mapping
        self._pdf_by_key = self._discover_pdfs(index_dir)
        if not self._pdf_by_key:
            raise FileNotFoundError(
                f"No regulation PDFs were found below {index_dir}. "
                "Expected them under jireumgil_index/pdfs."
            )

    @staticmethod
    def _discover_pdfs(index_dir: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for path in sorted(index_dir.rglob("*.pdf")):
            for key in (path.name, path.stem):
                found.setdefault(key.casefold(), path)
            match = _FR_CODE_RE.search(path.name)
            if match:
                found.setdefault(match.group(0).casefold(), path)
        return found

    def _resolve_pdf(self, source: str) -> Path | None:
        basename = re.split(r"[\\/]", source)[-1].strip()
        candidates = [basename, Path(basename).stem]
        match = _FR_CODE_RE.search(basename)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            path = self._pdf_by_key.get(candidate.casefold())
            if path is not None:
                return path
        return None

    @staticmethod
    def _pdf_url(path: Path, page: int | None) -> str:
        # The PDFs are mounted into Open WebUI's same-origin static directory.
        url = f"/static/gist-regulations/{quote(path.name, safe='')}"
        if page is not None:
            url += f"#page={page}"
        return url

    @staticmethod
    def _markdown_link(source: SourceCitation) -> str:
        if source.url:
            return f"[{source.title}]({source.url})"
        return source.title

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
                f"GIST query embedding dimension mismatch: got {vector.shape}, "
                f"expected {(int(self.index.d),)}"
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
            excerpt = str(getattr(document, "page_content", "")).strip()
            if not excerpt:
                continue

            if getattr(self.index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
                score = 1.0 / (1.0 + max(float(distance), 0.0))
            else:
                score = float(distance)

            pdf_path = self._resolve_pdf(source)
            raw_title = re.split(r"[\\/]", source)[-1] or "GIST regulation"
            title = pdf_path.stem if pdf_path is not None else Path(raw_title).stem
            results.append(
                SourceCitation(
                    chunk_id=int(faiss_id),
                    document_id=uuid5(NAMESPACE_URL, f"gist-regulation:{doc_id}"),
                    title=title or "GIST regulation",
                    page=page,
                    score=score,
                    excerpt=excerpt[:1600],
                    url=self._pdf_url(pdf_path, page) if pdf_path is not None else None,
                )
            )
        return results

    def context_from_sources(self, sources: list[SourceCitation]) -> str:
        blocks: list[str] = []
        used = 0
        for rank, source in enumerate(sources, start=1):
            citation = self._markdown_link(source)
            page = str(source.page) if source.page is not None else "unknown"
            provisions = list(dict.fromkeys(_PROVISION_RE.findall(source.excerpt)))[:6]
            provision_text = ", ".join(provisions) if provisions else "not pre-extracted"
            block = (
                f"[SOURCE {rank}]\n"
                f"Citation: {citation}\n"
                f"Regulation: {source.title}\n"
                f"PDF page: {page}\n"
                f"Detected provisions: {provision_text}\n"
                f"Evidence:\n{source.excerpt}"
            )
            if used + len(block) > self.settings.gist_regulations_context_chars:
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    def references_markdown(self, sources: list[SourceCitation]) -> str:
        lines = ["\n\n📌 **References**"]
        seen: set[tuple[str, str | None, int | None]] = set()
        for source in sources:
            key = (source.title, source.url, source.page)
            if key in seen:
                continue
            seen.add(key)
            provisions = list(dict.fromkeys(_PROVISION_RE.findall(source.excerpt)))[:4]
            detail = ", ".join(provisions)
            if not detail and source.page is not None:
                detail = f"PDF p.{source.page}"
            suffix = f" — {detail}" if detail else ""
            lines.append(f"- {self._markdown_link(source)}{suffix}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def has_references_section(self, answer: str) -> bool:
        return bool(_REFERENCES_RE.search(answer))

    def format_answer(
        self,
        answer: str,
        sources: list[SourceCitation],
    ) -> str:
        formatted = answer.strip()
        for rank, source in enumerate(sources, start=1):
            formatted = re.sub(
                rf"\[SOURCE\s+{rank}\]",
                lambda _: self._markdown_link(source),
                formatted,
                flags=re.IGNORECASE,
            )
        if not self.has_references_section(formatted):
            formatted += self.references_markdown(sources)
        return formatted.strip()
