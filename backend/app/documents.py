from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from fastapi import HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway
from .repositories import CollectionRepository, DocumentRepository
from .schemas import DocumentInfo
from .storage import ObjectStore, safe_filename

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParsedSection:
    text: str
    page_number: int | None = None
    heading: str | None = None


class DocumentParser:
    SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".json"}

    def parse(self, filename: str, data: bytes) -> list[ParsedSection]:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
        if suffix == ".pdf":
            return self._parse_pdf(data)
        if suffix == ".docx":
            return self._parse_docx(data)
        if suffix in {".html", ".htm"}:
            text = BeautifulSoup(data, "html.parser").get_text("\n", strip=True)
            return [ParsedSection(text=text)]
        text = data.decode("utf-8", errors="replace")
        return [ParsedSection(text=text)]

    @staticmethod
    def _parse_pdf(data: bytes) -> list[ParsedSection]:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("Encrypted PDFs are not supported") from exc
        sections: list[ParsedSection] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(ParsedSection(text=text, page_number=index))
        return sections

    @staticmethod
    def _parse_docx(data: bytes) -> list[ParsedSection]:
        document = DocxDocument(io.BytesIO(data))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return [ParsedSection(text="\n".join(paragraphs))]


class DocumentService:
    def __init__(
        self,
        *,
        settings: Settings,
        collections: CollectionRepository,
        documents: DocumentRepository,
        llm: LLMGateway,
        object_store: ObjectStore,
    ) -> None:
        self.settings = settings
        self.collections = collections
        self.documents = documents
        self.llm = llm
        self.object_store = object_store
        self.parser = DocumentParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest(self, upload: UploadFile, collection_id: UUID, user: UserContext) -> DocumentInfo:
        await self.collections.require_write(collection_id, user)
        data = await upload.read()
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"The file exceeds the {self.settings.max_upload_mb} MB limit",
            )
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")

        filename = safe_filename(upload.filename or "document")
        content_type = upload.content_type or "application/octet-stream"
        digest = hashlib.sha256(data).hexdigest()
        object_key = f"collections/{collection_id}/{uuid4().hex}_{filename}"
        storage_uri = await self.object_store.put(object_key, data, content_type)
        record = await self.documents.create(
            collection_id=collection_id,
            filename=filename,
            mime_type=content_type,
            storage_uri=storage_uri,
            sha256=digest,
            user_id=user.user_id,
        )

        try:
            sections = await asyncio.to_thread(self.parser.parse, filename, data)
            chunks = await asyncio.to_thread(self._split_sections, sections)
            if not chunks:
                raise ValueError("No extractable text was found")

            embeddings: list[list[float]] = []
            batch_size = 32
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                embeddings.extend(
                    await self.llm.embed_texts(
                        user=user,
                        texts=[chunk["content"] for chunk in batch],
                        run_id=f"index-{record.id}",
                        stage="document_embedding",
                    )
                )

            rows: list[dict] = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                rows.append(
                    {
                        **chunk,
                        "embedding": "[" + ",".join(format(value, ".10g") for value in embedding) + "]",
                    }
                )
            await self.documents.insert_chunks(record.id, collection_id, rows)
            return await self.documents.get(record.id)
        except Exception as exc:
            await self.documents.mark_failed(record.id, str(exc))
            logger.exception("Document indexing failed for document_id=%s", record.id)
            raise HTTPException(
                status_code=422,
                detail="Document indexing failed. Check the server log for the recorded cause.",
            ) from exc

    def _split_sections(self, sections: list[ParsedSection]) -> list[dict]:
        chunks: list[dict] = []
        chunk_index = 0
        for section in sections:
            for text in self.splitter.split_text(section.text):
                clean = text.strip()
                if not clean:
                    continue
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "page_number": section.page_number,
                        "heading": section.heading,
                        "content": clean,
                    }
                )
                chunk_index += 1
        return chunks
