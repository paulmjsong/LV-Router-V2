#!/usr/bin/env python3
"""Safely inspect, append to, or rebuild the LV-Router GIST regulations FAISS store.

Designed for LV-Router-V2's current jireumgil_index layout:
  jireumgil_index/
    index.faiss
    index.pkl
    pdfs/*.pdf

Compatibility rules preserved:
- index.pkl contains exactly (InMemoryDocstore, index_to_docstore_id)
- metadata['source'] is the PDF basename
- metadata['page'] is zero-based (runtime converts it to one-based display)
- new vectors use the same FAISS index class/metric as the current index
- current index dimensionality is treated as authoritative
- existing source names and FR codes cannot be appended twice
- installation is staged, validated, backed up, then swapped

Run `inspect` first. Stop the backend before `add` or `rebuild`, then restart it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


FR_CODE_RE = re.compile(r"FR\d+", re.IGNORECASE)
EXPECTED_LEGACY_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class RestrictedUnpickler(pickle.Unpickler):
    """Match the runtime's allowlist for the legacy LangChain pickle."""

    def find_class(self, module: str, name: str):  # noqa: D401
        if (module, name) == (
            "langchain_community.docstore.in_memory",
            "InMemoryDocstore",
        ):
            from langchain_community.docstore.in_memory import InMemoryDocstore

            return InMemoryDocstore
        if (module, name) == ("langchain_core.documents.base", "Document"):
            from langchain_core.documents.base import Document

            return Document
        raise pickle.UnpicklingError(f"Forbidden global in index.pkl: {module}.{name}")


def safe_load_pickle(path: Path) -> tuple[Any, dict[int, str]]:
    with path.open("rb") as handle:
        value = RestrictedUnpickler(handle).load()
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("index.pkl must contain (docstore, index_to_docstore_id)")
    docstore, mapping = value
    if not isinstance(mapping, dict) or not all(isinstance(k, int) for k in mapping):
        raise ValueError("index.pkl has an invalid FAISS mapping")
    return docstore, mapping


def load_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is None or not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def env_value(name: str, file_values: dict[str, str], default: str = "") -> str:
    return os.environ.get(name) or file_values.get(name) or default


def pdf_key(path_or_name: str) -> str:
    return Path(re.split(r"[\\/]", path_or_name)[-1]).name.casefold()


def fr_code(path_or_name: str) -> str | None:
    match = FR_CODE_RE.search(Path(path_or_name).name)
    return match.group(0).casefold() if match else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_documents(docstore: Any, mapping: dict[int, str]) -> dict[str, Any]:
    docs: dict[str, Any] = {}
    for doc_id in mapping.values():
        doc = docstore.search(doc_id)
        if isinstance(doc, str):
            raise ValueError(f"Missing document for docstore id {doc_id!r}")
        docs[doc_id] = doc
    return docs


def validate_loaded(index: Any, docstore: Any, mapping: dict[int, str]) -> None:
    if int(index.ntotal) != len(mapping):
        raise ValueError(
            f"FAISS/docstore mismatch: index has {index.ntotal} vectors but mapping has {len(mapping)}"
        )
    expected_positions = set(range(int(index.ntotal)))
    if set(mapping) != expected_positions:
        raise ValueError("FAISS mapping keys must be contiguous 0..ntotal-1")
    docs = current_documents(docstore, mapping)
    for faiss_id, doc_id in mapping.items():
        doc = docs[doc_id]
        text = str(getattr(doc, "page_content", "")).strip()
        if not text:
            raise ValueError(f"Mapped document {doc_id!r} at FAISS id {faiss_id} has empty text")
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = metadata.get("source")
        page = metadata.get("page")
        # Match the runtime: source/page may be absent in legacy chunks, but when
        # present they must be sane. New chunks written by this tool always set both.
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"Document {doc_id!r} has invalid metadata['source']")
        if page is not None and (not isinstance(page, int) or page < 0):
            raise ValueError(f"Document {doc_id!r} has invalid metadata['page']")


def load_store(index_dir: Path) -> tuple[Any, Any, dict[int, str]]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required") from exc
    index_path = index_dir / "index.faiss"
    pickle_path = index_dir / "index.pkl"
    if not index_path.is_file() or not pickle_path.is_file():
        raise FileNotFoundError(f"Expected {index_path} and {pickle_path}")
    index = faiss.read_index(str(index_path))
    docstore, mapping = safe_load_pickle(pickle_path)
    validate_loaded(index, docstore, mapping)
    return index, docstore, mapping


def listed_sources(docstore: Any, mapping: dict[int, str]) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    codes: set[str] = set()
    for doc in current_documents(docstore, mapping).values():
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = str(metadata.get("source") or "")
        if source:
            names.add(pdf_key(source))
            code = fr_code(source)
            if code:
                codes.add(code)
    return names, codes


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        hard_end = min(start + chunk_size, n)
        end = hard_end
        if hard_end < n:
            lower_bound = start + max(chunk_size // 2, 1)
            candidates = [
                text.rfind("\n\n", lower_bound, hard_end),
                text.rfind("\n", lower_bound, hard_end),
                text.rfind(" ", lower_bound, hard_end),
            ]
            split_at = max(candidates)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks


def extract_pdf_documents(
    pdf_path: Path,
    *,
    chunk_size: int,
    overlap: int,
) -> list[Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required: python -m pip install 'pypdf>=5,<7'") from exc
    from langchain_core.documents import Document

    reader = PdfReader(str(pdf_path))
    documents: list[Any] = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk_index, chunk in enumerate(
            split_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_index,
                        "chunk": chunk_index,
                    },
                )
            )
    if not documents:
        raise ValueError(
            f"No extractable text found in {pdf_path}. OCR scanned/image-only PDFs before indexing."
        )
    return documents


def stable_doc_id(document: Any) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    payload = "\0".join(
        [
            str(metadata.get("source") or ""),
            str(metadata.get("page") if metadata.get("page") is not None else ""),
            str(metadata.get("chunk") if metadata.get("chunk") is not None else ""),
            str(getattr(document, "page_content", "")),
        ]
    ).encode("utf-8")
    return "gist-" + hashlib.sha256(payload).hexdigest()


def embedding_client(args: argparse.Namespace, env_file_values: dict[str, str]):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai Python package is required") from exc

    api_key = args.api_key or env_value("EMBEDDING_API_KEY", env_file_values)
    if not api_key:
        raise ValueError("No embedding API key. Set EMBEDDING_API_KEY or pass --api-key.")

    base_url = args.base_url or env_value("EMBEDDING_API_BASE", env_file_values)
    if not base_url:
        base_url = "https://api.openai.com/v1"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1") and "api.openai.com" not in base_url:
        # Most OpenAI-compatible gateways, including LiteLLM, expose /v1.
        base_url += "/v1"

    raw_model = args.model or env_value(
        "INDEX_EMBEDDING_MODEL",
        env_file_values,
        env_value("EMBEDDING_MODEL", env_file_values, "openai/text-embedding-3-small"),
    )
    model = raw_model
    # LiteLLM config stores provider-qualified model names; direct OpenAI does not.
    if "api.openai.com" in base_url and model.startswith("openai/"):
        model = model.split("/", 1)[1]

    return OpenAI(api_key=api_key, base_url=base_url), model, base_url


def embed_documents(
    documents: Sequence[Any],
    *,
    client: Any,
    model: str,
    dimensions: int,
    batch_size: int,
) -> Any:
    import numpy as np

    rows: list[list[float]] = []
    texts = [str(getattr(doc, "page_content", "")) for doc in documents]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
            dimensions=dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        rows.extend(item.embedding for item in ordered)
        print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)} chunks", flush=True)
    matrix = np.asarray(rows, dtype="float32")
    if matrix.ndim != 2 or matrix.shape != (len(documents), dimensions):
        raise ValueError(
            f"Unexpected embedding matrix shape {matrix.shape}; expected {(len(documents), dimensions)}"
        )
    return matrix


def clone_docstore(docstore: Any, mapping: dict[int, str]) -> Any:
    from langchain_community.docstore.in_memory import InMemoryDocstore

    return InMemoryDocstore(dict(current_documents(docstore, mapping)))


def add_documents_to_store(
    index: Any,
    docstore: Any,
    mapping: dict[int, str],
    documents: Sequence[Any],
    vectors: Any,
) -> None:
    doc_ids = [stable_doc_id(doc) for doc in documents]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Generated duplicate document IDs in the new corpus")
    existing_ids = set(mapping.values())
    collisions = existing_ids.intersection(doc_ids)
    if collisions:
        raise ValueError(f"New documents collide with {len(collisions)} existing docstore IDs")

    start = int(index.ntotal)
    index.add(vectors)
    docstore.add(dict(zip(doc_ids, documents)))
    for offset, doc_id in enumerate(doc_ids):
        mapping[start + offset] = doc_id


def save_pair(index_dir: Path, index: Any, docstore: Any, mapping: dict[int, str]) -> None:
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with (index_dir / "index.pkl").open("wb") as handle:
        pickle.dump((docstore, mapping), handle, protocol=pickle.HIGHEST_PROTOCOL)


def validate_pair_on_disk(index_dir: Path, *, expected_dim: int) -> tuple[int, int]:
    index, docstore, mapping = load_store(index_dir)
    if int(index.d) != expected_dim:
        raise ValueError(f"Output dimension {index.d} != expected {expected_dim}")
    return int(index.ntotal), len(mapping)


def backup_current(index_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = index_dir / "backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for name in ("index.faiss", "index.pkl"):
        shutil.copy2(index_dir / name, backup / name)
    return backup


def install_staged_pair(index_dir: Path, staged: Path) -> Path:
    backup = backup_current(index_dir)
    try:
        os.replace(staged / "index.faiss", index_dir / "index.faiss")
        os.replace(staged / "index.pkl", index_dir / "index.pkl")
    except Exception:
        # Restore a coherent pair if either replacement fails.
        shutil.copy2(backup / "index.faiss", index_dir / "index.faiss")
        shutil.copy2(backup / "index.pkl", index_dir / "index.pkl")
        raise
    return backup


def ensure_pdf_inputs(pdf_paths: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    seen_names: set[str] = set()
    seen_codes: set[str] = set()
    for raw in pdf_paths:
        path = raw.expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() != ".pdf":
            raise FileNotFoundError(f"Not a PDF file: {path}")
        name_key = path.name.casefold()
        code = fr_code(path.name)
        if name_key in seen_names:
            raise ValueError(f"Duplicate input filename: {path.name}")
        if code and code in seen_codes:
            raise ValueError(f"Duplicate FR code among inputs: {code.upper()}")
        seen_names.add(name_key)
        if code:
            seen_codes.add(code)
        resolved.append(path)
    return resolved


def copy_new_pdfs(pdf_paths: Sequence[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in pdf_paths:
        target = target_dir / source.name
        if target.exists():
            if sha256_file(source) != sha256_file(target):
                raise ValueError(
                    f"{target.name} already exists in pdfs/ with different contents. Rebuild instead."
                )
            continue
        shutil.copy2(source, target)


def inspect_command(index_dir: Path) -> None:
    import faiss

    index, docstore, mapping = load_store(index_dir)
    sources: dict[str, int] = {}
    pages: dict[str, set[int]] = {}
    for doc in current_documents(docstore, mapping).values():
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = Path(str(metadata.get("source") or "GIST regulation")).name
        sources[source] = sources.get(source, 0) + 1
        page = metadata.get("page")
        if isinstance(page, int) and page >= 0:
            pages.setdefault(source, set()).add(page)
        else:
            pages.setdefault(source, set())

    metric = "L2" if getattr(index, "metric_type", None) == faiss.METRIC_L2 else str(
        getattr(index, "metric_type", "unknown")
    )
    print(f"INDEX_TYPE={type(index).__name__}")
    print(f"METRIC={metric}")
    print(f"DIMENSIONS={int(index.d)}")
    print(f"VECTORS={int(index.ntotal)}")
    print(f"MAPPING={len(mapping)}")
    print(f"SOURCES={len(sources)}")
    for source in sorted(sources):
        print(f"  {source}: chunks={sources[source]} pages={len(pages[source])}")


def build_documents_for_pdfs(
    paths: Sequence[Path], *, chunk_size: int, overlap: int
) -> list[Any]:
    documents: list[Any] = []
    for path in paths:
        docs = extract_pdf_documents(path, chunk_size=chunk_size, overlap=overlap)
        print(f"{path.name}: extracted {len(docs)} chunks")
        documents.extend(docs)
    return documents


def mutating_command(args: argparse.Namespace, *, rebuild: bool) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required") from exc
    from langchain_community.docstore.in_memory import InMemoryDocstore

    index_dir = args.index_dir.resolve()
    pdf_dir = index_dir / "pdfs"
    old_index, old_docstore, old_mapping = load_store(index_dir)
    dimensions = int(old_index.d)

    if dimensions != args.dimensions:
        raise ValueError(
            f"Configured --dimensions={args.dimensions}, but current index is {dimensions}. "
            "Use the current dimension; do not mix embedding spaces."
        )

    env_values = load_env_file(args.env_file)
    client, model, base_url = embedding_client(args, env_values)
    normalized_model = model.split("/", 1)[-1]
    if normalized_model != EXPECTED_LEGACY_MODEL:
        raise ValueError(
            f"Refusing model {model!r}. This maintenance tool intentionally locks the current "
            f"LV-Router store to {EXPECTED_LEGACY_MODEL!r}; changing embedding models also "
            "requires coordinated runtime configuration changes."
        )

    # Dimension probe catches endpoint/deployment mistakes before modifying anything.
    probe = client.embeddings.create(
        model=model,
        input=["GIST regulation embedding compatibility probe"],
        dimensions=dimensions,
        encoding_format="float",
    )
    probe_len = len(probe.data[0].embedding)
    if probe_len != dimensions:
        raise ValueError(f"Embedding endpoint returned {probe_len} dimensions; expected {dimensions}")
    print(f"EMBEDDING_ENDPOINT={base_url}")
    print(f"EMBEDDING_MODEL={model}")
    print(f"EMBEDDING_DIMENSIONS={probe_len}")

    if rebuild:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.casefold())
        if not pdf_paths:
            raise ValueError(f"No PDFs found in {pdf_dir}")
        documents = build_documents_for_pdfs(
            pdf_paths, chunk_size=args.chunk_size, overlap=args.chunk_overlap
        )
        new_index = faiss.clone_index(old_index)
        new_index.reset()
        if not getattr(new_index, "is_trained", True):
            raise ValueError(
                "Cloned FAISS index is not trained after reset. This script refuses to retrain an "
                "unknown legacy index type automatically."
            )
        new_docstore = InMemoryDocstore({})
        new_mapping: dict[int, str] = {}
    else:
        pdf_paths = ensure_pdf_inputs(args.pdf)
        existing_names, existing_codes = listed_sources(old_docstore, old_mapping)
        for path in pdf_paths:
            name_key = path.name.casefold()
            code = fr_code(path.name)
            if name_key in existing_names:
                raise ValueError(
                    f"{path.name} is already represented in the index. If this is a new revision, "
                    "replace the PDF in pdfs/ and run rebuild; do not append two versions."
                )
            if code and code in existing_codes:
                raise ValueError(
                    f"{code.upper()} already exists in the index under another filename. "
                    "Use rebuild for regulation revisions."
                )
        documents = build_documents_for_pdfs(
            pdf_paths, chunk_size=args.chunk_size, overlap=args.chunk_overlap
        )
        new_index = faiss.clone_index(old_index)
        new_docstore = clone_docstore(old_docstore, old_mapping)
        new_mapping = dict(old_mapping)

    vectors = embed_documents(
        documents,
        client=client,
        model=model,
        dimensions=dimensions,
        batch_size=args.batch_size,
    )
    add_documents_to_store(new_index, new_docstore, new_mapping, documents, vectors)
    validate_loaded(new_index, new_docstore, new_mapping)

    with tempfile.TemporaryDirectory(prefix="gist-index-build-", dir=str(index_dir)) as temp_name:
        staged = Path(temp_name)
        save_pair(staged, new_index, new_docstore, new_mapping)
        vector_count, mapping_count = validate_pair_on_disk(staged, expected_dim=dimensions)

        if not rebuild:
            # Copy PDFs before activating the new mapping so citation resolution succeeds on restart.
            copy_new_pdfs(pdf_paths, pdf_dir)

        backup = install_staged_pair(index_dir, staged)

    print("INDEX_UPDATE=PASS")
    print(f"MODE={'rebuild' if rebuild else 'add'}")
    print(f"VECTORS={vector_count}")
    print(f"MAPPING={mapping_count}")
    print(f"BACKUP={backup}")
    print("RESTART_REQUIRED=backend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inspect", "add", "rebuild"),
        help="inspect current store, append new PDFs, or rebuild from every PDF in pdfs/",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("jireumgil_index"),
        help="directory containing index.faiss, index.pkl and pdfs/",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        nargs="*",
        default=[],
        help="PDF(s) to append; required for add and ignored for rebuild",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--dimensions", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()
    if args.command == "add" and not args.pdf:
        parser.error("add requires at least one --pdf path")
    if args.command != "add" and args.pdf:
        parser.error("--pdf is only valid with add")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.chunk_size <= 0 or args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.command == "inspect":
            inspect_command(args.index_dir.resolve())
        elif args.command == "add":
            mutating_command(args, rebuild=False)
        else:
            mutating_command(args, rebuild=True)
        return 0
    except Exception as exc:
        print(f"INDEX_UPDATE_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
