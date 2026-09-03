#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8") if isinstance(path, str) else path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    raise AssertionError(message)


def call_name(node: ast.Call) -> str:
    parts: list[str] = []
    value = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def function(tree: ast.AST, class_name: str | None, name: str):
    parents = tree.body if isinstance(tree, ast.Module) else []
    if class_name:
        parent = next(
            (item for item in parents if isinstance(item, ast.ClassDef) and item.name == class_name),
            None,
        )
        if parent is None:
            fail(f"Missing class {class_name}")
        parents = parent.body
    found = next(
        (
            item
            for item in parents
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
        ),
        None,
    )
    if found is None:
        fail(f"Missing function {class_name + '.' if class_name else ''}{name}")
    return found


def main() -> None:
    # Router: exactly one semantic routing LLM call; no active grant/website route.
    routing_text = read("backend/app/routing.py")
    routing = ast.parse(routing_text)
    decide = function(routing, "LocalSemanticRouter", "decide")
    router_calls = [
        node for node in ast.walk(decide)
        if isinstance(node, ast.Call) and call_name(node).endswith("self.llm.chat")
    ]
    if len(router_calls) != 1:
        fail(f"LocalSemanticRouter.decide must contain exactly one LLM call, found {len(router_calls)}")
    if "used visible " not in routing_text or "fallback" not in routing_text:
        fail("Router must expose visible non-local fallback behavior")
    if "ModelTier.CLOUD_SMALL" not in routing_text or "ModelTier.LOCAL_FAST" not in routing_text:
        fail("Router difficulty-to-tier mapping is missing")

    # Parent graph: one parent with five active child subgraphs.
    parent_text = read("backend/app/workflows/parent.py")
    for term in ("subgraphs.direct", "subgraphs.pdf_document", "subgraphs.regulations", "subgraphs.web_search", "subgraphs.paper"):
        if term not in parent_text:
            fail(f"Parent graph is missing active child subgraph {term}")
    for term in ("subgraphs.grant", "subgraphs.website"):
        if term in parent_text:
            fail(f"Inactive child subgraph remains in parent graph: {term}")
    if "build_parent_graph" not in read("backend/app/runtime.py"):
        fail("WorkflowRuntime does not compile the parent graph")

    builders_text = read("backend/app/workflows/builders.py")
    builders = ast.parse(builders_text)
    direct = function(builders, None, "build_direct_subgraph")
    direct_calls = [
        node for node in ast.walk(direct)
        if isinstance(node, ast.Call) and call_name(node).endswith("services.llm.chat")
    ]
    if len(direct_calls) != 1:
        fail(f"Direct subgraph must contain exactly one answer LLM call, found {len(direct_calls)}")

    pdf_document = function(builders, None, "build_pdf_document_subgraph")
    pdf_document_source = ast.get_source_segment(builders_text, pdf_document) or ""
    for term in ("PDF_DOCUMENT_SYSTEM", "document_context", "WorkflowId.PDF", "services.llm.chat"):
        if term not in pdf_document_source:
            fail(f"PDF document subgraph is incomplete: missing {term}")

    web = function(builders, None, "build_web_search_subgraph")
    web_source = ast.get_source_segment(builders_text, web) or ""
    for term in ("services.web_search.search", "WEB_SEARCH_SYSTEM", "services.llm.chat"):
        if term not in web_source:
            fail(f"Web-search subgraph is incomplete: missing {term}")

    web_service = read("backend/app/web_search.py")
    for term in ("from ddgs import DDGS", ".text(", "backend=self.settings.web_search_backend", 'source_type="web"'):
        if term not in web_service:
            fail(f"DDGS web-search service is incomplete: missing {term}")
    for term in (
        "BeautifulSoup",
        "_ensure_public_url",
        "web_search_fetch_max_bytes",
        "web_search_fetch_timeout_seconds",
        "follow_redirects=False",
        "_weather_source",
        "Open-Meteo",
    ):
        if term not in web_service:
            fail(f"Safe live-evidence web retrieval is incomplete: {term}")

    paper = function(builders, None, "build_paper_subgraph")
    paper_source = ast.get_source_segment(builders_text, paper) or ""
    for node_name in ("orchestrator", "content_agent", "structure_agent", "draft", "validator", "final"):
        if f'graph.add_node("{node_name}"' not in paper_source:
            fail(f"Paper multi-agent graph is missing {node_name}")
    if 'graph.add_edge(["content_agent", "structure_agent"], "draft")' not in paper_source:
        fail("Paper subagents are not joined before drafting")

    # GIST RAG must use supplied FAISS pair and restricted deserialization.
    gist = read("backend/app/gist_regulations.py")
    required_gist = (
        "class _RestrictedUnpickler",
        "faiss.read_index",
        '"index.faiss"',
        '"index.pkl"',
        "self.index.search",
        "embed_texts",
    )
    missing_gist = [term for term in required_gist if term not in gist]
    if missing_gist:
        fail(f"GIST regulations retriever is incomplete: {missing_gist}")
    if "allow_dangerous_deserialization" in gist:
        fail("Unrestricted FAISS pickle deserialization is forbidden")
    for name in ("index.faiss", "index.pkl"):
        path = ROOT / "jireumgil_index" / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"Missing supplied GIST vectorstore file: {path}")

    # Open WebUI: six selectable modes; auto default; PDF-only upload path.
    compat = read("backend/app/openai_compat.py")
    expected_model_ids = {'"auto"', '"direct"', '"gist-regulations"', '"web-search"', '"research-paper"', '"pdf-document"'}
    for model_id in expected_model_ids:
        if model_id not in compat:
            fail(f"Missing active Open WebUI model ID: {model_id}")
    for old_id in ('"pdf"', '"chat"', '"regulations"', '"paper"', '"grant"', '"website"'):
        # Grant/website may appear in runtime metadata, but must not appear in provider MODEL_TO_WORKFLOW.
        model_block = compat.split("MODEL_TO_WORKFLOW", 1)[1].split("MODEL_DESCRIPTIONS", 1)[0]
        if old_id in model_block:
            fail(f"Inactive/old selectable model ID remains: {old_id}")
    for term in ("_document_context", "_has_document_attachment", "pdf_document_context_chars"):
        if term not in compat:
            fail(f"Backend PDF context forwarding is incomplete: {term}")
    if "File uploads are disabled" in compat:
        fail("Old file-upload rejection remains in the OpenAI adapter")

    compose = read("docker-compose.yml")
    required_compose = (
        'WEBUI_NAME: ${WEBUI_NAME:-Infonet AI Router}',
        '"auto","direct","web-search","pdf-document","gist-regulations","research-paper"',
        'DEFAULT_MODELS: "auto"',
        'DEFAULT_PROMPT_SUGGESTIONS:',
        'DEFAULT_MODEL_METADATA:',
        'USER_PERMISSIONS_CHAT_FILE_UPLOAD: "true"',
        'RAG_ALLOWED_FILE_EXTENSIONS: pdf',
        'RAG_EMBEDDING_ENGINE: openai',
        'RAG_EMBEDDING_MODEL: embedding',
        'RAG_SYSTEM_CONTEXT: "true"',
        'ENABLE_RETRIEVAL_QUERY_GENERATION: "false"',
        'context: ./infra/open-webui',
        'TASK_MODEL_EXTERNAL: "direct"',
        'USER_PERMISSIONS_CHAT_WEB_UPLOAD: "false"',
        'ENABLE_WEB_SEARCH: "false"',
        'USER_PERMISSIONS_FEATURES_WEB_SEARCH: "false"',
        'WEB_SEARCH_BACKEND: ${WEB_SEARCH_BACKEND:-duckduckgo}',
        "Grant — coming soon",
        "Website — coming soon",
        "./jireumgil_index:/app/jireumgil_index:ro",
    )
    missing_compose = [term for term in required_compose if term not in compose]
    if missing_compose:
        fail(f"Compose/UI settings missing: {missing_compose}")
    for service in ("minio:", "minio-init:"):
        if f"  {service}" in compose:
            fail(f"Unused upload infrastructure remains active: {service}")

    # Main runtime must initialize FAISS, not generic user upload/RAG services.
    main_source = read("backend/app/main.py")
    for term in ("GISTRegulationsRetriever", "await regulations.initialize()", "WebSearchService", "build_parent_graph"):
        if term == "build_parent_graph":
            continue
        if term not in main_source:
            fail(f"Main startup missing: {term}")
    for term in ("DocumentService", "RAGService", "build_object_store", "/api/documents/upload"):
        if term in main_source:
            fail(f"User document upload/RAG runtime remains active: {term}")

    # Branding + obsolete workflow IDs across user-visible/current runtime text.
    scan_roots = [ROOT / "backend", ROOT / "docs", ROOT / "README.md", ROOT / "docker-compose.yml", ROOT / ".env.example"]
    chunks: list[str] = []
    for item in scan_roots:
        if item.is_file():
            chunks.append(item.read_text(encoding="utf-8", errors="replace"))
        else:
            for path in item.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc"}:
                    chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    corpus = "\n".join(chunks)
    forbidden = ("SaeGyeol Lab AI", "lab-auto", "lab-direct", "lab-rag", "domain_rag")
    present = [term for term in forbidden if term in corpus]
    if present:
        fail(f"Old branding/workflow identifiers remain: {present}")

    # v0.6: user-facing route metadata, workflow steps, linked PDFs, and no
    # application-level ceilings on generated answers.
    compat_tree = ast.parse(compat)
    descriptions = next(
        node
        for node in compat_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "MODEL_DESCRIPTIONS"
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "MODEL_DESCRIPTIONS"
                    for target in node.targets
                )
            )
        )
    )
    description_source = ast.get_source_segment(compat, descriptions) or ""
    for label in (
        "Automatic Routing",
        "Direct Response",
        "Web Search",
        "GIST Regulations",
        "Research Paper Drafting",
        "Uploaded PDF",
    ):
        if label not in description_source:
            fail(f"Missing friendly Open WebUI route label: {label}")

    for term in (
        '"connection_type":"local"',
        "MODEL_ORDER_LIST:",
        "./jireumgil_index/pdfs:/app/backend/open_webui/static/gist-regulations:ro",
    ):
        if term not in compose:
            fail(f"Open WebUI route/PDF configuration missing: {term}")

    pdf_dir = ROOT / "jireumgil_index" / "pdfs"
    if not pdf_dir.is_dir() or not any(pdf_dir.glob("*.pdf")):
        fail(f"Missing GIST regulation PDFs: {pdf_dir}")

    for term in (
        "/static/gist-regulations/",
        "references_markdown",
        "Citation:",
        "#page=",
    ):
        if term not in gist:
            fail(f"GIST citation/PDF-link implementation missing: {term}")

    if pdf_document_source.count("_emit_step(") != 1:
        fail("PDF workflow must emit exactly one concise progress step")

    if "_emit_step" not in web_source:
        fail("Web-search subgraph does not emit workflow steps")
    if "_emit_step" not in paper_source:
        fail("Research-paper subgraph does not emit workflow steps")

    for token_cap in (
        "max_tokens=2400",
        "max_tokens=2200",
        "max_tokens=2600",
    ):
        if token_cap in builders_text:
            fail(
                "User-facing application-level answer token ceiling remains: "
                f"{token_cap}"
            )
    # v0.7: concise progress UI and canonical citations across turns.
    if "> **Status:** routing request" in compat:
        fail("Visible routing status noise remains in the OpenAI stream adapter")
    if "> **Served model:**" in compat:
        fail("Redundant served-model line remains in the OpenAI stream adapter")

    parent_source = read("backend/app/workflows/parent.py")
    if "difficulty {difficulty" in parent_source or "router {status}" in parent_source:
        fail("Verbose route diagnostics remain in the normal route line")

    regulations = function(builders, None, "build_regulations_subgraph")
    regulations_source = ast.get_source_segment(builders_text, regulations) or ""
    if direct_calls and "_emit_step" in (ast.get_source_segment(builders_text, direct) or ""):
        fail("Direct workflow should not emit a separate workflow-step line")
    if regulations_source.count("_emit_step(") != 1:
        fail("GIST workflow must emit exactly one concise progress step")
    if web_source.count("_emit_step(") != 1:
        fail("Web-search workflow must emit exactly one concise progress step")
    if paper_source.count("_emit_step(") != 2:
        fail("Research-paper workflow must emit exactly two concise progress steps")

    if "previous_queries=_prior_user_queries(state)" not in web_source:
        fail("Web-search workflow does not preserve news mode across follow-up turns")
    if "strip_citations=True" not in web_source or "strip_citations=True" not in regulations_source:
        fail("Prior citation lists are still being fed into grounded follow-up prompts")
    for term in ("_resolved_news_topic", "strip_generated_sources", "format_answer", "Published "):
        if term not in web_service:
            fail(f"Web citation consistency implementation missing: {term}")
    for term in ("Accessed ", "accessed_at", "kr-kr", "_weather_source"):
        if term not in web_service:
            fail(f"Web live-evidence consistency implementation missing: {term}")

    for term in ("strip_generated_references", "_provision_detail", "format_answer"):
        if term not in gist:
            fail(f"Canonical GIST reference implementation missing: {term}")
    prompts = read("backend/app/workflows/prompts.py")
    if "Do not write a References section" not in prompts:
        fail("GIST answer prompt does not delegate reference formatting to the backend")
    if "Source numbers reset on every turn" not in prompts:
        fail("Web-search prompt does not reset citation scope on each turn")

    runtime_source = read("backend/app/runtime.py")
    for term in (
        "self.settings.pdf_document_context_chars",
        "request.has_document_attachment",
        "bool(request.document_context.strip())",
    ):
        if term not in runtime_source:
            fail(f"PDF runtime state protection is incomplete: {term}")
    if "user-uploaded PDF files" not in prompts:
        fail("PDF answer prompt does not enforce uploaded-evidence grounding")
    suggestion_questions = (
        "지도학습과 비지도학습의 차이를 쉽게 설명해줘.",
        "최근 공개된 주요 AI 모델은 무엇이고, 이전 모델과 비교해 무엇이 달라졌는지 최신 출처와 함께 알려줘.",
        "업로드한 PDF의 핵심 주장과 근거, 그리고 한계를 정리해줘.",
        "GIST 규정에 따르면 석사과정 학생의 졸업 요건은 무엇이야?",
    )
    suggestion_positions = []
    for question in suggestion_questions:
        if compose.count(question) != 2:
            fail(f"Landing-page question must appear exactly as title and click content: {question}")
        suggestion_positions.append(compose.index(question))
    if suggestion_positions != sorted(suggestion_positions):
        fail("Landing-page workflow questions are not in the required order")
    for subtitle in ("Auto", "Direct", "Web Search", "PDF Document · upload first", "GIST Regulations"):
        if f'"{subtitle}"' not in compose:
            fail(f"Missing landing-page workflow subtitle: {subtitle}")
    if '"title":["Research Paper"' in compose or '"title":["Research Paper Drafting"' in compose:
        fail("Research-paper workflow must not have a landing-page suggestion")
    for rel in ("infra/open-webui/Dockerfile", "infra/open-webui/patch-suggestions.mjs"):
        if not (ROOT / rel).is_file():
            fail(f"Missing deterministic Open WebUI suggestion build file: {rel}")
    for term in (
        "_internal_task_name",
        "_workflow_for_request",
        "_document_inputs",
        "uuid4() if task_name",
    ):
        if term not in compat:
            fail(f"Open WebUI internal-task isolation is incomplete: {term}")
    if "_pdf_request_with_history" not in pdf_document_source:
        fail("PDF workflow does not sanitize retrieval-query task artifacts")

    llm_source = read("backend/app/llm.py")
    if "hidden reasoning suppressed" in llm_source:
        fail("Hidden-reasoning status text still leaks into the user-facing response")

    print("STATIC_INVARIANTS_PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"STATIC_INVARIANTS_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
