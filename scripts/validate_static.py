#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    routing = ast.parse((ROOT / "backend/app/routing.py").read_text(encoding="utf-8"))
    decide = function(routing, "LocalSemanticRouter", "decide")
    router_calls = [
        node
        for node in ast.walk(decide)
        if isinstance(node, ast.Call) and call_name(node).endswith("self.llm.chat")
    ]
    if len(router_calls) != 1:
        fail(f"LocalSemanticRouter.decide must contain exactly one LLM call, found {len(router_calls)}")

    builders_text = (ROOT / "backend/app/workflows/builders.py").read_text(encoding="utf-8")
    builders = ast.parse(builders_text)
    chat = function(builders, None, "build_chat_subgraph")
    chat_calls = [
        node
        for node in ast.walk(chat)
        if isinstance(node, ast.Call) and call_name(node).endswith("services.llm.chat")
    ]
    if len(chat_calls) != 1:
        fail(f"Chat subgraph must contain exactly one answer call, found {len(chat_calls)}")

    parent = (ROOT / "backend/app/workflows/parent.py").read_text(encoding="utf-8")
    required_subgraphs = [
        "subgraphs.chat",
        "subgraphs.pdf",
        "subgraphs.regulations",
        "subgraphs.paper",
        "subgraphs.grant",
        "subgraphs.website",
    ]
    missing_subgraphs = [term for term in required_subgraphs if term not in parent]
    if missing_subgraphs:
        fail(f"Parent graph is missing workflow subgraphs: {missing_subgraphs}")
    if "build_parent_graph" not in (ROOT / "backend/app/runtime.py").read_text(encoding="utf-8"):
        fail("WorkflowRuntime does not compile the parent graph")

    source = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "backend").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    forbidden = [
        "gradio",
        "FAISS",
        "ChatOpenAI",
        "OpenAIEmbeddings",
        "WorkflowId.DIRECT",
        "WorkflowId.DOMAIN_RAG",
        "build_direct_graph",
        "build_rag_graph",
        "GitHubPublisher",
    ]
    present = [term for term in forbidden if term in source]
    if present:
        fail(f"Legacy or bypass terms remain: {present}")

    whole_repo = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and not any(
            part.startswith((".patch", ".infonet-update-backup-", ".waiting-fix", ".startup-gate"))
            for part in path.parts
        )
        and path.suffix not in {".zip", ".pyc"}
        and path != ROOT / "scripts/validate_static.py"
    )
    branding_forbidden = ["SaeGyeol Lab AI", "lab-auto", "lab-direct", "lab-rag"]
    branding_present = [term for term in branding_forbidden if term in whole_repo]
    if branding_present:
        fail(f"Old branding or workflow IDs remain: {branding_present}")

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    required = [
        'WEBUI_NAME: ${WEBUI_NAME:-Infonet AI Router}',
        '"auto","chat","pdf","regulations","paper","grant","website"',
        'DEFAULT_MODELS: "auto"',
        'ENABLE_NOTES: "false"',
        'ENABLE_CALENDAR: "false"',
        'ENABLE_AUTOMATIONS: "false"',
        'ENABLE_SUBAGENTS: "false"',
        'ENABLE_FOLLOW_UP_GENERATION: "false"',
        'ENABLE_AUTOCOMPLETE_GENERATION: "false"',
        "OPENAI_API_BASE_URL: http://backend:8000/v1",
        "LOCAL_ROUTER_FALLBACK_TIER",
        "LOCAL_FAST_MIN_CONFIDENCE",
    ]
    missing = [term for term in required if term not in compose]
    if missing:
        fail(f"Open WebUI/routing settings missing: {missing}")

    main_source = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    if "ensure_system_collection" not in main_source or "/api/regulations/upload" not in main_source:
        fail("Reserved GIST regulations collection/ingestion path is missing")

    dead_placeholder_interfaces = [
        "RunDecisionRequest",
        "PendingAction",
        "claim_for_resume",
        "/api/runs/{run_id}/decision",
        "async def resume",
    ]
    app_source = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "backend/app").rglob("*.py")
    )
    dead_present = [term for term in dead_placeholder_interfaces if term in app_source]
    if dead_present:
        fail(f"Dead approval interfaces remain after placeholder simplification: {dead_present}")

    print("STATIC_INVARIANTS_PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"STATIC_INVARIANTS_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
