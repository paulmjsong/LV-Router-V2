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


def function(tree: ast.AST, class_name: str | None, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
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
        (item for item in parents if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name),
        None,
    )
    if found is None:
        fail(f"Missing function {class_name + '.' if class_name else ''}{name}")
    return found


def main() -> None:
    routing = ast.parse((ROOT / "backend/app/routing.py").read_text())
    decide = function(routing, "LocalSemanticRouter", "decide")
    router_calls = [
        node for node in ast.walk(decide)
        if isinstance(node, ast.Call) and call_name(node).endswith("self.llm.chat")
    ]
    if len(router_calls) != 1:
        fail(f"LocalSemanticRouter.decide must contain exactly one LLM call, found {len(router_calls)}")

    builders = ast.parse((ROOT / "backend/app/workflows/builders.py").read_text())
    direct = function(builders, None, "build_direct_graph")
    direct_calls = [
        node for node in ast.walk(direct)
        if isinstance(node, ast.Call) and call_name(node).endswith("services.llm.chat")
    ]
    if len(direct_calls) != 1:
        fail(f"Direct graph must contain exactly one downstream LLM call, found {len(direct_calls)}")

    source = "\n".join(
        path.read_text(errors="replace")
        for path in (ROOT / "backend").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    forbidden = ["gradio", "FAISS", "ChatOpenAI", "OpenAIEmbeddings"]
    present = [term for term in forbidden if term in source]
    if present:
        fail(f"Legacy/provider-bypass terms remain: {present}")

    compose = (ROOT / "docker-compose.yml").read_text()
    required = [
        'ENABLE_TITLE_GENERATION: "false"',
        'ENABLE_TAGS_GENERATION: "false"',
        "FORWARD_USER_INFO_HEADER_JWT_SECRET",
        "OPENAI_API_BASE_URL: http://backend:8000/v1",
    ]
    missing = [term for term in required if term not in compose]
    if missing:
        fail(f"Open WebUI hardening/integration settings missing: {missing}")

    print("STATIC_INVARIANTS_PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"STATIC_INVARIANTS_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
