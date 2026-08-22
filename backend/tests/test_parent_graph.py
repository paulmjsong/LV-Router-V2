import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.auth import UserContext
from app.llm import LLMResult
from app.routing import LocalSemanticRouter, StageModelPolicy
from app.workflows.builders import WorkflowServices
from app.workflows.parent import build_parent_graph


class FakeLLM:
    def __init__(self, route_payload: dict) -> None:
        self.route_payload = route_payload
        self.calls: list[dict] = []
        self.controls: list[str] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stage"] == "route":
            content = json.dumps(self.route_payload)
        else:
            content = f"answer from {kwargs['workflow_id']} via {kwargs['model_alias']}"
        return LLMResult(
            content=content,
            requested_alias=kwargs["model_alias"],
            served_model="fake-model",
        )

    async def emit_control(self, **kwargs):
        self.controls.append(kwargs["content"])

    @staticmethod
    def extract_json_object(content: str):
        return json.loads(content)


class FakeRAG:
    async def retrieve(self, **kwargs):
        return []

    @staticmethod
    def context_from_sources(sources):
        return ""


def settings():
    return SimpleNamespace(
        local_router_model_alias="local-router",
        local_router_max_tokens=64,
        local_router_min_confidence=0.55,
        local_router_fallback_tier="cloud-small",
        local_fast_min_confidence=0.80,
        pdf_attachment_context_chars=30000,
        gist_regulations_system_key="gist-regulations",
    )


def initial_state(**overrides):
    state = {
        "messages": [HumanMessage(content="Draft an abstract")],
        "query": "Draft an abstract",
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "team_id": "lab",
        "roles": ["member"],
        "requested_workflow": "auto",
        "allowed_workflows": ["chat", "pdf", "regulations", "paper", "grant", "website"],
        "quality": "balanced",
        "collection_ids": [],
        "attachment_context": "",
        "has_pdf_attachment": False,
        "workflow_id": "",
        "recommended_tier": "",
        "use_documents": False,
        "route_reason": "",
        "route_fallback": False,
        "route_confidence": 0.0,
        "route_difficulty": "",
        "router_served_model": "",
        "answer": "",
        "sources": [],
        "call_events": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_parent_graph_routes_to_isolated_paper_subgraph() -> None:
    cfg = settings()
    llm = FakeLLM(
        {
            "workflow": "paper",
            "difficulty": "simple",
            "confidence": 0.95,
        }
    )
    services = WorkflowServices(
        llm=llm,
        rag=FakeRAG(),
        policy=StageModelPolicy(),
        settings=cfg,
    )
    graph = build_parent_graph(
        services=services,
        semantic_router=LocalSemanticRouter(llm, cfg),
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        initial_state(),
        config={"configurable": {"thread_id": "thread-paper"}},
    )

    assert result["workflow_id"] == "paper"
    # Balanced specialist workflows have a cloud-small floor.
    assert result["recommended_tier"] == "cloud-small"
    assert result["route_difficulty"] == "standard"
    assert "answer from paper via cloud-small" in result["answer"]
    assert [call["stage"] for call in llm.calls] == ["route", "draft"]
    assert llm.controls and "paper" in llm.controls[0]


@pytest.mark.asyncio
async def test_pdf_attachment_hard_routes_without_router_call() -> None:
    cfg = settings()
    llm = FakeLLM(
        {
            "workflow": "chat",
            "difficulty": "simple",
            "confidence": 0.99,
        }
    )
    services = WorkflowServices(
        llm=llm,
        rag=FakeRAG(),
        policy=StageModelPolicy(),
        settings=cfg,
    )
    graph = build_parent_graph(
        services=services,
        semantic_router=LocalSemanticRouter(llm, cfg),
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        initial_state(
            query="Summarize it",
            messages=[HumanMessage(content="Summarize it")],
            has_pdf_attachment=True,
            attachment_context="The PDF says hello.",
        ),
        config={"configurable": {"thread_id": "thread-pdf"}},
    )

    assert result["workflow_id"] == "pdf"
    assert result["route_difficulty"] == "attachment"
    assert [call["stage"] for call in llm.calls] == ["answer"]
    assert "answer from pdf via cloud-small" in result["answer"]
