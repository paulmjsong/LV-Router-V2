from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.schemas import SourceCitation
from app.workflows.builders import (
    WorkflowServices,
    build_pdf_subgraph,
    build_regulations_subgraph,
    build_workflow_subgraphs,
)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(
            content=f"answer via {kwargs['model_alias']}",
            requested_alias=kwargs["model_alias"],
            served_model="fake",
        )


class FakeRAG:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SourceCitation(
                chunk_id=1,
                document_id=uuid4(),
                title="source.pdf",
                page=1,
                score=0.9,
                excerpt="Relevant evidence.",
            )
        ]

    @staticmethod
    def context_from_sources(sources):
        return "[SOURCE 1: source.pdf, page 1]\nRelevant evidence." if sources else ""


def _services():
    llm = FakeLLM()
    rag = FakeRAG()
    settings = SimpleNamespace(
        pdf_attachment_context_chars=30000,
        gist_regulations_system_key="gist-regulations",
    )
    return WorkflowServices(
        llm=llm,
        rag=rag,
        policy=StageModelPolicy(),
        settings=settings,
    ), llm, rag


def _common(workflow: str):
    return {
        "messages": [HumanMessage(content="What does it say?")],
        "query": "What does it say?",
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "team_id": "lab",
        "roles": ["member"],
        "workflow_id": workflow,
        "quality": "balanced",
        "recommended_tier": "cloud-small",
        "collection_ids": [],
        "attachment_context": "",
        "has_pdf_attachment": False,
        "use_documents": workflow in {"pdf", "regulations"},
        "call_events": [],
        "sources": [],
    }


@pytest.mark.asyncio
async def test_pdf_subgraph_queries_only_user_pdf_collections() -> None:
    services, llm, rag = _services()
    graph = build_pdf_subgraph(services)
    state = _common("pdf")
    state["collection_ids"] = [str(uuid4())]
    result = await graph.ainvoke(state)

    assert result["answer"] == "answer via cloud-small"
    assert len(rag.calls) == 1
    assert rag.calls[0]["mime_types"] == {"application/pdf", "application/x-pdf"}
    assert rag.calls[0]["exclude_system_collections"] is True
    assert not rag.calls[0].get("system_keys")
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_regulations_subgraph_queries_only_reserved_system_collection() -> None:
    services, llm, rag = _services()
    graph = build_regulations_subgraph(services)
    result = await graph.ainvoke(_common("regulations"))

    assert result["answer"] == "answer via cloud-small"
    assert len(rag.calls) == 1
    assert rag.calls[0]["collection_ids"] == []
    assert rag.calls[0]["system_keys"] == {"gist-regulations"}
    assert len(llm.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow", "stage"),
    [("paper", "draft"), ("grant", "draft"), ("website", "proposal")],
)
async def test_placeholder_subgraphs_are_single_stage_non_agentic_calls(
    workflow: str,
    stage: str,
) -> None:
    services, llm, _ = _services()
    subgraphs = build_workflow_subgraphs(services)
    graph = getattr(subgraphs, workflow)
    result = await graph.ainvoke(_common(workflow))

    assert len(llm.calls) == 1
    assert llm.calls[0]["stage"] == stage
    assert result["answer"] == "answer via cloud-small"
