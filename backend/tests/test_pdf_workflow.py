from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.schemas import WorkflowId
from app.workflows.builders import WorkflowServices, build_pdf_document_subgraph
from app.workflows.parent import build_parent_graph


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []
        self.controls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(
            content="The paper reports a controlled evaluation [1].",
            requested_alias=kwargs["model_alias"],
            served_model="fake",
        )

    async def emit_control(self, **kwargs):
        self.controls.append(kwargs)


class ExplodingSemanticRouter:
    async def decide(self, **kwargs):
        raise AssertionError("attachment routing must bypass the semantic router")


def services(llm: FakeLLM) -> WorkflowServices:
    return WorkflowServices(
        llm=llm,
        regulations=SimpleNamespace(),
        web_search=SimpleNamespace(),
        policy=StageModelPolicy(),
        settings=SimpleNamespace(local_router_model_alias="local-router"),
    )


def pdf_state() -> dict:
    return {
        "messages": [HumanMessage(content="What did the paper evaluate?")],
        "query": "What did the paper evaluate?",
        "document_context": (
            '<source id="1" name="paper.pdf">'
            "A controlled evaluation was conducted."
            "</source>"
        ),
        "has_document_attachment": True,
        "run_id": "r-pdf-1",
        "conversation_id": "c-pdf-1",
        "user_id": "u",
        "team_id": "lab",
        "roles": ["member"],
        "requested_workflow": WorkflowId.AUTO.value,
        "allowed_workflows": [
            WorkflowId.DIRECT.value,
            WorkflowId.PDF.value,
            WorkflowId.REGULATIONS.value,
            WorkflowId.WEB_SEARCH.value,
            WorkflowId.PAPER.value,
        ],
        "quality": "balanced",
        "workflow_id": "",
        "recommended_tier": "",
        "call_events": [],
    }


@pytest.mark.asyncio
async def test_pdf_subgraph_answers_only_after_context_is_supplied() -> None:
    llm = FakeLLM()
    graph = build_pdf_document_subgraph(services(llm))
    state = pdf_state()
    state["workflow_id"] = WorkflowId.PDF.value
    state["recommended_tier"] = "cloud-small"
    result = await graph.ainvoke(state)

    assert result["answer"].endswith("[1].")
    assert [call["stage"] for call in llm.calls] == ["answer"]
    assert "UPLOADED PDF SOURCE BLOCKS" in llm.calls[0]["messages"][1]["content"]
    assert len(llm.controls) == 1


@pytest.mark.asyncio
async def test_auto_pdf_attachment_bypasses_semantic_router() -> None:
    llm = FakeLLM()
    graph = build_parent_graph(
        services=services(llm),
        semantic_router=ExplodingSemanticRouter(),
        checkpointer=None,
    )
    result = await graph.ainvoke(pdf_state())

    assert result["workflow_id"] == WorkflowId.PDF.value
    assert result["route_confidence"] == 1.0
    assert "PDF attachment was detected" in result["route_reason"]
    assert [call["stage"] for call in llm.calls] == ["answer"]


@pytest.mark.asyncio
async def test_pdf_subgraph_refuses_to_guess_without_retrieved_context() -> None:
    llm = FakeLLM()
    graph = build_pdf_document_subgraph(services(llm))
    state = pdf_state()
    state["workflow_id"] = WorkflowId.PDF.value
    state["recommended_tier"] = "cloud-small"
    state["document_context"] = ""
    result = await graph.ainvoke(state)

    assert "No usable PDF text" in result["answer"]
    assert llm.calls == []
