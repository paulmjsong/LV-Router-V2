from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.schemas import SourceCitation
from app.workflows.builders import WorkflowServices, build_regulations_subgraph


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(content="Grounded answer [SOURCE 1].", requested_alias=kwargs["model_alias"], served_model="fake")


class FakeRegulations:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, **kwargs):
        self.calls += 1
        return [SourceCitation(
            chunk_id=1,
            document_id=uuid4(),
            title="regulation.pdf",
            page=2,
            score=0.9,
            excerpt="Relevant regulation text.",
        )]

    @staticmethod
    def context_from_sources(sources):
        return "[SOURCE 1] regulation.pdf p.2\nRelevant regulation text."


@pytest.mark.asyncio
async def test_regulations_is_straight_retrieve_then_answer_rag() -> None:
    llm = FakeLLM()
    regulations = FakeRegulations()
    services = WorkflowServices(
        llm=llm,
        regulations=regulations,
        policy=StageModelPolicy(),
        settings=SimpleNamespace(embedding_model_alias="embedding"),
    )
    graph = build_regulations_subgraph(services)
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="What is the rule?")],
        "query": "What is the rule?",
        "run_id": "r1",
        "user_id": "u",
        "team_id": "lab",
        "roles": ["member"],
        "workflow_id": "gist-regulations",
        "quality": "balanced",
        "recommended_tier": "cloud-small",
        "sources": [],
        "call_events": [],
    })
    assert regulations.calls == 1
    assert [call["stage"] for call in llm.calls] == ["answer"]
    assert result["answer"].startswith("Grounded answer")
    assert len(result["sources"]) == 1
