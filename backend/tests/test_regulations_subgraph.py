from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.gist_regulations import GISTRegulationsRetriever
from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.schemas import SourceCitation
from app.workflows.builders import WorkflowServices, build_regulations_subgraph


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []
        self.controls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(
            content=(
                "Graduate students receive academic probation below the stated GPA "
                "threshold [SOURCE 1].\n\n"
                "📌 References\n- old model-generated reference"
            ),
            requested_alias=kwargs["model_alias"],
            served_model="fake",
        )

    async def emit_control(self, **kwargs):
        self.controls.append(kwargs)


class FakeRegulations:
    def __init__(self) -> None:
        self.calls = 0
        self.formatter = object.__new__(GISTRegulationsRetriever)

    async def retrieve(self, **kwargs):
        self.calls += 1
        return [SourceCitation(
            chunk_id=1,
            document_id=uuid4(),
            title="FR00401 광주과학기술원 학칙",
            page=14,
            score=0.9,
            excerpt=(
                "제52조(학사경고) ① 학사경고를 한다. "
                "② 수강학점수를 제한할 수 있다. "
                "③ 일정 횟수의 학사경고를 받은 자를 제적한다. "
                "④ 지도교수에게 통지한다."
            ),
            url=(
                "/static/gist-regulations/"
                "FR00401%20%EA%B4%91%EC%A3%BC%EA%B3%BC%ED%95%99%EA%B8%B0%EC%88%A0%EC%9B%90%20%ED%95%99%EC%B9%99.pdf#page=14"
            ),
        )]

    def context_from_sources(self, sources):
        return self.formatter.context_from_sources(sources)

    def references_markdown(self, sources, *, answer=""):
        return self.formatter.references_markdown(sources, answer=answer)

    def format_answer(self, answer, sources):
        return self.formatter.format_answer(answer, sources)


@pytest.mark.asyncio
async def test_regulations_is_straight_rag_with_canonical_references() -> None:
    llm = FakeLLM()
    regulations = FakeRegulations()
    services = WorkflowServices(
        llm=llm,
        regulations=regulations,
        web_search=SimpleNamespace(),
        policy=StageModelPolicy(),
        settings=SimpleNamespace(embedding_model_alias="embedding"),
    )
    graph = build_regulations_subgraph(services)
    result = await graph.ainvoke({
        "messages": [
            HumanMessage(content="What is the rule?"),
            AIMessage(content="Old answer.\n\n### 📌 References\n- stale reference"),
            HumanMessage(content="And what happens after a warning?"),
        ],
        "query": "And what happens after a warning?",
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
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "stale reference" not in prompt
    assert result["answer"].count("### 📌 References") == 1
    assert "old model-generated reference" not in result["answer"]
    assert "제52조(학사경고) 제1항–제4항" in result["answer"]
    assert "/static/gist-regulations/" in result["answer"]
    assert [item["event_type"] for item in llm.controls] == [
        "workflow-step",
        "sources",
    ]
    assert "Checking GIST regulations" in llm.controls[0]["content"]


def test_gist_reference_formatter_is_deduplicated_and_compact() -> None:
    formatter = object.__new__(GISTRegulationsRetriever)
    url = "/static/gist-regulations/FR00401.pdf#page=14"
    source = SourceCitation(
        chunk_id=1,
        document_id=uuid4(),
        title="FR00401 광주과학기술원 학칙",
        page=14,
        score=0.9,
        excerpt="제52조(학사경고) ① 내용 ② 내용 ③ 내용 ④ 내용",
        url=url,
    )
    duplicate = source.model_copy(update={"chunk_id": 2, "document_id": uuid4()})

    rendered = formatter.references_markdown([source, duplicate])

    assert rendered.count("FR00401 광주과학기술원 학칙") == 1
    assert "제52조(학사경고) 제1항–제4항" in rendered
    assert "① 내용" not in rendered
