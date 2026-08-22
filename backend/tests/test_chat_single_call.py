from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.workflows.builders import WorkflowServices, build_chat_subgraph


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return LLMResult(
            content="Paris.",
            requested_alias=kwargs["model_alias"],
            served_model="fake",
        )


@pytest.mark.asyncio
async def test_chat_subgraph_has_exactly_one_answer_call() -> None:
    llm = FakeLLM()
    services = WorkflowServices(
        llm=llm,
        rag=SimpleNamespace(),
        policy=StageModelPolicy(),
        settings=SimpleNamespace(),
    )
    graph = build_chat_subgraph(services)
    state = {
        "messages": [HumanMessage(content="What is the capital of France?")],
        "query": "What is the capital of France?",
        "run_id": "run-1",
        "user_id": "user-1",
        "team_id": "lab",
        "roles": ["member"],
        "workflow_id": "chat",
        "quality": "balanced",
        "recommended_tier": "cloud-small",
        "collection_ids": [],
        "use_documents": False,
        "call_events": [],
    }
    result = await graph.ainvoke(state)
    assert llm.calls == 1
    assert result["answer"] == "Paris."
    assert [event["alias"] for event in result["call_events"]] == ["cloud-small"]
