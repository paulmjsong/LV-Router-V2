from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.workflows.builders import WorkflowServices, build_direct_subgraph


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(content="Paris.", requested_alias=kwargs["model_alias"], served_model="fake")


@pytest.mark.asyncio
async def test_direct_subgraph_is_one_answer_call() -> None:
    llm = FakeLLM()
    services = WorkflowServices(
        llm=llm,
        regulations=SimpleNamespace(),
        policy=StageModelPolicy(),
        settings=SimpleNamespace(),
    )
    graph = build_direct_subgraph(services)
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Capital of France?")],
        "query": "Capital of France?",
        "run_id": "r1",
        "user_id": "u",
        "team_id": "lab",
        "roles": ["member"],
        "workflow_id": "direct",
        "quality": "balanced",
        "recommended_tier": "cloud-small",
        "call_events": [],
    })
    assert result["answer"] == "Paris."
    assert [call["stage"] for call in llm.calls] == ["answer"]
