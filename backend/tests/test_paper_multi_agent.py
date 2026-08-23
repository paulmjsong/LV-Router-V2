import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.workflows.builders import WorkflowServices, build_paper_subgraph


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        stage = kwargs["stage"]
        if stage == "orchestrator":
            content = json.dumps({
                "objective": "draft abstract",
                "target_section": "abstract",
                "content_tasks": ["state problem", "state contribution"],
                "structure_tasks": ["motivation before method"],
                "constraints": ["no invented results"],
            })
        elif stage == "validator":
            content = json.dumps({
                "status": "revise",
                "issues": ["missing evidence placeholder"],
                "revision_instructions": "Add [RESULT NEEDED].",
            })
        elif stage == "final":
            content = "Final validated abstract with [RESULT NEEDED]."
        else:
            content = f"{stage} output"
        return LLMResult(content=content, requested_alias=kwargs["model_alias"], served_model="fake")

    @staticmethod
    def extract_json_object(content: str):
        return json.loads(content)


@pytest.mark.asyncio
async def test_paper_graph_runs_orchestrator_parallel_subagents_validator_and_finalizer() -> None:
    llm = FakeLLM()
    services = WorkflowServices(
        llm=llm,
        regulations=SimpleNamespace(),
        policy=StageModelPolicy(),
        settings=SimpleNamespace(),
    )
    graph = build_paper_subgraph(services)
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Draft my abstract")],
        "query": "Draft my abstract",
        "run_id": "r1",
        "user_id": "u",
        "team_id": "lab",
        "roles": ["member"],
        "workflow_id": "research-paper",
        "quality": "balanced",
        "recommended_tier": "cloud-small",
        "paper_agent_outputs": [],
        "call_events": [],
    })
    stages = [call["stage"] for call in llm.calls]
    assert stages[0] == "orchestrator"
    assert set(stages[1:3]) == {"content_agent", "structure_agent"}
    assert stages[3:] == ["draft", "validator", "final"]
    assert result["answer"] == "Final validated abstract with [RESULT NEEDED]."
    assert {item["agent"] for item in result["paper_agent_outputs"]} == {"content", "structure"}
