import json
from types import SimpleNamespace

import pytest

from app.auth import UserContext
from app.llm import LLMResult
from app.routing import LocalSemanticRouter, RouteDifficulty
from app.schemas import ModelTier, WorkflowId


def workflow_member(*names: str) -> WorkflowId:
    for name in names:
        member = getattr(WorkflowId, name, None)
        if member is not None:
            return member
    raise AssertionError(f"Missing WorkflowId member from {names}")


DIRECT = workflow_member("DIRECT", "CHAT")
REGULATIONS = workflow_member("REGULATIONS")
PAPER = workflow_member("PAPER")


class FakeRouterLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.kwargs = None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return LLMResult(
            content=json.dumps(self.payload),
            requested_alias=kwargs["model_alias"],
            served_model="router-test",
        )

    @staticmethod
    def extract_json_object(content: str):
        return json.loads(content)


def settings():
    return SimpleNamespace(
        local_router_model_alias="local-router",
        local_router_max_tokens=64,
        local_router_min_confidence=0.55,
        local_router_fallback_tier="cloud-small",
    )


@pytest.mark.asyncio
async def test_router_schema_has_no_self_reported_confidence() -> None:
    llm = FakeRouterLLM({"workflow": DIRECT.value, "difficulty": "simple"})
    outcome = await LocalSemanticRouter(llm, settings()).decide(
        query="What is gradient descent?",
        history=[],
        allowed_workflows=[DIRECT, REGULATIONS, PAPER],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="schema-test",
    )
    assert outcome.used_fallback is False
    assert outcome.decision.workflow == DIRECT
    assert outcome.decision.difficulty == RouteDifficulty.SIMPLE
    assert outcome.decision.model_tier == ModelTier.LOCAL_FAST
    schema = llm.kwargs["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == {"workflow", "difficulty"}
    assert "confidence" not in schema["properties"]


@pytest.mark.asyncio
async def test_legacy_zero_confidence_cannot_force_cloud_fallback() -> None:
    llm = FakeRouterLLM({
        "workflow": DIRECT.value,
        "difficulty": "simple",
        "confidence": 0.0,
    })
    outcome = await LocalSemanticRouter(llm, settings()).decide(
        query="What is gradient descent?",
        history=[],
        allowed_workflows=[DIRECT, REGULATIONS, PAPER],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="zero-confidence-test",
    )
    assert outcome.used_fallback is False
    assert outcome.decision.model_tier == ModelTier.LOCAL_FAST


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_workflow", "expected_difficulty", "expected_tier"),
    [
        ({"route": "chat", "complexity": "easy"}, DIRECT, RouteDifficulty.SIMPLE, ModelTier.LOCAL_FAST),
        ({"workflow": "gist_regulations", "difficulty": "medium"}, REGULATIONS, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL),
        ({"workflow": "paper", "level": "hard"}, PAPER, RouteDifficulty.ADVANCED, ModelTier.CLOUD_LARGE),
    ],
)
async def test_router_repairs_common_small_model_variants(
    payload,
    expected_workflow,
    expected_difficulty,
    expected_tier,
) -> None:
    outcome = await LocalSemanticRouter(FakeRouterLLM(payload), settings()).decide(
        query="Representative request",
        history=[],
        allowed_workflows=[DIRECT, REGULATIONS, PAPER],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="repair-test",
    )
    assert outcome.used_fallback is False
    assert outcome.decision.workflow == expected_workflow
    assert outcome.decision.difficulty == expected_difficulty
    assert outcome.decision.model_tier == expected_tier
