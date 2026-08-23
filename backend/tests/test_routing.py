import json
from types import SimpleNamespace

import pytest

from app.auth import UserContext
from app.llm import LLMResult
from app.routing import LocalSemanticRouter, RouteDifficulty, StageModelPolicy
from app.schemas import ModelTier, Quality, WorkflowId


class FakeRouterLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return LLMResult(
            content=json.dumps(self.payload),
            requested_alias=kwargs["model_alias"],
            served_model="router-test",
        )

    @staticmethod
    def extract_json_object(content: str):
        return json.loads(content)


def settings(min_confidence: float = 0.55):
    return SimpleNamespace(
        local_router_model_alias="local-router",
        local_router_max_tokens=64,
        local_router_min_confidence=min_confidence,
        local_router_fallback_tier="cloud-small",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow", "difficulty", "tier", "uses_docs"),
    [
        (WorkflowId.DIRECT, RouteDifficulty.SIMPLE, ModelTier.LOCAL_FAST, False),
        (WorkflowId.DIRECT, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL, False),
        (WorkflowId.DIRECT, RouteDifficulty.ADVANCED, ModelTier.CLOUD_LARGE, False),
        # Specialists are never accepted as simple/local-fast.
        (WorkflowId.REGULATIONS, RouteDifficulty.SIMPLE, ModelTier.CLOUD_SMALL, True),
        (WorkflowId.PAPER, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL, False),
        (WorkflowId.PAPER, RouteDifficulty.ADVANCED, ModelTier.CLOUD_LARGE, False),
    ],
)
async def test_router_maps_supported_outcomes(workflow, difficulty, tier, uses_docs) -> None:
    llm = FakeRouterLLM({
        "workflow": workflow.value,
        "difficulty": difficulty.value,
        "confidence": 0.96,
    })
    router = LocalSemanticRouter(llm, settings())
    outcome = await router.decide(
        query="Representative request",
        history=[],
        allowed_workflows=[WorkflowId.DIRECT, WorkflowId.REGULATIONS, WorkflowId.PAPER],
        user=UserContext(user_id="u", team_id="lab", roles={"member"}),
        run_id="run-1",
    )
    assert llm.calls == 1
    assert outcome.used_fallback is False
    assert outcome.decision.workflow == workflow
    assert outcome.decision.model_tier == tier
    assert outcome.decision.use_documents is uses_docs


@pytest.mark.asyncio
async def test_invalid_or_low_confidence_router_never_falls_back_to_local_fast() -> None:
    llm = FakeRouterLLM({
        "workflow": "direct",
        "difficulty": "simple",
        "confidence": 0.10,
    })
    outcome = await LocalSemanticRouter(llm, settings()).decide(
        query="Ambiguous",
        history=[],
        allowed_workflows=[WorkflowId.DIRECT, WorkflowId.REGULATIONS, WorkflowId.PAPER],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="run-2",
    )
    assert outcome.used_fallback is True
    assert outcome.decision.workflow == WorkflowId.DIRECT
    assert outcome.decision.model_tier == ModelTier.CLOUD_SMALL


def test_balanced_specialists_have_cloud_small_floor() -> None:
    policy = StageModelPolicy()
    for workflow in (WorkflowId.REGULATIONS, WorkflowId.PAPER):
        assert policy.stage_alias(
            workflow=workflow,
            recommended_tier=ModelTier.LOCAL_FAST,
            quality=Quality.BALANCED,
            stage="answer",
        ) == "cloud-small"
