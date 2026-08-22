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
        self.kwargs = None

    async def chat(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return LLMResult(
            content=json.dumps(self.payload),
            requested_alias=kwargs["model_alias"],
            served_model="local-test-model",
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
    ("workflow", "difficulty", "expected_tier", "use_documents"),
    [
        (WorkflowId.CHAT, RouteDifficulty.SIMPLE, ModelTier.LOCAL_FAST, False),
        (WorkflowId.CHAT, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL, False),
        (WorkflowId.CHAT, RouteDifficulty.ADVANCED, ModelTier.CLOUD_LARGE, False),
        # A specialist/simple classification is normalized upward rather than
        # silently creating another local-fast outcome.
        (WorkflowId.PDF, RouteDifficulty.SIMPLE, ModelTier.CLOUD_SMALL, True),
        (WorkflowId.REGULATIONS, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL, True),
        (WorkflowId.PAPER, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL, False),
        (WorkflowId.GRANT, RouteDifficulty.ADVANCED, ModelTier.CLOUD_LARGE, False),
        (WorkflowId.WEBSITE, RouteDifficulty.STANDARD, ModelTier.CLOUD_SMALL, False),
    ],
)
async def test_router_accepts_and_normalizes_every_supported_outcome(
    workflow: WorkflowId,
    difficulty: RouteDifficulty,
    expected_tier: ModelTier,
    use_documents: bool,
) -> None:
    llm = FakeRouterLLM(
        {
            "workflow": workflow.value,
            "difficulty": difficulty.value,
            "confidence": 0.95,
        }
    )
    router = LocalSemanticRouter(llm, settings())
    outcome = await router.decide(
        query="Representative request",
        history=[],
        collection_count=1 if workflow == WorkflowId.PDF else 0,
        has_pdf_attachment=workflow == WorkflowId.PDF,
        allowed_workflows=list(WorkflowId)[1:],
        user=UserContext(user_id="u", team_id="lab", roles={"member"}),
        run_id="run-1",
    )
    assert llm.calls == 1
    assert outcome.used_fallback is False
    assert outcome.decision.workflow == workflow
    assert outcome.decision.model_tier == expected_tier
    assert outcome.decision.use_documents is use_documents


@pytest.mark.asyncio
async def test_low_confidence_fallback_is_visible_cloud_small_not_local_fast() -> None:
    llm = FakeRouterLLM(
        {
            "workflow": "chat",
            "difficulty": "simple",
            "confidence": 0.10,
        }
    )
    router = LocalSemanticRouter(llm, settings())
    outcome = await router.decide(
        query="Ambiguous request",
        history=[],
        collection_count=0,
        has_pdf_attachment=False,
        allowed_workflows=[WorkflowId.CHAT],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="run-2",
    )
    assert outcome.used_fallback is True
    assert outcome.decision.workflow == WorkflowId.CHAT
    assert outcome.decision.model_tier == ModelTier.CLOUD_SMALL
    assert "fallback" in outcome.reason.lower()


@pytest.mark.asyncio
async def test_old_model_tier_schema_is_rejected_instead_of_defaulting_local() -> None:
    llm = FakeRouterLLM(
        {
            "workflow": "chat",
            "model_tier": "local-fast",
            "use_documents": False,
            "confidence": 0.99,
        }
    )
    router = LocalSemanticRouter(llm, settings())
    outcome = await router.decide(
        query="Ambiguous request",
        history=[],
        collection_count=0,
        has_pdf_attachment=False,
        allowed_workflows=[WorkflowId.CHAT],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="run-3",
    )
    assert outcome.used_fallback is True
    assert outcome.decision.model_tier == ModelTier.CLOUD_SMALL


def test_stage_policy_prevents_specialist_workflows_from_collapsing_to_local_fast() -> None:
    policy = StageModelPolicy()
    for workflow in {
        WorkflowId.PDF,
        WorkflowId.REGULATIONS,
        WorkflowId.PAPER,
        WorkflowId.GRANT,
        WorkflowId.WEBSITE,
    }:
        assert policy.stage_alias(
            workflow=workflow,
            recommended_tier=ModelTier.LOCAL_FAST,
            quality=Quality.BALANCED,
            stage="answer",
        ) == "cloud-small"

    assert policy.stage_alias(
        workflow=WorkflowId.CHAT,
        recommended_tier=ModelTier.LOCAL_FAST,
        quality=Quality.BALANCED,
        stage="answer",
    ) == "local-fast"
    assert policy.stage_alias(
        workflow=WorkflowId.PAPER,
        recommended_tier=ModelTier.CLOUD_SMALL,
        quality=Quality.FAST,
        stage="draft",
    ) == "local-fast"
    assert policy.stage_alias(
        workflow=WorkflowId.CHAT,
        recommended_tier=ModelTier.LOCAL_FAST,
        quality=Quality.HIGH,
        stage="answer",
    ) == "cloud-large"
