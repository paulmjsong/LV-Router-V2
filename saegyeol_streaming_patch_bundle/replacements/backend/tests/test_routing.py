import json
from types import SimpleNamespace

import pytest

from app.auth import UserContext
from app.llm import LLMResult
from app.routing import LocalSemanticRouter, StageModelPolicy
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
    )


@pytest.mark.asyncio
async def test_local_router_returns_compact_decision_only() -> None:
    llm = FakeRouterLLM(
        {
            "workflow": "direct",
            "model_tier": "local-fast",
            "use_documents": False,
            "confidence": 0.95,
        }
    )
    router = LocalSemanticRouter(llm, settings())
    outcome = await router.decide(
        query="What is the capital of France?",
        history=[],
        collection_count=0,
        allowed_workflows=[WorkflowId.DIRECT, WorkflowId.DOMAIN_RAG],
        user=UserContext(user_id="u", team_id="lab", roles={"member"}),
        run_id="run-1",
    )
    assert llm.calls == 1
    assert outcome.decision.workflow == WorkflowId.DIRECT
    assert outcome.decision.model_tier == ModelTier.LOCAL_FAST
    assert llm.kwargs["model_alias"] == "local-router"
    assert llm.kwargs["max_tokens"] == 64


@pytest.mark.asyncio
async def test_low_confidence_decision_falls_back_to_local_direct() -> None:
    llm = FakeRouterLLM(
        {
            "workflow": "direct",
            "model_tier": "local-fast",
            "use_documents": False,
            "confidence": 0.40,
        }
    )
    router = LocalSemanticRouter(llm, settings())
    outcome = await router.decide(
        query="Difficult question",
        history=[],
        collection_count=0,
        allowed_workflows=[WorkflowId.DIRECT],
        user=UserContext(user_id="u", roles={"member"}),
        run_id="run-2",
    )
    assert outcome.used_fallback is True
    assert outcome.decision.workflow == WorkflowId.DIRECT
    assert outcome.decision.model_tier == ModelTier.LOCAL_FAST


def test_stage_policy_uses_router_tier_not_query_scoring() -> None:
    policy = StageModelPolicy()
    assert policy.stage_alias(
        recommended_tier=ModelTier.CLOUD_LARGE,
        quality=Quality.BALANCED,
        stage="outline",
    ) == "cloud-small"
    assert policy.stage_alias(
        recommended_tier=ModelTier.CLOUD_LARGE,
        quality=Quality.BALANCED,
        stage="draft",
    ) == "cloud-large"
    assert policy.stage_alias(
        recommended_tier=ModelTier.LOCAL_FAST,
        quality=Quality.FAST,
        stage="answer",
    ) == "local-fast"
