from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway, LLMResult
from .schemas import ModelTier, Quality, WorkflowId


LOCAL_ROUTER_SYSTEM = """Route one request for a research-lab AI platform.

Workflows: direct, domain_rag, paper, grant, website.
Model tiers: local-fast, cloud-small, cloud-large.

Rules:
- Documents are required -> domain_rag and use_documents=true.
- Research-paper work -> paper.
- Grant/proposal work -> grant.
- Website or repository work -> website.
- Otherwise -> direct.
- Prefer local-fast when sufficient; use cloud-small for nontrivial work and cloud-large only for difficult synthesis or important review.
- Select only an allowed workflow.
- Do not answer the request. Do not explain the decision.

Return JSON only with exactly: workflow, model_tier, use_documents, confidence.
"""


class LocalRouteDecision(BaseModel):
    workflow: WorkflowId
    model_tier: ModelTier
    use_documents: bool = False
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_contract(self) -> "LocalRouteDecision":
        if self.workflow == WorkflowId.AUTO:
            raise ValueError("The local router cannot return workflow=auto")
        if self.workflow == WorkflowId.DOMAIN_RAG and not self.use_documents:
            raise ValueError("domain_rag requires use_documents=true")
        if self.workflow == WorkflowId.DIRECT and self.use_documents:
            raise ValueError("direct cannot use documents")
        return self


@dataclass(frozen=True, slots=True)
class RouterOutcome:
    decision: LocalRouteDecision
    llm_result: LLMResult
    reason: str
    used_fallback: bool = False


class LocalSemanticRouter:
    """Use one small local-model call for a compact routing decision only."""

    def __init__(self, llm: LLMGateway, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    @staticmethod
    def _history_text(history: Sequence[dict[str, str]], max_chars: int = 1200) -> str:
        blocks: list[str] = []
        used = 0
        for item in reversed(history):
            role = item.get("role", "user").upper()
            content = item.get("content", "").strip()
            if not content:
                continue
            block = f"{role}: {content}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block) + 1
        blocks.reverse()
        return "\n".join(blocks)

    @staticmethod
    def _fallback() -> LocalRouteDecision:
        # An invalid router output must not silently create a paid cloud call.
        return LocalRouteDecision(
            workflow=WorkflowId.DIRECT,
            model_tier=ModelTier.LOCAL_FAST,
            use_documents=False,
            confidence=0.0,
        )

    async def decide(
        self,
        *,
        query: str,
        history: Sequence[dict[str, str]],
        collection_count: int,
        allowed_workflows: Sequence[WorkflowId],
        user: UserContext,
        run_id: str,
    ) -> RouterOutcome:
        allowed = ",".join(item.value for item in allowed_workflows)
        prior = self._history_text(history) or "none"
        user_prompt = (
            f"allowed={allowed}\n"
            f"selected_document_collections={collection_count}\n"
            f"history={prior}\n"
            f"request={query}\n"
            "/no_think"
        )
        result = await self.llm.chat(
            user=user,
            model_alias=self.settings.local_router_model_alias,
            messages=[
                {"role": "system", "content": LOCAL_ROUTER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            run_id=run_id,
            workflow_id=WorkflowId.AUTO.value,
            stage="route",
            temperature=0.0,
            max_tokens=self.settings.local_router_max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            decision = LocalRouteDecision.model_validate(
                self.llm.extract_json_object(result.content)
            )
        except (ValueError, ValidationError) as exc:
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=f"Invalid local routing output ({type(exc).__name__}); used local direct fallback.",
                used_fallback=True,
            )

        if decision.workflow not in set(allowed_workflows):
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason="The local router selected a disallowed workflow; used local direct fallback.",
                used_fallback=True,
            )

        if decision.confidence < self.settings.local_router_min_confidence:
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason="The local router was uncertain; used local direct fallback.",
                used_fallback=True,
            )

        reason = (
            f"Local router selected {decision.workflow.value} with "
            f"{decision.model_tier.value} (confidence {decision.confidence:.2f})."
        )
        return RouterOutcome(decision, result, reason=reason, used_fallback=False)


class StageModelPolicy:
    """Map a semantic router recommendation and explicit quality setting to aliases.

    This policy never examines query text. The semantic local router chooses the base tier;
    the workflow stage and explicit user quality setting only cap or promote that tier.
    LiteLLM then chooses the concrete provider deployment inside the alias.
    """

    _ORDER = {
        ModelTier.LOCAL_FAST: 0,
        ModelTier.CLOUD_SMALL: 1,
        ModelTier.CLOUD_LARGE: 2,
    }
    _BY_ORDER = {value: key for key, value in _ORDER.items()}

    @classmethod
    def explicit_tier(cls, quality: Quality) -> ModelTier:
        if quality == Quality.FAST:
            return ModelTier.LOCAL_FAST
        if quality == Quality.HIGH:
            return ModelTier.CLOUD_LARGE
        return ModelTier.CLOUD_SMALL

    @classmethod
    def _quality_adjusted(cls, tier: ModelTier, quality: Quality) -> ModelTier:
        level = cls._ORDER[tier]
        if quality == Quality.FAST:
            level = min(level, cls._ORDER[ModelTier.CLOUD_SMALL])
        elif quality == Quality.HIGH:
            level = max(level, cls._ORDER[ModelTier.CLOUD_SMALL])
        return cls._BY_ORDER[level]

    def stage_alias(
        self,
        *,
        recommended_tier: ModelTier,
        quality: Quality,
        stage: str,
    ) -> str:
        tier = self._quality_adjusted(recommended_tier, quality)

        # Planning/extraction is deliberately one tier cheaper than a large-model draft.
        if stage in {"outline", "requirements"} and tier == ModelTier.CLOUD_LARGE:
            tier = ModelTier.CLOUD_SMALL

        # A high-quality final synthesis is the only automatic stage promotion.
        if stage == "final" and quality == Quality.HIGH:
            tier = ModelTier.CLOUD_LARGE

        return tier.value
