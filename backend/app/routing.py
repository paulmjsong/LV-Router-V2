from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway, LLMResult
from .schemas import ModelTier, Quality, WorkflowId

logger = logging.getLogger("infonet.routing")

LOCAL_ROUTER_SYSTEM = """Classify one request for Infonet AI Router. Do not answer it.
Active workflows:
- direct: general questions, explanation, rewriting, translation, coding, calculation, planning
- gist-regulations: questions about GIST rules, degree requirements, graduation, academic or institutional regulations
- research-paper: drafting, rewriting, structuring, or reviewing research-paper text
Disabled workflows:
- grant and website are not available yet; never select them
Difficulty:
- simple: one-step, low-risk, short answer; a small local model is sufficient
- standard: coding/debugging, comparisons, plans, multi-part reasoning, drafting, or document synthesis
- advanced: research-grade synthesis, difficult reasoning, or important final review
Rules:
- GIST regulations or institutional-policy questions -> gist-regulations.
- Drafting/revising research-paper content -> research-paper.
- Grant or website requests -> direct/standard because those specialist workflows are disabled.
- Otherwise -> direct.
- research-paper and gist-regulations are never simple.
- Select only an allowed workflow.
Examples:
"What is overfitting?" -> direct/simple
"Debug this traceback and propose a robust fix" -> direct/standard
"Compare three methods and design a publication-grade experiment" -> direct/advanced
"GIST master's graduation requirements" -> gist-regulations/standard
"Rewrite my abstract" -> research-paper/standard
Return JSON only with exactly: workflow, difficulty, confidence.
"""


class RouteDifficulty(StrEnum):
    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"


class RouterClassification(BaseModel):
    workflow: WorkflowId
    difficulty: RouteDifficulty
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_contract(self) -> "RouterClassification":
        if self.workflow in {WorkflowId.AUTO, WorkflowId.GRANT, WorkflowId.WEBSITE}:
            raise ValueError("The router returned a non-routable workflow")
        return self


class LocalRouteDecision(BaseModel):
    workflow: WorkflowId
    model_tier: ModelTier
    use_documents: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    difficulty: RouteDifficulty = RouteDifficulty.STANDARD

    @model_validator(mode="after")
    def validate_contract(self) -> "LocalRouteDecision":
        if self.workflow in {WorkflowId.AUTO, WorkflowId.GRANT, WorkflowId.WEBSITE}:
            raise ValueError("The resolved route is not active")
        if self.workflow == WorkflowId.REGULATIONS and not self.use_documents:
            raise ValueError("gist-regulations requires retrieval")
        if self.workflow == WorkflowId.DIRECT and self.use_documents:
            raise ValueError("direct cannot use documents")
        if self.workflow != WorkflowId.DIRECT and self.model_tier == ModelTier.LOCAL_FAST:
            raise ValueError("Specialist workflows cannot resolve to local-fast")
        return self


@dataclass(frozen=True, slots=True)
class RouterOutcome:
    decision: LocalRouteDecision
    llm_result: LLMResult
    reason: str
    used_fallback: bool = False


class LocalSemanticRouter:
    _DIFFICULTY_TO_TIER = {
        RouteDifficulty.SIMPLE: ModelTier.LOCAL_FAST,
        RouteDifficulty.STANDARD: ModelTier.CLOUD_SMALL,
        RouteDifficulty.ADVANCED: ModelTier.CLOUD_LARGE,
    }

    def __init__(self, llm: LLMGateway, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    @staticmethod
    def _history_text(history: Sequence[dict[str, str]], max_chars: int = 900) -> str:
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

    def _fallback(self) -> LocalRouteDecision:
        tier = ModelTier(self.settings.local_router_fallback_tier)
        difficulty = RouteDifficulty.STANDARD if tier == ModelTier.CLOUD_SMALL else RouteDifficulty.ADVANCED
        return LocalRouteDecision(
            workflow=WorkflowId.DIRECT,
            model_tier=tier,
            use_documents=False,
            confidence=0.0,
            difficulty=difficulty,
        )

    @classmethod
    def _normalize(cls, classification: RouterClassification) -> LocalRouteDecision:
        difficulty = classification.difficulty
        if classification.workflow != WorkflowId.DIRECT and difficulty == RouteDifficulty.SIMPLE:
            difficulty = RouteDifficulty.STANDARD
        return LocalRouteDecision(
            workflow=classification.workflow,
            model_tier=cls._DIFFICULTY_TO_TIER[difficulty],
            use_documents=classification.workflow == WorkflowId.REGULATIONS,
            confidence=classification.confidence,
            difficulty=difficulty,
        )

    async def decide(
        self,
        *,
        query: str,
        history: Sequence[dict[str, str]],
        allowed_workflows: Sequence[WorkflowId],
        user: UserContext,
        run_id: str,
    ) -> RouterOutcome:
        allowed = ",".join(item.value for item in allowed_workflows)
        prior = self._history_text(history) or "none"
        result = await self.llm.chat(
            user=user,
            model_alias=self.settings.local_router_model_alias,
            messages=[
                {"role": "system", "content": LOCAL_ROUTER_SYSTEM},
                {
                    "role": "user",
                    "content": f"allowed={allowed}\nhistory={prior}\nrequest={query}\n/no_think",
                },
            ],
            run_id=run_id,
            workflow_id=WorkflowId.AUTO.value,
            stage="route",
            temperature=0.0,
            max_tokens=self.settings.local_router_max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            classification = RouterClassification.model_validate(
                self.llm.extract_json_object(result.content)
            )
            decision = self._normalize(classification)
        except (ValueError, ValidationError) as exc:
            logger.warning("Router output invalid; using visible fallback: %s", exc)
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=f"Router output invalid ({type(exc).__name__}); used visible cloud fallback.",
                used_fallback=True,
            )
        if decision.workflow not in set(allowed_workflows):
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason="Router selected a disabled/disallowed workflow; used visible cloud fallback.",
                used_fallback=True,
            )
        if decision.confidence < self.settings.local_router_min_confidence:
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=(
                    f"Router confidence {decision.confidence:.2f} below "
                    f"{self.settings.local_router_min_confidence:.2f}; used visible cloud fallback."
                ),
                used_fallback=True,
            )
        return RouterOutcome(
            decision=decision,
            llm_result=result,
            reason=(
                f"Local router selected {decision.workflow.value}/{decision.difficulty.value} "
                f"as {decision.model_tier.value} (confidence {decision.confidence:.2f})."
            ),
            used_fallback=False,
        )


class StageModelPolicy:
    _ORDER = {
        ModelTier.LOCAL_FAST: 0,
        ModelTier.CLOUD_SMALL: 1,
        ModelTier.CLOUD_LARGE: 2,
    }
    _BY_ORDER = {value: key for key, value in _ORDER.items()}
    _SPECIALIST = {WorkflowId.REGULATIONS, WorkflowId.PAPER}

    @classmethod
    def explicit_tier(cls, quality: Quality) -> ModelTier:
        if quality == Quality.FAST:
            return ModelTier.LOCAL_FAST
        if quality == Quality.HIGH:
            return ModelTier.CLOUD_LARGE
        return ModelTier.CLOUD_SMALL

    @classmethod
    def resolve_tier(
        cls,
        *,
        workflow: WorkflowId,
        recommended_tier: ModelTier,
        quality: Quality,
    ) -> ModelTier:
        if quality == Quality.FAST:
            return ModelTier.LOCAL_FAST
        if quality == Quality.HIGH:
            return ModelTier.CLOUD_LARGE
        if workflow in cls._SPECIALIST:
            return cls._BY_ORDER[max(cls._ORDER[recommended_tier], 1)]
        return recommended_tier

    def stage_alias(
        self,
        *,
        workflow: WorkflowId,
        recommended_tier: ModelTier,
        quality: Quality,
        stage: str,
    ) -> str:
        tier = self.resolve_tier(
            workflow=workflow,
            recommended_tier=recommended_tier,
            quality=quality,
        )
        # Keep paper control/validation agents cheap even when the final draft is advanced.
        if workflow == WorkflowId.PAPER and stage in {
            "orchestrator",
            "content_agent",
            "structure_agent",
            "validator",
        }:
            if tier == ModelTier.CLOUD_LARGE:
                tier = ModelTier.CLOUD_SMALL
        return tier.value
