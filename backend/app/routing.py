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

Workflows:
- chat: general questions, explanation, rewriting, translation, coding, calculation
- pdf: answer from an attached or selected PDF
- regulations: GIST rules, degree requirements, graduation, academic policy, institutional regulations
- paper: drafting or revising a research paper
- grant: drafting or managing a grant proposal
- website: planning or drafting website changes

Difficulty:
- simple: one-step, low-risk, short answer; a small local model is sufficient
- standard: coding/debugging, comparisons, plans, multi-part reasoning, drafting, or document synthesis
- advanced: research-grade synthesis, difficult reasoning, or important final review

Rules:
- PDF evidence means pdf. GIST 규정, 학위, 졸업, 학사 policy, or institutional rules mean regulations.
- Ordinary research questions are chat unless the user is drafting or revising a paper.
- Ordinary funding questions are chat unless the user is drafting or managing a proposal.
- Specialist workflows are never simple.
- Select only an allowed workflow.

Examples:
"What is overfitting?" -> chat/simple
"Debug this traceback and propose a robust fix" -> chat/standard
"Compare three methods and design a publication-grade experiment" -> chat/advanced
"Summarize the attached PDF" -> pdf/standard
"GIST master's graduation requirements" -> regulations/standard
"Rewrite my abstract" -> paper/standard

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
        if self.workflow == WorkflowId.AUTO:
            raise ValueError("The router cannot return workflow=auto")
        return self


class LocalRouteDecision(BaseModel):
    """Normalized application route derived from the local semantic classification."""

    workflow: WorkflowId
    model_tier: ModelTier
    use_documents: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    difficulty: RouteDifficulty = RouteDifficulty.STANDARD

    @model_validator(mode="after")
    def validate_contract(self) -> "LocalRouteDecision":
        if self.workflow == WorkflowId.AUTO:
            raise ValueError("The router cannot return workflow=auto")
        if self.workflow in {WorkflowId.PDF, WorkflowId.REGULATIONS} and not self.use_documents:
            raise ValueError(f"{self.workflow.value} requires use_documents=true")
        if self.workflow == WorkflowId.CHAT and self.use_documents:
            raise ValueError("chat cannot use documents")
        if self.workflow != WorkflowId.CHAT and self.model_tier == ModelTier.LOCAL_FAST:
            raise ValueError("Specialist workflows cannot resolve to local-fast")
        return self


@dataclass(frozen=True, slots=True)
class RouterOutcome:
    decision: LocalRouteDecision
    llm_result: LLMResult
    reason: str
    used_fallback: bool = False


class LocalSemanticRouter:
    """Use one small local-model call for a compact semantic classification."""

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
        difficulty = {
            ModelTier.LOCAL_FAST: RouteDifficulty.SIMPLE,
            ModelTier.CLOUD_SMALL: RouteDifficulty.STANDARD,
            ModelTier.CLOUD_LARGE: RouteDifficulty.ADVANCED,
        }[tier]
        return LocalRouteDecision(
            workflow=WorkflowId.CHAT,
            model_tier=tier,
            use_documents=False,
            confidence=0.0,
            difficulty=difficulty,
        )

    @classmethod
    def _normalize(cls, classification: RouterClassification) -> LocalRouteDecision:
        difficulty = classification.difficulty
        # The local model only decides semantic difficulty. The application enforces
        # that every specialist workflow has at least the standard/cloud-small tier.
        if classification.workflow != WorkflowId.CHAT and difficulty == RouteDifficulty.SIMPLE:
            difficulty = RouteDifficulty.STANDARD
        return LocalRouteDecision(
            workflow=classification.workflow,
            model_tier=cls._DIFFICULTY_TO_TIER[difficulty],
            use_documents=classification.workflow in {WorkflowId.PDF, WorkflowId.REGULATIONS},
            confidence=classification.confidence,
            difficulty=difficulty,
        )

    async def decide(
        self,
        *,
        query: str,
        history: Sequence[dict[str, str]],
        collection_count: int,
        has_pdf_attachment: bool,
        allowed_workflows: Sequence[WorkflowId],
        user: UserContext,
        run_id: str,
    ) -> RouterOutcome:
        allowed = ",".join(item.value for item in allowed_workflows)
        prior = self._history_text(history) or "none"
        user_prompt = (
            f"allowed={allowed}\n"
            f"selected_pdf_collections={collection_count}\n"
            f"pdf_attached={str(has_pdf_attachment).lower()}\n"
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
            classification = RouterClassification.model_validate(
                self.llm.extract_json_object(result.content)
            )
            decision = self._normalize(classification)
        except (ValueError, ValidationError) as exc:
            logger.warning(
                "Router output invalid; applying %s fallback: %s",
                self.settings.local_router_fallback_tier,
                exc,
            )
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=(
                    f"Router output was invalid ({type(exc).__name__}); "
                    f"used visible {self.settings.local_router_fallback_tier} chat fallback."
                ),
                used_fallback=True,
            )
        if decision.workflow not in set(allowed_workflows):
            logger.warning(
                "Router selected disallowed workflow=%s; applying %s fallback",
                decision.workflow.value,
                self.settings.local_router_fallback_tier,
            )
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=(
                    "Router selected a disallowed workflow; "
                    f"used visible {self.settings.local_router_fallback_tier} chat fallback."
                ),
                used_fallback=True,
            )
        if decision.confidence < self.settings.local_router_min_confidence:
            logger.warning(
                "Router confidence %.2f below %.2f; applying %s fallback",
                decision.confidence,
                self.settings.local_router_min_confidence,
                self.settings.local_router_fallback_tier,
            )
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=(
                    f"Router confidence {decision.confidence:.2f} was below "
                    f"{self.settings.local_router_min_confidence:.2f}; used visible "
                    f"{self.settings.local_router_fallback_tier} chat fallback."
                ),
                used_fallback=True,
            )
        logger.info(
            "Router decision workflow=%s difficulty=%s tier=%s documents=%s confidence=%.2f",
            decision.workflow.value,
            decision.difficulty.value,
            decision.model_tier.value,
            decision.use_documents,
            decision.confidence,
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
    """Convert workflow, quality, and router recommendation to a LiteLLM alias."""

    _ORDER = {
        ModelTier.LOCAL_FAST: 0,
        ModelTier.CLOUD_SMALL: 1,
        ModelTier.CLOUD_LARGE: 2,
    }
    _BY_ORDER = {value: key for key, value in _ORDER.items()}
    _SPECIALIST_WORKFLOWS = {
        WorkflowId.PDF,
        WorkflowId.REGULATIONS,
        WorkflowId.PAPER,
        WorkflowId.GRANT,
        WorkflowId.WEBSITE,
    }

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
        if workflow in cls._SPECIALIST_WORKFLOWS:
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
        del stage  # Placeholder workflows currently have one visible model stage.
        return self.resolve_tier(
            workflow=workflow,
            recommended_tier=recommended_tier,
            quality=quality,
        ).value
