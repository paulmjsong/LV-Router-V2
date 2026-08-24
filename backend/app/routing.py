from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway, LLMResult
from .schemas import ModelTier, Quality, WorkflowId

logger = logging.getLogger("infonet.routing")

LOCAL_ROUTER_SYSTEM = """Classify one request for Infonet AI Router. Do not answer it.

The user message supplies allowed_workflows. Select exactly one value from that list.

Intent rules:
- General questions, explanations, rewriting, translation, coding, calculations, and planning use the direct workflow.
- GIST rules, graduation, degree requirements, academic policy, and institutional regulations use the GIST regulations workflow.
- Drafting, revising, structuring, or reviewing research-paper text uses the research-paper workflow.

Difficulty values:
- simple: one-step, low-risk, short answer; the local model is sufficient.
- standard: debugging, comparison, planning, multi-part reasoning, drafting, or synthesis.
- advanced: difficult research-grade synthesis, demanding reasoning, or important final review.

Specialist workflows are never simple.

Return exactly one JSON object with exactly these required keys:
workflow, difficulty.

Do not return confidence, explanation, Markdown, or any extra key.
"""


class RouteDifficulty(StrEnum):
    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"


class RouterClassification(BaseModel):
    workflow: WorkflowId
    difficulty: RouteDifficulty

    @model_validator(mode="after")
    def validate_contract(self) -> "RouterClassification":
        if self.workflow == WorkflowId.AUTO:
            raise ValueError("The router cannot return workflow=auto")
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
    """Use one small local-model call for compact semantic classification.

    The model selects workflow and difficulty. It does not self-report confidence;
    generative confidence numbers are not calibrated and must not control fallback.
    """

    _DIFFICULTY_TO_TIER = {
        RouteDifficulty.SIMPLE: ModelTier.LOCAL_FAST,
        RouteDifficulty.STANDARD: ModelTier.CLOUD_SMALL,
        RouteDifficulty.ADVANCED: ModelTier.CLOUD_LARGE,
    }

    def __init__(self, llm: LLMGateway, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    @staticmethod
    def _workflow_member(*member_names: str) -> WorkflowId:
        for member_name in member_names:
            member = getattr(WorkflowId, member_name, None)
            if member is not None:
                return member
        raise RuntimeError(f"WorkflowId is missing all expected members: {member_names}")

    @classmethod
    def _direct_workflow(cls) -> WorkflowId:
        return cls._workflow_member("DIRECT", "CHAT")

    @classmethod
    def _regulations_workflow(cls) -> WorkflowId:
        return cls._workflow_member("REGULATIONS")

    @classmethod
    def _paper_workflow(cls) -> WorkflowId:
        return cls._workflow_member("PAPER")

    @classmethod
    def _active_workflows(cls) -> tuple[WorkflowId, ...]:
        return (
            cls._direct_workflow(),
            cls._regulations_workflow(),
            cls._paper_workflow(),
        )

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

    @staticmethod
    def _token(value: Any) -> str:
        token = str(value).strip().casefold().replace("_", "-")
        return "-".join(token.split())

    @classmethod
    def _workflow_aliases(cls) -> dict[str, str]:
        direct = cls._direct_workflow().value
        regulations = cls._regulations_workflow().value
        paper = cls._paper_workflow().value
        return {
            "direct": direct,
            "chat": direct,
            "general": direct,
            "general-chat": direct,
            "direct-inference": direct,
            "gist-regulations": regulations,
            "gist-regulation": regulations,
            "regulations": regulations,
            "regulation": regulations,
            "gist-rules": regulations,
            "jireumgil": regulations,
            "research-paper": paper,
            "research-paper-drafting": paper,
            "paper": paper,
            "paper-drafting": paper,
        }

    _DIFFICULTY_ALIASES = {
        "simple": RouteDifficulty.SIMPLE.value,
        "easy": RouteDifficulty.SIMPLE.value,
        "basic": RouteDifficulty.SIMPLE.value,
        "low": RouteDifficulty.SIMPLE.value,
        "standard": RouteDifficulty.STANDARD.value,
        "medium": RouteDifficulty.STANDARD.value,
        "moderate": RouteDifficulty.STANDARD.value,
        "normal": RouteDifficulty.STANDARD.value,
        "advanced": RouteDifficulty.ADVANCED.value,
        "hard": RouteDifficulty.ADVANCED.value,
        "complex": RouteDifficulty.ADVANCED.value,
        "high": RouteDifficulty.ADVANCED.value,
    }

    @classmethod
    def _response_format(
        cls,
        allowed_workflows: Sequence[WorkflowId],
    ) -> dict[str, Any]:
        active = set(cls._active_workflows())
        workflow_values = [item.value for item in allowed_workflows if item in active]
        if not workflow_values:
            workflow_values = [cls._direct_workflow().value]
        workflow_values = list(dict.fromkeys(workflow_values))
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "infonet_router_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "workflow": {
                            "type": "string",
                            "enum": workflow_values,
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": [item.value for item in RouteDifficulty],
                        },
                    },
                    "required": ["workflow", "difficulty"],
                    "additionalProperties": False,
                },
            },
        }

    @classmethod
    def _normalize_payload(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("router response must be a JSON object")
        workflow_raw = payload.get("workflow", payload.get("route", payload.get("intent")))
        difficulty_raw = payload.get(
            "difficulty",
            payload.get("complexity", payload.get("level")),
        )
        if workflow_raw is None or difficulty_raw is None:
            raise ValueError("router response is missing workflow or difficulty")
        workflow_token = cls._token(workflow_raw)
        difficulty_token = cls._token(difficulty_raw)
        return {
            "workflow": cls._workflow_aliases().get(workflow_token, workflow_token),
            "difficulty": cls._DIFFICULTY_ALIASES.get(difficulty_token, difficulty_token),
        }

    @staticmethod
    def _validation_summary(exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            errors = exc.errors()
            if errors:
                first = errors[0]
                location = ".".join(str(item) for item in first.get("loc", ())) or "response"
                return f"{location}: {first.get('msg', 'invalid value')}"
        return str(exc) or type(exc).__name__

    def _fallback(self) -> LocalRouteDecision:
        tier = ModelTier(self.settings.local_router_fallback_tier)
        difficulty = {
            ModelTier.LOCAL_FAST: RouteDifficulty.SIMPLE,
            ModelTier.CLOUD_SMALL: RouteDifficulty.STANDARD,
            ModelTier.CLOUD_LARGE: RouteDifficulty.ADVANCED,
        }[tier]
        return LocalRouteDecision(
            workflow=self._direct_workflow(),
            model_tier=tier,
            use_documents=False,
            confidence=0.0,
            difficulty=difficulty,
        )

    @classmethod
    def _normalize(cls, classification: RouterClassification) -> LocalRouteDecision:
        difficulty = classification.difficulty
        if classification.workflow != cls._direct_workflow() and difficulty == RouteDifficulty.SIMPLE:
            difficulty = RouteDifficulty.STANDARD
        return LocalRouteDecision(
            workflow=classification.workflow,
            model_tier=cls._DIFFICULTY_TO_TIER[difficulty],
            use_documents=classification.workflow.value in {
                "pdf",
                "regulations",
                "gist-regulations",
            },
            # Retained only to preserve the existing state schema. It means
            # schema-valid, not calibrated model confidence.
            confidence=1.0,
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
        collection_count: int = 0,
        has_pdf_attachment: bool = False,
    ) -> RouterOutcome:
        allowed = ",".join(item.value for item in allowed_workflows)
        prior = self._history_text(history) or "none"
        user_prompt = (
            f"allowed_workflows={allowed}\n"
            f"selected_collections={collection_count}\n"
            f"file_attached={str(has_pdf_attachment).lower()}\n"
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
            response_format=self._response_format(allowed_workflows),
        )

        parsed: Any = None
        normalized: dict[str, Any] | None = None
        try:
            parsed = self.llm.extract_json_object(result.content)
            normalized = self._normalize_payload(parsed)
            classification = RouterClassification.model_validate(normalized)
            decision = self._normalize(classification)
        except (TypeError, ValueError, ValidationError) as exc:
            summary = self._validation_summary(exc)
            logger.warning(
                "Router output invalid; applying %s fallback. raw=%r parsed=%r normalized=%r error=%s",
                self.settings.local_router_fallback_tier,
                result.content[:500],
                parsed,
                normalized,
                summary,
            )
            return RouterOutcome(
                decision=self._fallback(),
                llm_result=result,
                reason=(
                    f"Router output invalid ({summary}); used visible "
                    f"{self.settings.local_router_fallback_tier} fallback."
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
                    "Router selected a disabled/disallowed workflow; used visible "
                    f"{self.settings.local_router_fallback_tier} fallback."
                ),
                used_fallback=True,
            )

        logger.info(
            "Router decision workflow=%s difficulty=%s tier=%s documents=%s",
            decision.workflow.value,
            decision.difficulty.value,
            decision.model_tier.value,
            decision.use_documents,
        )
        return RouterOutcome(
            decision=decision,
            llm_result=result,
            reason=(
                f"Validated local router selected {decision.workflow.value}/"
                f"{decision.difficulty.value} as {decision.model_tier.value}."
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
