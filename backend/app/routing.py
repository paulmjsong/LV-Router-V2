from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator

from .auth import UserContext
from .config import Settings
from .llm import LLMGateway, LLMResult
from .schemas import ModelTier, Quality, WorkflowId


LOCAL_ROUTER_SYSTEM = """You are the local router for a research-lab AI platform.

Choose exactly one action:

- answer: Only for a simple general request that can be answered correctly from the conversation alone in 40 words or fewer.
- delegate: Use when the request needs documents, tools, persistent state, website/repository access, a paper/grant workflow, a longer answer, or stronger reasoning.

Workflows:
- direct: general inference
- domain_rag: answer using authorized documents
- paper: research-paper work
- grant: grant/proposal work
- website: website/repository work

Model tiers:
- local-fast: easy local inference
- cloud-small: routine stronger reasoning/drafting
- cloud-large: difficult reasoning or important final review

Rules:
- Never claim access to documents, websites, repositories, or databases unless delegated to the appropriate workflow.
- Document-dependent requests → domain_rag with use_documents=true.
- Paper work → paper.
- Grant work → grant.
- Website/repository changes → website.
- Longer but easy general requests → direct + local-fast.
- Difficult general requests → direct + cloud-small or cloud-large.
- Prefer local-fast when sufficient.

If action=answer:
workflow=direct, model_tier=local-fast, use_documents=false.

If action=delegate:
answer must be empty.

Return only valid JSON with:
action, workflow, model_tier, use_documents, confidence, reason, answer.

Keep reason under 10 words. Do not output reasoning or Markdown.
"""


class LocalRouteDecision(BaseModel):
    action: Literal["answer", "delegate"]
    workflow: WorkflowId
    model_tier: ModelTier
    use_documents: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=1000)
    answer: str = Field(default="", max_length=50000)

    @model_validator(mode="after")
    def validate_contract(self) -> "LocalRouteDecision":
        if self.workflow == WorkflowId.AUTO:
            raise ValueError("The local router cannot return workflow=auto")
        if self.action == "answer":
            if self.workflow != WorkflowId.DIRECT:
                raise ValueError("A local answer must use workflow=direct")
            if self.model_tier != ModelTier.LOCAL_FAST:
                raise ValueError("A local answer must use model_tier=local-fast")
            if self.use_documents:
                raise ValueError("A local answer cannot claim document use")
            if not self.answer.strip():
                raise ValueError("A local answer must include answer text")
        else:
            if self.answer.strip():
                raise ValueError("A delegated decision must leave answer empty")
            if self.workflow == WorkflowId.DOMAIN_RAG and not self.use_documents:
                raise ValueError("domain_rag requires use_documents=true")
            if self.workflow == WorkflowId.DIRECT and self.model_tier == ModelTier.LOCAL_FAST:
                raise ValueError("Delegated direct inference must use a cloud tier")
        return self


@dataclass(frozen=True, slots=True)
class RouterOutcome:
    decision: LocalRouteDecision
    llm_result: LLMResult
    used_fallback: bool = False


class LocalSemanticRouter:
    """One local-model call that either returns the final answer or delegates.

    The call still goes through LiteLLM, so it is budgeted, logged, and auditable. The
    application does not inspect query length or keyword counts.
    """

    def __init__(self, llm: LLMGateway, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    @staticmethod
    def _history_text(history: Sequence[dict[str, str]], max_chars: int = 2000) -> str:
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
            used += len(block) + 2
        blocks.reverse()
        return "\n\n".join(blocks)

    @staticmethod
    def _fallback(reason: str) -> LocalRouteDecision:
        return LocalRouteDecision(
            action="delegate",
            workflow=WorkflowId.DIRECT,
            model_tier=ModelTier.CLOUD_SMALL,
            use_documents=False,
            confidence=0.0,
            reason=reason,
            answer="",
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
        allowed = ", ".join(item.value for item in allowed_workflows)
        prior = self._history_text(history) or "None"
        user_prompt = (
            f"ALLOWED WORKFLOWS: {allowed}\n"
            f"AUTHORIZED DOCUMENT COLLECTIONS SELECTED: {collection_count}\n\n"
            f"PRIOR CONVERSATION:\n{prior}\n\n"
            f"CURRENT USER REQUEST:\n{query}\n\n"
            "/no_think\n"
            "Return the required JSON immediately. "
            "For action=answer, keep the answer concise."
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
            stage="answer_or_delegate",
            temperature=0.0,
            max_tokens=self.settings.local_router_max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            decision = LocalRouteDecision.model_validate(
                self.llm.extract_json_object(result.content)
            )
        except (ValueError, ValidationError) as exc:
            fallback = self._fallback(
                f"The local router returned an invalid structured decision; "
                f"falling back to direct cloud inference ({type(exc).__name__})."
            )
            return RouterOutcome(fallback, result, used_fallback=True)

        allowed_set = set(allowed_workflows)
        if decision.workflow not in allowed_set:
            fallback = self._fallback(
                "The local router selected a workflow that this user is not allowed to run; "
                "falling back to direct cloud inference."
            )
            return RouterOutcome(fallback, result, used_fallback=True)

        if decision.action == "answer" and decision.confidence < self.settings.local_answer_min_confidence:
            fallback = self._fallback(
                "The local model was not confident enough to return its own answer; "
                "delegating to direct cloud inference."
            )
            return RouterOutcome(fallback, result, used_fallback=True)

        return RouterOutcome(decision, result, used_fallback=False)


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
