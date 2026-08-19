from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from .auth import UserContext
from .config import Settings
from .llm import LLMStreamChunk
from .repositories import ConversationRepository, RunRepository
from .routing import LocalSemanticRouter, StageModelPolicy
from .schemas import (
    ChatRequest,
    ChatResponse,
    ModelTier,
    PendingAction,
    Quality,
    SourceCitation,
    WorkflowId,
    WorkflowInfo,
)
from .workflows.builders import (
    WorkflowServices,
    build_direct_graph,
    build_grant_graph,
    build_paper_graph,
    build_rag_graph,
    build_website_graph,
)


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    id: WorkflowId
    name: str
    description: str
    allowed_roles: frozenset[str]
    mutating: bool = False


WORKFLOW_SPECS: dict[WorkflowId, WorkflowSpec] = {
    WorkflowId.DIRECT: WorkflowSpec(
        WorkflowId.DIRECT,
        "Direct inference",
        "One model call for a general request selected by the local router or the user.",
        frozenset({"member", "editor", "admin"}),
    ),
    WorkflowId.DOMAIN_RAG: WorkflowSpec(
        WorkflowId.DOMAIN_RAG,
        "Domain RAG",
        "Hybrid retrieval over authorized lab collections followed by one answer call.",
        frozenset({"member", "editor", "admin"}),
    ),
    WorkflowId.PAPER: WorkflowSpec(
        WorkflowId.PAPER,
        "Research paper",
        "Evidence-aware outlining, drafting, and optional review.",
        frozenset({"member", "editor", "admin"}),
    ),
    WorkflowId.GRANT: WorkflowSpec(
        WorkflowId.GRANT,
        "Grant proposal",
        "Requirements extraction, drafting, and optional compliance review.",
        frozenset({"member", "editor", "admin"}),
    ),
    WorkflowId.WEBSITE: WorkflowSpec(
        WorkflowId.WEBSITE,
        "Website management",
        "Propose a repository change and require approval before opening a pull request.",
        frozenset({"editor", "admin"}),
        mutating=True,
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    workflow: WorkflowId
    recommended_tier: ModelTier
    use_documents: bool
    reason: str
    router_alias: str | None = None


TokenSink = Callable[[LLMStreamChunk], Awaitable[None]]
RouteSink = Callable[[ResolvedRoute, str], Awaitable[None]]


class WorkflowRuntime:
    def __init__(
        self,
        *,
        services: WorkflowServices,
        checkpointer: Any,
        runs: RunRepository,
        conversations: ConversationRepository,
        settings: Settings,
    ) -> None:
        self.services = services
        self.semantic_router = LocalSemanticRouter(services.llm, settings)
        self.policy: StageModelPolicy = services.policy
        self.runs = runs
        self.conversations = conversations
        self.settings = settings
        self.graphs = {
            WorkflowId.DIRECT: build_direct_graph(services, checkpointer),
            WorkflowId.DOMAIN_RAG: build_rag_graph(services, checkpointer),
            WorkflowId.PAPER: build_paper_graph(services, checkpointer),
            WorkflowId.GRANT: build_grant_graph(services, checkpointer),
            WorkflowId.WEBSITE: build_website_graph(services, checkpointer),
        }

    def list_workflows(self, user: UserContext) -> list[WorkflowInfo]:
        return [
            WorkflowInfo(
                id=spec.id,
                name=spec.name,
                description=spec.description,
                allowed_roles=sorted(spec.allowed_roles),
                mutating=spec.mutating,
            )
            for spec in WORKFLOW_SPECS.values()
            if user.has_any_role(set(spec.allowed_roles))
        ]

    def _allowed_workflow_ids(self, user: UserContext) -> list[WorkflowId]:
        return [
            spec.id
            for spec in WORKFLOW_SPECS.values()
            if user.has_any_role(set(spec.allowed_roles))
        ]

    @staticmethod
    def _authorize(workflow: WorkflowId, user: UserContext) -> None:
        spec = WORKFLOW_SPECS[workflow]
        if not user.has_any_role(set(spec.allowed_roles)):
            raise HTTPException(status_code=403, detail=f"Role not permitted for workflow: {workflow.value}")

    @staticmethod
    def _thread_id(user: UserContext, run_id: UUID, workflow: WorkflowId) -> str:
        raw = f"{user.user_id}:{run_id}:{workflow.value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _clip_history_text(value: str, max_chars: int) -> str:
        value = value.strip()
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        marker = "\n...[older content truncated]...\n"
        if max_chars <= len(marker):
            return value[:max_chars]
        remaining = max_chars - len(marker)
        head = remaining // 2
        tail = remaining - head
        return f"{value[:head]}{marker}{value[-tail:]}"

    def _bounded_history_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        remaining = self.settings.conversation_history_chars
        for row in reversed(rows):
            if remaining <= 0:
                break
            content = self._clip_history_text(row["content"], remaining)
            if not content:
                continue
            selected.append({"role": row["role"], "content": content})
            remaining -= len(content)
        selected.reverse()
        return selected

    @staticmethod
    def _history_messages(rows: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
        return [
            HumanMessage(content=row["content"])
            if row["role"] == "user"
            else AIMessage(content=row["content"])
            for row in rows
        ]

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 30,
        }

    @staticmethod
    def _visible_stage(workflow: WorkflowId, quality: Quality) -> str:
        if workflow in {WorkflowId.DIRECT, WorkflowId.DOMAIN_RAG}:
            return "answer"
        if workflow == WorkflowId.PAPER:
            if quality == Quality.FAST:
                return "fast_draft"
            if quality == Quality.HIGH:
                return "final"
            return "draft"
        if workflow == WorkflowId.GRANT:
            if quality == Quality.FAST:
                return "fast_draft"
            if quality == Quality.HIGH:
                return "final"
            return "draft"
        return "proposal"

    async def _resolve_route(
        self,
        *,
        request: ChatRequest,
        user: UserContext,
        run_id: UUID,
        history_rows: list[dict[str, str]],
    ) -> ResolvedRoute:
        if request.workflow != WorkflowId.AUTO:
            self._authorize(request.workflow, user)
            use_documents = (
                request.use_documents
                if request.use_documents is not None
                else request.workflow == WorkflowId.DOMAIN_RAG or bool(request.collection_ids)
            )
            return ResolvedRoute(
                workflow=request.workflow,
                recommended_tier=self.policy.explicit_tier(request.quality),
                use_documents=bool(use_documents),
                reason="The user explicitly selected this workflow and quality level.",
            )

        outcome = await self.semantic_router.decide(
            query=request.query,
            history=history_rows,
            collection_count=len(request.collection_ids),
            allowed_workflows=self._allowed_workflow_ids(user),
            user=user,
            run_id=str(run_id),
        )
        decision = outcome.decision

        workflow = decision.workflow
        documents_requested = bool(request.collection_ids) or request.use_documents is True
        if documents_requested and workflow == WorkflowId.DIRECT:
            workflow = WorkflowId.DOMAIN_RAG

        self._authorize(workflow, user)
        use_documents = request.use_documents if request.use_documents is not None else decision.use_documents
        if request.collection_ids:
            use_documents = True
        if workflow == WorkflowId.DOMAIN_RAG:
            use_documents = True

        return ResolvedRoute(
            workflow=workflow,
            recommended_tier=decision.model_tier,
            use_documents=bool(use_documents),
            reason=outcome.reason,
            router_alias=self.settings.local_router_model_alias,
        )

    async def execute(
        self,
        request: ChatRequest,
        user: UserContext,
        *,
        token_sink: TokenSink | None = None,
        route_sink: RouteSink | None = None,
    ) -> ChatResponse:
        run_id = uuid4()
        conversation_id = request.conversation_id or uuid4()
        raw_history = await self.conversations.list_recent(
            conversation_id=conversation_id,
            user_id=user.user_id,
            limit=self.settings.conversation_history_messages,
        )
        history_rows = self._bounded_history_rows(raw_history)
        route = await self._resolve_route(
            request=request,
            user=user,
            run_id=run_id,
            history_rows=history_rows,
        )
        self._authorize(route.workflow, user)

        thread_id = self._thread_id(user, run_id, route.workflow)
        await self.runs.create(
            run_id=run_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            user=user,
            workflow_id=route.workflow,
            route_reason=route.reason,
            quality=request.quality.value,
        )

        visible_stage = self._visible_stage(route.workflow, request.quality)
        visible_alias = self.policy.stage_alias(
            recommended_tier=route.recommended_tier,
            quality=request.quality,
            stage=visible_stage,
        )
        if route_sink is not None:
            await route_sink(route, visible_alias)

        graph = self.graphs[route.workflow]
        config = self._config(thread_id)
        initial_events: list[dict[str, str]] = []
        if route.router_alias:
            initial_events.append(
                {"run_id": str(run_id), "alias": route.router_alias, "stage": "route"}
            )
        history = self._history_messages(history_rows)
        initial_state = {
            "messages": [*history, HumanMessage(content=request.query)],
            "query": request.query,
            "run_id": str(run_id),
            "user_id": user.user_id,
            "team_id": user.team_id,
            "roles": sorted(user.roles),
            "workflow_id": route.workflow.value,
            "quality": request.quality.value,
            "recommended_tier": route.recommended_tier.value,
            "collection_ids": [str(item) for item in request.collection_ids],
            "use_documents": route.use_documents,
            "route_reason": route.reason,
            "context": "",
            "sources": [],
            "outline": "",
            "draft": "",
            "review": "",
            "answer": "",
            "pending_action": None,
            "approval_status": None,
            "publication_url": None,
            "call_events": initial_events,
        }

        try:
            if token_sink is not None and route.workflow != WorkflowId.WEBSITE:
                async with self.services.llm.stream_run(
                    run_id=str(run_id),
                    sink=token_sink,
                    stages={visible_stage},
                ):
                    result = await graph.ainvoke(initial_state, config=config)
            else:
                result = await graph.ainvoke(initial_state, config=config)

            response = await self._response_from_graph(
                graph=graph,
                config=config,
                result=result,
                run_id=run_id,
                conversation_id=conversation_id,
                workflow=route.workflow,
                route_reason=route.reason,
            )
            await self.runs.update(
                run_id,
                status=response.status,
                answer=response.answer,
                pending_action=(response.pending_action.model_dump(mode="json") if response.pending_action else None),
            )
            await self.conversations.append_turn(
                conversation_id=conversation_id,
                user_id=user.user_id,
                query=request.query,
                answer=response.answer,
                workflow_id=route.workflow,
                run_id=run_id,
            )
            return response
        except Exception as exc:
            await self.runs.update(run_id, status="failed", error_message=str(exc))
            raise

    async def resume(
        self,
        *,
        run_id: UUID,
        decision: str,
        feedback: str | None,
        user: UserContext,
    ) -> ChatResponse:
        preview = await self.runs.get_for_user(run_id, user)
        workflow = WorkflowId(preview["workflow_id"])
        self._authorize(workflow, user)
        run = await self.runs.claim_for_resume(run_id, user)

        graph = self.graphs[workflow]
        config = self._config(run["thread_id"])
        try:
            result = await graph.ainvoke(
                Command(resume={"decision": decision, "feedback": feedback}),
                config=config,
            )
            response = await self._response_from_graph(
                graph=graph,
                config=config,
                result=result,
                run_id=run_id,
                conversation_id=run["conversation_id"],
                workflow=workflow,
                route_reason=run["route_reason"],
            )
            await self.runs.update(
                run_id,
                status=response.status,
                answer=response.answer,
                pending_action=(response.pending_action.model_dump(mode="json") if response.pending_action else None),
            )
            await self.conversations.append(
                conversation_id=run["conversation_id"],
                user_id=user.user_id,
                role="assistant",
                content=response.answer,
                workflow_id=workflow,
                run_id=run_id,
            )
            return response
        except Exception as exc:
            await self.runs.update(run_id, status="failed", error_message=str(exc))
            raise

    async def _response_from_graph(
        self,
        *,
        graph: Any,
        config: dict[str, Any],
        result: Any,
        run_id: UUID,
        conversation_id: UUID,
        workflow: WorkflowId,
        route_reason: str,
    ) -> ChatResponse:
        snapshot = await graph.aget_state(config)
        result_value = getattr(result, "value", result)
        values = dict(snapshot.values or result_value or {})
        interrupts = list(getattr(snapshot, "interrupts", ()) or ())
        if not interrupts:
            result_interrupts = getattr(result, "interrupts", ()) or ()
            if result_interrupts:
                interrupts = list(result_interrupts)
            elif isinstance(result_value, dict):
                raw_interrupts = result_value.get("__interrupt__")
                if raw_interrupts:
                    interrupts = list(raw_interrupts)

        pending_raw = values.get("pending_action")
        pending_action = PendingAction.model_validate(pending_raw) if pending_raw else None
        awaiting = bool(interrupts) or (
            workflow == WorkflowId.WEBSITE
            and pending_action is not None
            and values.get("approval_status") is None
        )
        status = "awaiting_approval" if awaiting else "completed"

        source_items = [SourceCitation.model_validate(source) for source in values.get("sources", [])]
        events = [
            event
            for event in values.get("call_events", [])
            if str(event.get("run_id")) == str(run_id)
        ]
        model_tiers = [event["alias"] for event in events]
        answer = str(values.get("answer") or "Workflow completed without producing an answer.")

        return ChatResponse(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow=workflow,
            route_reason=route_reason,
            answer=answer,
            model_tiers=model_tiers,
            sources=source_items,
            status=status,
            pending_action=pending_action,
        )
