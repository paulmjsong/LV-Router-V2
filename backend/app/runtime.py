from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException
from langchain_core.messages import HumanMessage

from .auth import UserContext
from .config import Settings
from .repositories import ConversationRepository, RunRepository
from .routing import LocalSemanticRouter
from .schemas import ChatRequest, ChatResponse, SourceCitation, WorkflowId, WorkflowInfo
from .workflows.builders import WorkflowServices
from .workflows.parent import build_parent_graph


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    id: WorkflowId
    name: str
    description: str
    allowed_roles: frozenset[str]
    enabled: bool = True
    placeholder: bool = False


_ALL_ROLES = frozenset({"member", "editor", "admin"})
WORKFLOW_SPECS: dict[WorkflowId, WorkflowSpec] = {
    WorkflowId.DIRECT: WorkflowSpec(
        WorkflowId.DIRECT,
        "Direct",
        "General questions and direct model inference.",
        _ALL_ROLES,
    ),
    WorkflowId.REGULATIONS: WorkflowSpec(
        WorkflowId.REGULATIONS,
        "GIST Regulations",
        "Straightforward RAG over the supplied Jireumgil FAISS vectorstore.",
        _ALL_ROLES,
    ),
    WorkflowId.PAPER: WorkflowSpec(
        WorkflowId.PAPER,
        "Research Paper Drafting",
        "Basic multi-agent drafting with orchestrator, specialist subagents, and validator.",
        _ALL_ROLES,
    ),
    WorkflowId.GRANT: WorkflowSpec(
        WorkflowId.GRANT,
        "Grant",
        "Coming soon.",
        _ALL_ROLES,
        enabled=False,
        placeholder=True,
    ),
    WorkflowId.WEBSITE: WorkflowSpec(
        WorkflowId.WEBSITE,
        "Website",
        "Coming soon.",
        _ALL_ROLES,
        enabled=False,
        placeholder=True,
    ),
}


class WorkflowRuntime:
    _STREAMED_STAGES = frozenset({"answer", "final"})

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
        self.settings = settings
        self.runs = runs
        self.conversations = conversations
        self.semantic_router = LocalSemanticRouter(services.llm, settings)
        self.graph = build_parent_graph(
            services=services,
            semantic_router=self.semantic_router,
            checkpointer=checkpointer,
        )

    def list_workflows(self, user: UserContext) -> list[WorkflowInfo]:
        return [
            WorkflowInfo(
                id=spec.id,
                name=spec.name,
                description=spec.description,
                allowed_roles=sorted(spec.allowed_roles),
                enabled=spec.enabled,
                placeholder=spec.placeholder,
            )
            for spec in WORKFLOW_SPECS.values()
            if user.has_any_role(set(spec.allowed_roles))
        ]

    def _allowed_workflow_ids(self, user: UserContext) -> list[WorkflowId]:
        return [
            spec.id for spec in WORKFLOW_SPECS.values()
            if spec.enabled and user.has_any_role(set(spec.allowed_roles))
        ]

    @staticmethod
    def _thread_id(user: UserContext, conversation_id: UUID) -> str:
        raw = f"infonet-parent-v2:{user.user_id}:{conversation_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    @staticmethod
    def _source_items(values: dict[str, Any]) -> list[SourceCitation]:
        return [SourceCitation.model_validate(source) for source in values.get("sources", [])]

    @staticmethod
    def _model_tiers(values: dict[str, Any], run_id: UUID) -> list[str]:
        aliases: list[str] = []
        for event in values.get("call_events", []):
            if str(event.get("run_id")) != str(run_id):
                continue
            alias = str(event.get("alias", "")).strip()
            if alias and alias not in aliases:
                aliases.append(alias)
        return aliases

    async def execute(self, request: ChatRequest, user: UserContext, *, token_sink: Any | None = None) -> ChatResponse:
        if request.workflow in {WorkflowId.GRANT, WorkflowId.WEBSITE}:
            raise HTTPException(status_code=503, detail=f"Workflow is coming soon: {request.workflow.value}")
        run_id = uuid4()
        conversation_id = request.conversation_id or uuid4()
        thread_id = self._thread_id(user, conversation_id)
        config = self._config(thread_id)
        await self.runs.create(
            run_id=run_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            user=user,
            workflow_id=request.workflow,
            route_reason=(
                "Routing is performed inside the parent LangGraph."
                if request.workflow == WorkflowId.AUTO
                else "The user explicitly selected this workflow."
            ),
            quality=request.quality.value,
        )
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "query": request.query,
            "run_id": str(run_id),
            "conversation_id": str(conversation_id),
            "user_id": user.user_id,
            "team_id": user.team_id,
            "roles": sorted(user.roles),
            "requested_workflow": request.workflow.value,
            "allowed_workflows": [item.value for item in self._allowed_workflow_ids(user)],
            "quality": request.quality.value,
            "workflow_id": "",
            "recommended_tier": "",
            "use_documents": False,
            "route_reason": "",
            "route_fallback": False,
            "route_confidence": 0.0,
            "route_difficulty": "",
            "router_served_model": "",
            "answer": "",
            "sources": [],
            "call_events": [],
            "context": "",
            "retrieval_error": "",
            "paper_plan": "",
            "paper_agent_outputs": [],
            "paper_draft": "",
            "paper_validation": "",
        }
        try:
            if token_sink is None:
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                async with self.services.llm.stream_run(
                    run_id=str(run_id), sink=token_sink, stages=self._STREAMED_STAGES
                ):
                    result = await self.graph.ainvoke(initial_state, config=config)
            snapshot = await self.graph.aget_state(config)
            result_value = getattr(result, "value", result)
            values = dict(snapshot.values or result_value or {})
            workflow = WorkflowId(values["workflow_id"])
            route_reason = str(values.get("route_reason") or "Route completed.")
            answer = str(values.get("answer") or "Workflow completed without producing an answer.")
            await self.runs.update_route(run_id, workflow_id=workflow, route_reason=route_reason)
            await self.runs.update(run_id, status="completed", answer=answer)
            await self.conversations.append_turn(
                conversation_id=conversation_id,
                user_id=user.user_id,
                query=request.query,
                answer=answer,
                workflow_id=workflow,
                run_id=run_id,
            )
            return ChatResponse(
                run_id=run_id,
                conversation_id=conversation_id,
                workflow=workflow,
                route_reason=route_reason,
                route_fallback=bool(values.get("route_fallback", False)),
                route_difficulty=str(values.get("route_difficulty", "")),
                answer=answer,
                model_tiers=self._model_tiers(values, run_id),
                sources=self._source_items(values),
            )
        except Exception as exc:
            await self.runs.update(run_id, status="failed", error_message=str(exc))
            raise
