from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..auth import UserContext
from ..routing import LocalSemanticRouter
from ..schemas import ModelTier, Quality, WorkflowId
from .builders import WorkflowServices, build_workflow_subgraphs
from .state import ParentState


def _user(state: ParentState) -> UserContext:
    return UserContext(
        user_id=state["user_id"],
        team_id=state.get("team_id"),
        roles=set(state.get("roles", [])),
    )


def _quality(state: ParentState) -> Quality:
    return Quality(state.get("quality", Quality.BALANCED.value))


def _allowed(state: ParentState) -> set[WorkflowId]:
    return {WorkflowId(value) for value in state.get("allowed_workflows", [])}


def _history_rows(state: ParentState, limit: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    messages = list(state.get("messages", []))
    if messages and isinstance(messages[-1], HumanMessage):
        messages = messages[:-1]
    for message in messages[-limit:]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        rows.append({"role": role, "content": content})
    return rows


def build_parent_graph(
    *,
    services: WorkflowServices,
    semantic_router: LocalSemanticRouter,
    checkpointer: Any,
):
    subgraphs = build_workflow_subgraphs(services)

    async def route(state: ParentState) -> ParentState:
        requested = WorkflowId(state.get("requested_workflow", WorkflowId.AUTO.value))
        allowed = _allowed(state)
        quality = _quality(state)
        if requested != WorkflowId.AUTO:
            if requested not in allowed:
                raise HTTPException(status_code=403, detail=f"Workflow is disabled: {requested.value}")
            return {
                "workflow_id": requested.value,
                "recommended_tier": services.policy.explicit_tier(quality).value,
                "use_documents": requested == WorkflowId.REGULATIONS,
                "route_reason": "The user explicitly selected this workflow.",
                "route_fallback": False,
                "route_confidence": 1.0,
                "route_difficulty": "explicit",
                "router_served_model": "",
                "call_events": [],
            }
        outcome = await semantic_router.decide(
            query=state["query"],
            history=_history_rows(state),
            allowed_workflows=sorted(allowed, key=lambda item: item.value),
            user=_user(state),
            run_id=state["run_id"],
        )
        decision = outcome.decision
        return {
            "workflow_id": decision.workflow.value,
            "recommended_tier": decision.model_tier.value,
            "use_documents": decision.use_documents,
            "route_reason": outcome.reason,
            "route_fallback": outcome.used_fallback,
            "route_confidence": decision.confidence,
            "route_difficulty": decision.difficulty.value,
            "router_served_model": outcome.llm_result.served_model,
            "call_events": [{
                "run_id": state["run_id"],
                "alias": services.settings.local_router_model_alias,
                "stage": "route",
            }],
        }

    async def validate_route(state: ParentState) -> ParentState:
        workflow = WorkflowId(state["workflow_id"])
        if workflow not in _allowed(state):
            raise HTTPException(status_code=403, detail=f"Resolved workflow is disabled: {workflow.value}")
        recommended = ModelTier(state["recommended_tier"])
        quality = _quality(state)
        resolved = services.policy.resolve_tier(
            workflow=workflow,
            recommended_tier=recommended,
            quality=quality,
        )
        return {
            "recommended_tier": resolved.value,
            "use_documents": workflow == WorkflowId.REGULATIONS,
            "route_reason": str(state.get("route_reason", "")),
        }

    async def announce_route(state: ParentState) -> ParentState:
        is_fallback = bool(state.get("route_fallback"))
        auto_route = state.get("requested_workflow") == WorkflowId.AUTO.value
        difficulty = str(state.get("route_difficulty", "")).strip()
        diagnostics = ""
        if auto_route:
            status = "fallback" if is_fallback else "validated"
            diagnostics = f" · difficulty {difficulty or 'unknown'} · router {status}"
        content = (
            f"> **Route:** `{state['workflow_id']}` → `{state['recommended_tier']}`"
            f"{diagnostics}\n"
        )
        if is_fallback:
            content += f"> **Fallback reason:** {state.get('route_reason', 'router validation failed')}\n"
        content += "\n"
        await services.llm.emit_control(
            run_id=state["run_id"],
            content=content,
            event_type="route",
            requested_alias=services.settings.local_router_model_alias,
            served_model=state.get("router_served_model", ""),
        )
        return {}

    def select_subgraph(state: ParentState) -> str:
        return state["workflow_id"]

    async def finalize(state: ParentState) -> ParentState:
        return {"answer": str(state.get("answer") or "Workflow completed without producing an answer.")}

    graph = StateGraph(ParentState)
    graph.add_node("route", route)
    graph.add_node("validate_route", validate_route)
    graph.add_node("announce_route", announce_route)
    graph.add_node(WorkflowId.DIRECT.value, subgraphs.direct)
    graph.add_node(WorkflowId.REGULATIONS.value, subgraphs.regulations)
    graph.add_node(WorkflowId.WEB_SEARCH.value, subgraphs.web_search)
    graph.add_node(WorkflowId.PAPER.value, subgraphs.paper)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "route")
    graph.add_edge("route", "validate_route")
    graph.add_edge("validate_route", "announce_route")
    graph.add_conditional_edges(
        "announce_route",
        select_subgraph,
        {
            WorkflowId.DIRECT.value: WorkflowId.DIRECT.value,
            WorkflowId.REGULATIONS.value: WorkflowId.REGULATIONS.value,
            WorkflowId.WEB_SEARCH.value: WorkflowId.WEB_SEARCH.value,
            WorkflowId.PAPER.value: WorkflowId.PAPER.value,
        },
    )
    for workflow in (
        WorkflowId.DIRECT,
        WorkflowId.REGULATIONS,
        WorkflowId.WEB_SEARCH,
        WorkflowId.PAPER,
    ):
        graph.add_edge(workflow.value, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
