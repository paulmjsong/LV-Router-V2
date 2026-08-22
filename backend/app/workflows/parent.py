from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..auth import UserContext
from ..routing import LocalRouteDecision, LocalSemanticRouter
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
    # Exclude the current user message; it is already supplied as `query`.
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
    """Build one control graph with separately compiled workflow subgraphs."""
    subgraphs = build_workflow_subgraphs(services)

    async def route(state: ParentState) -> ParentState:
        requested = WorkflowId(state.get("requested_workflow", WorkflowId.AUTO.value))
        allowed = _allowed(state)
        quality = _quality(state)

        if requested != WorkflowId.AUTO:
            if requested not in allowed:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role not permitted for workflow: {requested.value}",
                )
            use_documents = requested in {WorkflowId.PDF, WorkflowId.REGULATIONS}
            return {
                "workflow_id": requested.value,
                "recommended_tier": services.policy.explicit_tier(quality).value,
                "use_documents": use_documents,
                "route_reason": "The user explicitly selected this workflow.",
                "route_fallback": False,
                "route_confidence": 1.0,
                "route_difficulty": "explicit",
                "router_served_model": "",
                "call_events": [],
            }

        # Concrete file/collection input is a stronger signal than semantic
        # classification and should not consume a router call.
        has_pdf_attachment = bool(state.get("has_pdf_attachment", False))
        has_selected_collection = bool(state.get("collection_ids"))
        if WorkflowId.PDF in allowed and (has_pdf_attachment or has_selected_collection):
            source_kind = "attachment" if has_pdf_attachment else "collection"
            return {
                "workflow_id": WorkflowId.PDF.value,
                "recommended_tier": services.policy.explicit_tier(quality).value,
                "use_documents": True,
                "route_reason": f"A PDF {source_kind} was supplied.",
                "route_fallback": False,
                "route_confidence": 1.0,
                "route_difficulty": source_kind,
                "router_served_model": "",
                "call_events": [],
            }

        outcome = await semantic_router.decide(
            query=state["query"],
            history=_history_rows(state),
            collection_count=len(state.get("collection_ids", [])),
            has_pdf_attachment=state.get("has_pdf_attachment", False),
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
            "call_events": [
                {
                    "run_id": state["run_id"],
                    "alias": services.settings.local_router_model_alias,
                    "stage": "route",
                }
            ],
        }

    async def validate_route(state: ParentState) -> ParentState:
        workflow = WorkflowId(state["workflow_id"])
        allowed = _allowed(state)
        if workflow == WorkflowId.AUTO or workflow not in allowed:
            raise HTTPException(status_code=403, detail="Resolved workflow is not permitted")
        recommended = ModelTier(state["recommended_tier"])
        quality = _quality(state)
        resolved = services.policy.resolve_tier(
            workflow=workflow,
            recommended_tier=recommended,
            quality=quality,
        )
        reason = str(state.get("route_reason", ""))
        confidence = float(state.get("route_confidence", 0.0))
        if (
            quality == Quality.BALANCED
            and resolved == ModelTier.LOCAL_FAST
            and confidence < services.settings.local_fast_min_confidence
        ):
            resolved = ModelTier.CLOUD_SMALL
            reason = (
                f"{reason} local-fast confidence {confidence:.2f} was below "
                f"{services.settings.local_fast_min_confidence:.2f}; promoted to cloud-small."
            ).strip()
        return {
            "recommended_tier": resolved.value,
            "use_documents": workflow in {WorkflowId.PDF, WorkflowId.REGULATIONS},
            "route_reason": reason,
        }

    async def announce_route(state: ParentState) -> ParentState:
        fallback = " · fallback" if state.get("route_fallback") else ""
        confidence = float(state.get("route_confidence", 0.0))
        auto_route = state.get("requested_workflow") == WorkflowId.AUTO.value
        difficulty = str(state.get("route_difficulty", "")).strip()
        diagnostics = ""
        if auto_route:
            diagnostics = f" · difficulty {difficulty or 'unknown'} · confidence {confidence:.2f}"
        content = (
            f"> **Route:** `{state['workflow_id']}` → `{state['recommended_tier']}`"
            f"{diagnostics}{fallback}\n"
        )
        if state.get("route_fallback"):
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
        answer = str(state.get("answer") or "Workflow completed without producing an answer.")
        return {"answer": answer}

    graph = StateGraph(ParentState)
    graph.add_node("route", route)
    graph.add_node("validate_route", validate_route)
    graph.add_node("announce_route", announce_route)
    graph.add_node(WorkflowId.CHAT.value, subgraphs.chat)
    graph.add_node(WorkflowId.PDF.value, subgraphs.pdf)
    graph.add_node(WorkflowId.REGULATIONS.value, subgraphs.regulations)
    graph.add_node(WorkflowId.PAPER.value, subgraphs.paper)
    graph.add_node(WorkflowId.GRANT.value, subgraphs.grant)
    graph.add_node(WorkflowId.WEBSITE.value, subgraphs.website)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "route")
    graph.add_edge("route", "validate_route")
    graph.add_edge("validate_route", "announce_route")
    graph.add_conditional_edges(
        "announce_route",
        select_subgraph,
        {
            WorkflowId.CHAT.value: WorkflowId.CHAT.value,
            WorkflowId.PDF.value: WorkflowId.PDF.value,
            WorkflowId.REGULATIONS.value: WorkflowId.REGULATIONS.value,
            WorkflowId.PAPER.value: WorkflowId.PAPER.value,
            WorkflowId.GRANT.value: WorkflowId.GRANT.value,
            WorkflowId.WEBSITE.value: WorkflowId.WEBSITE.value,
        },
    )
    for workflow in (
        WorkflowId.CHAT,
        WorkflowId.PDF,
        WorkflowId.REGULATIONS,
        WorkflowId.PAPER,
        WorkflowId.GRANT,
        WorkflowId.WEBSITE,
    ):
        graph.add_edge(workflow.value, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
