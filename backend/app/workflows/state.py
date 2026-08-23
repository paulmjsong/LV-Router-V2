from __future__ import annotations

import operator
from typing import Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class CommonWorkflowState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    run_id: str
    conversation_id: str
    user_id: str
    team_id: str | None
    roles: list[str]
    requested_workflow: str
    allowed_workflows: list[str]
    quality: str

    workflow_id: str
    recommended_tier: str
    use_documents: bool
    route_reason: str
    route_fallback: bool
    route_confidence: float
    route_difficulty: str
    router_served_model: str

    answer: str
    sources: list[dict[str, Any]]
    call_events: Annotated[list[dict[str, str]], operator.add]


class ParentState(CommonWorkflowState, total=False):
    """Control-plane state shared between the parent graph and active subgraphs."""


class DirectState(CommonWorkflowState, total=False):
    pass


class RegulationsState(CommonWorkflowState, total=False):
    context: str
    retrieval_error: str


class PaperState(CommonWorkflowState, total=False):
    paper_plan: str
    paper_agent_outputs: Annotated[list[dict[str, str]], operator.add]
    paper_draft: str
    paper_validation: str
