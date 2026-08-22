from __future__ import annotations

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
    collection_ids: list[str]
    attachment_context: str
    has_pdf_attachment: bool

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
    call_events: list[dict[str, str]]


class ParentState(CommonWorkflowState, total=False):
    """Control-plane state shared between the parent graph and subgraphs."""


class ChatState(CommonWorkflowState, total=False):
    """State visible to the general-chat subgraph."""


class PdfState(CommonWorkflowState, total=False):
    context: str
    retrieval_error: str


class RegulationsState(CommonWorkflowState, total=False):
    context: str
    retrieval_error: str


class PlaceholderState(CommonWorkflowState, total=False):
    draft: str
