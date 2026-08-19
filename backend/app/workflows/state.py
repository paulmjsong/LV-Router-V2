from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    run_id: str
    user_id: str
    team_id: str | None
    roles: list[str]
    workflow_id: str
    quality: str
    recommended_tier: str
    collection_ids: list[str]
    use_documents: bool
    route_reason: str

    context: str
    sources: list[dict[str, Any]]
    outline: str
    draft: str
    review: str
    answer: str
    pending_action: dict[str, Any] | None
    approval_status: str | None
    publication_url: str | None
    call_events: Annotated[list[dict[str, str]], operator.add]
