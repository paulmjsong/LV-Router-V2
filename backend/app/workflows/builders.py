from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from ..auth import UserContext
from ..github_publisher import GitHubPublisher
from ..llm import LLMGateway
from ..rag import RAGService
from ..routing import StageModelPolicy
from ..schemas import ModelTier, Quality, WorkflowId
from .prompts import (
    DIRECT_SYSTEM,
    GRANT_COMPLIANCE_SYSTEM,
    GRANT_DRAFT_SYSTEM,
    GRANT_FINAL_SYSTEM,
    GRANT_REQUIREMENTS_SYSTEM,
    PAPER_DRAFT_SYSTEM,
    PAPER_FINAL_SYSTEM,
    PAPER_OUTLINE_SYSTEM,
    PAPER_REVIEW_SYSTEM,
    RAG_SYSTEM,
    WEBSITE_SYSTEM,
)
from .state import WorkflowState


@dataclass(slots=True)
class WorkflowServices:
    llm: LLMGateway
    rag: RAGService
    policy: StageModelPolicy
    github: GitHubPublisher


class WebsiteProposal(BaseModel):
    summary: str = Field(min_length=1, max_length=2000)
    path: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=200000)
    commit_message: str = Field(min_length=1, max_length=300)
    pr_title: str = Field(min_length=1, max_length=300)
    pr_body: str = Field(default="", max_length=5000)


def _user(state: WorkflowState) -> UserContext:
    return UserContext(
        user_id=state["user_id"],
        team_id=state.get("team_id"),
        roles=set(state.get("roles", [])),
    )


def _quality(state: WorkflowState) -> Quality:
    return Quality(state.get("quality", Quality.BALANCED.value))


def _recommended_tier(state: WorkflowState) -> ModelTier:
    return ModelTier(state.get("recommended_tier", ModelTier.CLOUD_SMALL.value))


def _stage_alias(state: WorkflowState, services: WorkflowServices, stage: str) -> str:
    return services.policy.stage_alias(
        recommended_tier=_recommended_tier(state),
        quality=_quality(state),
        stage=stage,
    )


def _message_text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _clip_text(value: str, max_chars: int) -> str:
    value = value.strip()
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    marker = "\n...[truncated]...\n"
    if max_chars <= len(marker):
        return value[:max_chars]
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return f"{value[:head]}{marker}{value[-tail:]}"


def _conversation(
    state: WorkflowState,
    *,
    limit: int = 16,
    char_limit: int = 24000,
) -> list[dict[str, str]]:
    """Return the newest bounded chat history for a direct model call."""
    converted_reversed: list[dict[str, str]] = []
    remaining = char_limit
    for message in reversed(state.get("messages", [])[-limit:]):
        if remaining <= 0:
            break
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = _clip_text(_message_text(message), remaining)
        converted_reversed.append({"role": role, "content": content})
        remaining -= len(content)
    converted_reversed.reverse()
    return converted_reversed


def _prior_history_text(
    state: WorkflowState,
    *,
    max_messages: int = 8,
    max_chars: int = 6000,
) -> str:
    """Format prior turns without repeating the current user request."""
    messages = list(state.get("messages", []))
    if messages and isinstance(messages[-1], HumanMessage):
        latest = _message_text(messages[-1]).strip()
        if latest == state.get("query", "").strip():
            messages.pop()

    blocks_reversed: list[str] = []
    remaining = max_chars
    for message in reversed(messages[-max_messages:]):
        if remaining <= 0:
            break
        if isinstance(message, HumanMessage):
            label = "USER"
        elif isinstance(message, AIMessage):
            label = "ASSISTANT"
        else:
            continue
        prefix = f"{label}: "
        content = _clip_text(_message_text(message), max(0, remaining - len(prefix)))
        block = f"{prefix}{content}"
        blocks_reversed.append(block)
        remaining -= len(block) + 2
    blocks_reversed.reverse()
    return "\n\n".join(blocks_reversed)


def _request_text(state: WorkflowState, max_chars: int = 20000) -> str:
    return _clip_text(state["query"], max_chars)


def _request_with_history(state: WorkflowState) -> str:
    history = _prior_history_text(state)
    if not history:
        return f"CURRENT REQUEST:\n{_request_text(state)}"
    return (
        "PRIOR CONVERSATION CONTEXT (use as context, not as external evidence):\n"
        f"{history}\n\nCURRENT REQUEST:\n{_request_text(state)}"
    )


def _event(state: WorkflowState, alias: str, stage: str) -> list[dict[str, str]]:
    return [{"run_id": state["run_id"], "alias": alias, "stage": stage}]


async def _retrieve(state: WorkflowState, services: WorkflowServices) -> WorkflowState:
    collection_ids = [UUID(item) for item in state.get("collection_ids", [])]
    # The local semantic router (or explicit API request) decides whether documents are
    # needed. When true and no collection IDs are supplied, retrieval searches all
    # collections accessible to the current user.
    if not state.get("use_documents", False):
        return {"context": "", "sources": []}
    retrieval_query = _request_text(state, 5000)
    prior_context = _prior_history_text(state, max_messages=4, max_chars=2500)
    if prior_context:
        retrieval_query += f"\n\nRECENT CONVERSATION CONTEXT:\n{prior_context}"
    sources = await services.rag.retrieve(
        query=retrieval_query,
        collection_ids=collection_ids,
        user=_user(state),
        run_id=state["run_id"],
    )
    return {
        "sources": [source.model_dump(mode="json") for source in sources],
        "context": services.rag.context_from_sources(sources),
        "call_events": _event(state, "embedding", "query_embedding"),
    }


def build_direct_graph(services: WorkflowServices, checkpointer: Any):
    async def answer(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "answer")
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[{"role": "system", "content": DIRECT_SYSTEM}, *_conversation(state)],
            run_id=state["run_id"],
            workflow_id=WorkflowId.DIRECT.value,
            stage="answer",
            temperature=0.2,
            max_tokens=2400,
        )
        return {
            "answer": result.content,
            "messages": [AIMessage(content=result.content)],
            "call_events": _event(state, alias, "answer"),
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)


def build_rag_graph(services: WorkflowServices, checkpointer: Any):
    async def retrieve(state: WorkflowState) -> WorkflowState:
        return await _retrieve(state, services)

    async def answer(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "answer")
        context = state.get("context") or "No relevant document passages were retrieved."
        prompt = (
            f"{_request_with_history(state)}\n\nRETRIEVED DOCUMENT EVIDENCE:\n{context}"
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"],
            workflow_id=WorkflowId.DOMAIN_RAG.value,
            stage="answer",
            temperature=0.1,
            max_tokens=2600,
        )
        return {
            "answer": result.content,
            "messages": [AIMessage(content=result.content)],
            "call_events": _event(state, alias, "answer"),
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)


def build_paper_graph(services: WorkflowServices, checkpointer: Any):
    async def retrieve(state: WorkflowState) -> WorkflowState:
        return await _retrieve(state, services)

    async def fast_draft(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "draft")
        prompt = f"{_request_with_history(state)}\n\nEVIDENCE:\n{state.get('context') or 'None supplied.'}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": PAPER_DRAFT_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.PAPER.value, stage="fast_draft",
            temperature=0.2, max_tokens=4000,
        )
        return {"draft": result.content, "answer": result.content,
                "messages": [AIMessage(content=result.content)],
                "call_events": _event(state, alias, "fast_draft")}

    async def outline(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "outline")
        prompt = f"{_request_with_history(state)}\n\nEVIDENCE:\n{state.get('context') or 'None supplied.'}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": PAPER_OUTLINE_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.PAPER.value, stage="outline",
            temperature=0.1, max_tokens=2200,
        )
        return {"outline": result.content, "call_events": _event(state, alias, "outline")}

    async def draft(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "draft")
        prompt = (
            f"CURRENT REQUEST:\n{_request_text(state)}\n\nOUTLINE:\n{state.get('outline', '')}\n\n"
            f"EVIDENCE:\n{state.get('context') or 'None supplied.'}"
        )
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": PAPER_DRAFT_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.PAPER.value, stage="draft",
            temperature=0.2, max_tokens=6000,
        )
        return {"draft": result.content, "call_events": _event(state, alias, "draft")}

    async def review(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "review")
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": PAPER_REVIEW_SYSTEM},
                      {"role": "user", "content": state.get("draft", "")}],
            run_id=state["run_id"], workflow_id=WorkflowId.PAPER.value, stage="review",
            temperature=0.0, max_tokens=2500,
        )
        return {"review": result.content, "call_events": _event(state, alias, "review")}

    async def final(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "final")
        prompt = f"DRAFT:\n{state.get('draft', '')}\n\nREVIEW:\n{state.get('review', '')}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": PAPER_FINAL_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.PAPER.value, stage="final",
            temperature=0.1, max_tokens=6000,
        )
        return {"answer": result.content, "messages": [AIMessage(content=result.content)],
                "call_events": _event(state, alias, "final")}

    async def deliver(state: WorkflowState) -> WorkflowState:
        answer = state.get("draft", "")
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    def after_retrieve(state: WorkflowState) -> Literal["fast_draft", "outline"]:
        return "fast_draft" if _quality(state) == Quality.FAST else "outline"

    def after_draft(state: WorkflowState) -> Literal["review", "deliver"]:
        return "review" if _quality(state) == Quality.HIGH else "deliver"

    graph = StateGraph(WorkflowState)
    for name, node in {
        "retrieve": retrieve, "fast_draft": fast_draft, "outline": outline,
        "draft": draft, "review": review, "final": final, "deliver": deliver,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve)
    graph.add_edge("fast_draft", END)
    graph.add_edge("outline", "draft")
    graph.add_conditional_edges("draft", after_draft)
    graph.add_edge("review", "final")
    graph.add_edge("final", END)
    graph.add_edge("deliver", END)
    return graph.compile(checkpointer=checkpointer)


def build_grant_graph(services: WorkflowServices, checkpointer: Any):
    async def retrieve(state: WorkflowState) -> WorkflowState:
        return await _retrieve(state, services)

    async def fast_draft(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "draft")
        prompt = f"{_request_with_history(state)}\n\nCALL/INSTITUTIONAL EVIDENCE:\n{state.get('context') or 'None supplied.'}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": GRANT_DRAFT_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.GRANT.value, stage="fast_draft",
            temperature=0.2, max_tokens=4500,
        )
        return {"draft": result.content, "answer": result.content,
                "messages": [AIMessage(content=result.content)],
                "call_events": _event(state, alias, "fast_draft")}

    async def requirements(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "requirements")
        prompt = f"{_request_with_history(state)}\n\nCALL DOCUMENTS:\n{state.get('context') or 'None supplied.'}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": GRANT_REQUIREMENTS_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.GRANT.value, stage="requirements",
            temperature=0.0, max_tokens=2500,
        )
        return {"outline": result.content, "call_events": _event(state, alias, "requirements")}

    async def draft(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "draft")
        prompt = (
            f"CURRENT REQUEST:\n{_request_text(state)}\n\nREQUIREMENTS:\n{state.get('outline', '')}\n\n"
            f"EVIDENCE:\n{state.get('context') or 'None supplied.'}"
        )
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": GRANT_DRAFT_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.GRANT.value, stage="draft",
            temperature=0.2, max_tokens=6500,
        )
        return {"draft": result.content, "call_events": _event(state, alias, "draft")}

    async def compliance(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "compliance")
        prompt = f"REQUIREMENTS:\n{state.get('outline', '')}\n\nDRAFT:\n{state.get('draft', '')}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": GRANT_COMPLIANCE_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.GRANT.value, stage="compliance",
            temperature=0.0, max_tokens=2800,
        )
        return {"review": result.content, "call_events": _event(state, alias, "compliance")}

    async def final(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "final")
        prompt = f"DRAFT:\n{state.get('draft', '')}\n\nCOMPLIANCE REVIEW:\n{state.get('review', '')}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": GRANT_FINAL_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.GRANT.value, stage="final",
            temperature=0.1, max_tokens=6500,
        )
        return {"answer": result.content, "messages": [AIMessage(content=result.content)],
                "call_events": _event(state, alias, "final")}

    async def deliver(state: WorkflowState) -> WorkflowState:
        answer = state.get("draft", "")
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    def after_retrieve(state: WorkflowState) -> Literal["fast_draft", "requirements"]:
        return "fast_draft" if _quality(state) == Quality.FAST else "requirements"

    def after_draft(state: WorkflowState) -> Literal["compliance", "deliver"]:
        return "compliance" if _quality(state) == Quality.HIGH else "deliver"

    graph = StateGraph(WorkflowState)
    for name, node in {
        "retrieve": retrieve, "fast_draft": fast_draft, "requirements": requirements,
        "draft": draft, "compliance": compliance, "final": final, "deliver": deliver,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve)
    graph.add_edge("fast_draft", END)
    graph.add_edge("requirements", "draft")
    graph.add_conditional_edges("draft", after_draft)
    graph.add_edge("compliance", "final")
    graph.add_edge("final", END)
    graph.add_edge("deliver", END)
    return graph.compile(checkpointer=checkpointer)


def build_website_graph(services: WorkflowServices, checkpointer: Any):
    async def retrieve(state: WorkflowState) -> WorkflowState:
        return await _retrieve(state, services)

    async def propose(state: WorkflowState) -> WorkflowState:
        alias = _stage_alias(state, services, "proposal")
        prompt = f"{_request_with_history(state)}\n\nCURRENT WEBSITE EVIDENCE:\n{state.get('context') or 'None supplied.'}"
        result = await services.llm.chat(
            user=_user(state), model_alias=alias,
            messages=[{"role": "system", "content": WEBSITE_SYSTEM}, {"role": "user", "content": prompt}],
            run_id=state["run_id"], workflow_id=WorkflowId.WEBSITE.value, stage="proposal",
            temperature=0.1, max_tokens=6000,
            response_format={"type": "json_object"},
        )
        try:
            proposal = WebsiteProposal.model_validate(services.llm.extract_json_object(result.content))
            services.github.validate_path(proposal.path)
        except (ValueError, ValidationError) as exc:
            answer = f"The website proposal could not be converted into a safe file change: {exc}"
            return {"answer": answer, "messages": [AIMessage(content=answer)],
                    "approval_status": "invalid",
                    "call_events": _event(state, alias, "proposal")}
        action = {
            "action_type": "github_pull_request",
            "summary": proposal.summary,
            "payload": proposal.model_dump(),
        }
        answer = f"Proposed website change:\n\n{proposal.summary}\n\nFile: `{proposal.path}`\n\nApproval is required before publishing."
        return {"draft": result.content, "answer": answer, "pending_action": action,
                "messages": [AIMessage(content=answer)],
                "call_events": _event(state, alias, "proposal")}

    async def approval(state: WorkflowState) -> WorkflowState:
        if state.get("approval_status") == "invalid" or not state.get("pending_action"):
            return {}
        decision = interrupt(
            {
                "type": "approval_required",
                "run_id": state["run_id"],
                "action": state["pending_action"],
            }
        )
        if decision.get("decision") == "approve":
            return {"approval_status": "approved"}
        feedback = str(decision.get("feedback") or "No reason supplied.")
        answer = f"The website change was rejected. Feedback: {feedback}"
        return {"approval_status": "rejected", "answer": answer,
                "messages": [AIMessage(content=answer)], "pending_action": None}

    async def publish(state: WorkflowState) -> WorkflowState:
        proposal = (state.get("pending_action") or {}).get("payload", {})
        result = await services.github.publish(proposal, state["run_id"])
        answer = result.message
        if result.pull_request_url:
            answer += f"\n\nPull request: {result.pull_request_url}"
        return {"answer": answer, "messages": [AIMessage(content=answer)],
                "publication_url": result.pull_request_url, "pending_action": None}

    def after_propose(state: WorkflowState) -> Literal["approval", "end"]:
        return "approval" if state.get("pending_action") else "end"

    def after_approval(state: WorkflowState) -> Literal["publish", "end"]:
        return "publish" if state.get("approval_status") == "approved" else "end"

    graph = StateGraph(WorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("propose", propose)
    graph.add_node("approval", approval)
    graph.add_node("publish", publish)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "propose")
    graph.add_conditional_edges("propose", after_propose, {"approval": "approval", "end": END})
    graph.add_conditional_edges("approval", after_approval, {"publish": "publish", "end": END})
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)
