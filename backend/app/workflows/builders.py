from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..auth import UserContext
from ..config import Settings
from ..llm import LLMGateway
from ..rag import RAGService
from ..routing import StageModelPolicy
from ..schemas import ModelTier, Quality, WorkflowId
from .prompts import (
    CHAT_SYSTEM,
    GRANT_PLACEHOLDER_SYSTEM,
    PAPER_PLACEHOLDER_SYSTEM,
    PDF_SYSTEM,
    REGULATIONS_SYSTEM,
    WEBSITE_PLACEHOLDER_SYSTEM,
)
from .state import ChatState, PdfState, PlaceholderState, RegulationsState


@dataclass(slots=True)
class WorkflowServices:
    llm: LLMGateway
    rag: RAGService
    policy: StageModelPolicy
    settings: Settings


@dataclass(frozen=True, slots=True)
class WorkflowSubgraphs:
    chat: Any
    pdf: Any
    regulations: Any
    paper: Any
    grant: Any
    website: Any


def _user(state: dict[str, Any]) -> UserContext:
    return UserContext(
        user_id=state["user_id"],
        team_id=state.get("team_id"),
        roles=set(state.get("roles", [])),
    )


def _quality(state: dict[str, Any]) -> Quality:
    return Quality(state.get("quality", Quality.BALANCED.value))


def _workflow(state: dict[str, Any]) -> WorkflowId:
    return WorkflowId(state["workflow_id"])


def _recommended_tier(state: dict[str, Any]) -> ModelTier:
    return ModelTier(state.get("recommended_tier", ModelTier.CLOUD_SMALL.value))


def _stage_alias(state: dict[str, Any], services: WorkflowServices, stage: str) -> str:
    return services.policy.stage_alias(
        workflow=_workflow(state),
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
    state: dict[str, Any],
    *,
    limit: int = 16,
    char_limit: int = 24000,
) -> list[dict[str, str]]:
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


def _request_with_history(state: dict[str, Any]) -> str:
    messages = _conversation(state, limit=8, char_limit=8000)
    blocks = [f"{item['role'].upper()}: {item['content']}" for item in messages[:-1]]
    history = "\n\n".join(blocks)
    request = _clip_text(state["query"], 20000)
    if not history:
        return f"CURRENT REQUEST:\n{request}"
    return f"PRIOR CONVERSATION:\n{history}\n\nCURRENT REQUEST:\n{request}"


def _events_with(state: dict[str, Any], alias: str, stage: str) -> list[dict[str, str]]:
    return [
        *state.get("call_events", []),
        {"run_id": state["run_id"], "alias": alias, "stage": stage},
    ]


def _answer_update(
    state: dict[str, Any],
    *,
    answer: str,
    alias: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }
    if alias and stage:
        update["call_events"] = _events_with(state, alias, stage)
    return update


def build_chat_subgraph(services: WorkflowServices):
    async def answer(state: ChatState) -> ChatState:
        alias = _stage_alias(state, services, "answer")
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[{"role": "system", "content": CHAT_SYSTEM}, *_conversation(state)],
            run_id=state["run_id"],
            workflow_id=WorkflowId.CHAT.value,
            stage="answer",
            temperature=0.2,
            max_tokens=2400,
        )
        return _answer_update(state, answer=result.content, alias=alias, stage="answer")

    graph = StateGraph(ChatState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def build_pdf_subgraph(services: WorkflowServices):
    async def retrieve(state: PdfState) -> PdfState:
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        events = list(state.get("call_events", []))

        attached = _clip_text(
            state.get("attachment_context", ""),
            services.settings.pdf_attachment_context_chars,
        )
        if attached:
            context_parts.append("[UPLOADED PDF]\n" + attached)

        collection_ids = [UUID(item) for item in state.get("collection_ids", [])]
        if collection_ids:
            retrieved = await services.rag.retrieve(
                query=state["query"],
                collection_ids=collection_ids,
                user=_user(state),
                run_id=state["run_id"],
                mime_types={"application/pdf", "application/x-pdf"},
                exclude_system_collections=True,
            )
            sources = [source.model_dump(mode="json") for source in retrieved]
            db_context = services.rag.context_from_sources(retrieved)
            if db_context:
                context_parts.append(db_context)
            events.append(
                {
                    "run_id": state["run_id"],
                    "alias": "embedding",
                    "stage": "pdf_query_embedding",
                }
            )

        context = "\n\n".join(context_parts).strip()
        if not context:
            return {
                "context": "",
                "sources": [],
                "retrieval_error": (
                    "No usable PDF content was supplied. Attach a PDF with Open WebUI File Context "
                    "enabled, or index the PDF into a collection and select that collection."
                ),
                "call_events": events,
            }
        return {
            "context": context,
            "sources": sources,
            "retrieval_error": "",
            "call_events": events,
        }

    async def answer(state: PdfState) -> PdfState:
        alias = _stage_alias(state, services, "answer")
        prompt = (
            f"{_request_with_history(state)}\n\nPDF EVIDENCE:\n"
            f"{state.get('context', '')}"
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PDF_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PDF.value,
            stage="answer",
            temperature=0.1,
            max_tokens=2600,
        )
        return _answer_update(state, answer=result.content, alias=alias, stage="answer")

    async def missing(state: PdfState) -> PdfState:
        return _answer_update(state, answer=state.get("retrieval_error", "No PDF evidence was found."))

    def after_retrieve(state: PdfState) -> Literal["answer", "missing"]:
        return "answer" if state.get("context") else "missing"

    graph = StateGraph(PdfState)
    graph.add_node("retrieve_pdf", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("missing_pdf", missing)
    graph.add_edge(START, "retrieve_pdf")
    graph.add_conditional_edges(
        "retrieve_pdf",
        after_retrieve,
        {"answer": "answer", "missing": "missing_pdf"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("missing_pdf", END)
    return graph.compile()


def build_regulations_subgraph(services: WorkflowServices):
    async def retrieve(state: RegulationsState) -> RegulationsState:
        sources = await services.rag.retrieve(
            query=state["query"],
            collection_ids=[],
            system_keys={services.settings.gist_regulations_system_key},
            user=_user(state),
            run_id=state["run_id"],
        )
        context = services.rag.context_from_sources(sources)
        events = _events_with(state, "embedding", "regulations_query_embedding")
        if not context:
            return {
                "context": "",
                "sources": [],
                "retrieval_error": (
                    "The GIST Regulations collection contains no relevant indexed passages. "
                    "An administrator must index the source regulation documents with the "
                    "`upload-regulations` command."
                ),
                "call_events": events,
            }
        return {
            "context": context,
            "sources": [source.model_dump(mode="json") for source in sources],
            "retrieval_error": "",
            "call_events": events,
        }

    async def answer(state: RegulationsState) -> RegulationsState:
        alias = _stage_alias(state, services, "answer")
        prompt = (
            f"{_request_with_history(state)}\n\nGIST REGULATION EVIDENCE:\n"
            f"{state.get('context', '')}"
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": REGULATIONS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.REGULATIONS.value,
            stage="answer",
            temperature=0.0,
            max_tokens=2400,
        )
        return _answer_update(state, answer=result.content, alias=alias, stage="answer")

    async def missing(state: RegulationsState) -> RegulationsState:
        return _answer_update(
            state,
            answer=state.get("retrieval_error", "No GIST regulation evidence was found."),
        )

    def after_retrieve(state: RegulationsState) -> Literal["answer", "missing"]:
        return "answer" if state.get("context") else "missing"

    graph = StateGraph(RegulationsState)
    graph.add_node("retrieve_regulations", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("missing_regulations", missing)
    graph.add_edge(START, "retrieve_regulations")
    graph.add_conditional_edges(
        "retrieve_regulations",
        after_retrieve,
        {"answer": "answer", "missing": "missing_regulations"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("missing_regulations", END)
    return graph.compile()


def _build_placeholder_subgraph(
    services: WorkflowServices,
    *,
    workflow: WorkflowId,
    system_prompt: str,
    stage: str,
):
    async def run_placeholder(state: PlaceholderState) -> PlaceholderState:
        alias = _stage_alias(state, services, stage)
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _request_with_history(state)},
            ],
            run_id=state["run_id"],
            workflow_id=workflow.value,
            stage=stage,
            temperature=0.2,
            max_tokens=1800,
        )
        return {
            **_answer_update(state, answer=result.content, alias=alias, stage=stage),
            "draft": result.content,
        }

    graph = StateGraph(PlaceholderState)
    graph.add_node(stage, run_placeholder)
    graph.add_edge(START, stage)
    graph.add_edge(stage, END)
    return graph.compile()


def build_workflow_subgraphs(services: WorkflowServices) -> WorkflowSubgraphs:
    return WorkflowSubgraphs(
        chat=build_chat_subgraph(services),
        pdf=build_pdf_subgraph(services),
        regulations=build_regulations_subgraph(services),
        paper=_build_placeholder_subgraph(
            services,
            workflow=WorkflowId.PAPER,
            system_prompt=PAPER_PLACEHOLDER_SYSTEM,
            stage="draft",
        ),
        grant=_build_placeholder_subgraph(
            services,
            workflow=WorkflowId.GRANT,
            system_prompt=GRANT_PLACEHOLDER_SYSTEM,
            stage="draft",
        ),
        website=_build_placeholder_subgraph(
            services,
            workflow=WorkflowId.WEBSITE,
            system_prompt=WEBSITE_PLACEHOLDER_SYSTEM,
            stage="proposal",
        ),
    )
