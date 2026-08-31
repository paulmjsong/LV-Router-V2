from types import ModuleType, SimpleNamespace
import sys
from uuid import NAMESPACE_URL, uuid5

import pytest
from langchain_core.messages import HumanMessage

from app.llm import LLMResult
from app.routing import StageModelPolicy
from app.schemas import SourceCitation
from app.web_search import WebSearchService
from app.workflows.builders import WorkflowServices, build_web_search_subgraph


def web_settings():
    return SimpleNamespace(
        web_search_query_chars=1000,
        web_search_timeout_seconds=5.0,
        web_search_proxy=None,
        web_search_region="us-en",
        web_search_safesearch="moderate",
        web_search_max_results=5,
        web_search_backend="duckduckgo",
        web_search_snippet_chars=1200,
        web_search_context_chars=12000,
    )


@pytest.mark.asyncio
async def test_general_web_search_uses_text_and_deduplicates(monkeypatch) -> None:
    class FakeDDGS:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def news(self, *args, **kwargs):
            raise AssertionError("general queries must not use news search")

        def text(self, query, **kwargs):
            assert query == "Python release notes"
            assert kwargs["backend"] == "duckduckgo"
            return [
                {
                    "title": " Release notes ",
                    "href": "https://example.com/release#section",
                    "body": " Current release information. ",
                },
                {
                    "title": "Duplicate",
                    "href": "https://example.com/release",
                    "body": "Duplicate source.",
                },
            ]

    module = ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)

    results = await WebSearchService(web_settings()).search("Python release notes")

    assert len(results) == 1
    assert results[0].title == "Release notes"
    assert results[0].url == "https://example.com/release"
    assert results[0].published_at is None


@pytest.mark.asyncio
async def test_latest_news_uses_news_endpoint_and_sorts_dated_articles(monkeypatch) -> None:
    class FakeDDGS:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def text(self, *args, **kwargs):
            raise AssertionError("news queries must not use text search")

        def news(self, query, **kwargs):
            assert query.casefold() == "morocco"
            assert kwargs["backend"] == "auto"
            assert kwargs["timelimit"] == "w"
            return [
                {
                    "date": "2026-08-29T10:00:00+00:00",
                    "title": "Older concrete event",
                    "url": "https://news.example.com/2026/08/29/older-event",
                    "body": "An older event happened in Morocco.",
                    "source": "Example News",
                },
                {
                    "date": "2026-08-31T09:00:00+00:00",
                    "title": "Newest concrete event",
                    "url": "https://news.example.com/2026/08/31/newest-event",
                    "body": "A newer event happened in Morocco.",
                    "source": "Example News",
                },
                {
                    "date": "",
                    "title": "Morocco latest news homepage",
                    "url": "https://news.example.com/",
                    "body": "Generic publisher homepage.",
                    "source": "Example News",
                },
            ]

    module = ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)

    service = WebSearchService(web_settings())
    results = await service.search("Tell me the latest new on Morocco")

    assert [result.title for result in results] == [
        "Newest concrete event",
        "Older concrete event",
    ]
    assert results[0].published_at == "2026-08-31 09:00 UTC"
    assert results[0].publisher == "Example News"
    context = service.context_from_sources(results)
    assert "SEARCH MODE: NEWS" in context
    assert "Published: 2026-08-31 09:00 UTC" in context
    assert "Generic publisher homepage" not in context


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []
        self.controls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(
            content="A concrete event was reported on August 31 [1].",
            requested_alias=kwargs["model_alias"],
            served_model="fake-cloud-model",
        )

    async def emit_control(self, **kwargs):
        self.controls.append(kwargs)


class FakeWebSearch:
    async def search(self, query: str):
        assert query == "What is the latest news?"
        url = "https://example.com/2026/08/31/event"
        return [
            SourceCitation(
                source_type="web",
                chunk_id=1,
                document_id=uuid5(NAMESPACE_URL, f"web-search:{url}"),
                title="Concrete event",
                score=1.0,
                excerpt="A concrete event occurred.",
                url=url,
                published_at="2026-08-31 09:00 UTC",
                publisher="Example News",
            )
        ]

    @staticmethod
    def context_from_sources(sources):
        source = sources[0]
        return (
            "SEARCH MODE: NEWS\n\n"
            f"[1] {source.title}\nPublished: {source.published_at}\n"
            f"Publisher: {source.publisher}\nURL: {source.url}\nSummary: {source.excerpt}"
        )

    @staticmethod
    def sources_markdown(sources):
        source = sources[0]
        return (
            f"\n\n### Sources\n1. [{source.title}]({source.url}) — "
            f"{source.publisher}, {source.published_at}"
        )


@pytest.mark.asyncio
async def test_web_search_subgraph_answers_with_dated_sources() -> None:
    llm = FakeLLM()
    services = WorkflowServices(
        llm=llm,
        regulations=SimpleNamespace(),
        web_search=FakeWebSearch(),
        policy=StageModelPolicy(),
        settings=SimpleNamespace(),
    )
    graph = build_web_search_subgraph(services)
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="What is the latest news?")],
        "query": "What is the latest news?",
        "run_id": "r-web",
        "user_id": "u",
        "team_id": "lab",
        "roles": ["member"],
        "workflow_id": "web-search",
        "quality": "balanced",
        "recommended_tier": "cloud-small",
        "sources": [],
        "call_events": [],
    })

    assert [call["stage"] for call in llm.calls] == ["answer"]
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "SEARCH MODE: NEWS" in prompt
    assert result["answer"].endswith(
        "[Concrete event](https://example.com/2026/08/31/event) — "
        "Example News, 2026-08-31 09:00 UTC"
    )
    assert llm.controls[0]["event_type"] == "sources"
