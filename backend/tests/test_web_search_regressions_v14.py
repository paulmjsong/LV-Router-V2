from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.routing import LocalSemanticRouter
from app.web_search import WebSearchService


def _settings():
    return SimpleNamespace(
        web_search_query_chars=4000,
        web_search_timeout_seconds=10.0,
        web_search_proxy="",
        web_search_region="us-en",
        web_search_auto_region=False,
        web_search_safesearch="moderate",
        web_search_max_results=5,
        web_search_backend="duckduckgo",
        web_search_snippet_chars=1200,
        web_search_fetch_pages=False,
    )


def _install_fake_ddgs(monkeypatch, fake_cls):
    module = ModuleType("ddgs")
    module.DDGS = fake_cls
    monkeypatch.setitem(sys.modules, "ddgs", module)


def test_explicit_korean_search_command_is_deterministic():
    assert LocalSemanticRouter.explicitly_requests_web_search(
        "지도학습의 동향이 어떻게 되는지 검색해줘"
    )
    assert LocalSemanticRouter.explicitly_requests_web_search(
        "최근 공개된 주요 AI 모델을 웹에서 검색해서, 이전 모델과 비교해 "
        "무엇이 달라졌는지 최신 출처와 함께 알려줘."
    )

    assert not LocalSemanticRouter.explicitly_requests_web_search(
        "이진 검색 알고리즘을 설명해줘"
    )


@pytest.mark.asyncio
async def test_korean_trend_query_uses_text_search(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, *args, **kwargs):
            pass

        def news(self, *args, **kwargs):
            raise AssertionError("trend query must not use the news endpoint")

        def text(self, query, **kwargs):
            calls.append(("text", query))
            return [
                {
                    "title": "Supervised learning trends",
                    "href": "https://example.com/trends",
                    "body": "A current overview of supervised learning research.",
                }
            ]

    _install_fake_ddgs(monkeypatch, FakeDDGS)
    service = WebSearchService(_settings())
    results = await service.search("지도학습의 동향이 어떻게 되는지 검색해줘")

    assert calls == [("text", "지도학습의 동향이 어떻게 되는지 검색해줘")]
    assert results
    assert results[0].url == "https://example.com/trends"


@pytest.mark.asyncio
async def test_news_endpoint_failure_falls_back_to_text(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, *args, **kwargs):
            pass

        def news(self, query, **kwargs):
            calls.append(("news", query))
            raise RuntimeError("synthetic news backend failure")

        def text(self, query, **kwargs):
            calls.append(("text", query))
            return [
                {
                    "title": "AI news fallback",
                    "href": "https://example.com/ai-news",
                    "body": "Fallback ordinary web result.",
                }
            ]

    _install_fake_ddgs(monkeypatch, FakeDDGS)
    service = WebSearchService(_settings())
    results = await service.search("AI 관련 뉴스를 검색해줘")

    assert [kind for kind, _ in calls] == ["news", "text"]
    assert results
    assert results[0].url == "https://example.com/ai-news"
