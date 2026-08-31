from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import re
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5

from .config import Settings
from .schemas import SourceCitation

logger = logging.getLogger("infonet.web_search")


class WebSearchError(RuntimeError):
    """Raised when the live search provider cannot return usable results."""


SearchKind = Literal["text", "news"]


@dataclass(frozen=True, slots=True)
class _SearchItem:
    title: str
    url: str
    snippet: str
    kind: SearchKind = "text"
    published_at: str = ""
    publisher: str = ""
    published_sort: float = 0.0


class WebSearchService:
    """Retrieve bounded live search evidence through ``ddgs``.

    General web questions use DDGS text search. News/headline requests use DDGS
    news search. Short follow-ups retain news mode and the preceding topic, so
    dates and source formatting do not disappear midway through a conversation.
    """

    _NEWS_INTENT_RE = re.compile(
        r"(?ix)(?:"
        r"\b(?:latest|recent|breaking|current|today'?s?)\s+"
        r"(?:news|new|headlines?|updates?|developments?|current\s+events?)\b"
        r"|\b(?:news|headlines?|breaking\s+news)\s+(?:about|on|from|in|regarding)\b"
        r"|\bwhat(?:'s|\s+is)\s+happening\s+(?:in|with)\b"
        r"|(?:최신|최근)\s*(?:뉴스|소식|동향)"
        r"|\b(?:뉴스|속보)\b"
        r")"
    )
    _LEADING_REQUEST_RE = re.compile(
        r"(?ix)^\s*(?:please\s+)?(?:tell|show|give|find)\s+(?:me|us)\s+"
    )
    _LEADING_NEWS_RE = re.compile(
        r"(?ix)^\s*(?:the\s+)?"
        r"(?:(?:latest|recent|breaking|current|today'?s?)\s+)?"
        r"(?:news|new|headlines?|updates?|developments?|current\s+events?)"
        r"(?:\s+(?:about|on|from|in|regarding))?\s*"
    )
    _FOLLOWUP_PREFIX_RE = re.compile(
        r"(?ix)^\s*(?:"
        r"(?:and\s+)?(?:what|how)\s+about\s+"
        r"|(?:and|also|then)\s+"
        r"|(?:tell|show|give)\s+me\s+(?:more|other)\s*"
        r"|(?:more|other|another|additional)\s*"
        r"|(?:anything|something)\s+else\s*"
        r"|continue\s*"
        r"|same\s+(?:for|with)\s+"
        r"|(?:그럼|그러면|그리고|또|더|다른|추가로)\s*"
        r")"
    )
    _GENERIC_FOLLOWUP_RE = re.compile(
        r"(?ix)^\s*(?:"
        r"more|more\s+news|other\s+news|another|anything\s+else|"
        r"what\s+happened\s+next|continue|updates?|developments?|"
        r"더|더\s*알려줘|다른\s*뉴스|또|계속"
        r")\s*[?.!]*\s*$"
    )
    _TRAILING_SOURCES_RE = re.compile(
        r"(?ims)\n+(?:#{1,6}\s*)?(?:📌\s*)?(?:\*\*)?"
        r"(?:sources|references|참조\s*조항|참고\s*문헌)(?:\*\*)?\s*\n.*\Z"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(
        self,
        query: str,
        *,
        previous_queries: Sequence[str] = (),
    ) -> list[SourceCitation]:
        cleaned = " ".join(query.split())[: self.settings.web_search_query_chars]
        if not cleaned:
            return []
        prior = tuple(
            " ".join(item.split())[: self.settings.web_search_query_chars]
            for item in previous_queries
            if isinstance(item, str) and item.strip()
        )
        try:
            items = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, cleaned, prior),
                timeout=self.settings.web_search_timeout_seconds + 4.0,
            )
        except TimeoutError as exc:
            raise WebSearchError("Live web search timed out") from exc
        except Exception as exc:
            raise WebSearchError(f"Live web search failed: {type(exc).__name__}") from exc

        return [
            SourceCitation(
                source_type="web",
                chunk_id=rank,
                document_id=uuid5(NAMESPACE_URL, f"web-search:{item.url}"),
                title=item.title,
                page=None,
                score=1.0 / rank,
                excerpt=item.snippet,
                url=item.url,
                published_at=item.published_at or None,
                publisher=item.publisher or None,
            )
            for rank, item in enumerate(items, start=1)
        ]

    def _search_sync(
        self,
        query: str,
        previous_queries: Sequence[str] = (),
    ) -> list[_SearchItem]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError("The ddgs package is required for web search") from exc

        client_kwargs: dict[str, Any] = {
            "timeout": int(max(2, round(self.settings.web_search_timeout_seconds))),
        }
        if self.settings.web_search_proxy:
            client_kwargs["proxy"] = self.settings.web_search_proxy
        client = DDGS(**client_kwargs)

        news_topic = self._resolved_news_topic(query, previous_queries)
        if news_topic:
            return self._search_news(client, news_topic)
        return self._search_text(client, query)

    def _search_news(self, client: Any, topic: str) -> list[_SearchItem]:
        candidate_limit = max(10, self.settings.web_search_max_results * 3)
        raw_results = client.news(
            topic,
            region=self.settings.web_search_region,
            safesearch=self.settings.web_search_safesearch,
            timelimit="w",
            max_results=candidate_limit,
            backend="auto",
        )

        items: list[_SearchItem] = []
        seen_urls: set[str] = set()
        for raw in raw_results or []:
            if not isinstance(raw, dict):
                continue
            title = self._clean_text(raw.get("title"), 300)
            url = self._safe_url(raw.get("url") or raw.get("href"))
            snippet = self._clean_text(
                raw.get("body") or raw.get("snippet") or raw.get("description"),
                self.settings.web_search_snippet_chars,
            )
            published_at, published_sort = self._published(
                raw.get("date") or raw.get("published")
            )
            publisher = self._clean_text(raw.get("source") or raw.get("publisher"), 160)
            if not publisher and url:
                publisher = urlsplit(url).netloc.removeprefix("www.")
            if (
                not title
                or not url
                or not snippet
                or not published_at
                or not self._looks_like_article_url(url)
                or url in seen_urls
            ):
                continue
            seen_urls.add(url)
            items.append(
                _SearchItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    kind="news",
                    published_at=published_at,
                    publisher=publisher,
                    published_sort=published_sort,
                )
            )
        items.sort(key=lambda item: item.published_sort, reverse=True)
        return items[: self.settings.web_search_max_results]

    def _search_text(self, client: Any, query: str) -> list[_SearchItem]:
        raw_results = client.text(
            query,
            region=self.settings.web_search_region,
            safesearch=self.settings.web_search_safesearch,
            max_results=self.settings.web_search_max_results,
            backend=self.settings.web_search_backend,
        )
        items: list[_SearchItem] = []
        seen_urls: set[str] = set()
        for raw in raw_results or []:
            if not isinstance(raw, dict):
                continue
            title = self._clean_text(raw.get("title"), 300)
            url = self._safe_url(raw.get("href") or raw.get("url"))
            snippet = self._clean_text(
                raw.get("body") or raw.get("snippet") or raw.get("description"),
                self.settings.web_search_snippet_chars,
            )
            if not title or not url or not snippet or url in seen_urls:
                continue
            seen_urls.add(url)
            items.append(_SearchItem(title=title, url=url, snippet=snippet))
            if len(items) >= self.settings.web_search_max_results:
                break
        return items

    @classmethod
    def _resolved_news_topic(
        cls,
        query: str,
        previous_queries: Sequence[str],
    ) -> str | None:
        if cls._is_news_query(query):
            return cls._news_topic(query)

        prior_news = next(
            (item for item in reversed(previous_queries) if cls._is_news_query(item)),
            None,
        )
        if prior_news is None or not cls._is_news_followup(query):
            return None

        base_topic = cls._news_topic(prior_news)
        detail = cls._followup_detail(query)
        if not detail:
            return base_topic
        return " ".join(dict.fromkeys(f"{base_topic} {detail}".split()))

    @classmethod
    def _is_news_query(cls, query: str) -> bool:
        return bool(cls._NEWS_INTENT_RE.search(query))

    @classmethod
    def _is_news_followup(cls, query: str) -> bool:
        cleaned = " ".join(query.split())
        if not cleaned:
            return False
        if cls._FOLLOWUP_PREFIX_RE.search(cleaned) or cls._GENERIC_FOLLOWUP_RE.fullmatch(cleaned):
            return True
        # A short noun phrase such as "the economy?" is usually a topic refinement
        # when it immediately follows a news request. Avoid reclassifying full questions.
        words = cleaned.strip(" ?.!,:;-").split()
        question_verbs = re.search(
            r"(?ix)\b(?:who|what\s+is|what\s+are|why|how\s+does|how\s+do|"
            r"when|where|define|explain|calculate|translate)\b",
            cleaned,
        )
        return 0 < len(words) <= 5 and question_verbs is None

    @classmethod
    def _followup_detail(cls, query: str) -> str:
        detail = cls._FOLLOWUP_PREFIX_RE.sub("", query, count=1)
        detail = detail.strip(" ?.!,:;-")
        if cls._GENERIC_FOLLOWUP_RE.fullmatch(detail):
            return ""
        detail = re.sub(
            r"(?ix)\b(?:latest|recent|breaking|current|news|headlines?|updates?|developments?)\b",
            " ",
            detail,
        )
        return " ".join(detail.split())

    @classmethod
    def _news_topic(cls, query: str) -> str:
        topic = cls._LEADING_REQUEST_RE.sub("", query, count=1)
        topic = cls._LEADING_NEWS_RE.sub("", topic, count=1)
        topic = topic.strip(" ?.!,:;-")
        return topic or query

    @staticmethod
    def _published(value: Any) -> tuple[str, float]:
        if not isinstance(value, str) or not value.strip():
            return "", 0.0
        raw = value.strip()
        parsed: datetime | None = None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
            except (TypeError, ValueError, OverflowError):
                for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d %b %Y %H:%M:%S"):
                    try:
                        parsed = datetime.strptime(raw, pattern)
                        break
                    except ValueError:
                        continue
        if parsed is None:
            return "", 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%Y-%m-%d %H:%M UTC"), parsed.timestamp()

    @staticmethod
    def _looks_like_article_url(url: str) -> bool:
        parsed = urlsplit(url)
        path = parsed.path.strip("/")
        if not path:
            return False
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            return True
        return len(parts[0]) >= 24 or bool(re.search(r"\d{4}", parts[0]))

    @staticmethod
    def _clean_text(value: Any, max_chars: int) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:max_chars].strip()

    @staticmethod
    def _safe_url(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        try:
            parsed = urlsplit(value.strip())
        except ValueError:
            return ""
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))

    def context_from_sources(self, sources: list[SourceCitation]) -> str:
        is_news = any(source.published_at for source in sources)
        blocks: list[str] = [
            "SEARCH MODE: NEWS" if is_news else "SEARCH MODE: GENERAL WEB",
            f"SEARCHED AT: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "CITATION SCOPE: Use only the numbered sources in this current evidence block.",
        ]
        used = sum(len(block) + 2 for block in blocks)
        for rank, source in enumerate(sources, start=1):
            fields = [f"[{rank}] {source.title}"]
            if source.published_at:
                fields.append(f"Published: {source.published_at}")
            if source.publisher:
                fields.append(f"Publisher: {source.publisher}")
            fields.extend([f"URL: {source.url}", f"Summary: {source.excerpt}"])
            block = "\n".join(fields)
            if used + len(block) > self.settings.web_search_context_chars:
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    @classmethod
    def strip_generated_sources(cls, answer: str) -> str:
        return cls._TRAILING_SOURCES_RE.sub("", answer.strip()).strip()

    @staticmethod
    def sources_markdown(sources: list[SourceCitation]) -> str:
        lines = ["\n\n### Sources"]
        for rank, source in enumerate(sources, start=1):
            if not source.url:
                continue
            details: list[str] = []
            if source.publisher:
                details.append(source.publisher)
            if source.published_at:
                details.append(f"Published {source.published_at}")
            suffix = f" — {' · '.join(details)}" if details else ""
            lines.append(f"{rank}. [{source.title}]({source.url}){suffix}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def format_answer(self, answer: str, sources: list[SourceCitation]) -> str:
        body = self.strip_generated_sources(answer)
        return f"{body}{self.sources_markdown(sources)}".strip()
