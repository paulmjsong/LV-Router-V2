from app.config import Settings
from app.web_search import WebSearchService


def service() -> WebSearchService:
    return WebSearchService(Settings())


def test_korean_news_particle():
    assert service()._intent("오늘 뉴스는 뭐가 있어?") == "news"


def test_ranking_is_not_plain_news():
    assert service()._intent("오늘 가장 조회가 많이된 뉴스는?") == "ranking"


def test_korean_weather_location():
    svc = service()
    assert svc._intent("지금 광주의 날씨는 어때?") == "weather"
    assert svc._weather_location("지금 광주의 날씨는 어때?") == "광주"


def test_korean_region():
    assert service()._effective_region("광주 날씨") == "kr-kr"


def test_v07_followup_contract():
    topic = service()._resolved_news_topic(
        "그럼 경제는?",
        ["모로코 최신 뉴스 알려줘"],
    )
    assert topic is not None
    assert "모로코" in topic


def test_v07_answer_formatter_contract():
    svc = service()
    assert hasattr(svc, "strip_generated_sources")
    assert hasattr(svc, "format_answer")
