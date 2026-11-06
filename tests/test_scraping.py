"""Unit tests for the scraping layer."""

from dataclasses import replace

import pytest

from config import CONFIG
from models import Article
from scraping import (
    BaseNewsScraper,
    CorpusBuilder,
    LentaScraper,
    RobotsChecker,
    RSSNewsScraper,
    RobotsDisallowedError,
    ScrapingError,
)


class FakeResponse:
    """Small HTTP response substitute for isolated unit tests."""

    def __init__(
        self,
        status_code: int,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = headers or {}


class FakeSession:
    """Fake requests.Session with predetermined responses."""

    def __init__(
        self,
        responses: dict[str, list[FakeResponse]] | None = None,
    ) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses or {}
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        **kwargs: object,
    ) -> FakeResponse:
        self.calls.append(url)

        available = self.responses.get(url, [])

        if not available:
            raise AssertionError(
                f"Unexpected HTTP request in test: {url}"
            )

        return available.pop(0)

    def close(self) -> None:
        """Implement the close method expected by the scraper."""


class DummyScraper(BaseNewsScraper):
    """Minimal scraper used to test the abstract base class."""

    def scrape(
        self,
        limit: int | None = None,
    ) -> list[Article]:
        return []


class StaticScraper(BaseNewsScraper):
    """Return predetermined Article objects without using the network."""

    def __init__(
        self,
        articles: list[Article],
        config=CONFIG,
        session=None,
    ) -> None:
        super().__init__(
            config=config,
            session=session or FakeSession(),
        )

        self.articles = articles

    def scrape(
        self,
        limit: int | None = None,
    ) -> list[Article]:
        if limit is None:
            return list(self.articles)

        return list(self.articles[:limit])


def make_test_config():
    """Create a fast configuration suitable for isolated tests."""

    return replace(
        CONFIG,
        request_timeout_seconds=1,
        request_delay_seconds=1.0,
        max_retries=3,
        min_articles=2,
        min_article_words=1,
        min_total_words=2,
        max_articles=10,
    )


def make_article(
    url: str,
    title: str,
    text: str,
) -> Article:
    """Create a valid article for tests."""

    return Article(
        url=url,
        title=title,
        text=text,
        source="Test Source",
    )


def test_robots_checker_allows_when_robots_file_is_missing() -> None:
    """A 404 robots.txt response is treated as no published rules."""

    session = FakeSession(
        {
            "https://example.com/robots.txt": [
                FakeResponse(404)
            ]
        }
    )

    checker = RobotsChecker(
        config=make_test_config(),
        session=session,
    )

    assert checker.can_fetch(
        "https://example.com/news/1"
    ) is True


def test_robots_checker_blocks_disallowed_path() -> None:
    """A Disallow rule must prevent access to the specified path."""

    robots_content = (
        "User-agent: *\n"
        "Disallow: /private\n"
    )

    session = FakeSession(
        {
            "https://example.com/robots.txt": [
                FakeResponse(200, robots_content)
            ]
        }
    )

    checker = RobotsChecker(
        config=make_test_config(),
        session=session,
    )

    assert checker.can_fetch(
        "https://example.com/private/article"
    ) is False

    assert checker.can_fetch(
        "https://example.com/news/article"
    ) is True


def test_robots_checker_caches_policy() -> None:
    """robots.txt should be retrieved only once per website."""

    session = FakeSession(
        {
            "https://example.com/robots.txt": [
                FakeResponse(404)
            ]
        }
    )

    checker = RobotsChecker(
        config=make_test_config(),
        session=session,
    )

    checker.can_fetch("https://example.com/news/1")
    checker.can_fetch("https://example.com/news/2")

    assert session.calls.count(
        "https://example.com/robots.txt"
    ) == 1


def test_robots_checker_fails_closed_on_server_error() -> None:
    """An unavailable robots.txt policy must block crawling."""

    session = FakeSession(
        {
            "https://example.com/robots.txt": [
                FakeResponse(503)
            ]
        }
    )

    checker = RobotsChecker(
        config=make_test_config(),
        session=session,
    )

    assert checker.can_fetch(
        "https://example.com/news/1"
    ) is False


def test_base_scraper_does_not_fetch_disallowed_page() -> None:
    """A blocked page must never be requested."""

    robots_content = (
        "User-agent: *\n"
        "Disallow: /private\n"
    )

    session = FakeSession(
        {
            "https://example.com/robots.txt": [
                FakeResponse(200, robots_content)
            ]
        }
    )

    scraper = DummyScraper(
        config=make_test_config(),
        session=session,
    )

    with pytest.raises(RobotsDisallowedError):
        scraper.fetch_response(
            "https://example.com/private/article"
        )

    assert (
        "https://example.com/private/article"
        not in session.calls
    )


def test_base_scraper_retries_temporary_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temporary 503 response should be retried."""

    monkeypatch.setattr(
        "scraping.base_scraper.time.sleep",
        lambda _seconds: None,
    )

    session = FakeSession(
        {
            "https://example.com/robots.txt": [
                FakeResponse(404)
            ],
            "https://example.com/news/1": [
                FakeResponse(503),
                FakeResponse(200, "<html><body>OK</body></html>"),
            ],
        }
    )

    scraper = DummyScraper(
        config=make_test_config(),
        session=session,
    )

    response = scraper.fetch_response(
        "https://example.com/news/1"
    )

    assert response.status_code == 200

    assert session.calls.count(
        "https://example.com/news/1"
    ) == 2


def test_lenta_scraper_recognizes_article_urls() -> None:
    """Only supported Lenta article paths should be accepted."""

    scraper = LentaScraper(
        listing_urls=["https://lenta.ru/news/"],
        config=make_test_config(),
        session=FakeSession(),
    )

    assert LentaScraper._is_article_url(
        "https://lenta.ru/news/2026/11/06/example/"
    )

    assert not LentaScraper._is_article_url(
        "https://example.com/news/2026/11/06/example/"
    )

    assert not LentaScraper._is_article_url(
        "https://lenta.ru/"
    )


def test_lenta_scraper_rejects_other_domains() -> None:
    """A Lenta scraper must not accept an unrelated website."""

    with pytest.raises(ValueError):
        LentaScraper(
            listing_urls=["https://example.com/news/"],
            config=make_test_config(),
            session=FakeSession(),
        )


def test_rss_scraper_requires_a_feed_url() -> None:
    """An RSS scraper cannot collect articles without configured feeds."""

    scraper = RSSNewsScraper(
        feed_urls=(),
        config=make_test_config(),
        session=FakeSession(),
    )

    with pytest.raises(ValueError, match="No RSS feed URLs"):
        scraper.scrape()


def test_corpus_builder_removes_duplicate_urls() -> None:
    """The same URL should only be represented once."""

    first = make_article(
        "https://example.com/news/1",
        "First article",
        "Первый текст новости.",
    )

    duplicate = make_article(
        "https://example.com/news/1/",
        "Different title",
        "Другой текст.",
    )

    scraper = StaticScraper(
        [first, duplicate],
        config=make_test_config(),
    )

    builder = CorpusBuilder(
        [scraper],
        config=make_test_config(),
    )

    result = builder.build(limit=10)

    assert len(result) == 1


def test_corpus_builder_removes_duplicate_content() -> None:
    """Identical title and text should not be stored twice."""

    first = make_article(
        "https://example.com/news/1",
        "Одинаковая новость",
        "Текст одной и той же новости.",
    )

    duplicate = make_article(
        "https://another-example.com/news/2",
        "Одинаковая новость",
        "Текст одной и той же новости.",
    )

    scraper = StaticScraper(
        [first, duplicate],
        config=make_test_config(),
    )

    builder = CorpusBuilder(
        [scraper],
        config=make_test_config(),
    )

    result = builder.build(limit=10)

    assert len(result) == 1


def test_corpus_builder_obeys_article_limit() -> None:
    """The builder must never return more than the requested limit."""

    articles = [
        make_article(
            f"https://example.com/news/{index}",
            f"Новость {index}",
            f"Текст новости номер {index}.",
        )
        for index in range(1, 6)
    ]

    scraper = StaticScraper(
        articles,
        config=make_test_config(),
    )

    builder = CorpusBuilder(
        [scraper],
        config=make_test_config(),
    )

    result = builder.build(limit=2)

    assert len(result) == 2


def test_corpus_builder_rejects_zero_limit() -> None:
    """A corpus collection limit must be positive."""

    builder = CorpusBuilder(
        [
            StaticScraper(
                [],
                config=make_test_config(),
            )
        ],
        config=make_test_config(),
    )

    with pytest.raises(ValueError):
        builder.build(limit=0)