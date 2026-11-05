"""Concrete scraper implementation for Lenta.ru news articles."""

from __future__ import annotations

import logging
import re
from typing import Optional

import feedparser

import config
from models.article import Article
from scraping.base_scraper import BaseNewsScraper
from scraping.robots_checker import RobotsChecker

logger = logging.getLogger(__name__)


class LentaScraper(BaseNewsScraper):
    """Scraper specialized for extracting Russian news articles from Lenta.ru."""

    DEFAULT_BASE_URL: str = "https://lenta.ru"
    RSS_ENDPOINTS: tuple[str, ...] = (
        "https://lenta.ru/rss/news",
        "https://lenta.ru/rss/articles",
        "https://lenta.ru/rss/top7",
    )
    ARCHIVE_PATHS: tuple[str, ...] = (
        "/parts/news/",
        "/rubrics/russia/",
        "/rubrics/world/",
        "/rubrics/economics/",
        "/rubrics/science/",
    )

    TITLE_SELECTORS: list[str] = [
        "span.topic-body__title",
        "h1.topic-body__titles",
        "h1",
    ]
    BODY_SELECTORS: list[str] = [
        "div.topic-body__content",
        "div.b-topic__content",
        "article",
    ]
    DATE_SELECTORS: list[str] = [
        "a.topic-header__time",
        "time.topic-header__time",
        "time",
    ]
    ARTICLE_PATH_PATTERN: re.Pattern[str] = re.compile(r"/news/\d{4}/\d{2}/\d{2}/[^/]+/?$")

    def __init__(
        self,
        robots_checker: Optional[RobotsChecker] = None,
        request_delay: float = config.REQUEST_DELAY_SECONDS,
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
    ) -> None:
        """Initialize the Lenta.ru scraper.

        Args:
            robots_checker: Optional shared RobotsChecker instance.
            request_delay: Polite delay between HTTP requests in seconds.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum retry attempts on transient network failures.
        """
        super().__init__(
            source_name="Lenta.ru",
            base_url=self.DEFAULT_BASE_URL,
            robots_checker=robots_checker,
            request_delay=request_delay,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._rss_fallback_cache: dict[str, dict[str, str]] = {}

    def discover_article_urls(self, max_urls: int = 150) -> list[str]:
        """Discover Lenta.ru article URLs via RSS feeds and rubric index pages.

        Args:
            max_urls: Maximum number of unique URLs to collect.

        Returns:
            List of normalized Lenta.ru article URLs.
        """
        discovered: list[str] = []
        seen: set[str] = set()

        # 1. Primary discovery through official Lenta.ru RSS feeds
        for rss_url in self.RSS_ENDPOINTS:
            if len(discovered) >= max_urls:
                break
            xml_text = self._fetch_html(rss_url, check_robots=True)
            if not xml_text:
                continue

            feed = feedparser.parse(xml_text)
            for entry in feed.entries:
                link = getattr(entry, "link", "") or ""
                if not isinstance(link, str) or not link.strip():
                    continue
                norm_url = self.normalize_url(link)
                if norm_url not in seen:
                    seen.add(norm_url)
                    discovered.append(norm_url)
                    self._rss_fallback_cache[norm_url] = {
                        "title": str(getattr(entry, "title", "") or "").strip(),
                        "published": str(getattr(entry, "published", "") or "").strip(),
                    }
                if len(discovered) >= max_urls:
                    break

        # 2. Secondary discovery through Lenta.ru section pages if needed
        for path in self.ARCHIVE_PATHS:
            if len(discovered) >= max_urls:
                break
            page_url = self.normalize_url(path)
            html = self._fetch_html(page_url, check_robots=True)
            if not html:
                continue

            soup = self._create_soup(html)
            for anchor in soup.find_all("a", href=True):
                href = anchor.get("href")
                if not isinstance(href, str):
                    continue
                if self.ARTICLE_PATH_PATTERN.search(href):
                    norm_url = self.normalize_url(href)
                    if norm_url not in seen:
                        seen.add(norm_url)
                        discovered.append(norm_url)
                        if len(discovered) >= max_urls:
                            break

        return discovered

    def parse_article(self, url: str) -> Optional[Article]:
        """Download and parse a single Lenta.ru article page.

        Args:
            url: Normalized Lenta.ru article URL.

        Returns:
            Validated `Article` instance, or `None` if extraction fails or article is too short.
        """
        norm_url = self.normalize_url(url)
        html = self._fetch_html(norm_url, check_robots=True)
        if not html:
            return None

        soup = self._create_soup(html)
        rss_meta = self._rss_fallback_cache.get(norm_url, {})

        title = self._extract_title(soup, self.TITLE_SELECTORS) or rss_meta.get("title")
        body_text = self._extract_body_text(soup, self.BODY_SELECTORS)
        pub_date = self._extract_date(soup, self.DATE_SELECTORS) or rss_meta.get("published")

        if not title or not body_text:
            logger.debug("Skipping Lenta.ru URL due to missing title or body: %s", norm_url)
            return None

        try:
            article = Article(
                url=norm_url,
                title=title,
                text=body_text,
                date=pub_date,
                source=self.source_name,
            )
        except ValueError as exc:
            logger.debug("Validation failed for Lenta.ru article %s: %s", norm_url, exc)
            return None

        if article.word_count < config.MIN_WORDS_PER_ARTICLE:
            logger.debug(
                "Skipping short Lenta.ru article (%d words < %d): %s",
                article.word_count,
                config.MIN_WORDS_PER_ARTICLE,
                norm_url,
            )
            return None

        return article