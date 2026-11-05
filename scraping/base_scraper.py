"""Abstract base scraper providing HTTP session management, rate limiting, and HTML parsing.

Academic Control Question Reference (Q1 & Q2):
    Q1 (What is parsing? Which tools are used?):
        Web parsing (scraping) is the automated extraction of structured data from
        unstructured or semi-structured HTML/XML web pages. This module uses `requests`
        for HTTP transport and session pooling, and `BeautifulSoup4` backed by the C-based
        `lxml` parser for fast DOM traversal[cite: 2, 3].
    Q2 (What ethical principles must be respected during web parsing?):
        Ethical scraping mandates verifying permissions via `robots.txt` prior to fetching,
        enforcing polite inter-request delays (0.5–1.0s) to avoid server overload,
        identifying the client via headers, and never bypassing HTTP 403/401 blocks[cite: 3].
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

import config
from models.article import Article
from scraping.robots_checker import RobotsChecker

logger = logging.getLogger(__name__)


class BaseNewsScraper(ABC):
    """Abstract base class for Russian news website scrapers.

    Provides shared infrastructure for HTTP retries, polite rate limiting,
    `robots.txt` verification, URL normalization, and common HTML extraction helpers.
    """

    def __init__(
        self,
        source_name: str,
        base_url: str,
        robots_checker: Optional[RobotsChecker] = None,
        request_delay: float = config.REQUEST_DELAY_SECONDS,
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize the base news scraper.

        Args:
            source_name: Human-readable identifier of the news publication.
            base_url: Root URL of the target website (e.g., 'https://lenta.ru').
            robots_checker: Optional shared RobotsChecker instance.
            request_delay: Delay in seconds enforced between consecutive HTTP requests.
            timeout: Socket timeout in seconds for each HTTP GET request.
            max_retries: Maximum number of retry attempts for transient network errors.
            session: Optional pre-configured `requests.Session` instance.

        Raises:
            ValueError: If `source_name` or `base_url` is empty, or numeric parameters are invalid.
        """
        if not source_name or not source_name.strip():
            raise ValueError("source_name must be a non-empty string.")
        if not base_url or not base_url.strip().startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url: {base_url!r}. Must start with http:// or https://.")
        if request_delay < 0:
            raise ValueError("request_delay must be non-negative.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        self.source_name: str = source_name.strip()
        self.base_url: str = base_url.strip().rstrip("/")
        self.request_delay: float = float(request_delay)
        self.timeout: int = int(timeout)
        self.max_retries: int = int(max_retries)

        self._session: requests.Session = session or requests.Session()
        self._session.headers.update(config.DEFAULT_HEADERS)
        self._robots_checker: RobotsChecker = robots_checker or RobotsChecker(
            session=self._session,
            user_agent=config.USER_AGENT,
            timeout=self.timeout,
        )
        self._last_request_ts: float = 0.0
        self._blocked_urls: set[str] = set()

    @abstractmethod
    def discover_article_urls(self, max_urls: int = 150) -> list[str]:
        """Discover candidate news article URLs from the source.

        Args:
            max_urls: Maximum number of unique URLs to discover.

        Returns:
            List of normalized absolute article URLs.
        """

    @abstractmethod
    def parse_article(self, url: str) -> Optional[Article]:
        """Fetch and parse a single news article page into an `Article` instance.

        Args:
            url: Full HTTP/HTTPS URL of the news article.

        Returns:
            Populated `Article` instance if extraction succeeds, or `None` if invalid/unavailable.
        """

    def scrape_articles(self, max_articles: int = 100) -> list[Article]:
        """Discover URLs and scrape up to `max_articles` valid news articles.

        Args:
            max_articles: Target number of valid articles to collect from this source.

        Returns:
            List of validated `Article` objects.
        """
        if max_articles <= 0:
            return []

        logger.info(
            "Starting article discovery for source '%s' (target=%d)...",
            self.source_name,
            max_articles,
        )
        candidate_urls = self.discover_article_urls(max_urls=max_articles * 2)
        logger.info(
            "Discovered %d candidate URLs from '%s'.",
            len(candidate_urls),
            self.source_name,
        )

        collected: list[Article] = []
        seen_urls: set[str] = set()

        for url in candidate_urls:
            if len(collected) >= max_articles:
                break

            norm_url = self.normalize_url(url)
            if norm_url in seen_urls or norm_url in self._blocked_urls:
                continue
            seen_urls.add(norm_url)

            article = self.parse_article(norm_url)
            if article is not None:
                collected.append(article)
                if len(collected) % 10 == 0:
                    logger.info(
                        "[%s] Collected %d / %d articles.",
                        self.source_name,
                        len(collected),
                        max_articles,
                    )

        logger.info(
            "Finished scraping '%s': %d valid articles collected.",
            self.source_name,
            len(collected),
        )
        return collected

    def _respect_rate_limit(self) -> None:
        """Sleep if necessary to enforce the configured delay between HTTP requests."""
        if self.request_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        if self._last_request_ts > 0 and elapsed < self.request_delay:
            sleep_duration = self.request_delay - elapsed
            time.sleep(sleep_duration)

    def _fetch_html(self, url: str, check_robots: bool = True) -> Optional[str]:
        """Perform an HTTP GET request with retries, backoff, and `robots.txt` verification.

        Never retries permanent client errors such as HTTP 401, 403, or 404.

        Args:
            url: Target URL to request.
            check_robots: Whether to verify `robots.txt` allowance prior to fetching.

        Returns:
            Decoded response text if HTTP 200 is returned, otherwise `None`.
        """
        norm_url = self.normalize_url(url)
        if norm_url in self._blocked_urls:
            logger.debug("Skipping previously blocked/forbidden URL: %s", norm_url)
            return None

        if check_robots and not self._robots_checker.can_fetch(norm_url):
            logger.warning("Blocked by robots.txt policy: %s", norm_url)
            self._blocked_urls.add(norm_url)
            return None

        for attempt in range(1, self.max_retries + 1):
            self._respect_rate_limit()
            try:
                self._last_request_ts = time.monotonic()
                response = self._session.get(norm_url, timeout=self.timeout)
                status = response.status_code

                if status == 200:
                    response.encoding = response.apparent_encoding or "utf-8"
                    return response.text

                if status in (401, 403, 404, 410, 451):
                    logger.warning(
                        "Permanent HTTP %d on %s — aborting without retry.",
                        status,
                        norm_url,
                    )
                    self._blocked_urls.add(norm_url)
                    return None

                logger.warning(
                    "HTTP %d on %s (attempt %d/%d).",
                    status,
                    norm_url,
                    attempt,
                    self.max_retries,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Network exception fetching %s (attempt %d/%d): %s",
                    norm_url,
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                backoff = config.RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1))
                time.sleep(backoff)

        logger.error("Exhausted all %d attempts for URL: %s", self.max_retries, norm_url)
        return None

    @staticmethod
    def _create_soup(html_content: str) -> BeautifulSoup:
        """Create a BeautifulSoup DOM tree using `lxml` with automatic fallback to `html.parser`.

        Args:
            html_content: Raw HTML markup string.

        Returns:
            Parsed `BeautifulSoup` instance.
        """
        try:
            return BeautifulSoup(html_content, "lxml")
        except Exception:
            return BeautifulSoup(html_content, "html.parser")

    def normalize_url(self, url: str) -> str:
        """Normalize a relative or absolute URL and strip tracking query parameters/fragments.

        Args:
            url: Raw URL or path string.

        Returns:
            Normalized absolute HTTP/HTTPS URL string.
        """
        full_url = urljoin(self.base_url + "/", url.strip())
        parsed = urlparse(full_url)
        clean_path = re.sub(r"/{2,}", "/", parsed.path)
        return str(
            urlunparse(
                (
                    parsed.scheme.lower() or "https",
                    parsed.netloc.lower(),
                    clean_path,
                    "",
                    "",
                    "",
                )
            )
        )

    @staticmethod
    def _strip_noise_elements(container: Tag) -> None:
        """Remove non-article DOM elements (menus, scripts, ads, social widgets) in place.

        Args:
            container: BeautifulSoup `Tag` representing the article container.
        """
        for tag_name in (
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "iframe",
            "noscript",
            "svg",
            "button",
            "figure",
        ):
            for node in container.find_all(tag_name):
                node.decompose()

    @staticmethod
    def _extract_title(soup: BeautifulSoup, selectors: list[str]) -> Optional[str]:
        """Extract the article headline using a prioritized list of CSS selectors.

        Args:
            soup: Parsed BeautifulSoup document.
            selectors: Ordered list of CSS selectors to try.

        Returns:
            Cleaned headline string, or `None` if not found.
        """
        for selector in selectors:
            element = soup.select_one(selector)
            if isinstance(element, Tag):
                title_text = element.get_text(separator=" ", strip=True)
                if title_text:
                    return title_text

        og_title = soup.find("meta", attrs={"property": "og:title"})
        if isinstance(og_title, Tag):
            content_attr = og_title.get("content")
            if isinstance(content_attr, str) and content_attr.strip():
                return content_attr.strip()
        return None

    @classmethod
    def _extract_body_text(cls, soup: BeautifulSoup, selectors: list[str]) -> Optional[str]:
        """Extract and concatenate article body paragraphs while excluding navigation noise.

        Args:
            soup: Parsed BeautifulSoup document.
            selectors: Ordered list of CSS selectors identifying article body containers.

        Returns:
            Concatenated article body text, or `None` if extraction fails.
        """
        for selector in selectors:
            containers = soup.select(selector)
            if not containers:
                continue

            paragraphs: list[str] = []
            for container in containers:
                if not isinstance(container, Tag):
                    continue
                cls._strip_noise_elements(container)
                p_nodes = container.find_all("p")
                if p_nodes:
                    for p in p_nodes:
                        text = p.get_text(separator=" ", strip=True)
                        if len(text) >= 20:
                            paragraphs.append(text)
                else:
                    direct_text = container.get_text(separator=" ", strip=True)
                    if len(direct_text) >= 40:
                        paragraphs.append(direct_text)

            combined = " ".join(paragraphs).strip()
            if len(combined.split()) >= config.MIN_WORDS_PER_ARTICLE:
                return combined
        return None

    @staticmethod
    def _extract_date(soup: BeautifulSoup, selectors: list[str]) -> Optional[str]:
        """Extract article publication timestamp from HTML tags or meta attributes.

        Args:
            soup: Parsed BeautifulSoup document.
            selectors: Ordered list of CSS selectors for date/time elements.

        Returns:
            Publication date string if available, otherwise `None`.
        """
        for meta_prop in ("article:published_time", "og:published_time", "pubdate"):
            meta_node = soup.find("meta", attrs={"property": meta_prop}) or soup.find(
                "meta", attrs={"name": meta_prop}
            )
            if isinstance(meta_node, Tag):
                content_val = meta_node.get("content")
                if isinstance(content_val, str) and content_val.strip():
                    return content_val.strip()

        for selector in selectors:
            element = soup.select_one(selector)
            if isinstance(element, Tag):
                datetime_attr = element.get("datetime") or element.get("content")
                if isinstance(datetime_attr, str) and datetime_attr.strip():
                    return datetime_attr.strip()
                text = element.get_text(separator=" ", strip=True)
                if text:
                    return text
        return None