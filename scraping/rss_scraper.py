"""Multi-source RSS and HTML news scraper for RIA Novosti, Gazeta.ru, and TASS."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import feedparser
from bs4 import Tag

import config
from models.article import Article
from scraping.base_scraper import BaseNewsScraper
from scraping.robots_checker import RobotsChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsSourceProfile:
    """Configuration profile describing an RSS-backed Russian news source."""

    source_name: str
    base_url: str
    rss_urls: tuple[str, ...]
    title_selectors: tuple[str, ...]
    body_selectors: tuple[str, ...]
    date_selectors: tuple[str, ...]


DEFAULT_SOURCE_PROFILES: dict[str, NewsSourceProfile] = {
    "ria.ru": NewsSourceProfile(
        source_name="РИА Новости",
        base_url="https://ria.ru",
        rss_urls=("https://ria.ru/export/rss2/archive/index.xml",),
        title_selectors=(
            "div.article__title",
            "h1.article__title",
            "h1",
        ),
        body_selectors=(
            "div.article__body",
            "div.article__text",
            "article",
        ),
        date_selectors=(
            "div.article__info-date a",
            "time",
        ),
    ),
    "gazeta.ru": NewsSourceProfile(
        source_name="Газета.Ru",
        base_url="https://www.gazeta.ru",
        rss_urls=("https://www.gazeta.ru/export/rss/lenta.xml",),
        title_selectors=(
            "h1.headline",
            "div.b_article-header h1",
            "h1",
        ),
        body_selectors=(
            "div.b_article-text",
            "div.article-text-body",
            "article",
        ),
        date_selectors=(
            "time.time",
            "time",
        ),
    ),
    "tass.ru": NewsSourceProfile(
        source_name="ТАСС",
        base_url="https://tass.ru",
        rss_urls=("https://tass.ru/rss/v2.xml",),
        title_selectors=(
            "h1.ds_ext_title",
            "div.news-header__title h1",
            "h1",
        ),
        body_selectors=(
            "article",
            "div.text-content",
            "div.news-content",
        ),
        date_selectors=(
            "time",
            "span.ds_ext_date",
        ),
    ),
}


class RSSNewsScraper(BaseNewsScraper):
    """Flexible RSS + HTML article scraper supporting RIA.ru, Gazeta.ru, and TASS.ru."""

    def __init__(
        self,
        profile: NewsSourceProfile,
        robots_checker: Optional[RobotsChecker] = None,
        request_delay: float = config.REQUEST_DELAY_SECONDS,
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
    ) -> None:
        """Initialize the RSS news scraper for a specific source profile.

        Args:
            profile: `NewsSourceProfile` with RSS endpoints and DOM selectors.
            robots_checker: Optional shared RobotsChecker instance.
            request_delay: Delay in seconds between HTTP requests.
            timeout: Socket timeout in seconds.
            max_retries: Retry attempts for temporary errors.
        """
        super().__init__(
            source_name=profile.source_name,
            base_url=profile.base_url,
            robots_checker=robots_checker,
            request_delay=request_delay,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.profile: NewsSourceProfile = profile
        self._entry_cache: dict[str, dict[str, str]] = {}

    @classmethod
    def from_domain(
        cls,
        domain_key: str,
        robots_checker: Optional[RobotsChecker] = None,
    ) -> RSSNewsScraper:
        """Create an `RSSNewsScraper` from a registered domain key.

        Args:
            domain_key: One of 'ria.ru', 'gazeta.ru', or 'tass.ru'.
            robots_checker: Optional shared RobotsChecker instance.

        Returns:
            Configured `RSSNewsScraper` instance.

        Raises:
            KeyError: If `domain_key` is not in `DEFAULT_SOURCE_PROFILES`.
        """
        if domain_key not in DEFAULT_SOURCE_PROFILES:
            raise KeyError(
                f"Unknown source domain {domain_key!r}. "
                f"Available profiles: {list(DEFAULT_SOURCE_PROFILES.keys())}"
            )
        return cls(
            profile=DEFAULT_SOURCE_PROFILES[domain_key],
            robots_checker=robots_checker,
        )

    def discover_article_urls(self, max_urls: int = 150) -> list[str]:
        """Discover article URLs from the profile's RSS feeds.

        Args:
            max_urls: Maximum number of unique URLs to return.

        Returns:
            List of normalized article URLs.
        """
        urls: list[str] = []
        seen: set[str] = set()

        for rss_url in self.profile.rss_urls:
            if len(urls) >= max_urls:
                break

            xml_content = self._fetch_html(rss_url, check_robots=True)
            if not xml_content:
                logger.warning(
                    "[%s] Could not retrieve RSS feed: %s",
                    self.source_name,
                    rss_url,
                )
                continue

            feed = feedparser.parse(xml_content)
            for entry in feed.entries:
                link = getattr(entry, "link", "") or ""
                if not isinstance(link, str) or not link.strip():
                    continue
                norm_url = self.normalize_url(link)
                if norm_url in seen:
                    continue
                seen.add(norm_url)
                urls.append(norm_url)

                # Extract any embedded full text or summary provided directly in the RSS feed
                rss_body = ""
                if hasattr(entry, "content") and entry.content:
                    first_content = entry.content[0]
                    if isinstance(first_content, dict):
                        rss_body = str(first_content.get("value", ""))
                if not rss_body:
                    rss_body = str(
                        getattr(entry, "summary", "")
                        or getattr(entry, "description", "")
                        or ""
                    )

                self._entry_cache[norm_url] = {
                    "title": str(getattr(entry, "title", "") or "").strip(),
                    "published": str(getattr(entry, "published", "") or "").strip(),
                    "rss_body": self._clean_rss_html(rss_body),
                }

                if len(urls) >= max_urls:
                    break

        return urls

    def parse_article(self, url: str) -> Optional[Article]:
        """Fetch and parse an article page, falling back to RSS full-text if HTML is restricted.

        Args:
            url: Normalized article URL.

        Returns:
            Populated `Article` instance or `None` if extraction fails.
        """
        norm_url = self.normalize_url(url)
        cached = self._entry_cache.get(norm_url, {})

        html = self._fetch_html(norm_url, check_robots=True)
        title: Optional[str] = None
        body_text: Optional[str] = None
        pub_date: Optional[str] = None

        if html:
            soup = self._create_soup(html)
            title = self._extract_title(soup, list(self.profile.title_selectors))
            body_text = self._extract_Ria_or_generic_body(soup)
            pub_date = self._extract_date(soup, list(self.profile.date_selectors))

        if not title:
            title = cached.get("title")
        if not pub_date:
            pub_date = cached.get("published")
        if not body_text or len(body_text.split()) < config.MIN_WORDS_PER_ARTICLE:
            rss_body = cached.get("rss_body", "")
            if len(rss_body.split()) >= config.MIN_WORDS_PER_ARTICLE:
                body_text = rss_body

        if not title or not body_text:
            return None

        try:
            article = Article(
                url=norm_url,
                title=title,
                text=body_text,
                date=pub_date or None,
                source=self.source_name,
            )
        except ValueError:
            return None

        if article.word_count < config.MIN_WORDS_PER_ARTICLE:
            return None

        return article

    def _extract_Ria_or_generic_body(self, soup: Tag) -> Optional[str]:
        """Extract article text supporting RIA.ru `div.article__text` blocks and generic `<p>` tags.

        Args:
            soup: Parsed BeautifulSoup document.

        Returns:
            Extracted body text or `None`.
        """
        # RIA Novosti frequently places paragraphs inside <div class="article__text">
        ria_blocks = soup.select("div.article__text")
        if ria_blocks:
            chunks = [
                blk.get_text(separator=" ", strip=True)
                for blk in ria_blocks
                if isinstance(blk, Tag) and len(blk.get_text(strip=True)) >= 20
            ]
            combined_ria = " ".join(chunks).strip()
            if len(combined_ria.split()) >= config.MIN_WORDS_PER_ARTICLE:
                return combined_ria

        return self._extract_body_text(soup, list(self.profile.body_selectors))

    def _clean_rss_html(self, raw_fragment: str) -> str:
        """Strip HTML tags from an RSS summary/content payload.

        Args:
            raw_fragment: Raw string from RSS entry content or summary.

        Returns:
            Plain text string.
        """
        if not raw_fragment or not raw_fragment.strip():
            return ""
        soup = self._create_soup(raw_fragment)
        return soup.get_text(separator=" ", strip=True)