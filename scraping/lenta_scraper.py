"""Парсер новостных статей для сайта Lenta.ru."""

from typing import Dict, List, Optional
import feedparser

import config
from models.article import Article
from scraping.base_scraper import BaseNewsScraper


class LentaScraper(BaseNewsScraper):
    """Извлекает новости с сайта Lenta.ru через RSS и HTML-страницы."""

    RSS_FEEDS = (
        "https://lenta.ru/rss/news",
        "https://lenta.ru/rss/articles",
        "https://lenta.ru/rss/top7",
    )

    def __init__(self) -> None:
        super().__init__(source_name="Lenta.ru", base_url="https://lenta.ru")
        self._rss_cache: Dict[str, Dict[str, str]] = {}

    def discover_article_urls(self, max_urls: int = 150) -> List[str]:
        """Собирает ссылки на статьи из RSS-лент Lenta.ru."""
        urls: List[str] = []
        seen = set()
        for rss_url in self.RSS_FEEDS:
            xml_text = self._fetch_html(rss_url)
            if not xml_text:
                continue
            feed = feedparser.parse(xml_text)
            for entry in feed.entries:
                link = getattr(entry, "link", "").strip()
                if link and link not in seen:
                    seen.add(link)
                    urls.append(link)
                    self._rss_cache[link] = {
                        "title": getattr(entry, "title", "").strip(),
                        "date": getattr(entry, "published", "").strip(),
                        "summary": getattr(entry, "summary", "").strip(),
                    }
                if len(urls) >= max_urls:
                    return urls
        return urls

    def parse_article(self, url: str) -> Optional[Article]:
        """Парсит страницу статьи на Lenta.ru."""
        html_text = self._fetch_html(url)
        meta = self._rss_cache.get(url, {})
        if not html_text:
            return None

        soup = self._make_soup(html_text)
        title_node = (
            soup.select_one("span.topic-body__title")
            or soup.find("h1")
        )
        title = title_node.get_text(strip=True) if title_node else meta.get("title", "")

        body_node = (
            soup.select_one("div.topic-body__content")
            or soup.find("article")
            or soup
        )
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in body_node.find_all("p")
            if len(p.get_text(strip=True)) >= 20
        ]
        text = " ".join(paragraphs).strip()
        if len(text.split()) < config.MIN_WORDS_PER_ARTICLE:
            text = meta.get("summary", "")

        if not title or len(text.split()) < config.MIN_WORDS_PER_ARTICLE:
            return None

        time_node = soup.find("time") or soup.select_one("a.topic-header__time")
        pub_date = time_node.get_text(strip=True) if time_node else meta.get("date")

        return Article(
            url=url,
            title=title,
            text=text,
            date=pub_date or None,
            source=self.source_name,
        )