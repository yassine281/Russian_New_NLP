"""Универсальный RSS и HTML парсер для РИА Новости, Газета.Ru и ТАСС."""

from typing import Dict, List, Optional
import feedparser

import config
from models.article import Article
from scraping.base_scraper import BaseNewsScraper


class RSSNewsScraper(BaseNewsScraper):
    """Собирает статьи из RSS-лент и HTML-страниц российских СМИ."""

    def __init__(self, source_name: str, base_url: str, rss_url: str) -> None:
        super().__init__(source_name=source_name, base_url=base_url)
        self.rss_url = rss_url
        self._rss_cache: Dict[str, Dict[str, str]] = {}

    def discover_article_urls(self, max_urls: int = 150) -> List[str]:
        """Извлекает ссылки на новости из RSS-ленты."""
        urls: List[str] = []
        xml_text = self._fetch_html(self.rss_url)
        if not xml_text:
            return urls

        feed = feedparser.parse(xml_text)
        for entry in feed.entries:
            link = getattr(entry, "link", "").strip()
            if not link:
                continue
            urls.append(link)
            self._rss_cache[link] = {
                "title": getattr(entry, "title", "").strip(),
                "date": getattr(entry, "published", "").strip(),
                "summary": getattr(entry, "summary", "").strip(),
            }
            if len(urls) >= max_urls:
                break
        return urls

    def parse_article(self, url: str) -> Optional[Article]:
        """Извлекает полный текст статьи со страницы или из RSS."""
        html_text = self._fetch_html(url)
        meta = self._rss_cache.get(url, {})
        title = meta.get("title", "")
        pub_date = meta.get("date", "")
        text = ""

        if html_text:
            soup = self._make_soup(html_text)
            if not title:
                h1 = soup.find("h1") or soup.select_one("div.article__title")
                if h1:
                    title = h1.get_text(strip=True)

            # Поддержка блоков РИА Новости (div.article__text) и обычных тегов <p>
            ria_blocks = soup.select("div.article__text")
            if ria_blocks:
                chunks = [b.get_text(" ", strip=True) for b in ria_blocks if len(b.get_text(strip=True)) >= 20]
            else:
                chunks = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) >= 20]
            text = " ".join(chunks).strip()

        if len(text.split()) < config.MIN_WORDS_PER_ARTICLE:
            text = meta.get("summary", "")

        if not title or len(text.split()) < config.MIN_WORDS_PER_ARTICLE:
            return None

        return Article(
            url=url,
            title=title,
            text=text,
            date=pub_date or None,
            source=self.source_name,
        )