"""Базовый класс для парсинга новостных сайтов."""

import time
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

import config
from models.article import Article
from scraping.robots_checker import RobotsChecker


class BaseNewsScraper:
    """Базовый парсер с поддержкой сессий, задержек и BeautifulSoup."""

    def __init__(self, source_name: str, base_url: str) -> None:
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(config.DEFAULT_HEADERS)
        self.robots = RobotsChecker()

    def _fetch_html(self, url: str) -> Optional[str]:
        """Отправляет GET-запрос с соблюдением задержки и проверкой robots.txt."""
        if not self.robots.can_fetch(url):
            return None
        time.sleep(config.REQUEST_DELAY_SECONDS)
        for _ in range(config.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
                if response.status_code == 200:
                    response.encoding = response.apparent_encoding or "utf-8"
                    return response.text
                if response.status_code in (403, 404):
                    return None
            except requests.RequestException:
                time.sleep(0.5)
        return None

    @staticmethod
    def _make_soup(html_text: str) -> BeautifulSoup:
        """Создает объект BeautifulSoup с парсером lxml."""
        try:
            return BeautifulSoup(html_text, "lxml")
        except Exception:
            return BeautifulSoup(html_text, "html.parser")

    def discover_article_urls(self, max_urls: int = 150) -> List[str]:
        """Возвращает список ссылок на статьи."""
        raise NotImplementedError

    def parse_article(self, url: str) -> Optional[Article]:
        """Извлекает заголовок, текст и дату статьи по URL."""
        raise NotImplementedError