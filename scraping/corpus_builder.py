"""Сборщик корпуса новостных текстов и сохранение в JSON."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import config
from models.article import Article
from scraping.base_scraper import BaseNewsScraper
from scraping.lenta_scraper import LentaScraper
from scraping.rss_scraper import RSSNewsScraper

logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Координирует сбор статей из нескольких источников до достижения 100 статей и 30 000 слов."""

    def __init__(
        self,
        scrapers: Optional[List[BaseNewsScraper]] = None,
        raw_corpus_path: Path = config.RAW_CORPUS_JSON_PATH,
    ) -> None:
        self.raw_corpus_path = Path(raw_corpus_path)
        self.scrapers = scrapers or [
            LentaScraper(),
            RSSNewsScraper("РИА Новости", "https://ria.ru", "https://ria.ru/export/rss2/archive/index.xml"),
            RSSNewsScraper("Газета.Ru", "https://www.gazeta.ru", "https://www.gazeta.ru/export/rss/lenta.xml"),
            RSSNewsScraper("ТАСС", "https://tass.ru", "https://tass.ru/rss/v2.xml"),
        ]

    def build_corpus(self, target_articles: int = config.MIN_CORPUS_ARTICLES) -> Tuple[List[Article], bool]:
        """Собирает корпус новостных статей и проверяет его объем."""
        articles: List[Article] = []
        seen_urls = set()

        for scraper in self.scrapers:
            total_words = sum(a.word_count for a in articles)
            if len(articles) >= target_articles and total_words >= config.MIN_CORPUS_WORDS:
                break

            logger.info("Сбор ссылок из источника: %s", scraper.source_name)
            urls = scraper.discover_article_urls(max_urls=150)

            for url in urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                article = scraper.parse_article(url)
                if article:
                    articles.append(article)
                    if len(articles) % 10 == 0:
                        logger.info(
                            "Обработано статей: %d (слов: %d)",
                            len(articles),
                            sum(a.word_count for a in articles),
                        )

                total_words = sum(a.word_count for a in articles)
                if len(articles) >= target_articles and total_words >= config.MIN_CORPUS_WORDS:
                    break

        if articles:
            self.save_raw_corpus(articles)

        total_words = sum(a.word_count for a in articles)
        is_compliant = (
            len(articles) >= config.MIN_CORPUS_ARTICLES
            and total_words >= config.MIN_CORPUS_WORDS
        )
        return articles, is_compliant

    def save_raw_corpus(self, articles: List[Article]) -> Path:
        """Сохраняет собранные статьи в файл corpus.json (UTF-8)."""
        self.raw_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [a.to_dict() for a in articles]
        with open(self.raw_corpus_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Корпус сохранен в %s (%d статей)", self.raw_corpus_path, len(articles))
        return self.raw_corpus_path

    def load_raw_corpus(self) -> List[Article]:
        """Загружает ранее сохраненный корпус из corpus.json."""
        if not self.raw_corpus_path.exists():
            return []
        with open(self.raw_corpus_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [Article.from_dict(item) for item in data if isinstance(item, dict)]