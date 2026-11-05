"""Corpus builder coordinating multi-source collection, caching, deduplication, and validation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from models.article import Article
from scraping.base_scraper import BaseNewsScraper
from scraping.lenta_scraper import LentaScraper
from scraping.robots_checker import RobotsChecker
from scraping.rss_scraper import DEFAULT_SOURCE_PROFILES, RSSNewsScraper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusValidationReport:
    """Summary of corpus size validation against academic requirements."""

    total_articles: int
    total_words: int
    average_words_per_article: float
    min_articles_required: int
    min_words_required: int
    meets_article_target: bool
    meets_word_target: bool
    sources_breakdown: dict[str, int] = field(default_factory=dict)
    source_failures: list[str] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        """Return True if both article count and total word count targets are satisfied."""
        return self.meets_article_target and self.meets_word_target


class CorpusBuilder:
    """Orchestrates news scraping across multiple Russian news sources with disk caching."""

    def __init__(
        self,
        scrapers: Optional[list[BaseNewsScraper]] = None,
        raw_corpus_path: Path = config.RAW_CORPUS_JSON_PATH,
        cache_dir: Path = config.CACHE_DIR,
        min_articles: int = config.MIN_CORPUS_ARTICLES,
        min_words: int = config.MIN_CORPUS_WORDS,
    ) -> None:
        """Initialize the CorpusBuilder.

        Args:
            scrapers: Optional list of custom `BaseNewsScraper` instances.
            raw_corpus_path: Path where the final `corpus.json` file is saved.
            cache_dir: Directory path for intermediate article caching.
            min_articles: Minimum number of articles required (default 100)[cite: 5].
            min_words: Minimum total words required across the corpus (default 30,000)[cite: 5].
        """
        self.raw_corpus_path: Path = Path(raw_corpus_path)
        self.cache_dir: Path = Path(cache_dir)
        self.cache_file_path: Path = self.cache_dir / "articles_cache.json"
        self.min_articles: int = int(min_articles)
        self.min_words: int = int(min_words)

        self.raw_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if scrapers is not None:
            self.scrapers: list[BaseNewsScraper] = scrapers
        else:
            shared_robots = RobotsChecker()
            self.scrapers = [
                LentaScraper(robots_checker=shared_robots),
                RSSNewsScraper.from_domain("ria.ru", robots_checker=shared_robots),
                RSSNewsScraper.from_domain("gazeta.ru", robots_checker=shared_robots),
                RSSNewsScraper.from_domain("tass.ru", robots_checker=shared_robots),
            ]

        self._source_failures: list[str] = []

    def build_corpus(
        self,
        use_cache: bool = True,
        max_articles_per_source: int = 80,
    ) -> tuple[list[Article], CorpusValidationReport]:
        """Collect articles across configured sources until academic targets are met.

        Args:
            use_cache: Whether to preload already cached articles from disk first.
            max_articles_per_source: Maximum articles to request from each individual scraper.

        Returns:
            Tuple of `(articles, validation_report)`.
        """
        articles: list[Article] = []
        seen_urls: set[str] = set()
        seen_fingerprints: set[str] = set()
        self._source_failures.clear()

        if use_cache:
            cached_articles = self.load_cached_articles()
            for article in cached_articles:
                self._register_unique_article(
                    article, articles, seen_urls, seen_fingerprints
                )
            if articles:
                logger.info(
                    "Preloaded %d unique articles (%d words) from local cache.",
                    len(articles),
                    sum(a.word_count for a in articles),
                )

        for scraper in self.scrapers:
            total_words = sum(a.word_count for a in articles)
            if len(articles) >= self.min_articles and total_words >= self.min_words:
                logger.info(
                    "Academic targets reached (%d articles, %d words). Stopping scraping.",
                    len(articles),
                    total_words,
                )
                break

            needed = max(self.min_articles - len(articles) + 25, max_articles_per_source)
            try:
                scraped_batch = scraper.scrape_articles(max_articles=needed)
                added_count = 0
                for article in scraped_batch:
                    if self._register_unique_article(
                        article, articles, seen_urls, seen_fingerprints
                    ):
                        added_count += 1

                if added_count == 0:
                    msg = f"{scraper.source_name}: 0 new valid articles extracted."
                    logger.warning(msg)
                    self._source_failures.append(msg)
                else:
                    logger.info(
                        "Added %d unique articles from %s (running total: %d articles, %d words).",
                        added_count,
                        scraper.source_name,
                        len(articles),
                        sum(a.word_count for a in articles),
                    )
                    self._save_cache(articles)
            except Exception as exc:
                failure_msg = f"{scraper.source_name} failed with error: {exc}"
                logger.error(failure_msg)
                self._source_failures.append(failure_msg)

        if articles:
            self.save_raw_corpus(articles)

        report = self.validate_corpus(articles)
        return articles, report

    def validate_corpus(self, articles: list[Article]) -> CorpusValidationReport:
        """Evaluate collected articles against the minimum article and word count targets.

        Args:
            articles: List of collected `Article` objects.

        Returns:
            Populated `CorpusValidationReport`.
        """
        total_articles = len(articles)
        total_words = sum(article.word_count for article in articles)
        avg_words = total_words / total_articles if total_articles > 0 else 0.0

        breakdown: dict[str, int] = {}
        for article in articles:
            breakdown[article.source] = breakdown.get(article.source, 0) + 1

        report = CorpusValidationReport(
            total_articles=total_articles,
            total_words=total_words,
            average_words_per_article=avg_words,
            min_articles_required=self.min_articles,
            min_words_required=self.min_words,
            meets_article_target=total_articles >= self.min_articles,
            meets_word_target=total_words >= self.min_words,
            sources_breakdown=breakdown,
            source_failures=list(self._source_failures),
        )

        if report.is_compliant:
            logger.info(
                "Corpus validation PASSED: %d articles, %d total words (avg %.1f words/article).",
                report.total_articles,
                report.total_words,
                report.average_words_per_article,
            )
        else:
            logger.warning(
                "Corpus is BELOW academic target: %d/%d articles, %d/%d words. Failures: %s",
                report.total_articles,
                report.min_articles_required,
                report.total_words,
                report.min_words_required,
                report.source_failures,
            )
        return report

    def save_raw_corpus(
        self,
        articles: list[Article],
        target_path: Optional[Path] = None,
    ) -> Path:
        """Serialize the collected articles to structured UTF-8 JSON format[cite: 5].

        Args:
            articles: List of `Article` objects to save.
            target_path: Optional override path for the JSON output.

        Returns:
            Path to the written JSON file.
        """
        out_path = Path(target_path) if target_path else self.raw_corpus_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [article.to_dict() for article in articles]
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Saved %d raw articles to %s", len(articles), out_path)
        return out_path

    def load_raw_corpus(self, source_path: Optional[Path] = None) -> list[Article]:
        """Load previously saved articles from `corpus.json`.

        Args:
            source_path: Optional path to the JSON corpus file.

        Returns:
            List of validated `Article` instances.

        Raises:
            FileNotFoundError: If the corpus JSON file does not exist.
        """
        in_path = Path(source_path) if source_path else self.raw_corpus_path
        if not in_path.exists():
            raise FileNotFoundError(f"Raw corpus file not found at: {in_path}")

        with in_path.open("r", encoding="utf-8") as fh:
            raw_list = json.load(fh)

        if not isinstance(raw_list, list):
            raise ValueError(f"Expected JSON list in {in_path}, got {type(raw_list).__name__}")

        loaded: list[Article] = []
        for item in raw_list:
            if isinstance(item, dict):
                try:
                    loaded.append(Article.from_dict(item))
                except ValueError as exc:
                    logger.warning("Skipping invalid article record in %s: %s", in_path, exc)

        logger.info("Loaded %d articles from %s", len(loaded), in_path)
        return loaded

    def load_cached_articles(self) -> list[Article]:
        """Load cached articles from `data/cache/articles_cache.json` if present."""
        if not self.cache_file_path.exists():
            return []
        try:
            return self.load_raw_corpus(source_path=self.cache_file_path)
        except Exception as exc:
            logger.warning("Failed to read article cache (%s): %s", self.cache_file_path, exc)
            return []

    def _save_cache(self, articles: list[Article]) -> None:
        """Persist intermediate articles to the cache directory."""
        try:
            self.save_raw_corpus(articles, target_path=self.cache_file_path)
        except Exception as exc:
            logger.warning("Could not update article cache: %s", exc)

    @staticmethod
    def _compute_fingerprint(text: str) -> str:
        """Compute a SHA-256 fingerprint of normalized article text to prevent duplicate content."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _register_unique_article(
        cls,
        article: Article,
        target_list: list[Article],
        seen_urls: set[str],
        seen_fingerprints: set[str],
    ) -> bool:
        """Add an article to `target_list` only if its URL and text fingerprint are unique."""
        norm_url = article.url.strip().rstrip("/").lower()
        fingerprint = cls._compute_fingerprint(article.text)
        if norm_url in seen_urls or fingerprint in seen_fingerprints:
            return False
        seen_urls.add(norm_url)
        seen_fingerprints.add(fingerprint)
        target_list.append(article)
        return True