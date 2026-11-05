"""Scraping package providing ethical news collection, RSS parsing, and corpus building."""

from scraping.base_scraper import BaseNewsScraper
from scraping.corpus_builder import CorpusBuilder, CorpusValidationReport
from scraping.lenta_scraper import LentaScraper
from scraping.robots_checker import RobotsChecker
from scraping.rss_scraper import DEFAULT_SOURCE_PROFILES, NewsSourceProfile, RSSNewsScraper

__all__ = [
    "BaseNewsScraper",
    "CorpusBuilder",
    "CorpusValidationReport",
    "DEFAULT_SOURCE_PROFILES",
    "LentaScraper",
    "NewsSourceProfile",
    "RSSNewsScraper",
    "RobotsChecker",
]