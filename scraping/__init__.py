"""Пакет для сбора новостных статей."""

from scraping.base_scraper import BaseNewsScraper
from scraping.corpus_builder import CorpusBuilder
from scraping.lenta_scraper import LentaScraper
from scraping.robots_checker import RobotsChecker
from scraping.rss_scraper import RSSNewsScraper

__all__ = [
    "BaseNewsScraper",
    "CorpusBuilder",
    "LentaScraper",
    "RSSNewsScraper",
    "RobotsChecker",
]