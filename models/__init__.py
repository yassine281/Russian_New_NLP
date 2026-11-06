"""Paquet models exposant les entites de domaine Article, ProcessedArticle et FrequencyEntry."""

from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry

__all__ = [
    "Article",
    "FrequencyEntry",
    "ProcessedArticle",
]