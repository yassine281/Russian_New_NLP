"""Пакет моделей данных проекта."""

from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry

__all__ = ["Article", "ProcessedArticle", "FrequencyEntry"]