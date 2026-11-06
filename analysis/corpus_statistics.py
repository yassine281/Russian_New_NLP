"""Сводная статистика новостного корпуса."""

from typing import Any, Dict, Sequence
from models.article import Article


class CorpusStatistics:
    """Вычисляет базовые показатели собранного корпуса (Задание 1, Шаг 6)."""

    @staticmethod
    def compute(articles: Sequence[Article], stopword_stats: Dict[str, Any], vocab_size: int) -> Dict[str, Any]:
        lengths = [a.word_count for a in articles]
        total_words = sum(lengths)
        avg_words = (total_words / len(lengths)) if lengths else 0.0

        return {
            "total_articles": len(articles),
            "total_words": total_words,
            "average_words": avg_words,
            "min_words": min(lengths) if lengths else 0,
            "max_words": max(lengths) if lengths else 0,
            "vocab_size": vocab_size,
            "tokens_before_stopwords": stopword_stats.get("tokens_before", 0),
            "tokens_after_stopwords": stopword_stats.get("tokens_after", 0),
            "tokens_removed": stopword_stats.get("tokens_removed", 0),
            "percentage_removed": stopword_stats.get("percentage_removed", 0.0),
        }