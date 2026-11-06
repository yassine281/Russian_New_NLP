"""Частотный анализ текста на базе collections.Counter (Задание 5, Шаг 1)."""

from collections import Counter
from typing import Iterable, List, Optional

from models.frequency import FrequencyEntry


class FrequencyAnalyzer:
    """Строит частотный словарь корпуса и определяет ранги слов."""

    def __init__(self, corpus_tokens: Optional[Iterable[Iterable[str]]] = None) -> None:
        self.counter: Counter = Counter()
        self.total_tokens: int = 0
        if corpus_tokens is not None:
            self.fit(corpus_tokens)

    def fit(self, corpus_tokens: Iterable[Iterable[str]]) -> None:
        """Подсчитывает частоту всех слов в корпусе."""
        self.counter.clear()
        self.total_tokens = 0
        for doc in corpus_tokens:
            clean_tokens = [w.strip().lower() for w in doc if w and w.strip()]
            self.counter.update(clean_tokens)
            self.total_tokens += len(clean_tokens)

    @property
    def vocabulary_size(self) -> int:
        """Возвращает число уникальных слов в словаре."""
        return len(self.counter)

    def get_ranked_entries(self, limit: Optional[int] = None) -> List[FrequencyEntry]:
        """Возвращает список FrequencyEntry, отсортированный по убыванию частоты."""
        most_common = self.counter.most_common(limit)
        return [
            FrequencyEntry(word=word, count=count, rank=idx)
            for idx, (word, count) in enumerate(most_common, start=1)
        ]

    def get_top_30(self) -> List[FrequencyEntry]:
        """Возвращает 30 наиболее частотных слов."""
        return self.get_ranked_entries(limit=30)

    def get_top_1000(self) -> List[FrequencyEntry]:
        """Возвращает 1000 наиболее частотных слов."""
        return self.get_ranked_entries(limit=1000)