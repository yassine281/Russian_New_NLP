"""Counter-based lexical frequency analyzer."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from models.frequency import FrequencyEntry


class FrequencyAnalyzer:
    """Computes word frequency distributions and rank tables across a tokenized corpus."""

    def __init__(self, corpus_tokens: Iterable[Iterable[str]] | None = None) -> None:
        """Initialize the analyzer optionally with a corpus of token sequences.

        Args:
            corpus_tokens: Optional iterable of token lists per document.
        """
        self._counter: Counter[str] = Counter()
        self._total_tokens: int = 0
        if corpus_tokens is not None:
            self.fit(corpus_tokens)

    def fit(self, corpus_tokens: Iterable[Iterable[str]]) -> FrequencyAnalyzer:
        """Populate frequency counts from a collection of document token lists.

        Args:
            corpus_tokens: Iterable of token lists.

        Returns:
            Self for method chaining.
        """
        self._counter.clear()
        total = 0
        for doc in corpus_tokens:
            cleaned_doc = [w.strip().lower() for w in doc if w and w.strip()]
            self._counter.update(cleaned_doc)
            total += len(cleaned_doc)
        self._total_tokens = total
        return self

    @property
    def total_tokens(self) -> int:
        """Return total number of tokens processed."""
        return self._total_tokens

    @property
    def vocabulary_size(self) -> int:
        """Return number of unique words in the vocabulary[cite: 8]."""
        return len(self._counter)

    @property
    def counter(self) -> Counter[str]:
        """Return a copy of the underlying Counter object."""
        return Counter(self._counter)

    def get_ranked_entries(self, limit: int | None = None) -> list[FrequencyEntry]:
        """Return `FrequencyEntry` objects ordered by descending frequency and ascending rank[cite: 8].

        Args:
            limit: Maximum number of top entries to return (`None` returns all).

        Returns:
            List of ranked `FrequencyEntry` records.
        """
        most_common = self._counter.most_common(limit)
        return [
            FrequencyEntry(word=word, count=count, rank=idx)
            for idx, (word, count) in enumerate(most_common, start=1)
        ]

    def get_top_n(self, n: int = 30) -> list[FrequencyEntry]:
        """Return the top `n` most frequent words[cite: 8].

        Args:
            n: Number of top entries to retrieve (e.g., 30 or 1000).

        Returns:
            List of top `n` `FrequencyEntry` items.
        """
        if n <= 0:
            return []
        return self.get_ranked_entries(limit=n)

    def get_top_30(self) -> list[FrequencyEntry]:
        """Return the top 30 most frequent words[cite: 8]."""
        return self.get_top_n(30)

    def get_top_1000(self) -> list[FrequencyEntry]:
        """Return the top 1000 most frequent words for `frequency_dict.txt` export."""
        return self.get_top_n(1000)