"""Domain data model for lexical frequency and rank entries.

This module provides the immutable `FrequencyEntry` dataclass used in
frequency dictionaries, Zipf's law regression, and report generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FrequencyEntry:
    """Immutable entry representing a word's frequency and rank in the corpus.

    Attributes:
        word: The normalized lemma or token string.
        count: Number of occurrences in the corpus (must be >= 1).
        rank: Statistical rank ordered by descending frequency (must be >= 1).
    """

    word: str
    count: int
    rank: int

    def __post_init__(self) -> None:
        """Validate frequency entry constraints.

        Raises:
            ValueError: If word is empty, count < 1, or rank < 1.
        """
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("FrequencyEntry 'word' must be a non-empty string.")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("FrequencyEntry 'count' must be a positive integer >= 1.")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("FrequencyEntry 'rank' must be a positive integer >= 1.")

    def to_tsv_line(self) -> str:
        """Format the entry as a tab-separated line for frequency_dict.txt.

        Returns:
            String formatted as 'word\\tcount'.
        """
        return f"{self.word}\t{self.count}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry into a dictionary representation.

        Returns:
            Dictionary with rank, word, and count.
        """
        return {
            "rank": self.rank,
            "word": self.word,
            "count": self.count,
        }

    def __repr__(self) -> str:
        """Return developer representation of FrequencyEntry."""
        return f"FrequencyEntry(rank={self.rank}, word={self.word!r}, count={self.count})"

    def __str__(self) -> str:
        """Return human-readable rank, word, and frequency."""
        return f"#{self.rank}: {self.word} ({self.count})"