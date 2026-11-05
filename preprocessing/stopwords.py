"""Russian stop-word management and corpus filtering metrics.

Academic Control Question Reference (Q6):
    Q6 (What are stop words, and why are they removed?):
        Stop words are high-frequency functional words—such as prepositions ('в', 'на'),
        conjunctions ('и', 'что'), particles ('не', 'бы'), pronouns ('это', 'который',
        'свой'), and auxiliary verbs—that serve grammatical connectivity rather than
        carrying topical semantic meaning[cite: 3, 7]. Removing them reduces corpus noise and
        dimensionality so frequency dictionaries and word clouds reflect genuine subject-matter
        concepts rather than functional syntax[cite: 7, 8].
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import nltk
from nltk.corpus import stopwords

import config

logger = logging.getLogger(__name__)

DEFAULT_CUSTOM_STOPWORDS: tuple[str, ...] = (
    "это",
    "который",
    "свой",
    "весь",
    "год",
    "также",
    "мочь",
    "очень",
    "еще",
    "ещё",
    "такой",
    "один",
    "другой",
    "наш",
    "ваш",
    "самый",
    "день",
    "время",
    "человек",
    "стать",
    "говорить",
    "сообщить",
    "заявить",
    "рассказать",
    "отметить",
    "риа",
    "тасс",
    "лента",
)


@dataclass(frozen=True)
class StopWordStats:
    """Statistical summary of token counts before and after stop-word removal."""

    tokens_before: int
    tokens_after: int
    tokens_removed: int
    percentage_removed: float
    unique_stopwords_removed: int
    total_stopword_list_size: int


class StopWordFilter:
    """Filters Russian stop words and short non-lexical tokens from lemma lists[cite: 7]."""

    CYRILLIC_WORD_RE: re.Pattern[str] = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$")

    def __init__(
        self,
        custom_stopwords: Optional[Iterable[str]] = None,
        min_word_length: int = 3,
    ) -> None:
        """Initialize the Russian stop-word filter[cite: 7].

        Args:
            custom_stopwords: Additional domain-specific Russian stop words[cite: 7].
            min_word_length: Minimum character length for retained lemmas (default 3, i.e. `len > 2`)[cite: 7].
        """
        self.min_word_length: int = max(1, int(min_word_length))
        base_words = self._load_nltk_russian_stopwords()
        configured_custom = getattr(
            config, "CUSTOM_RUSSIAN_STOPWORDS", DEFAULT_CUSTOM_STOPWORDS
        )
        extra_words = set(configured_custom)
        if custom_stopwords is not None:
            extra_words.update(w.strip().lower() for w in custom_stopwords if w.strip())

        self._stop_words: set[str] = base_words | extra_words
        self._encountered_stopwords: set[str] = set()

    @staticmethod
    def _load_nltk_russian_stopwords() -> set[str]:
        """Load NLTK's Russian stop-word list, downloading it if missing[cite: 7]."""
        try:
            return set(stopwords.words("russian"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("russian"))
            except Exception as exc:
                logger.warning("Could not download NLTK stopwords (%s); using fallback set.", exc)
                return {
                    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
                    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
                    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
                    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
                }

    @property
    def stop_words(self) -> frozenset[str]:
        """Return an immutable view of the extended stop-word set."""
        return frozenset(self._stop_words)

    def get_sorted_stopwords(self) -> list[str]:
        """Return the complete extended stop-word list sorted alphabetically."""
        return sorted(self._stop_words)

    def filter_lemmas(self, lemmas: list[str]) -> list[str]:
        """Remove stop words, short tokens (`len <= 2`), and non-Cyrillic items[cite: 7].

        Args:
            lemmas: List of normalized Russian lemmas.

        Returns:
            Filtered list of content-bearing Russian lemmas.
        """
        filtered: list[str] = []
        for lemma in lemmas:
            word = lemma.strip().lower()
            if not word:
                continue
            if word in self._stop_words:
                self._encountered_stopwords.add(word)
                continue
            if len(word) < self.min_word_length:
                continue
            if not self.CYRILLIC_WORD_RE.match(word):
                continue
            filtered.append(word)
        return filtered

    def filter_corpus(
        self,
        corpus_lemmas: list[list[str]],
    ) -> tuple[list[list[str]], StopWordStats]:
        """Filter stop words across all documents and compute removal statistics[cite: 7, 8].

        Formula: `Percentage removed = (removed / before) * 100`.

        Args:
            corpus_lemmas: List of lemma lists (one list per article).

        Returns:
            Tuple of `(filtered_corpus_lemmas, StopWordStats)`.
        """
        self._encountered_stopwords.clear()
        filtered_corpus: list[list[str]] = []
        total_before = 0
        total_after = 0

        for doc_lemmas in corpus_lemmas:
            total_before += len(doc_lemmas)
            doc_filtered = self.filter_lemmas(doc_lemmas)
            total_after += len(doc_filtered)
            filtered_corpus.append(doc_filtered)

        removed = total_before - total_after
        percentage = (removed / total_before) * 100.0 if total_before > 0 else 0.0

        stats = StopWordStats(
            tokens_before=total_before,
            tokens_after=total_after,
            tokens_removed=removed,
            percentage_removed=percentage,
            unique_stopwords_removed=len(self._encountered_stopwords),
            total_stopword_list_size=len(self._stop_words),
        )
        return filtered_corpus, stats