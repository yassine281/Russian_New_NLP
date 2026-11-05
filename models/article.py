"""Domain data models for raw and preprocessed Russian news articles.

This module implements immutable dataclasses representing raw scraped news
articles (`Article`) and linguistically enriched documents (`ProcessedArticle`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Article:
    """Immutable representation of a raw Russian news article.

    Attributes:
        url: Source URL of the article.
        title: Headline title of the article.
        text: Full main body text of the article.
        date: Publication date string (ISO or source format), if available.
        source: Domain identifier of the news agency (e.g., 'lenta.ru').
    """

    url: str
    title: str
    text: str
    date: Optional[str] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Validate required fields upon instantiation.

        Raises:
            ValueError: If url, title, or text is empty or non-string.
        """
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("Article 'url' must be a non-empty string.")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Article 'title' must be a non-empty string.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Article 'text' must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Article 'source' must be a non-empty string.")

    @property
    def word_count(self) -> int:
        """Calculate the whitespace-delimited word count of the raw article text.

        Returns:
            Number of words in the article text.
        """
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Article instance into a JSON-compatible dictionary.

        Returns:
            Dictionary containing article attributes and computed word_count.
        """
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "date": self.date,
            "source": self.source,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Article:
        """Deserialize an Article instance from a dictionary.

        Args:
            data: Dictionary containing raw article fields.

        Returns:
            Validated Article instance.

        Raises:
            ValueError: If required keys are missing or invalid.
        """
        if not isinstance(data, dict):
            raise ValueError("Input data for Article.from_dict must be a dictionary.")
        return cls(
            url=str(data.get("url", "")).strip(),
            title=str(data.get("title", "")).strip(),
            text=str(data.get("text", "")).strip(),
            date=str(data["date"]).strip() if data.get("date") else None,
            source=str(data.get("source", "unknown")).strip() or "unknown",
        )

    def __repr__(self) -> str:
        """Return an unambiguous developer representation of the Article."""
        short_title = (
            self.title[:40] + "..." if len(self.title) > 40 else self.title
        )
        return (
            f"Article(source={self.source!r}, url={self.url!r}, "
            f"title={short_title!r}, word_count={self.word_count})"
        )

    def __str__(self) -> str:
        """Return a readable summary string of the Article."""
        date_part = f" [{self.date}]" if self.date else ""
        return f"[{self.source}]{date_part} {self.title} ({self.word_count} words)"


@dataclass(frozen=True)
class ProcessedArticle:
    """Immutable representation of an article after NLP preprocessing stages.

    Uses tuples internally to preserve true immutability while providing
    convenient factory and update methods via `dataclasses.replace()`.

    Attributes:
        article: Original immutable Article instance.
        clean_text: Normalized lowercase Russian text.
        tokens: Tokens produced by SimpleTokenizer.
        nltk_tokens: Tokens produced by NLTKTokenizer.
        spacy_tokens: Tokens produced by SpacyTokenizer.
        pymorphy_lemmas: Lemmas produced by PyMorphyLemmatizer.
        spacy_lemmas: Lemmas produced by SpacyLemmatizer.
        snowball_stems: Stems produced by SnowballStemmerWrapper.
        filtered_lemmas: Primary lemmas after stop-word removal.
    """

    article: Article
    clean_text: str = ""
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    nltk_tokens: Tuple[str, ...] = field(default_factory=tuple)
    spacy_tokens: Tuple[str, ...] = field(default_factory=tuple)
    pymorphy_lemmas: Tuple[str, ...] = field(default_factory=tuple)
    spacy_lemmas: Tuple[str, ...] = field(default_factory=tuple)
    snowball_stems: Tuple[str, ...] = field(default_factory=tuple)
    filtered_lemmas: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate types and convert any list sequences into immutable tuples.

        Raises:
            TypeError: If `article` is not an Article instance.
        """
        if not isinstance(self.article, Article):
            raise TypeError(
                f"Expected 'article' to be an Article instance, got {type(self.article).__name__}."
            )
        if not isinstance(self.clean_text, str):
            raise TypeError("'clean_text' must be a string.")

        # Coerce any passed lists to tuples to guarantee deep immutability
        sequence_fields = (
            "tokens",
            "nltk_tokens",
            "spacy_tokens",
            "pymorphy_lemmas",
            "spacy_lemmas",
            "snowball_stems",
            "filtered_lemmas",
        )
        for field_name in sequence_fields:
            val = getattr(self, field_name)
            if isinstance(val, list):
                object.__setattr__(self, field_name, tuple(val))
            elif not isinstance(val, tuple):
                raise TypeError(f"Field '{field_name}' must be a list or tuple of strings.")

    def with_updates(self, **kwargs: Any) -> ProcessedArticle:
        """Create a new ProcessedArticle with updated fields without mutating self.

        Args:
            **kwargs: Fields to update on the new ProcessedArticle copy.

        Returns:
            New ProcessedArticle instance with updated values.
        """
        return replace(self, **kwargs)

    def to_csv_row(self) -> Dict[str, str]:
        """Serialize fields required for corpus_processed.csv export.

        Returns:
            Dictionary with 'url', 'title', 'clean', and 'lemmas_filtered'.
        """
        return {
            "url": self.article.url,
            "title": self.article.title,
            "clean": self.clean_text,
            "lemmas_filtered": " ".join(self.filtered_lemmas),
        }

    def __repr__(self) -> str:
        """Return developer representation of the ProcessedArticle."""
        return (
            f"ProcessedArticle(url={self.article.url!r}, "
            f"tokens={len(self.tokens)}, "
            f"filtered_lemmas={len(self.filtered_lemmas)})"
        )

    def __str__(self) -> str:
        """Return readable summary of the ProcessedArticle."""
        return (
            f"ProcessedArticle({self.article.title[:35]!r}: "
            f"{len(self.tokens)} raw tokens -> {len(self.filtered_lemmas)} filtered lemmas)"
        )