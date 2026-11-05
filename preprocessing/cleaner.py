"""Russian text cleaning and normalization module."""

from __future__ import annotations

import html
import re


class TextCleaner:
    """Performs modular cleaning and normalization on raw Russian news text."""

    HTML_TAG_PATTERN: re.Pattern[str] = re.compile(r"<[^>]+>")
    URL_PATTERN: re.Pattern[str] = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
    EMAIL_PATTERN: re.Pattern[str] = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    # Preserves Cyrillic, digits, whitespace, and hyphens (strict lexical mode)
    STRICT_CLEAN_PATTERN: re.Pattern[str] = re.compile(r"[^а-яА-ЯёЁ0-9\s\-]")
    # Preserves Cyrillic, digits, whitespace, hyphens, and standard sentence punctuation
    PUNCT_PRESERVING_PATTERN: re.Pattern[str] = re.compile(
        r"[^а-яА-ЯёЁ0-9\s\-\.,!?:;«»\"'()]"
    )
    STANDALONE_HYPHEN_PATTERN: re.Pattern[str] = re.compile(r"(?<!\w)-+|-(?!\w)")
    MULTI_HYPHEN_PATTERN: re.Pattern[str] = re.compile(r"-{2,}")
    WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")

    @classmethod
    def remove_html(cls, text: str) -> str:
        """Unescape HTML entities and strip HTML tags.

        Args:
            text: Input string potentially containing HTML markup.

        Returns:
            String with HTML tags removed.
        """
        if not text:
            return ""
        unescaped = html.unescape(text)
        return cls.HTML_TAG_PATTERN.sub(" ", unescaped)

    @classmethod
    def remove_urls(cls, text: str) -> str:
        """Remove HTTP/HTTPS and www URLs from text.

        Args:
            text: Input text string.

        Returns:
            Text without URLs.
        """
        if not text:
            return ""
        return cls.URL_PATTERN.sub(" ", text)

    @classmethod
    def remove_emails(cls, text: str) -> str:
        """Remove email addresses from text[cite: 6].

        Args:
            text: Input text string.

        Returns:
            Text without email addresses.
        """
        if not text:
            return ""
        return cls.EMAIL_PATTERN.sub(" ", text)

    @classmethod
    def remove_unwanted_chars(
        cls,
        text: str,
        preserve_punctuation: bool = False,
    ) -> str:
        """Remove non-Cyrillic characters while preserving digits and internal hyphens[cite: 6].

        Args:
            text: Input text string.
            preserve_punctuation: If True, retains punctuation for tokenizer comparisons.

        Returns:
            Filtered text string.
        """
        if not text:
            return ""
        normalized_dashes = text.replace("—", " - ").replace("–", " - ")
        if preserve_punctuation:
            cleaned = cls.PUNCT_PRESERVING_PATTERN.sub(" ", normalized_dashes)
        else:
            cleaned = cls.STRICT_CLEAN_PATTERN.sub(" ", normalized_dashes)
            cleaned = cls.STANDALONE_HYPHEN_PATTERN.sub(" ", cleaned)
        return cls.MULTI_HYPHEN_PATTERN.sub("-", cleaned)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Collapse multiple whitespace characters into single spaces and strip ends[cite: 6].

        Args:
            text: Input text string.

        Returns:
            Whitespace-normalized string.
        """
        if not text:
            return ""
        return cls.WHITESPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def to_lowercase(text: str) -> str:
        """Convert text to lowercase[cite: 6].

        Args:
            text: Input text string.

        Returns:
            Lowercase string.
        """
        return text.lower() if text else ""

    @classmethod
    def clean(cls, text: str, preserve_punctuation: bool = False) -> str:
        """Execute the full text cleaning pipeline[cite: 6].

        Args:
            text: Raw input text.
            preserve_punctuation: Whether to retain sentence punctuation marks.

        Returns:
            Cleaned, normalized lowercase Russian text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        step = cls.remove_html(text)
        step = cls.remove_urls(step)
        step = cls.remove_emails(step)
        step = cls.remove_unwanted_chars(step, preserve_punctuation=preserve_punctuation)
        step = cls.to_lowercase(step)
        return cls.normalize_whitespace(step)