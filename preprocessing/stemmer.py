"""Russian Snowball stemmer wrapper built on NLTK."""

from __future__ import annotations

from nltk.stem.snowball import SnowballStemmer


class SnowballStemmerWrapper:
    """Wrapper around NLTK's `SnowballStemmer('russian')` for word-stem truncation."""

    def __init__(self, language: str = "russian") -> None:
        """Initialize the Russian Snowball stemmer[cite: 7].

        Args:
            language: Stemmer language identifier (default 'russian').
        """
        self.language: str = language
        self._stemmer: SnowballStemmer = SnowballStemmer(language)
        self._cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Return stemmer name."""
        return "SnowballStemmer"

    def stem_token(self, token: str) -> str:
        """Compute the Snowball stem for a single word token[cite: 7].

        Args:
            token: Input Russian word.

        Returns:
            Truncated word stem.
        """
        cleaned = token.strip().lower()
        if not cleaned:
            return ""
        cached = self._cache.get(cleaned)
        if cached is not None:
            return cached
        stem = str(self._stemmer.stem(cleaned))
        self._cache[cleaned] = stem
        return stem

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Compute Snowball stems for a list of tokens[cite: 7].

        Args:
            tokens: List of word tokens.

        Returns:
            List of stemmed tokens.
        """
        return [self.stem_token(t) for t in tokens if t and t.strip()]