"""Tokenization strategies for Russian text processing.

Academic Control Question Reference (Q3):
    Q3 (What is the difference between tokenization and lemmatization?):
        Tokenization is the structural segmentation of a continuous character stream into
        discrete lexical units (words, numbers, punctuation tokens) without altering their
        inflected morphological form[cite: 3]. Lemmatization is a subsequent morphological analysis
        step that maps each inflected token to its canonical dictionary headword (lemma),
        such as transforming 'красивая' -> 'красивый' or 'бегущий' -> 'бежать'.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import nltk
from nltk.tokenize import word_tokenize

import config

logger = logging.getLogger(__name__)


class BaseTokenizer(ABC):
    """Abstract interface for all text tokenizers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable tokenizer name."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Segment input text into a list of tokens.

        Args:
            text: Input string to tokenize.

        Returns:
            List of extracted token strings.
        """


class SimpleTokenizer(BaseTokenizer):
    """Whitespace-based Python `.split()` tokenizer[cite: 6]."""

    @property
    def name(self) -> str:
        """Return tokenizer name."""
        return "SimpleTokenizer"

    def tokenize(self, text: str) -> list[str]:
        """Split text on whitespace using Python's built-in `str.split()`[cite: 6].

        Args:
            text: Cleaned or raw Russian text.

        Returns:
            List of whitespace-separated tokens.
        """
        if not text or not text.strip():
            return []
        return text.split()


class NLTKTokenizer(BaseTokenizer):
    """NLTK `word_tokenize` wrapper configured for Russian text[cite: 6]."""

    FALLBACK_REGEX: re.Pattern[str] = re.compile(r"[а-яА-ЯёЁ0-9]+(?:-[а-яА-ЯёЁ0-9]+)*|[^\w\s]")

    def __init__(self, language: str = "russian") -> None:
        """Initialize the NLTK tokenizer and ensure Punkt resources are available.

        Args:
            language: Language model for Punkt sentence/word tokenization.
        """
        self.language: str = language
        self._resources_ready: bool = False

    @property
    def name(self) -> str:
        """Return tokenizer name."""
        return "NLTKTokenizer"

    def _ensure_nltk_resources(self) -> None:
        """Ensure NLTK `punkt` and `punkt_tab` packages are available locally[cite: 6]."""
        if self._resources_ready:
            return
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as exc:
                    logger.warning("Could not download NLTK resource %s: %s", resource, exc)
        self._resources_ready = True

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using `nltk.tokenize.word_tokenize`[cite: 6].

        Args:
            text: Input text string.

        Returns:
            List of tokens produced by NLTK.
        """
        if not text or not text.strip():
            return []
        self._ensure_nltk_resources()
        try:
            return list(word_tokenize(text, language=self.language))
        except LookupError:
            logger.warning("NLTK punkt lookup failed; using regex fallback tokenizer.")
            return self.FALLBACK_REGEX.findall(text)


class SpacyTokenizer(BaseTokenizer):
    """spaCy Russian tokenizer wrapper reusing a single loaded pipeline[cite: 6]."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the spaCy tokenizer with lazy model loading.

        Args:
            model_name: Name of the spaCy Russian model (default `ru_core_news_sm`)[cite: 4, 6].
        """
        self.model_name: str = model_name or getattr(
            config, "SPACY_MODEL_NAME", "ru_core_news_sm"
        )
        self._nlp = None

    @property
    def name(self) -> str:
        """Return tokenizer name."""
        return "SpacyTokenizer"

    def _get_nlp(self):
        """Lazily load the spaCy Russian pipeline or fallback to `spacy.blank('ru')`."""
        if self._nlp is None:
            import spacy

            try:
                self._nlp = spacy.load(
                    self.model_name,
                    disable=["ner", "parser", "attribute_ruler", "lemmatizer"],
                )
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found; falling back to spacy.blank('ru') for tokenization.",
                    self.model_name,
                )
                self._nlp = spacy.blank("ru")
        return self._nlp

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using spaCy's morphological tokenizer[cite: 6].

        Args:
            text: Input text string.

        Returns:
            List of non-whitespace token strings.
        """
        if not text or not text.strip():
            return []
        nlp = self._get_nlp()
        doc = nlp.make_doc(text)
        return [token.text for token in doc if not token.is_space]


class TokenizerFactory:
    """Factory for instantiating tokenizers by identifier."""

    _instances: dict[str, BaseTokenizer] = {}

    @classmethod
    def create(cls, tokenizer_type: str) -> BaseTokenizer:
        """Return a cached tokenizer instance matching `tokenizer_type`.

        Args:
            tokenizer_type: Identifier ('simple', 'nltk', or 'spacy').

        Returns:
            Concrete `BaseTokenizer` implementation.

        Raises:
            ValueError: If `tokenizer_type` is unknown.
        """
        key = tokenizer_type.strip().lower()
        if key in cls._instances:
            return cls._instances[key]

        if key == "simple":
            instance: BaseTokenizer = SimpleTokenizer()
        elif key == "nltk":
            instance = NLTKTokenizer()
        elif key == "spacy":
            instance = SpacyTokenizer()
        else:
            raise ValueError(
                f"Unknown tokenizer_type {tokenizer_type!r}. Choose from: 'simple', 'nltk', 'spacy'."
            )

        cls._instances[key] = instance
        return instance