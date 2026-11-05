"""Morphological lemmatization implementations for Russian text.

Academic Control Question Reference (Q4 & Q5):
    Q4 (What is the difference between lemmatization and stemming?):
        Lemmatization uses morphological vocabularies and grammatical rules to resolve a
        word to its true dictionary base form (lemma), accounting for part of speech and
        suppletion (e.g., 'бегущий' -> 'бежать', 'лето' -> 'лето')[cite: 3, 7]. Stemming applies
        heuristic suffix-stripping rules without morphological knowledge, yielding a truncated
        stem that may not be a valid word and can conflate unrelated words ('летать' -> 'лет'
        and 'лето' -> 'лет')[cite: 3, 7].
    Q5 (What Russian-language NLP preprocessing tools exist?):
        1. Pymorphy3: Dictionary-based (OpenCorpora) analyzer + probabilistic prefix/suffix
           prediction for out-of-vocabulary words[cite: 3].
        2. spaCy (`ru_core_news_sm`): Context-aware neural pipeline supporting tokenization,
           POS-tagging, lemmatization, dependency parsing, and NER[cite: 3, 4].
        3. Yandex Mystem: Statistical/dictionary morphological analyzer capable of resolving
           homonymy and handling unknown Russian words[cite: 3].
        4. NLTK (`SnowballStemmer`, `word_tokenize`, `stopwords`): Rule-based Russian stemmer,
           Punkt tokenizer, and stop-word lists[cite: 3, 4].
        5. Natasha (Slovnet / Razdel / Navek): Rule-based and compact neural toolkit optimized
           for Russian segmentation, morphology, and named entity recognition[cite: 3].
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import pymorphy3

import config

logger = logging.getLogger(__name__)


class BaseLemmatizer(ABC):
    """Abstract base class for Russian lemmatizers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the lemmatizer identifier."""

    @abstractmethod
    def lemmatize_token(self, token: str) -> str:
        """Lemmatize a single Russian word token.

        Args:
            token: Single word string.

        Returns:
            Normalized lemma string.
        """

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Lemmatize a sequence of tokens.

        Args:
            tokens: List of word tokens.

        Returns:
            List of lemmas corresponding to input tokens.
        """
        return [self.lemmatize_token(t) for t in tokens if t and t.strip()]

    def lemmatize_text(self, text: str) -> list[str]:
        """Tokenize and lemmatize a cleaned text string.

        Args:
            text: Cleaned Russian text.

        Returns:
            List of lemmas.
        """
        if not text or not text.strip():
            return []
        return self.lemmatize_tokens(text.split())


class PyMorphyLemmatizer(BaseLemmatizer):
    """Dictionary-backed Russian lemmatizer powered by `pymorphy3.MorphAnalyzer`[cite: 3, 6]."""

    _shared_analyzer: Optional[pymorphy3.MorphAnalyzer] = None

    def __init__(self) -> None:
        """Initialize the Pymorphy3 analyzer once and maintain an internal lemma cache."""
        if PyMorphyLemmatizer._shared_analyzer is None:
            PyMorphyLemmatizer._shared_analyzer = pymorphy3.MorphAnalyzer()
        self._morph: pymorphy3.MorphAnalyzer = PyMorphyLemmatizer._shared_analyzer
        self._cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Return lemmatizer name."""
        return "Pymorphy3"

    def lemmatize_token(self, token: str) -> str:
        """Return the highest-probability normal form from Pymorphy3[cite: 6].

        Args:
            token: Input Russian word token.

        Returns:
            Normal form (`normal_form`) of the word.
        """
        cleaned = token.strip().lower()
        if not cleaned:
            return ""
        cached = self._cache.get(cleaned)
        if cached is not None:
            return cached

        parsed = self._morph.parse(cleaned)
        lemma = parsed[0].normal_form if parsed else cleaned
        lemma = lemma.replace("ё", "е") if getattr(config, "NORMALIZE_YO", False) else lemma
        self._cache[cleaned] = lemma
        return lemma


class SpacyLemmatizer(BaseLemmatizer):
    """Context-sensitive neural lemmatizer using spaCy `ru_core_news_sm`[cite: 3, 7]."""

    _shared_nlp = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the spaCy lemmatizer with lazy singleton model loading.

        Args:
            model_name: Name of the spaCy model (`ru_core_news_sm`)[cite: 4].
        """
        self.model_name: str = model_name or getattr(
            config, "SPACY_MODEL_NAME", "ru_core_news_sm"
        )
        self._fallback_morph: Optional[PyMorphyLemmatizer] = None

    @property
    def name(self) -> str:
        """Return lemmatizer name."""
        return "spaCy"

    def _get_nlp(self):
        """Load and cache the `ru_core_news_sm` model with `ner` and `parser` disabled for speed."""
        if SpacyLemmatizer._shared_nlp is None:
            import spacy

            try:
                SpacyLemmatizer._shared_nlp = spacy.load(
                    self.model_name,
                    disable=["ner", "parser"],
                )
            except OSError as exc:
                logger.warning(
                    "spaCy model '%s' is not installed (%s). Using blank('ru') with Pymorphy3 fallback.",
                    self.model_name,
                    exc,
                )
                SpacyLemmatizer._shared_nlp = spacy.blank("ru")
                self._fallback_morph = PyMorphyLemmatizer()
        return SpacyLemmatizer._shared_nlp

    def lemmatize_token(self, token: str) -> str:
        """Lemmatize a single isolated word using spaCy.

        Args:
            token: Single word token.

        Returns:
            Lemma string produced by spaCy.
        """
        cleaned = token.strip()
        if not cleaned:
            return ""
        nlp = self._get_nlp()
        if self._fallback_morph is not None:
            return self._fallback_morph.lemmatize_token(cleaned)
        doc = nlp(cleaned)
        for tok in doc:
            if not tok.is_space:
                return (tok.lemma_ or tok.text).lower()
        return cleaned.lower()

    def lemmatize_text(self, text: str) -> list[str]:
        """Lemmatize full text in context using spaCy's `doc` pipeline.

        Args:
            text: Cleaned Russian text string.

        Returns:
            List of non-space, non-punctuation lemmas.
        """
        if not text or not text.strip():
            return []
        nlp = self._get_nlp()
        if self._fallback_morph is not None:
            return self._fallback_morph.lemmatize_text(text)
        doc = nlp(text)
        return [
            (tok.lemma_ or tok.text).lower()
            for tok in doc
            if not tok.is_space and not tok.is_punct
        ]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Lemmatize a list of tokens by joining them into a spaCy document.

        Args:
            tokens: List of word tokens.

        Returns:
            List of lemmas.
        """
        if not tokens:
            return []
        return self.lemmatize_text(" ".join(tokens))


class MystemLemmatizer(BaseLemmatizer):
    """Optional adapter for Yandex Mystem (`pymystem3`)[cite: 3].

    Explicitly detects whether `pymystem3` is installed and operable on Windows.
    Never fabricates output if the binary or package is absent.
    """

    WORD_RE: re.Pattern[str] = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9-]+")

    def __init__(self) -> None:
        """Attempt to initialize Yandex Mystem if available in the environment."""
        self._mystem = None
        self.is_available: bool = False
        self.unavailable_reason: str = ""
        try:
            from pymystem3 import Mystem  # type: ignore[import-not-found]

            self._mystem = Mystem()
            self.is_available = True
        except Exception as exc:
            self.is_available = False
            self.unavailable_reason = (
                f"pymystem3/Mystem binary not installed in current environment ({exc})."
            )
            logger.info("Optional Mystem adapter unavailable: %s", self.unavailable_reason)

    @property
    def name(self) -> str:
        """Return lemmatizer name."""
        return "Mystem"

    def lemmatize_token(self, token: str) -> str:
        """Lemmatize a single token using Mystem if installed.

        Args:
            token: Input Russian word.

        Returns:
            Mystem lemma.

        Raises:
            RuntimeError: If Mystem is not installed in the current environment.
        """
        if not self.is_available or self._mystem is None:
            raise RuntimeError(f"Mystem is unavailable: {self.unavailable_reason}")
        lemmas = self._mystem.lemmatize(token.strip())
        for item in lemmas:
            stripped = item.strip()
            if stripped and self.WORD_RE.fullmatch(stripped):
                return stripped.lower()
        return token.strip().lower()