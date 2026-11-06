"""Лемматизация русского текста с помощью Pymorphy3 и spaCy (Задание 3)."""

import inspect
from collections import namedtuple
from typing import Dict, List

# Патч совместимости inspect.getargspec для Python 3.11 - 3.14
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]

import pymorphy3
import spacy

import config


class PyMorphyLemmatizer:
    """Морфологическая лемматизация с помощью библиотеки Pymorphy3."""

    _morph = None

    def __init__(self) -> None:
        if PyMorphyLemmatizer._morph is None:
            PyMorphyLemmatizer._morph = pymorphy3.MorphAnalyzer()
        self._cache: Dict[str, str] = {}

    def lemmatize_token(self, token: str) -> str:
        """Возвращает начальную форму (normal_form) одного слова."""
        word = token.strip().lower()
        if not word:
            return ""
        if word in self._cache:
            return self._cache[word]
        parsed = self._morph.parse(word)
        lemma = parsed[0].normal_form if parsed else word
        self._cache[word] = lemma
        return lemma

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Лемматизирует список токенов."""
        return [self.lemmatize_token(t) for t in tokens if t and t.strip()]


class SpacyLemmatizer:
    """Нейросетевая лемматизация с помощью модели spaCy ru_core_news_sm."""

    _nlp = None

    def __init__(self) -> None:
        if SpacyLemmatizer._nlp is None:
            try:
                SpacyLemmatizer._nlp = spacy.load(
                    config.SPACY_MODEL_NAME,
                    disable=["ner", "parser"],
                )
            except Exception:
                SpacyLemmatizer._nlp = spacy.blank("ru")

    def lemmatize_token(self, token: str) -> str:
        """Лемматизирует одно слово с помощью spaCy."""
        word = token.strip().lower()
        if not word:
            return ""
        doc = self._nlp(word)
        for tok in doc:
            if not tok.is_space:
                return (tok.lemma_ or tok.text).lower()
        return word

    def lemmatize_text(self, text: str) -> List[str]:
        """Лемматизирует текст целиком."""
        if not text or not text.strip():
            return []
        doc = self._nlp(text)
        return [
            (t.lemma_ or t.text).lower()
            for t in doc
            if not t.is_space and not t.is_punct
        ]