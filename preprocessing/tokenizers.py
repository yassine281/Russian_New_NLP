"""Инструменты токенизации русского текста (Задание 2, Шаг 2)."""

from typing import List
import nltk
from nltk.tokenize import word_tokenize
import spacy

import config


class SimpleTokenizer:
    """Простая токенизация методом str.split()."""

    @property
    def name(self) -> str:
        return "SimpleTokenizer"

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return text.split()


class NLTKTokenizer:
    """Токенизация с помощью nltk.tokenize.word_tokenize."""

    def __init__(self) -> None:
        self._ready = False

    @property
    def name(self) -> str:
        return "NLTKTokenizer"

    def _ensure_punkt(self) -> None:
        if not self._ready:
            for pkg in ("punkt", "punkt_tab"):
                try:
                    nltk.data.find(f"tokenizers/{pkg}")
                except LookupError:
                    try:
                        nltk.download(pkg, quiet=True)
                    except Exception:
                        pass
            self._ready = True

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        self._ensure_punkt()
        try:
            return list(word_tokenize(text, language="russian"))
        except Exception:
            return text.split()


class SpacyTokenizer:
    """Токенизация с использованием модели spaCy (ru_core_news_sm)."""

    _nlp = None

    def __init__(self) -> None:
        if SpacyTokenizer._nlp is None:
            try:
                SpacyTokenizer._nlp = spacy.load(
                    config.SPACY_MODEL_NAME,
                    disable=["ner", "parser", "lemmatizer"],
                )
            except Exception:
                SpacyTokenizer._nlp = spacy.blank("ru")

    @property
    def name(self) -> str:
        return "SpacyTokenizer"

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        doc = self._nlp.make_doc(text)
        return [t.text for t in doc if not t.is_space]


class TokenizerFactory:
    """Фабрика для создания токенизаторов по имени."""

    @staticmethod
    def create(tokenizer_type: str):
        key = tokenizer_type.strip().lower()
        if key == "simple":
            return SimpleTokenizer()
        if key == "nltk":
            return NLTKTokenizer()
        if key == "spacy":
            return SpacyTokenizer()
        raise ValueError(f"Неизвестный токенизатор: {tokenizer_type}")