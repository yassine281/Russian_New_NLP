"""Стемминг русского текста с помощью NLTK SnowballStemmer (Задание 3, Шаг 3)."""

from typing import Dict, List
from nltk.stem.snowball import SnowballStemmer


class SnowballStemmerWrapper:
    """Обертка над русским стеммером SnowballStemmer из библиотеки NLTK."""

    def __init__(self, language: str = "russian") -> None:
        self.stemmer = SnowballStemmer(language)
        self._cache: Dict[str, str] = {}

    def stem_token(self, token: str) -> str:
        """Отсекает окончания слова и возвращает его основу (стем)."""
        word = token.strip().lower()
        if not word:
            return ""
        if word in self._cache:
            return self._cache[word]
        stem = str(self.stemmer.stem(word))
        self._cache[word] = stem
        return stem

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Выполняет стемминг списка токенов."""
        return [self.stem_token(t) for t in tokens if t and t.strip()]