"""Удаление русских стоп-слов (Задание 4)."""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import nltk
from nltk.corpus import stopwords

import config


class StopWordFilter:
    """Фильтрует стоп-слова NLTK, пользовательские стоп-слова и короткие токены (len <= 2)."""

    def __init__(
        self,
        custom_stopwords: Optional[Iterable[str]] = None,
        min_word_length: int = 3,
    ) -> None:
        self.min_word_length = min_word_length
        self.stop_words: Set[str] = self._load_nltk_stopwords()
        self.stop_words.update(config.CUSTOM_RUSSIAN_STOPWORDS)
        if custom_stopwords:
            self.stop_words.update(w.strip().lower() for w in custom_stopwords if w.strip())

    @staticmethod
    def _load_nltk_stopwords() -> Set[str]:
        try:
            return set(stopwords.words("russian"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("russian"))
            except Exception:
                return {"и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то"}

    def filter_lemmas(self, lemmas: List[str]) -> List[str]:
        """Удаляет стоп-слова и слова длиной <= 2 символов."""
        result: List[str] = []
        for lemma in lemmas:
            word = lemma.strip().lower()
            if (
                word
                and word not in self.stop_words
                and len(word) >= self.min_word_length
                and word.isalpha()
            ):
                result.append(word)
        return result

    def filter_corpus(self, docs_lemmas: List[List[str]]) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Фильтрует весь корпус и возвращает статистику до и после удаления."""
        filtered_docs: List[List[str]] = []
        before_count = 0
        after_count = 0

        for doc in docs_lemmas:
            before_count += len(doc)
            filtered = self.filter_lemmas(doc)
            after_count += len(filtered)
            filtered_docs.append(filtered)

        removed = before_count - after_count
        pct = (removed / before_count * 100.0) if before_count > 0 else 0.0
        stats = {
            "tokens_before": before_count,
            "tokens_after": after_count,
            "tokens_removed": removed,
            "percentage_removed": pct,
        }
        return filtered_docs, stats