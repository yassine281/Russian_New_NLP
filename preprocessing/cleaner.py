"""Модуль очистки и нормализации текста (Задание 2, Шаг 1)."""

import html
import re


class TextCleaner:
    """Очищает текст от HTML-тегов, ссылок, email и спецсимволов."""

    @staticmethod
    def clean(text: str, preserve_punctuation: bool = False) -> str:
        """Выполняет полную очистку текста и приводит его к нижнему регистру."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        if preserve_punctuation:
            text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s\-\.,!?:;«»\"']", " ", text)
        else:
            text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s\-]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()