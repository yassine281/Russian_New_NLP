"""Модели данных для сырой и обработанной новостной статьи."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Article:
    """Представляет исходную новостную статью."""

    url: str
    title: str
    text: str
    date: Optional[str] = None
    source: str = "Unknown"

    def __post_init__(self) -> None:
        if not self.url or not self.url.strip():
            raise ValueError("Поле 'url' не может быть пустым.")
        if not self.title or not self.title.strip():
            raise ValueError("Поле 'title' не может быть пустым.")
        if not self.text or not self.text.strip():
            raise ValueError("Поле 'text' не может быть пустым.")

    @property
    def word_count(self) -> int:
        """Возвращает количество слов в тексте статьи."""
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует статью в словарь для сохранения в JSON."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "date": self.date,
            "source": self.source,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Article":
        """Создает объект Article из словаря."""
        return cls(
            url=str(data.get("url", "")),
            title=str(data.get("title", "")),
            text=str(data.get("text", "")),
            date=data.get("date"),
            source=str(data.get("source", "Unknown")),
        )


@dataclass(frozen=True)
class ProcessedArticle:
    """Представляет статью после всех этапов лингвистической предобработки."""

    article: Article
    clean_text: str
    tokens: List[str] = field(default_factory=list)
    nltk_tokens: List[str] = field(default_factory=list)
    spacy_tokens: List[str] = field(default_factory=list)
    pymorphy_lemmas: List[str] = field(default_factory=list)
    spacy_lemmas: List[str] = field(default_factory=list)
    snowball_stems: List[str] = field(default_factory=list)
    filtered_lemmas: List[str] = field(default_factory=list)