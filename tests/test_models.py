"""Unit tests for domain data models (`Article`, `ProcessedArticle`, `FrequencyEntry`)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry


def test_article_creation_and_word_count() -> None:
    """Verify Article creation, word_count calculation, and serialization."""
    article = Article(
        url="https://lenta.ru/news/2026/05/01/test/",
        title="Научное открытие в России",
        text="Российские учёные разработали новый квантовый процессор для сложных вычислений.",
        date="2026-05-01",
        source="Lenta.ru",
    )
    assert article.word_count == 9
    assert "Lenta.ru" in repr(article) or "lenta.ru" in str(article)

    as_dict = article.to_dict()
    restored = Article.from_dict(as_dict)
    assert restored == article


def test_article_immutability_and_validation() -> None:
    """Verify Article is frozen and rejects empty required fields."""
    article = Article(
        url="https://ria.ru/2026/test.html",
        title="Заголовок",
        text="Текст новости из нескольких слов для проверки.",
        date=None,
        source="РИА Новости",
    )
    with pytest.raises(FrozenInstanceError):
        article.title = "Новый заголовок"  # type: ignore[misc]

    with pytest.raises(ValueError):
        Article(url="", title="Заголовок", text="Текст", date=None, source="РИА")


def test_frequency_entry_validation() -> None:
    """Verify FrequencyEntry validates positive count and rank."""
    entry = FrequencyEntry(word="россия", count=150, rank=1)
    assert entry.word == "россия"
    assert entry.count == 150
    assert entry.rank == 1

    with pytest.raises(ValueError):
        FrequencyEntry(word="", count=10, rank=1)
    with pytest.raises(ValueError):
        FrequencyEntry(word="слово", count=0, rank=1)


def test_processed_article_creation() -> None:
    """Verify ProcessedArticle stores all linguistic representations."""
    raw = Article(
        url="https://lenta.ru/news/1",
        title="Тест",
        text="Красивая быстрая машина ехала по дороге.",
        date="2026-05-01",
        source="Lenta.ru",
    )
    proc = ProcessedArticle(
        article=raw,
        clean_text="красивая быстрая машина ехала по дороге",
        tokens=["красивая", "быстрая", "машина", "ехала", "по", "дороге"],
        nltk_tokens=["красивая", "быстрая", "машина", "ехала", "по", "дороге"],
        spacy_tokens=["красивая", "быстрая", "машина", "ехала", "по", "дороге"],
        pymorphy_lemmas=["красивый", "быстрый", "машина", "ехать", "по", "дорога"],
        spacy_lemmas=["красивый", "быстрый", "машина", "ехать", "по", "дорога"],
        snowball_stems=["красив", "быстр", "машин", "ех", "по", "дорог"],
        filtered_lemmas=["красивый", "быстрый", "машина", "ехать", "дорога"],
    )
    assert len(proc.filtered_lemmas) == 5