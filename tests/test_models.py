"""Тесты моделей данных Article и FrequencyEntry."""

import pytest
from models.article import Article
from models.frequency import FrequencyEntry


def test_article_word_count() -> None:
    article = Article(
        url="https://lenta.ru/news/1",
        title="Тестовый заголовок",
        text="Один два три четыре пять",
        source="Lenta.ru",
    )
    assert article.word_count == 5
    restored = Article.from_dict(article.to_dict())
    assert restored.word_count == 5


def test_frequency_entry_validation() -> None:
    entry = FrequencyEntry(word="россия", count=25, rank=1)
    assert entry.word == "россия"
    with pytest.raises(ValueError):
        FrequencyEntry(word="", count=0, rank=0)