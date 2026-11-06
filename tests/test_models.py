"""Tests unitaires pour config.py et les modeles Article, ProcessedArticle et FrequencyEntry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import config
from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry


def test_config_directories_and_thresholds() -> None:
    """Verifie que les repertoires et seuils academiques de config.py sont valides."""
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.PLOTS_DIR.exists()
    assert config.REPORTS_DIR.exists()
    assert config.MIN_CORPUS_ARTICLES == 100
    assert config.MIN_CORPUS_WORDS == 30000
    assert config.REQUEST_DELAY_SECONDS >= 0.5


def test_article_creation_and_word_count() -> None:
    """Verifie la creation d'un Article, le comptage de mots et la serialisation JSON."""
    article = Article(
        url="https://lenta.ru/news/2026/05/01/test/",
        title="Научное открытие в России",
        text="Российские учёные разработали новый квантовый процессор для сложных вычислений.",
        date="2026-05-01",
        source="Lenta.ru",
    )
    assert article.word_count == 9
    assert len(article.article_id) == 12
    assert article.status == "collected"
    assert "Lenta.ru" in repr(article) or "lenta.ru" in str(article)

    as_dict = article.to_dict()
    assert as_dict["word_count"] == 9
    restored = Article.from_dict(as_dict)
    assert restored.url == article.url
    assert restored.title == article.title
    assert restored.text == article.text
    assert restored.word_count == 9


def test_article_immutability_and_validation() -> None:
    """Verifie l'immuabilite de Article et le rejet des champs vides."""
    article = Article(
        url="https://ria.ru/2026/test.html",
        title="Заголовок",
        text="Текст новости из нескольких слов для проверки.",
        date=None,
        source="РИА Новости",
    )
    with pytest.raises(FrozenInstanceError):
        article.title = "Новый заголовок"  # type: ignore[misc]

    with pytest.raises(ValueError, match="url"):
        Article(url="   ", title="Заголовок", text="Текст", date=None, source="РИА")
    with pytest.raises(ValueError, match="title"):
        Article(url="https://ria.ru/1", title="", text="Текст", date=None, source="РИА")
    with pytest.raises(ValueError, match="text"):
        Article(url="https://ria.ru/1", title="Заголовок", text="  ", date=None, source="РИА")


def test_frequency_entry_validation_and_tsv() -> None:
    """Verifie la validation de FrequencyEntry et le formatage TSV."""
    entry = FrequencyEntry(word="Россия", count=150, rank=1, relative_frequency=0.025)
    assert entry.word == "россия"
    assert entry.count == 150
    assert entry.rank == 1
    assert entry.to_tsv_line() == "россия\t150"

    with pytest.raises(ValueError, match="word"):
        FrequencyEntry(word="", count=10, rank=1)
    with pytest.raises(ValueError, match="count"):
        FrequencyEntry(word="слово", count=0, rank=1)
    with pytest.raises(ValueError, match="rank"):
        FrequencyEntry(word="слово", count=10, rank=0)


def test_processed_article_creation_and_csv_row() -> None:
    """Verifie que ProcessedArticle conserve toutes les couches linguistiques et exporte en CSV."""
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
    assert proc.filtered_word_count == 5
    csv_row = proc.to_csv_row()
    assert csv_row["url"] == "https://lenta.ru/news/1"
    assert csv_row["lemmas_filtered"] == "красивый быстрый машина ехать дорога"