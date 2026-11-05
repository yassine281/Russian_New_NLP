"""Unit tests for `TextCleaner`."""

from __future__ import annotations

from preprocessing.cleaner import TextCleaner


def test_cleaner_strips_html_urls_and_emails() -> None:
    """Verify HTML tags, URLs, and emails are removed while preserving Cyrillic."""
    raw = (
        "<div class='body'><p>Привет, мир! Подробности на https://lenta.ru/news/123 "
        "и по почте info@ria.ru в 2026 году по-новому.</p></div>"
    )
    cleaned = TextCleaner.clean(raw, preserve_punctuation=False)
    assert "<div>" not in cleaned
    assert "https://" not in cleaned
    assert "info@ria.ru" not in cleaned
    assert "привет" in cleaned
    assert "2026" in cleaned
    assert "по-новому" in cleaned


def test_cleaner_preserve_punctuation_mode() -> None:
    """Verify punctuation is preserved when requested for tokenizer comparison."""
    raw = "В Москве, Санкт-Петербурге и Казани прошла выставка!"
    with_punct = TextCleaner.clean(raw, preserve_punctuation=True)
    without_punct = TextCleaner.clean(raw, preserve_punctuation=False)

    assert "," in with_punct and "!" in with_punct
    assert "," not in without_punct and "!" not in without_punct
    assert "санкт-петербурге" in without_punct


def test_cleaner_empty_input() -> None:
    """Verify empty or whitespace strings return empty string safely."""
    assert TextCleaner.clean("") == ""
    assert TextCleaner.clean("   \n\t  ") == ""