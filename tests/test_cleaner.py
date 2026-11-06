"""Тесты модуля очистки текста TextCleaner."""

from preprocessing.cleaner import TextCleaner


def test_text_cleaner_removes_html_and_urls() -> None:
    raw = "<p>Привет, мир! Ссылка: https://lenta.ru и почта test@mail.ru</p>"
    cleaned = TextCleaner.clean(raw)
    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "test@mail.ru" not in cleaned
    assert "привет" in cleaned
    assert "мир" in cleaned