"""Модель записи частотного словаря."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrequencyEntry:
    """Хранит слово, его частоту и ранг в корпусе."""

    word: str
    count: int
    rank: int

    def __post_init__(self) -> None:
        if not self.word or not self.word.strip():
            raise ValueError("Поле 'word' не может быть пустым.")
        if self.count <= 0:
            raise ValueError("Частота 'count' должна быть больше нуля.")
        if self.rank <= 0:
            raise ValueError("Ранг 'rank' должен быть больше нуля.")