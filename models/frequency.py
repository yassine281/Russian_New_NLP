"""Modele de donnees immuable pour une entree du dictionnaire frequentiel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrequencyEntry:
    """Represente un mot (lemme), sa frequence absolue et son rang statistique.

    Attributes:
        word: Lemme ou token normalise en russe.
        count: Frequence d'apparition absolue dans le corpus (> 0).
        rank: Rang dans la distribution de Zipf (>= 1).
        relative_frequency: Frequence relative optionnelle dans le corpus (entre 0.0 et 1.0).
    """

    word: str
    count: int
    rank: int
    relative_frequency: float = 0.0

    def __post_init__(self) -> None:
        """Valide que le mot est non vide et que le compte et le rang sont strictement positifs."""
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("FrequencyEntry 'word' must be a non-empty string.")
        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("FrequencyEntry 'count' must be a positive integer (> 0).")
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("FrequencyEntry 'rank' must be a positive integer (>= 1).")
        if not isinstance(self.relative_frequency, (int, float)) or self.relative_frequency < 0.0:
            raise ValueError("FrequencyEntry 'relative_frequency' must be non-negative.")

        object.__setattr__(self, "word", self.word.strip().lower())
        object.__setattr__(self, "relative_frequency", float(self.relative_frequency))

    def to_tsv_line(self) -> str:
        """Formate l'entree sous la forme 'word\\tfrequency' pour frequency_dict.txt."""
        return f"{self.word}\t{self.count}"

    def to_dict(self) -> dict[str, Any]:
        """Convertit l'entree en dictionnaire pour l'export CSV/JSON."""
        return {
            "rank": self.rank,
            "word": self.word,
            "count": self.count,
            "relative_frequency": round(self.relative_frequency, 6),
        }

    def __str__(self) -> str:
        """Representation textuelle claire du rang, du mot et de sa frequence."""
        return f"#{self.rank}: {self.word} ({self.count})"

    def __repr__(self) -> str:
        """Representation technique complete de l'instance."""
        return (
            f"FrequencyEntry(rank={self.rank}, word={self.word!r}, "
            f"count={self.count}, relative_frequency={self.relative_frequency:.6f})"
        )