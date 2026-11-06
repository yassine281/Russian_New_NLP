"""Modeles de donnees immuables pour les articles de presse russes bruts et pretraites."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _default_timestamp() -> str:
    """Genere un horodatage ISO-8601 UTC pour la collecte d'un article."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Article:
    """Represente un article de presse russe brut collecte depuis une source publique.

    Attributes:
        url: Adresse URL complete de l'article.
        title: Titre principal de l'article.
        text: Contenu textuel complet de l'article.
        date: Date de publication extraite (optionnelle).
        source: Nom du media source (ex. 'Lenta.ru', 'РИА Новости').
        article_id: Identifiant unique deterministe base sur l'URL normalisee.
        collected_at: Horodatage ISO-8601 du moment de la collecte.
        status: Statut de collecte ('collected' par defaut).
    """

    url: str
    title: str
    text: str
    date: Optional[str] = None
    source: str = "Unknown"
    article_id: str = ""
    collected_at: str = field(default_factory=_default_timestamp)
    status: str = "collected"

    def __post_init__(self) -> None:
        """Valide les champs obligatoires et normalise les chaines de caracteres."""
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("Article 'url' must be a non-empty string.")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Article 'title' must be a non-empty string.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Article 'text' must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Article 'source' must be a non-empty string.")

        clean_url = self.url.strip()
        clean_title = self.title.strip()
        clean_text = self.text.strip()
        clean_source = self.source.strip()
        clean_date = (
            self.date.strip()
            if isinstance(self.date, str) and self.date.strip()
            else None
        )

        computed_id = (
            self.article_id.strip()
            if isinstance(self.article_id, str) and self.article_id.strip()
            else hashlib.sha1(clean_url.lower().encode("utf-8")).hexdigest()[:12]
        )

        object.__setattr__(self, "url", clean_url)
        object.__setattr__(self, "title", clean_title)
        object.__setattr__(self, "text", clean_text)
        object.__setattr__(self, "source", clean_source)
        object.__setattr__(self, "date", clean_date)
        object.__setattr__(self, "article_id", computed_id)

    @property
    def word_count(self) -> int:
        """Retourne le nombre de mots de l'article par decoupage sur les espaces."""
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        """Convertit l'instance Article en dictionnaire serialisable en JSON."""
        return {
            "article_id": self.article_id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "date": self.date,
            "source": self.source,
            "word_count": self.word_count,
            "collected_at": self.collected_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Article:
        """Reconstruit une instance Article a partir d'un dictionnaire JSON."""
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a dictionary.")
        return cls(
            url=str(data.get("url", "")),
            title=str(data.get("title", "")),
            text=str(data.get("text", "")),
            date=str(data["date"]) if data.get("date") is not None else None,
            source=str(data.get("source") or "Unknown"),
            article_id=str(data.get("article_id") or ""),
            collected_at=str(data.get("collected_at") or _default_timestamp()),
            status=str(data.get("status") or "collected"),
        )

    def __str__(self) -> str:
        """Representation lisible de l'article."""
        return f"[{self.source}] {self.title} ({self.word_count} words) - {self.url}"

    def __repr__(self) -> str:
        """Representation technique detaillee de l'article."""
        return (
            f"Article(article_id={self.article_id!r}, source={self.source!r}, "
            f"title={self.title[:40]!r}, word_count={self.word_count}, url={self.url!r})"
        )


@dataclass(frozen=True)
class ProcessedArticle:
    """Represente un article enrichi par toutes les etapes de pretraitement NLP.

    Attributes:
        article: Instance Article brute d'origine.
        clean_text: Texte nettoye et normalise en minuscules.
        tokens: Tokens issus de SimpleTokenizer (.split()).
        nltk_tokens: Tokens issus de NLTK word_tokenize.
        spacy_tokens: Tokens issus de SpacyTokenizer.
        pymorphy_lemmas: Lemmes produits par Pymorphy3.
        spacy_lemmas: Lemmes produits par spaCy (ru_core_news_sm).
        snowball_stems: Racines produites par NLTK SnowballStemmer.
        filtered_lemmas: Lemmes finaux apres suppression des mots vides (stop words).
    """

    article: Article
    clean_text: str
    tokens: list[str] = field(default_factory=list)
    nltk_tokens: list[str] = field(default_factory=list)
    spacy_tokens: list[str] = field(default_factory=list)
    pymorphy_lemmas: list[str] = field(default_factory=list)
    spacy_lemmas: list[str] = field(default_factory=list)
    snowball_stems: list[str] = field(default_factory=list)
    filtered_lemmas: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Verifie la validite de l'article source."""
        if not isinstance(self.article, Article):
            raise TypeError("ProcessedArticle 'article' must be an Article instance.")
        if not isinstance(self.clean_text, str):
            raise TypeError("ProcessedArticle 'clean_text' must be a string.")

    @property
    def filtered_word_count(self) -> int:
        """Retourne le nombre de lemmes apres filtrage des mots vides."""
        return len(self.filtered_lemmas)

    def to_csv_row(self) -> dict[str, str]:
        """Retourne les colonnes requises pour l'export corpus_processed.csv."""
        return {
            "url": self.article.url,
            "title": self.article.title,
            "clean": self.clean_text,
            "lemmas_filtered": " ".join(self.filtered_lemmas),
        }

    def __str__(self) -> str:
        """Representation lisible de l'article pretraite."""
        return (
            f"ProcessedArticle(title={self.article.title[:35]!r}, "
            f"tokens={len(self.tokens)}, filtered_lemmas={len(self.filtered_lemmas)})"
        )