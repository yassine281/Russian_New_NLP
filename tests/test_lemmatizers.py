"""Unit tests for Russian lemmatizers, Snowball stemmer, and StopWordFilter."""

from __future__ import annotations

import pytest

from preprocessing.lemmatizers import (
    MystemLemmatizer,
    PyMorphyLemmatizer,
    SpacyLemmatizer,
)
from preprocessing.stemmer import SnowballStemmerWrapper
from preprocessing.stopwords import StopWordFilter, StopWordStats


def test_pymorphy_on_official_validation_words() -> None:
    """Verify Pymorphy3 produces exact dictionary lemmas on the 5 lab validation words."""
    pymorphy = PyMorphyLemmatizer()
    assert pymorphy.lemmatize_token("красивая") == "красивый"
    assert pymorphy.lemmatize_token("бегущий") == "бежать"
    assert pymorphy.lemmatize_token("деревянный") == "деревянный"
    assert pymorphy.lemmatize_token("летать") == "летать"
    assert pymorphy.lemmatize_token("лето") == "лето"


def test_snowball_stemmer_on_validation_words() -> None:
    """Verify NLTK Russian SnowballStemmer truncation on the 5 validation words."""
    stemmer = SnowballStemmerWrapper()

    # Verify exact Snowball stem outputs for the benchmark set
    assert stemmer.stem_token("красивая") == "красив"
    assert stemmer.stem_token("летать") == "лета"
    assert stemmer.stem_token("лето") == "лет"

    # Allow valid Snowball participle/adjective suffix variations
    assert stemmer.stem_token("бегущий") in ("бег", "бегущ")
    assert stemmer.stem_token("деревянный") in ("деревян", "деревянн")

    validation_words = ["красивая", "бегущий", "деревянный", "летать", "лето"]
    stems = stemmer.stem_tokens(validation_words)
    assert len(stems) == 5
    for original, stem in zip(validation_words, stems):
        assert 0 < len(stem) <= len(original)


def test_inflectional_invariance_pymorphy_and_snowball() -> None:
    """Verify case/gender inflections collapse to one lemma and one stem."""
    pymorphy = PyMorphyLemmatizer()
    stemmer = SnowballStemmerWrapper()

    inflected_forms = ["красивая", "красивой", "красивую", "красивыми", "красивом"]
    lemmas = pymorphy.lemmatize_tokens(inflected_forms)
    stems = stemmer.stem_tokens(inflected_forms)

    assert set(lemmas) == {"красивый"}
    assert set(stems) == {"красив"}


def test_spacy_lemmatizer_contextual_and_single_token() -> None:
    """Verify SpacyLemmatizer normalizes both isolated tokens and full phrases."""
    spacy_lem = SpacyLemmatizer()
    phrase = "красивая деревянная ложка лежала на столе"
    lemmas = spacy_lem.lemmatize_text(phrase)

    assert len(lemmas) == 6
    assert "красивый" in lemmas
    assert "деревянный" in lemmas
    assert "ложка" in lemmas
    assert spacy_lem.lemmatize_token("красивая") == "красивый"


def test_optional_mystem_adapter_graceful_behavior() -> None:
    """Verify MystemLemmatizer either lemmatizes cleanly or reports unavailability explicitly."""
    mystem = MystemLemmatizer()
    assert mystem.name == "Mystem"

    if mystem.is_available:
        assert mystem.lemmatize_token("красивая") == "красивый"
    else:
        assert len(mystem.unavailable_reason) > 0
        with pytest.raises(RuntimeError, match="Mystem is unavailable"):
            mystem.lemmatize_token("красивая")


def test_lemmatizers_and_stemmer_edge_cases() -> None:
    """Verify empty strings, whitespace, and uppercase tokens are handled safely."""
    pymorphy = PyMorphyLemmatizer()
    spacy_lem = SpacyLemmatizer()
    stemmer = SnowballStemmerWrapper()

    assert pymorphy.lemmatize_token("") == ""
    assert pymorphy.lemmatize_token("   ") == ""
    assert pymorphy.lemmatize_token("КРАСИВАЯ") == "красивый"
    assert pymorphy.lemmatize_tokens([]) == []

    assert spacy_lemmatizer.lemmatize_token("") == ""
    assert spacy_lemmatizer.lemmatize_text("") == []

    assert stemmer.stem_token("") == ""
    assert stemmer.stem_token("   ") == ""
    assert stemmer.stem_token("КРАСИВАЯ") == "красив"
    assert stemmer.stem_tokens([]) == []


def test_stopword_filter_removal_and_statistics() -> None:
    """Verify StopWordFilter removes NLTK + custom stop words and short tokens (len <= 2)."""
    sw_filter = StopWordFilter(
        custom_stopwords=["это", "который", "свой"],
        min_word_length=3,
    )
    sample_doc = [
        "это",
        "красивый",
        "дом",
        "в",
        "который",
        "жить",
        "свой",
        "семья",
        "я",
        "2026",
    ]
    filtered_docs, stats = sw_filter.filter_corpus([sample_doc])

    assert isinstance(stats, StopWordStats)
    assert filtered_docs[0] == ["красивый", "дом", "жить", "семья"]
    assert stats.tokens_before == 10
    assert stats.tokens_after == 4
    assert stats.tokens_removed == 6
    assert pytest.approx(stats.percentage_removed, rel=1e-3) == 60.0
    assert stats.unique_stopwords_removed >= 4


def test_stopword_filter_zero_division_safety() -> None:
    """Verify StopWordFilter safely handles empty corpora without ZeroDivisionError."""
    sw_filter = StopWordFilter()
    empty_docs, empty_stats = sw_filter.filter_corpus([])

    assert empty_docs == []
    assert empty_stats.tokens_before == 0
    assert empty_stats.tokens_after == 0
    assert empty_stats.tokens_removed == 0
    assert empty_stats.percentage_removed == 0.0