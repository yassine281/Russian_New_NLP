"""Тесты лемматизаторов Pymorphy3, spaCy, стеммера Snowball и фильтра стоп-слов."""

from preprocessing.lemmatizers import PyMorphyLemmatizer, SpacyLemmatizer
from preprocessing.stemmer import SnowballStemmerWrapper
from preprocessing.stopwords import StopWordFilter


def test_pymorphy_and_snowball() -> None:
    morph = PyMorphyLemmatizer()
    stemmer = SnowballStemmerWrapper()

    assert morph.lemmatize_token("красивая") == "красивый"
    assert morph.lemmatize_token("бегущий") == "бежать"
    assert morph.lemmatize_token("деревянный") == "деревянный"
    assert morph.lemmatize_token("летать") == "летать"
    assert morph.lemmatize_token("лето") == "лето"

    assert stemmer.stem_token("красивая") == "красив"
    assert stemmer.stem_token("лето") == "лет"
    assert stemmer.stem_token("летать").startswith("лет")


def test_spacy_lemmatizer_and_stopwords() -> None:
    spacy_lem = SpacyLemmatizer()
    lemmas = spacy_lem.lemmatize_text("красивая машина")
    assert len(lemmas) == 2

    sw = StopWordFilter(custom_stopwords=["это", "свой"])
    docs, stats = sw.filter_corpus([["это", "красивый", "дом", "и", "свой", "сад"]])
    assert docs[0] == ["красивый", "дом", "сад"]
    assert stats["tokens_before"] == 6
    assert stats["tokens_after"] == 3