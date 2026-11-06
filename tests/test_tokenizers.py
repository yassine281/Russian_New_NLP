"""Тесты трех токенизаторов: Simple, NLTK и spaCy."""

from preprocessing.tokenizers import SimpleTokenizer, NLTKTokenizer, SpacyTokenizer, TokenizerFactory


def test_all_tokenizers() -> None:
    text = "В Москве открылся новый научный центр."
    assert len(SimpleTokenizer().tokenize(text)) >= 6
    assert len(NLTKTokenizer().tokenize(text)) >= 6
    assert len(SpacyTokenizer().tokenize(text)) >= 6
    assert TokenizerFactory.create("simple").name == "SimpleTokenizer"