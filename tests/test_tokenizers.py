"""Unit tests for Simple, NLTK, and spaCy tokenizers and TokenizerFactory."""

from __future__ import annotations

import pytest

from preprocessing.tokenizers import (
    NLTKTokenizer,
    SimpleTokenizer,
    SpacyTokenizer,
    TokenizerFactory,
)


def test_tokenizers_on_russian_sentence() -> None:
    """Verify all three tokenizers segment Russian text and handle punctuation."""
    text = "В Санкт-Петербурге открылся новый научно-исследовательский центр, сообщили учёные."
    simple = SimpleTokenizer().tokenize(text)
    nltk_tokens = NLTKTokenizer().tokenize(text)
    spacy_tokens = SpacyTokenizer().tokenize(text)

    assert len(simple) > 0
    assert len(nltk_tokens) >= len(simple)
    assert len(spacy_tokens) >= len(simple)
    assert "," in nltk_tokens
    assert "." in spacy_tokens


def test_tokenizer_factory_and_invalid_type() -> None:
    """Verify TokenizerFactory returns correct classes and rejects unknown keys."""
    assert isinstance(TokenizerFactory.create("simple"), SimpleTokenizer)
    assert isinstance(TokenizerFactory.create("nltk"), NLTKTokenizer)
    assert isinstance(TokenizerFactory.create("spacy"), SpacyTokenizer)

    with pytest.raises(ValueError):
        TokenizerFactory.create("unknown_tokenizer")