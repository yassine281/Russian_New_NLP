"""Сравнение токенизаторов, лемматизаторов и стеммера (Задания 2 и 3)."""

from typing import Any, Dict, List
from models.article import Article
from preprocessing.cleaner import TextCleaner
from preprocessing.lemmatizers import PyMorphyLemmatizer, SpacyLemmatizer
from preprocessing.stemmer import SnowballStemmerWrapper
from preprocessing.tokenizers import NLTKTokenizer, SimpleTokenizer, SpacyTokenizer


class LinguisticEvaluator:
    """Проводит сравнение инструментов предобработки русского текста."""

    BENCHMARK_WORDS = (
        "красивая",
        "бегущий",
        "деревянный",
        "летать",
        "лето",
        "российских",
        "правительства",
        "исследователи",
        "экономического",
        "технологиями",
    )

    def __init__(self) -> None:
        self.simple_tok = SimpleTokenizer()
        self.nltk_tok = NLTKTokenizer()
        self.spacy_tok = SpacyTokenizer()
        self.pymorphy = PyMorphyLemmatizer()
        self.spacy_lem = SpacyLemmatizer()
        self.stemmer = SnowballStemmerWrapper()

    def evaluate_tokenizers(self, articles: List[Article], count: int = 3) -> List[Dict[str, Any]]:
        """Сравнивает 3 токенизатора на первых 3 статьях корпуса (Задание 2, Шаг 3)."""
        comparisons: List[Dict[str, Any]] = []
        for idx, article in enumerate(articles[:count], start=1):
            clean_txt = TextCleaner.clean(article.text, preserve_punctuation=True)
            comparisons.append({
                "index": idx,
                "title": article.title,
                "snippet_200": clean_txt[:200],
                "simple_15": self.simple_tok.tokenize(clean_txt)[:15],
                "nltk_15": self.nltk_tok.tokenize(clean_txt)[:15],
                "spacy_15": self.spacy_tok.tokenize(clean_txt)[:15],
            })
        return comparisons

    def evaluate_morphology(self) -> List[Dict[str, str]]:
        """Сравнивает Pymorphy3, spaCy и SnowballStemmer на 10 примерах (Задание 3, Шаг 4)."""
        rows: List[Dict[str, str]] = []
        for word in self.BENCHMARK_WORDS:
            rows.append({
                "word": word,
                "pymorphy3": self.pymorphy.lemmatize_token(word),
                "spacy": self.spacy_lem.lemmatize_token(word),
                "snowball": self.stemmer.stem_token(word),
            })
        return rows