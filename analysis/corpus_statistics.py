"""Aggregate corpus statistics calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from analysis.frequency import FrequencyAnalyzer
from models.article import Article, ProcessedArticle
from preprocessing.stopwords import StopWordStats
from scraping.corpus_builder import CorpusValidationReport


@dataclass(frozen=True)
class CorpusSummaryStats:
    """Comprehensive statistical profile of the raw and preprocessed Russian news corpus[cite: 5, 8]."""

    total_articles: int
    total_raw_words: int
    mean_raw_words_per_article: float
    median_raw_words_per_article: float
    min_raw_words_in_article: int
    max_raw_words_in_article: int
    total_tokens_before_stopwords: int
    total_tokens_after_stopwords: int
    stopwords_removed_count: int
    stopwords_removed_pct: float
    unique_stopwords_encountered: int
    unique_vocabulary_size: int
    lexical_diversity_ttr: float
    meets_academic_target: bool


class CorpusStatistics:
    """Computes descriptive statistics across raw and processed articles[cite: 5, 8]."""

    @staticmethod
    def compute(
        raw_articles: Sequence[Article],
        processed_articles: Sequence[ProcessedArticle],
        stopword_stats: StopWordStats,
        freq_analyzer: FrequencyAnalyzer,
        validation_report: CorpusValidationReport,
    ) -> CorpusSummaryStats:
        """Aggregate all corpus-level metrics into a single `CorpusSummaryStats` record.

        Args:
            raw_articles: Original `Article` instances.
            processed_articles: Preprocessed `ProcessedArticle` instances.
            stopword_stats: Stop-word removal statistics.
            freq_analyzer: Fitted `FrequencyAnalyzer`.
            validation_report: Corpus validation report.

        Returns:
            Populated `CorpusSummaryStats`.
        """
        word_counts = [a.word_count for a in raw_articles]
        if word_counts:
            total_raw = int(sum(word_counts))
            mean_raw = float(np.mean(word_counts))
            median_raw = float(np.median(word_counts))
            min_raw = int(min(word_counts))
            max_raw = int(max(word_counts))
        else:
            total_raw = 0
            mean_raw = 0.0
            median_raw = 0.0
            min_raw = 0
            max_raw = 0

        vocab_size = freq_analyzer.vocabulary_size
        after_tokens = stopword_stats.tokens_after
        ttr = (vocab_size / after_tokens) if after_tokens > 0 else 0.0

        return CorpusSummaryStats(
            total_articles=len(raw_articles),
            total_raw_words=total_raw,
            mean_raw_words_per_article=mean_raw,
            median_raw_words_per_article=median_raw,
            min_raw_words_in_article=min_raw,
            max_raw_words_in_article=max_raw,
            total_tokens_before_stopwords=stopword_stats.tokens_before,
            total_tokens_after_stopwords=after_tokens,
            stopwords_removed_count=stopword_stats.tokens_removed,
            stopwords_removed_pct=stopword_stats.percentage_removed,
            unique_stopwords_encountered=stopword_stats.unique_stopwords_removed,
            unique_vocabulary_size=vocab_size,
            lexical_diversity_ttr=ttr,
            meets_academic_target=validation_report.is_compliant,
        )