"""State container tracking intermediate and final artifacts across the NLP pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from analysis.corpus_statistics import CorpusSummaryStats
from analysis.evaluator import MorphEvaluationReport, TokenizerEvaluationReport
from analysis.law_fitter import HeapsFitResult, ZipfFitResult
from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry
from preprocessing.stopwords import StopWordStats
from scraping.corpus_builder import CorpusValidationReport


@dataclass
class PipelineState:
    """Mutable orchestration state holding artifacts produced by each pipeline stage."""

    mode: str = "scrape"
    raw_articles: list[Article] = field(default_factory=list)
    validation_report: Optional[CorpusValidationReport] = None
    processed_articles: list[ProcessedArticle] = field(default_factory=list)
    stopword_stats: Optional[StopWordStats] = None
    frequency_entries: list[FrequencyEntry] = field(default_factory=list)
    top_30_entries: list[FrequencyEntry] = field(default_factory=list)
    top_1000_entries: list[FrequencyEntry] = field(default_factory=list)
    tokenizer_evaluation: Optional[TokenizerEvaluationReport] = None
    morph_evaluation: Optional[MorphEvaluationReport] = None
    zipf_result: Optional[ZipfFitResult] = None
    heaps_result: Optional[HeapsFitResult] = None
    corpus_summary: Optional[CorpusSummaryStats] = None
    exported_files: dict[str, Path] = field(default_factory=dict)
    plot_files: dict[str, Path] = field(default_factory=dict)
    academic_report_path: Optional[Path] = None