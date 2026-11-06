"""Пакет частотного и статистического анализа корпуса."""

from analysis.corpus_statistics import CorpusStatistics
from analysis.evaluator import LinguisticEvaluator
from analysis.frequency import FrequencyAnalyzer
from analysis.law_fitter import LawFitter

__all__ = [
    "CorpusStatistics",
    "FrequencyAnalyzer",
    "LawFitter",
    "LinguisticEvaluator",
]