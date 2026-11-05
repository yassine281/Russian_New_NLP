"""Unit tests for FrequencyAnalyzer, Zipf regression, and Heaps' law estimation."""

from __future__ import annotations

import pytest

from analysis.frequency import FrequencyAnalyzer
from analysis.law_fitter import LawFitter


def test_frequency_analyzer_ranking() -> None:
    """Verify FrequencyAnalyzer computes accurate counts, vocabulary size, and ranks."""
    corpus = [
        ["россия", "экономика", "россия", "наука"],
        ["россия", "экономика", "технология"],
    ]
    analyzer = FrequencyAnalyzer(corpus)
    assert analyzer.total_tokens == 7
    assert analyzer.vocabulary_size == 4

    top = analyzer.get_top_30()
    assert top[0].word == "россия"
    assert top[0].count == 3
    assert top[0].rank == 1
    assert top[1].word == "экономика"
    assert top[1].count == 2
    assert top[1].rank == 2


def test_zipf_and_heaps_fitting() -> None:
    """Verify Zipf linear regression and Heaps power-law fitting on synthetic power-law data."""
    corpus = [
        ["слово1"] * 100 + ["слово2"] * 50 + ["слово3"] * 33 + ["слово4"] * 25,
        ["слово1"] * 20 + ["слово5"] * 20 + ["слово6"] * 16 + ["слово7"] * 14,
        ["слово8"] * 12 + ["слово9"] * 11 + ["слово10"] * 10,
    ]
    analyzer = FrequencyAnalyzer(corpus)
    entries = analyzer.get_ranked_entries()

    zipf_res = LawFitter.fit_zipf(entries)
    assert zipf_res.slope < 0.0
    assert 0.0 <= zipf_res.r_squared <= 1.0

    heaps_res = LawFitter.fit_heaps(corpus)
    assert heaps_res.k > 0.0
    assert heaps_res.beta > 0.0
    assert heaps_res.cumulative_tokens == [208, 278, 311]
    assert heaps_res.cumulative_vocab == [4, 7, 10]


def test_law_fitter_insufficient_data() -> None:
    """Verify LawFitter raises ValueError when given insufficient observations."""
    with pytest.raises(ValueError):
        LawFitter.fit_zipf([])
    with pytest.raises(ValueError):
        LawFitter.fit_heaps([["одно_слово"]])