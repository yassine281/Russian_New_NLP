"""Тесты частотного анализа и законов Ципфа и Хипса."""

from analysis.frequency import FrequencyAnalyzer
from analysis.law_fitter import LawFitter


def test_frequency_and_laws() -> None:
    docs = [
        ["россия", "наука", "россия", "технология", "россия"],
        ["наука", "экономика", "образование", "развитие"],
    ]
    analyzer = FrequencyAnalyzer(docs)
    entries = analyzer.get_ranked_entries()
    assert entries[0].word == "россия"
    assert entries[0].count == 3

    zipf = LawFitter.fit_zipf(entries)
    assert zipf["slope"] < 0

    heaps = LawFitter.fit_heaps(docs)
    assert heaps["k"] > 0
    assert heaps["n_values"] == [5, 9]