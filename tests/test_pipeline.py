"""Integration tests for mocked HTTP scraping, exporting, and end-to-end NLPPipeline execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from export.exporter import DataExporter
from export.report_generator import ReportGenerator
from models.article import Article
from pipeline.nlp_pipeline import NLPPipeline
from scraping.corpus_builder import CorpusBuilder
from visualization.heaps import HeapsPlotter
from visualization.wordcloud_gen import WordCloudGenerator
from visualization.zipf import ZipfPlotter


def _create_sample_articles(count: int = 6) -> list[Article]:
    """Create realistic Russian news articles for deterministic offline testing."""
    base_texts = [
        "Российские исследователи в Москве представили новую систему искусственного интеллекта для анализа больших текстовых данных и обработки естественного языка.",
        "Экономический форум в Санкт-Петербурге собрал представителей крупнейших технологических компаний и научных институтов страны для обсуждения цифровизации.",
        "Космический аппарат успешно вышел на расчетную орбиту и передал первые телеметрические данные в центр управления полетами.",
        "Министерство науки и высшего образования объявило о запуске новой программы поддержки молодых ученых и инженерных лабораторий.",
        "Развитие возобновляемых источников энергии и модернизация энергетической инфраструктуры стали главными темами международной конференции.",
        "Новые алгоритмы машинного обучения позволяют существенно повысить точность медицинской диагностики и прогнозирования сложных процессов.",
    ]
    articles: list[Article] = []
    for i in range(count):
        text = base_texts[i % len(base_texts)] + f" Дополнительный уникальный контекст статьи номер {i + 1}."
        articles.append(
            Article(
                url=f"https://lenta.ru/news/2026/05/01/article_{i + 1}/",
                title=f"Тестовая новостная статья номер {i + 1}",
                text=text,
                date="2026-05-01",
                source="Lenta.ru",
            )
        )
    return articles


def test_end_to_end_pipeline_with_mocked_scraper(tmp_path: Path) -> None:
    """Verify the entire NLPPipeline executes offline using mocked scrapers and temporary directories."""
    mock_scraper = MagicMock()
    mock_scraper.source_name = "Lenta.ru"
    mock_scraper.scrape_articles.return_value = _create_sample_articles(6)

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    plots_dir = tmp_path / "plots"
    reports_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"

    builder = CorpusBuilder(
        scrapers=[mock_scraper],
        raw_corpus_path=raw_dir / "corpus.json",
        cache_dir=cache_dir,
        min_articles=5,
        min_words=50,
    )
    exporter = DataExporter(raw_dir=raw_dir, processed_dir=processed_dir)
    report_gen = ReportGenerator(reports_dir=reports_dir)
    zipf_plotter = ZipfPlotter(output_path=plots_dir / "zipf_law.png")
    heaps_plotter = HeapsPlotter(output_path=plots_dir / "heaps_law.png")
    wc_gen = WordCloudGenerator(output_path=plots_dir / "wordcloud.png")

    pipeline = NLPPipeline(
        corpus_builder=builder,
        exporter=exporter,
        report_generator=report_gen,
        zipf_plotter=zipf_plotter,
        heaps_plotter=heaps_plotter,
        wordcloud_generator=wc_gen,
    )

    state = pipeline.run(mode="scrape", show_plots=False)

    assert len(state.raw_articles) == 6
    assert state.validation_report is not None and state.validation_report.is_compliant
    assert state.zipf_result is not None
    assert state.heaps_result is not None
    assert (raw_dir / "corpus.json").exists()
    assert (processed_dir / "corpus_processed.csv").exists()
    assert (processed_dir / "frequency_dict.txt").exists()
    assert (processed_dir / "stopwords_extended.txt").exists()
    assert (plots_dir / "zipf_law.png").exists()
    assert (plots_dir / "heaps_law.png").exists()
    assert (plots_dir / "wordcloud.png").exists()
    assert state.academic_report_path is not None and state.academic_report_path.exists()