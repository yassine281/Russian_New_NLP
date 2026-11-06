"""Интеграционный тест полного конвейера NLPPipeline без интернета."""

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


def test_pipeline_offline(tmp_path: Path) -> None:
    sample_articles = [
        Article(
            url=f"https://lenta.ru/news/{i}",
            title=f"Новость {i}",
            text="Российские ученые разработали новый метод анализа больших текстовых данных и систем.",
            source="Lenta.ru",
        )
        for i in range(1, 5)
    ]

    mock_builder = MagicMock(spec=CorpusBuilder)
    mock_builder.build_corpus.return_value = (sample_articles, True)

    pipeline = NLPPipeline(
        corpus_builder=mock_builder,
        exporter=DataExporter(processed_dir=tmp_path / "processed"),
        report_generator=ReportGenerator(reports_dir=tmp_path / "reports"),
        zipf_plotter=ZipfPlotter(output_path=tmp_path / "plots/zipf.png"),
        heaps_plotter=HeapsPlotter(output_path=tmp_path / "plots/heaps.png"),
        wordcloud_generator=WordCloudGenerator(output_path=tmp_path / "plots/wc.png"),
    )

    state = pipeline.run(mode="scrape", show_plots=False)
    assert len(state.raw_articles) == 4
    assert (tmp_path / "processed/corpus_processed.csv").exists()
    assert (tmp_path / "processed/frequency_dict.txt").exists()
    assert (tmp_path / "processed/stopwords_extended.txt").exists()
    assert state.report_path is not None and state.report_path.exists()