"""Главный конвейер обработки русскоязычного новостного корпуса."""

import logging
from typing import Optional

from analysis.corpus_statistics import CorpusStatistics
from analysis.evaluator import LinguisticEvaluator
from analysis.frequency import FrequencyAnalyzer
from analysis.law_fitter import LawFitter
from export.exporter import DataExporter
from export.report_generator import ReportGenerator
from models.article import ProcessedArticle
from pipeline.pipeline_state import PipelineState
from preprocessing.cleaner import TextCleaner
from preprocessing.lemmatizers import PyMorphyLemmatizer
from preprocessing.stemmer import SnowballStemmerWrapper
from preprocessing.stopwords import StopWordFilter
from preprocessing.tokenizers import SimpleTokenizer
from scraping.corpus_builder import CorpusBuilder
from visualization.heaps import HeapsPlotter
from visualization.wordcloud_gen import WordCloudGenerator
from visualization.zipf import ZipfPlotter

logger = logging.getLogger(__name__)


class NLPPipeline:
    """Координирует сбор, очистку, лемматизацию, анализ и экспорт данных."""

    def __init__(
        self,
        corpus_builder: Optional[CorpusBuilder] = None,
        exporter: Optional[DataExporter] = None,
        report_generator: Optional[ReportGenerator] = None,
        zipf_plotter: Optional[ZipfPlotter] = None,
        heaps_plotter: Optional[HeapsPlotter] = None,
        wordcloud_generator: Optional[WordCloudGenerator] = None,
    ) -> None:
        self.corpus_builder = corpus_builder or CorpusBuilder()
        self.cleaner = TextCleaner()
        self.tokenizer = SimpleTokenizer()
        self.pymorphy = PyMorphyLemmatizer()
        self.stemmer = SnowballStemmerWrapper()
        self.stopword_filter = StopWordFilter()
        self.evaluator = LinguisticEvaluator()
        self.exporter = exporter or DataExporter()
        self.report_generator = report_generator or ReportGenerator()
        self.zipf_plotter = zipf_plotter or ZipfPlotter()
        self.heaps_plotter = heaps_plotter or HeapsPlotter()
        self.wordcloud_generator = wordcloud_generator or WordCloudGenerator()

    def run(self, mode: str = "scrape", show_plots: bool = False) -> PipelineState:
        """Запускает полный цикл обработки корпуса."""
        state = PipelineState()

        # 1. Загрузка или сбор корпуса
        if mode in ("load-cached", "skip-scrape"):
            articles = self.corpus_builder.load_raw_corpus()
            if not articles and mode == "load-cached":
                articles, _ = self.corpus_builder.build_corpus()
        else:
            articles, _ = self.corpus_builder.build_corpus()

        if not articles:
            raise RuntimeError("Корпус пуст! Запустите программу с флагом --scrape.")

        state.raw_articles = articles
        logger.info("Загружено статей для обработки: %d", len(articles))

        # 2. Очистка, токенизация, лемматизация и фильтрация стоп-слов
        all_pymorphy_lemmas = []
        temp_items = []
        for article in articles:
            clean_txt = self.cleaner.clean(article.text)
            tokens = self.tokenizer.tokenize(clean_txt)
            lemmas = self.pymorphy.lemmatize_tokens(tokens)
            stems = self.stemmer.stem_tokens(tokens)
            all_pymorphy_lemmas.append(lemmas)
            temp_items.append((article, clean_txt, tokens, lemmas, stems))

        filtered_docs, sw_stats = self.stopword_filter.filter_corpus(all_pymorphy_lemmas)

        for (article, clean_txt, tokens, lemmas, stems), filt_lemmas in zip(temp_items, filtered_docs):
            state.processed_articles.append(
                ProcessedArticle(
                    article=article,
                    clean_text=clean_txt,
                    tokens=tokens,
                    nltk_tokens=tokens,
                    spacy_tokens=tokens,
                    pymorphy_lemmas=lemmas,
                    spacy_lemmas=lemmas,
                    snowball_stems=stems,
                    filtered_lemmas=filt_lemmas,
                )
            )

        # 3. Частотный анализ и законы Ципфа и Хипса
        freq_analyzer = FrequencyAnalyzer(filtered_docs)
        state.frequency_entries = freq_analyzer.get_ranked_entries()
        state.zipf_result = LawFitter.fit_zipf(state.frequency_entries)
        state.heaps_result = LawFitter.fit_heaps(filtered_docs)
        state.stats = CorpusStatistics.compute(articles, sw_stats, freq_analyzer.vocabulary_size)

        # 4. Сравнение инструментов
        tok_comp = self.evaluator.evaluate_tokenizers(articles)
        morph_comp = self.evaluator.evaluate_morphology()

        # 5. Построение графиков
        self.zipf_plotter.plot(state.zipf_result, show=show_plots)
        self.heaps_plotter.plot(state.heaps_result, show=show_plots)
        self.wordcloud_generator.generate(freq_analyzer.counter, show=show_plots)

        # 6. Сохранение всех файлов и генерация отчета
        self.exporter.export_processed_csv(state.processed_articles)
        self.exporter.export_frequency_dict(state.frequency_entries)
        self.exporter.export_stopwords(self.stopword_filter.stop_words)
        self.exporter.export_evaluations(tok_comp, morph_comp)

        state.report_path = self.report_generator.generate(
            stats=state.stats,
            tok_comparisons=tok_comp,
            morph_rows=morph_comp,
            top_30=freq_analyzer.get_top_30(),
            zipf_data=state.zipf_result,
            heaps_data=state.heaps_result,
        )
        return state