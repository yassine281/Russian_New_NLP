"""Central NLP pipeline controller coordinating scraping, preprocessing, analysis, and reporting."""

from __future__ import annotations

import logging
from typing import Optional

from analysis.corpus_statistics import CorpusStatistics
from analysis.evaluator import LinguisticEvaluator
from analysis.frequency import FrequencyAnalyzer
from analysis.law_fitter import LawFitter
from export.exporter import DataExporter
from export.report_generator import ReportGenerator
from models.article import Article, ProcessedArticle
from pipeline.pipeline_state import PipelineState
from preprocessing.cleaner import TextCleaner
from preprocessing.lemmatizers import PyMorphyLemmatizer, SpacyLemmatizer
from preprocessing.stemmer import SnowballStemmerWrapper
from preprocessing.stopwords import StopWordFilter
from preprocessing.tokenizers import BaseTokenizer, TokenizerFactory
from scraping.corpus_builder import CorpusBuilder
from visualization.heaps import HeapsPlotter
from visualization.wordcloud_gen import WordCloudGenerator
from visualization.zipf import ZipfPlotter

logger = logging.getLogger(__name__)


class NLPPipeline:
    """End-to-end Russian News NLP Pipeline orchestrated via Dependency Injection."""

    def __init__(
        self,
        corpus_builder: Optional[CorpusBuilder] = None,
        cleaner: Optional[TextCleaner] = None,
        simple_tokenizer: Optional[BaseTokenizer] = None,
        nltk_tokenizer: Optional[BaseTokenizer] = None,
        spacy_tokenizer: Optional[BaseTokenizer] = None,
        pymorphy_lemmatizer: Optional[PyMorphyLemmatizer] = None,
        spacy_lemmatizer: Optional[SpacyLemmatizer] = None,
        stemmer: Optional[SnowballStemmerWrapper] = None,
        stopword_filter: Optional[StopWordFilter] = None,
        evaluator: Optional[LinguisticEvaluator] = None,
        exporter: Optional[DataExporter] = None,
        report_generator: Optional[ReportGenerator] = None,
        zipf_plotter: Optional[ZipfPlotter] = None,
        heaps_plotter: Optional[HeapsPlotter] = None,
        wordcloud_generator: Optional[WordCloudGenerator] = None,
    ) -> None:
        """Initialize the NLP pipeline with injectable components."""
        self.corpus_builder: CorpusBuilder = corpus_builder or CorpusBuilder()
        self.cleaner: TextCleaner = cleaner or TextCleaner()
        self.simple_tokenizer: BaseTokenizer = simple_tokenizer or TokenizerFactory.create("simple")
        self.nltk_tokenizer: BaseTokenizer = nltk_tokenizer or TokenizerFactory.create("nltk")
        self.spacy_tokenizer: BaseTokenizer = spacy_tokenizer or TokenizerFactory.create("spacy")
        self.pymorphy: PyMorphyLemmatizer = pymorphy_lemmatizer or PyMorphyLemmatizer()
        self.spacy_lemma: SpacyLemmatizer = spacy_lemmatizer or SpacyLemmatizer()
        self.stemmer: SnowballStemmerWrapper = stemmer or SnowballStemmerWrapper()
        self.stopword_filter: StopWordFilter = stopword_filter or StopWordFilter()

        self.evaluator: LinguisticEvaluator = evaluator or LinguisticEvaluator(
            simple_tokenizer=self.simple_tokenizer,
            nltk_tokenizer=self.nltk_tokenizer,
            spacy_tokenizer=self.spacy_tokenizer,
            pymorphy_lemmatizer=self.pymorphy,
            spacy_lemmatizer=self.spacy_lemma,
            snowball_stemmer=self.stemmer,
        )
        self.exporter: DataExporter = exporter or DataExporter()
        self.report_generator: ReportGenerator = report_generator or ReportGenerator()
        self.zipf_plotter: ZipfPlotter = zipf_plotter or ZipfPlotter()
        self.heaps_plotter: HeapsPlotter = heaps_plotter or HeapsPlotter()
        self.wordcloud_generator: WordCloudGenerator = (
            wordcloud_generator or WordCloudGenerator()
        )

    def run(
        self,
        mode: str = "scrape",
        show_plots: bool = False,
        compute_spacy_on_full_corpus: bool = False,
    ) -> PipelineState:
        """Execute the complete NLP pipeline in the specified acquisition mode.

        Args:
            mode: One of 'scrape' (live scrape with cache), 'load-cached' (use cache/saved JSON
                and scrape only if missing), or 'skip-scrape' (strictly offline from saved JSON).
            show_plots: Whether to open interactive Matplotlib windows during execution.
            compute_spacy_on_full_corpus: If True, runs full neural spaCy lemmatization on every
                article in addition to the benchmark subset (Pymorphy3 is always run on 100% of
                articles as required by the lab manual[cite: 6, 7]).

        Returns:
            Populated `PipelineState` containing all results and file paths.

        Raises:
            RuntimeError: If no articles are available to process.
        """
        state = PipelineState(mode=mode)
        logger.info("=== STAGE 1: Corpus Acquisition (mode=%s) ===", mode)
        raw_articles = self._acquire_articles(mode=mode)
        if not raw_articles:
            raise RuntimeError(
                "No articles available for processing. Run with --scrape while connected to the internet."
            )

        state.raw_articles = raw_articles
        state.validation_report = self.corpus_builder.validate_corpus(raw_articles)

        logger.info("=== STAGE 2: Text Cleaning, Tokenization, Lemmatization & Stemming ===")
        state.processed_articles, state.stopword_stats = self._preprocess_corpus(
            raw_articles,
            compute_spacy_all=compute_spacy_on_full_corpus,
        )

        logger.info("=== STAGE 3: Linguistic Tool Evaluation (Tokenizers & Morphology) ===")
        state.tokenizer_evaluation = self.evaluator.evaluate_tokenizers(raw_articles, sample_count=3)
        state.morph_evaluation = self.evaluator.evaluate_morphology(raw_articles, corpus_sample_size=12)

        logger.info("=== STAGE 4: Frequency Dictionary & Statistical Laws (Zipf & Heaps) ===")
        filtered_docs = [pa.filtered_lemmas for pa in state.processed_articles]
        freq_analyzer = FrequencyAnalyzer(filtered_docs)
        state.frequency_entries = freq_analyzer.get_ranked_entries()
        state.top_30_entries = freq_analyzer.get_top_30()
        state.top_1000_entries = freq_analyzer.get_top_1000()

        state.zipf_result = LawFitter.fit_zipf(state.frequency_entries)
        state.heaps_result = LawFitter.fit_heaps(filtered_docs)
        state.corpus_summary = CorpusStatistics.compute(
            raw_articles=state.raw_articles,
            processed_articles=state.processed_articles,
            stopword_stats=state.stopword_stats,
            freq_analyzer=freq_analyzer,
            validation_report=state.validation_report,
        )

        logger.info("=== STAGE 5: Generating 300 DPI Visualizations ===")
        state.plot_files["zipf"] = self.zipf_plotter.plot(
            state.zipf_result, show_interactive=show_plots
        )
        state.plot_files["heaps"] = self.heaps_plotter.plot(
            state.heaps_result, show_interactive=show_plots
        )
        state.plot_files["wordcloud"] = self.wordcloud_generator.generate(
            freq_analyzer.counter, show_interactive=show_plots
        )

        logger.info("=== STAGE 6: Exporting Artifacts & Generating Academic Report ===")
        state.exported_files["raw_json"] = self.exporter.export_raw_corpus_json(state.raw_articles)
        state.exported_files["processed_csv"] = self.exporter.export_processed_corpus_csv(
            state.processed_articles
        )
        state.exported_files["frequency_txt"] = self.exporter.export_frequency_dict_txt(
            state.top_1000_entries
        )
        state.exported_files["stopwords_txt"] = self.exporter.export_stopwords_txt(
            self.stopword_filter.get_sorted_stopwords()
        )
        state.exported_files["linguistic_csv"] = self.exporter.export_linguistic_evaluation_csv(
            state.morph_evaluation
        )
        state.exported_files["tokenizer_txt"] = self.exporter.export_tokenizer_comparison_txt(
            state.tokenizer_evaluation
        )
        state.exported_files["stats_txt"] = self.exporter.export_statistical_analysis_txt(
            summary_stats=state.corpus_summary,
            zipf_result=state.zipf_result,
            heaps_result=state.heaps_result,
            top_30=state.top_30_entries,
        )
        state.academic_report_path = self.report_generator.generate_report(
            validation_report=state.validation_report,
            summary_stats=state.corpus_summary,
            tok_report=state.tokenizer_evaluation,
            morph_report=state.morph_evaluation,
            top_30=state.top_30_entries,
            zipf_result=state.zipf_result,
            heaps_result=state.heaps_result,
            plot_paths=state.plot_files,
        )

        logger.info("=== NLP PIPELINE COMPLETED SUCCESSFULLY ===")
        return state

    def _acquire_articles(self, mode: str) -> list[Article]:
        """Acquire articles according to the selected CLI execution mode."""
        normalized_mode = mode.strip().lower()
        if normalized_mode == "skip-scrape":
            if self.corpus_builder.raw_corpus_path.exists():
                return self.corpus_builder.load_raw_corpus()
            cached = self.corpus_builder.load_cached_articles()
            if cached:
                return cached
            raise FileNotFoundError(
                f"Cannot run --skip-scrape: no existing corpus at {self.corpus_builder.raw_corpus_path}"
            )

        if normalized_mode == "load-cached":
            if self.corpus_builder.raw_corpus_path.exists():
                loaded = self.corpus_builder.load_raw_corpus()
                if loaded:
                    return loaded
            cached = self.corpus_builder.load_cached_articles()
            if cached:
                return cached
            logger.info("No local cache found; falling back to live scraping.")

        articles, _ = self.corpus_builder.build_corpus(use_cache=True)
        return articles

    def _preprocess_corpus(
        self,
        articles: list[Article],
        compute_spacy_all: bool = False,
    ) -> tuple[list[ProcessedArticle], object]:
        """Clean, tokenize, lemmatize, stem, and filter stop words across the corpus[cite: 6, 7]."""
        raw_clean_texts: list[str] = []
        all_pymorphy_lemmas: list[list[str]] = []
        intermediate_records: list[dict[str, object]] = []

        for idx, article in enumerate(articles):
            clean_txt = self.cleaner.clean(article.text, preserve_punctuation=False)
            raw_clean_texts.append(clean_txt)

            simple_tokens = self.simple_tokenizer.tokenize(clean_txt)
            # Run NLTK & spaCy full token/lemma lists on the first 5 articles or all if requested
            if compute_spacy_all or idx < 5:
                nltk_tokens = self.nltk_tokenizer.tokenize(clean_txt)
                spacy_tokens = self.spacy_tokenizer.tokenize(clean_txt)
                spacy_lemmas = self.spacy_lemma.lemmatize_text(clean_txt)
            else:
                nltk_tokens = simple_tokens
                spacy_tokens = simple_tokens
                spacy_lemmas = []

            pym_lemmas = self.pymorphy.lemmatize_tokens(simple_tokens)
            stems = self.stemmer.stem_tokens(simple_tokens)

            all_pymorphy_lemmas.append(pym_lemmas)
            intermediate_records.append(
                {
                    "article": article,
                    "clean_text": clean_txt,
                    "tokens": simple_tokens,
                    "nltk_tokens": nltk_tokens,
                    "spacy_tokens": spacy_tokens,
                    "pymorphy_lemmas": pym_lemmas,
                    "spacy_lemmas": spacy_lemmas or pym_lemmas,
                    "snowball_stems": stems,
                }
            )

        filtered_corpus, stop_stats = self.stopword_filter.filter_corpus(all_pymorphy_lemmas)

        processed_list: list[ProcessedArticle] = []
        for rec, filt_lemmas in zip(intermediate_records, filtered_corpus):
            processed_list.append(
                ProcessedArticle(
                    article=rec["article"],  # type: ignore[arg-type]
                    clean_text=str(rec["clean_text"]),
                    tokens=list(rec["tokens"]),  # type: ignore[arg-type]
                    nltk_tokens=list(rec["nltk_tokens"]),  # type: ignore[arg-type]
                    spacy_tokens=list(rec["spacy_tokens"]),  # type: ignore[arg-type]
                    pymorphy_lemmas=list(rec["pymorphy_lemmas"]),  # type: ignore[arg-type]
                    spacy_lemmas=list(rec["spacy_lemmas"]),  # type: ignore[arg-type]
                    snowball_stems=list(rec["snowball_stems"]),  # type: ignore[arg-type]
                    filtered_lemmas=filt_lemmas,
                )
            )

        return processed_list, stop_stats