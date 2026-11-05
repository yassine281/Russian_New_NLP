"""Data exporter generating CSV, TXT, and JSON artifacts for the NLP pipeline."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

import config
from analysis.corpus_statistics import CorpusSummaryStats
from analysis.evaluator import MorphEvaluationReport, TokenizerEvaluationReport
from analysis.law_fitter import HeapsFitResult, ZipfFitResult
from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry

logger = logging.getLogger(__name__)


class DataExporter:
    """Exports processed corpora, frequency dictionaries, stop words, and evaluation tables[cite: 9]."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        """Initialize the DataExporter with output directory paths.

        Args:
            raw_dir: Directory for raw JSON corpus files (`data/raw`).
            processed_dir: Directory for processed CSV/TXT outputs (`data/processed`).
        """
        self.raw_dir: Path = Path(raw_dir or getattr(config, "RAW_DATA_DIR", Path("data/raw")))
        self.processed_dir: Path = Path(
            processed_dir or getattr(config, "PROCESSED_DATA_DIR", Path("data/processed"))
        )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def export_raw_corpus_json(
        self,
        articles: Sequence[Article],
        filename: str = "corpus.json",
    ) -> Path:
        """Export raw articles to `data/raw/corpus.json`[cite: 5]."""
        out_path = self.raw_dir / filename
        payload = [a.to_dict() for a in articles]
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Exported raw corpus JSON to %s", out_path)
        return out_path

    def export_processed_corpus_csv(
        self,
        processed_articles: Sequence[ProcessedArticle],
        filename: str = "corpus_processed.csv",
    ) -> Path:
        """Export cleaned and lemmatized articles to `corpus_processed.csv`[cite: 9]."""
        out_path = self.processed_dir / filename
        rows = [
            {
                "url": item.article.url,
                "title": item.article.title,
                "date": item.article.date or "",
                "source": item.article.source,
                "clean": item.clean_text,
                "lemmas_filtered": " ".join(item.filtered_lemmas),
            }
            for item in processed_articles
        ]
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Exported processed corpus CSV to %s", out_path)
        return out_path

    def export_frequency_dict_txt(
        self,
        entries: Sequence[FrequencyEntry],
        filename: str = "frequency_dict.txt",
    ) -> Path:
        """Export top 1000 words as `word\\tfrequency` lines to `frequency_dict.txt`[cite: 9]."""
        out_path = self.processed_dir / filename
        with out_path.open("w", encoding="utf-8") as fh:
            for entry in list(entries)[:1000]:
                fh.write(f"{entry.word}\t{entry.count}\n")
        logger.info("Exported frequency dictionary (%d words) to %s", min(1000, len(entries)), out_path)
        return out_path

    def export_stopwords_txt(
        self,
        stopwords_list: Sequence[str],
        filename: str = "stopwords_extended.txt",
    ) -> Path:
        """Export sorted extended stop-word list to `stopwords_extended.txt`[cite: 9]."""
        out_path = self.processed_dir / filename
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(set(stopwords_list))) + "\n")
        logger.info("Exported %d extended stop words to %s", len(stopwords_list), out_path)
        return out_path

    def export_linguistic_evaluation_csv(
        self,
        morph_report: MorphEvaluationReport,
        filename: str = "linguistic_evaluation.csv",
    ) -> Path:
        """Export validation and corpus morphological comparison tables to CSV[cite: 7]."""
        out_path = self.processed_dir / filename
        all_rows = list(morph_report.validation_rows) + list(morph_report.corpus_rows)
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["dataset_type", "word", "pymorphy3", "spacy", "snowball", "mystem"]
            )
            for row in all_rows:
                writer.writerow(
                    [
                        row.dataset_type,
                        row.original_word,
                        row.pymorphy_lemma,
                        row.spacy_lemma,
                        row.snowball_stem,
                        row.mystem_lemma,
                    ]
                )
        logger.info("Exported linguistic evaluation CSV to %s", out_path)
        return out_path

    def export_tokenizer_comparison_txt(
        self,
        tok_report: TokenizerEvaluationReport,
        filename: str = "tokenizer_comparison.txt",
    ) -> Path:
        """Export the 3-article tokenizer comparison and timing benchmarks to TXT[cite: 6]."""
        out_path = self.processed_dir / filename
        lines: list[str] = [
            "======================================================================",
            "ЭКСПЕРИМЕНТ ПО СРАВНЕНИЮ ТОКЕНИЗАТОРОВ (Simple vs NLTK vs spaCy)",
            "======================================================================",
            "",
        ]
        for sample in tok_report.samples:
            lines.extend(
                [
                    f"Статья #{sample.article_index}: {sample.article_title}",
                    f"URL: {sample.article_url}",
                    f"Текст (первые 200 символов): {sample.snippet_200}",
                    f"Simple (первые 15): {sample.simple_first_15}",
                    f"NLTK   (первые 15): {sample.nltk_first_15}",
                    f"spaCy  (первые 15): {sample.spacy_first_15}",
                    f"Всего токенов в статье -> Simple: {sample.simple_total_count}, "
                    f"NLTK: {sample.nltk_total_count}, spaCy: {sample.spacy_total_count}",
                    "-" * 70,
                ]
            )

        lines.append("\nСкорость обработки (мс):")
        for tool_name, ms in tok_report.timings_ms.items():
            lines.append(
                f"  - {tool_name}: {ms:.2f} мс (всего токенов: {tok_report.total_tokens_by_tool.get(tool_name, 0)})"
            )

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.info("Exported tokenizer comparison TXT to %s", out_path)
        return out_path

    def export_statistical_analysis_txt(
        self,
        summary_stats: CorpusSummaryStats,
        zipf_result: ZipfFitResult,
        heaps_result: HeapsFitResult,
        top_30: Sequence[FrequencyEntry],
        filename: str = "statistical_analysis.txt",
    ) -> Path:
        """Export corpus statistics, Zipf slope, Heaps parameters, and Top-30 table to TXT[cite: 8]."""
        out_path = self.processed_dir / filename
        lines: list[str] = [
            "======================================================================",
            "СТАТИСТИЧЕСКИЙ АНАЛИЗ КОРПУСА И ЗАКОНОВ ЦИПФА / ХИПСА",
            "======================================================================",
            f"Число собранных статей: {summary_stats.total_articles}",
            f"Общий объём сырого корпуса (слов): {summary_stats.total_raw_words}",
            f"Средняя длина статьи (слов): {summary_stats.mean_raw_words_per_article:.1f}",
            f"Медианная длина статьи (слов): {summary_stats.median_raw_words_per_article:.1f}",
            f"Минимум / Максимум слов в статье: {summary_stats.min_raw_words_in_article} / {summary_stats.max_raw_words_in_article}",
            "",
            "--- Фильтрация стоп-слов ---",
            f"Токенов до фильтрации: {summary_stats.total_tokens_before_stopwords}",
            f"Токенов после фильтрации: {summary_stats.total_tokens_after_stopwords}",
            f"Удалено токенов: {summary_stats.stopwords_removed_count} ({summary_stats.stopwords_removed_pct:.2f}%)",
            f"Уникальных стоп-слов встречено и удалено: {summary_stats.unique_stopwords_encountered}",
            "",
            "--- Частотный словарь ---",
            f"Размер уникального словаря V: {summary_stats.unique_vocabulary_size}",
            f"Коэффициент лексического разнообразия (TTR): {summary_stats.lexical_diversity_ttr:.4f}",
            "",
            "--- Параметры статистических законов ---",
            f"Zipf slope: {zipf_result.slope:.3f} (intercept={zipf_result.intercept:.3f}, R²={zipf_result.r_squared:.4f})",
            f"Heaps k: {heaps_result.k:.3f}",
            f"Heaps β: {heaps_result.beta:.3f} (R²={heaps_result.r_squared:.4f})",
            "",
            "--- Топ-30 наиболее частотных лемм ---",
            f"{'Ранг':<6} | {'Лемма':<25} | {'Частота':<10}",
            "-" * 46,
        ]
        for entry in top_30:
            lines.append(f"{entry.rank:<6} | {entry.word:<25} | {entry.count:<10}")

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.info("Exported statistical analysis TXT to %s", out_path)
        return out_path