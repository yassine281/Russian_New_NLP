"""Сохранение результатов обработки корпуса в CSV и TXT (Задание 6)."""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import pandas as pd

import config
from models.article import ProcessedArticle
from models.frequency import FrequencyEntry


class DataExporter:
    """Экспортирует обработанный корпус, частотный словарь и список стоп-слов."""

    def __init__(self, processed_dir: Path = config.PROCESSED_DATA_DIR) -> None:
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def export_processed_csv(self, articles: Sequence[ProcessedArticle]) -> Path:
        """Сохраняет очищенный корпус в corpus_processed.csv (Задание 6, Шаг 1)."""
        out_path = self.processed_dir / "corpus_processed.csv"
        rows = [
            {
                "url": item.article.url,
                "title": item.article.title,
                "clean": item.clean_text,
                "lemmas_filtered": " ".join(item.filtered_lemmas),
            }
            for item in articles
        ]
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def export_frequency_dict(self, entries: Sequence[FrequencyEntry]) -> Path:
        """Сохраняет топ-1000 слов в frequency_dict.txt (Задание 6, Шаг 2)."""
        out_path = self.processed_dir / "frequency_dict.txt"
        with open(out_path, "w", encoding="utf-8") as fh:
            for entry in list(entries)[:1000]:
                fh.write(f"{entry.word}\t{entry.count}\n")
        return out_path

    def export_stopwords(self, stop_words: Iterable[str]) -> Path:
        """Сохраняет расширенный список стоп-слов в stopwords_extended.txt (Задание 6, Шаг 3)."""
        out_path = self.processed_dir / "stopwords_extended.txt"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(set(stop_words))) + "\n")
        return out_path

    def export_evaluations(
        self,
        tok_comparisons: List[Dict[str, Any]],
        morph_rows: List[Dict[str, str]],
    ) -> None:
        """Сохраняет таблицы сравнения токенизаторов и лемматизаторов."""
        tok_path = self.processed_dir / "tokenizer_comparison.txt"
        with open(tok_path, "w", encoding="utf-8") as fh:
            for item in tok_comparisons:
                fh.write(f"Статья {item['index']}: {item['title']}\n")
                fh.write(f"Текст: {item['snippet_200']}\n")
                fh.write(f"Simple: {item['simple_15']}\n")
                fh.write(f"NLTK:   {item['nltk_15']}\n")
                fh.write(f"spaCy:  {item['spacy_15']}\n")
                fh.write("-" * 60 + "\n")

        morph_path = self.processed_dir / "linguistic_evaluation.csv"
        with open(morph_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["word", "pymorphy3", "spacy", "snowball"])
            writer.writeheader()
            writer.writerows(morph_rows)