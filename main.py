"""Точка входа для запуска проекта RussianNewsNLP."""

import argparse
import logging
import config
from pipeline.nlp_pipeline import NLPPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ русскоязычного новостного корпуса")
    parser.add_argument("--scrape", action="store_true", help="Собрать корпус с новостных сайтов")
    parser.add_argument("--load-cached", action="store_true", help="Использовать сохраненный корпус")
    parser.add_argument("--skip-scrape", action="store_true", help="Обработать существующий corpus.json")
    parser.add_argument("--show-plots", action="store_true", help="Показать графики на экране")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    if args.skip_scrape:
        mode = "skip-scrape"
    elif args.load_cached:
        mode = "load-cached"
    else:
        mode = "scrape"

    pipeline = NLPPipeline()
    state = pipeline.run(mode=mode, show_plots=args.show_plots)

    print("=" * 65)
    print(f"Число статей:       {state.stats['total_articles']}")
    print(f"Средняя длина:      {state.stats['average_words']:.1f} слов")
    print(f"Общий объём слов:   {state.stats['total_words']}")
    print(f"До стоп-слов:       {state.stats['tokens_before_stopwords']}, После: {state.stats['tokens_after_stopwords']} (Удалено: {state.stats['percentage_removed']:.1f}%)")
    print(f"Уникальных слов:    {state.stats['vocab_size']}")
    print(f"Закон Ципфа (slope): {state.zipf_result['slope']:.3f}")
    print(f"Закон Хипса (k, β):  k={state.heaps_result['k']:.2f}, beta={state.heaps_result['beta']:.3f}")
    print(f"Отчет сохранен в:   {state.report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()