"""Command-line entry point for the Russian News NLP Pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from pipeline.nlp_pipeline import NLPPipeline

LOG_FORMAT: str = getattr(
    config,
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
)


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure console and UTF-8 file logging with the mandatory log format."""
    logs_dir = Path(getattr(config, "LOGS_DIR", Path("logs")))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line execution flags."""
    parser = argparse.ArgumentParser(
        description="Professional Russian News Corpus Collection & NLP Analysis Pipeline."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--scrape",
        action="store_true",
        help="Collect articles from permitted Russian news sources (uses cache + live scraping).",
    )
    mode_group.add_argument(
        "--load-cached",
        action="store_true",
        help="Load previously saved/cached corpus if available, scraping only if missing.",
    )
    mode_group.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Process existing saved corpus in data/raw/corpus.json without network requests.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display Matplotlib visualizations interactively in addition to saving PNGs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Russian News NLP Pipeline CLI."""
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("main")

    if args.skip_scrape:
        mode = "skip-scrape"
    elif args.load_cached:
        mode = "load-cached"
    else:
        mode = "scrape"

    logger.info("Starting RussianNewsNLP pipeline in '%s' mode...", mode)
    try:
        pipeline = NLPPipeline()
        state = pipeline.run(mode=mode, show_plots=args.show_plots)

        if state.corpus_summary and state.zipf_result and state.heaps_result:
            logger.info(
                "Corpus Summary: %d articles | %d raw words | %d unique lemmas",
                state.corpus_summary.total_articles,
                state.corpus_summary.total_raw_words,
                state.corpus_summary.unique_vocabulary_size,
            )
            logger.info(
                "Stop-words removed: %d / %d (%.2f%%)",
                state.corpus_summary.stopwords_removed_count,
                state.corpus_summary.total_tokens_before_stopwords,
                state.corpus_summary.stopwords_removed_pct,
            )
            logger.info("Zipf slope: %.3f", state.zipf_result.slope)
            logger.info("Heaps k: %.3f | Heaps β: %.3f", state.heaps_result.k, state.heaps_result.beta)
            logger.info("Academic report generated at: %s", state.academic_report_path)
        return 0
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())