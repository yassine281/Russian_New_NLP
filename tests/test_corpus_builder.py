"""Unit tests for persistent corpus building and validation."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config import CONFIG
from models import Article
from scraping import (
    CorpusBuilder,
    CorpusStorageError,
    ScrapingError,
)


class StubScraper:
    """Simple scraper substitute that does not access the internet."""

    def __init__(
        self,
        articles: list[Article] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.articles = articles or []
        self.error = error
        self.calls = 0

    def scrape(
        self,
        limit: int | None = None,
    ) -> list[Article]:
        self.calls += 1

        if self.error is not None:
            raise self.error

        if limit is None:
            return list(self.articles)

        return list(self.articles[:limit])


def make_config(tmp_path: Path):
    """Create an isolated configuration for filesystem tests."""

    data_dir = tmp_path / "data"

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    plots_dir = data_dir / "plots"
    reports_dir = data_dir / "reports"
    cache_dir = data_dir / "cache"
    logs_dir = tmp_path / "logs"

    return replace(
        CONFIG,
        project_root=tmp_path,
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        raw_corpus_file=raw_dir / "articles.json",
        processed_corpus_file=(
            processed_dir / "processed_articles.csv"
        ),
        frequency_report_file=(
            reports_dir / "word_frequencies.txt"
        ),
        stopwords_report_file=(
            reports_dir / "stopwords.txt"
        ),
        zipf_plot_file=plots_dir / "zipf_law.png",
        heaps_plot_file=plots_dir / "heaps_law.png",
        wordcloud_plot_file=plots_dir / "word_cloud.png",
        academic_report_file=(
            reports_dir / "laboratory_report.docx"
        ),
        log_file=logs_dir / "project.log",
        max_articles=10,
        min_articles=2,
        min_article_words=1,
        min_total_words=10,
        rss_feed_urls=(),
    )


def make_article(
    number: int = 1,
    *,
    title: str | None = None,
    text: str | None = None,
    url: str | None = None,
    date: datetime | None = None,
) -> Article:
    """Create a valid Russian article for testing."""

    return Article(
        url=url or f"https://example.com/news/{number}",
        title=title or f"Российская новость номер {number}",
        text=(
            text
            if text is not None
            else (
                "Москва является важным городом России. "
                "Новости помогают людям узнавать события."
            )
        ),
        source="Test News",
        date=date,
    )


def test_normalize_url_removes_tracking_parameters() -> None:
    """Tracking parameters and fragments should not affect URL identity."""

    url = (
        "HTTPS://Example.COM/news/123/"
        "?utm_source=test&id=25#section"
    )

    normalized = CorpusBuilder.normalize_url(url)

    assert normalized == (
        "https://example.com/news/123?id=25"
    )


def test_normalize_url_preserves_meaningful_query_parameters() -> None:
    """Parameters that may identify an article must be retained."""

    first = "https://example.com/article?id=10"
    second = "https://example.com/article?id=20"

    assert (
        CorpusBuilder.normalize_url(first)
        != CorpusBuilder.normalize_url(second)
    )


def test_deduplicate_removes_duplicate_urls() -> None:
    """URLs differing only by tracking parameters are duplicates."""

    first = make_article(
        url="https://example.com/news/1?utm_source=rss",
    )

    duplicate = make_article(
        url="https://example.com/news/1?utm_medium=social",
        title="Different title",
        text="Different text from the same page.",
    )

    result = CorpusBuilder._deduplicate(
        [first, duplicate]
    )

    assert len(result) == 1
    assert result[0] == first


def test_deduplicate_removes_identical_content() -> None:
    """Identical title and body should not be counted twice."""

    first = make_article(
        number=1,
        title="Одинаковая новость",
        text="Москва является столицей России.",
    )

    duplicate = make_article(
        number=2,
        title="Одинаковая новость",
        text="Москва является столицей России.",
    )

    result = CorpusBuilder._deduplicate(
        [first, duplicate]
    )

    assert len(result) == 1


def test_validation_reports_actual_corpus_size(
    tmp_path: Path,
) -> None:
    """Validation must report actual counts, not invented values."""

    config = make_config(tmp_path)

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    articles = [make_article(1)]

    report = builder.validate(articles)

    assert report.article_count == 1
    assert report.required_articles == 2
    assert report.missing_articles == 1
    assert report.is_sufficient is False

    assert report.total_words == articles[0].word_count
    assert report.missing_words == max(
        0,
        config.min_total_words - articles[0].word_count,
    )


def test_validation_marks_sufficient_corpus(
    tmp_path: Path,
) -> None:
    """A corpus meeting both minimums must be marked sufficient."""

    config = make_config(tmp_path)

    articles = [
        make_article(
            number=index,
            text=(
                "Москва является большим городом. "
                "Люди читают российские новости ежедневно."
            ),
        )
        for index in range(1, 3)
    ]

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    report = builder.validate(articles)

    assert report.article_count == 2
    assert report.total_words >= 10
    assert report.is_sufficient is True
    assert report.warnings == ()


def test_save_and_load_preserves_article_data(
    tmp_path: Path,
) -> None:
    """JSON saving and loading must preserve Russian text and dates."""

    config = make_config(tmp_path)

    published = datetime(
        2026,
        11,
        6,
        10,
        30,
        tzinfo=timezone.utc,
    )

    original = make_article(
        number=1,
        title="Новости России",
        text="Москва является столицей Российской Федерации.",
        date=published,
    )

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    saved_path = builder.save([original])

    assert saved_path == config.raw_corpus_file
    assert saved_path.is_file()

    loaded_articles = builder.load()

    assert len(loaded_articles) == 1

    loaded = loaded_articles[0]

    assert loaded.url == original.url
    assert loaded.title == original.title
    assert loaded.text == original.text
    assert loaded.source == original.source
    assert loaded.date == original.date
    assert loaded.word_count == original.word_count


def test_saved_json_contains_structured_metadata(
    tmp_path: Path,
) -> None:
    """The raw corpus JSON should contain metadata and article records."""

    import json

    config = make_config(tmp_path)

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    builder.save([make_article(1)])

    with config.raw_corpus_file.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        payload = json.load(json_file)

    assert payload["schema_version"] == 1
    assert payload["article_count"] == 1
    assert payload["total_words"] > 0
    assert payload["sources"] == ["Test News"]
    assert isinstance(payload["articles"], list)
    assert payload["articles"][0]["title"] == (
        "Российская новость номер 1"
    )


def test_individual_article_cache_can_be_loaded(
    tmp_path: Path,
) -> None:
    """A cached article must be available without a network request."""

    config = make_config(tmp_path)

    article = make_article(1)

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    cache_path = builder.save_article_cache(article)

    assert cache_path.is_file()

    cached_articles = builder.load_article_caches()

    assert len(cached_articles) == 1
    assert cached_articles[0].url == article.url


def test_load_cached_works_without_network(
    tmp_path: Path,
) -> None:
    """Cached data must load without calling a scraper."""

    config = make_config(tmp_path)

    article = make_article(1)

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    builder.save([article])
    builder.save_article_cache(article)

    loaded = builder.load_cached()

    assert len(loaded) == 1
    assert loaded[0].title == article.title


def test_build_reuses_saved_corpus_without_scraping(
    tmp_path: Path,
) -> None:
    """Existing corpus data should prevent unnecessary downloads."""

    config = make_config(tmp_path)

    article = make_article(1)

    first_builder = CorpusBuilder(
        [StubScraper([article])],
        config=config,
    )

    first_builder.build(
        limit=1,
        use_cache=False,
        persist=True,
    )

    unused_scraper = StubScraper([])

    second_builder = CorpusBuilder(
        [unused_scraper],
        config=config,
    )

    result = second_builder.build(
        limit=1,
        use_cache=True,
        persist=False,
    )

    assert len(result) == 1
    assert unused_scraper.calls == 0


def test_build_respects_article_limit(
    tmp_path: Path,
) -> None:
    """The corpus builder must not exceed the requested limit."""

    config = make_config(tmp_path)

    articles = [
        make_article(number=index)
        for index in range(1, 6)
    ]

    scraper = StubScraper(articles)

    builder = CorpusBuilder(
        [scraper],
        config=config,
    )

    result = builder.build(
        limit=2,
        use_cache=False,
        persist=False,
    )

    assert len(result) == 2
    assert scraper.calls == 1


def test_build_continues_after_a_source_failure(
    tmp_path: Path,
) -> None:
    """A failed source must not stop other configured sources."""

    config = make_config(tmp_path)

    failed_scraper = StubScraper(
        error=ScrapingError("Simulated source failure")
    )

    working_scraper = StubScraper(
        [make_article(1), make_article(2)]
    )

    builder = CorpusBuilder(
        [failed_scraper, working_scraper],
        config=config,
    )

    result = builder.build(
        limit=2,
        use_cache=False,
        persist=False,
    )

    assert len(result) == 2
    assert failed_scraper.calls == 1
    assert working_scraper.calls == 1

    assert builder.last_validation_report is not None

    assert len(
        builder.last_validation_report.source_failures
    ) == 1


def test_load_rejects_missing_corpus(
    tmp_path: Path,
) -> None:
    """Loading a non-existent corpus should raise a clear error."""

    builder = CorpusBuilder(
        [StubScraper()],
        config=make_config(tmp_path),
    )

    with pytest.raises(FileNotFoundError):
        builder.load()


def test_load_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON must not be treated as a valid corpus."""

    config = make_config(tmp_path)
    config.create_directories()

    config.raw_corpus_file.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    with pytest.raises(CorpusStorageError):
        builder.load()


def test_load_rejects_unsupported_schema(
    tmp_path: Path,
) -> None:
    """Unknown schema versions must produce a clear storage error."""

    import json

    config = make_config(tmp_path)
    config.create_directories()

    config.raw_corpus_file.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "article_count": 0,
                "articles": [],
            }
        ),
        encoding="utf-8",
    )

    builder = CorpusBuilder(
        [StubScraper()],
        config=config,
    )

    with pytest.raises(CorpusStorageError):
        builder.load()


def test_load_cached_fails_when_no_cache_exists(
    tmp_path: Path,
) -> None:
    """Offline loading must explain when there is no saved corpus."""

    builder = CorpusBuilder(
        [StubScraper()],
        config=make_config(tmp_path),
    )

    with pytest.raises(FileNotFoundError):
        builder.load_cached()


def test_save_rejects_non_article_objects(
    tmp_path: Path,
) -> None:
    """Only validated Article objects may be stored."""

    builder = CorpusBuilder(
        [StubScraper()],
        config=make_config(tmp_path),
    )

    with pytest.raises(TypeError):
        builder.save(["not an article"])  # type: ignore[list-item]