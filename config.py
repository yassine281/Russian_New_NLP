"""Centralized configuration for the RussianNewsNLP project."""

from __future__ import annotations

from pathlib import Path

# 1. Base Project & Data Directories
BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
PLOTS_DIR: Path = DATA_DIR / "plots"
REPORTS_DIR: Path = DATA_DIR / "reports"
CACHE_DIR: Path = DATA_DIR / "cache"
LOGS_DIR: Path = BASE_DIR / "logs"

for _directory in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    CACHE_DIR,
    LOGS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# 2. Standard Output File Paths
RAW_CORPUS_JSON_PATH: Path = RAW_DATA_DIR / "corpus.json"
PROCESSED_CORPUS_CSV_PATH: Path = PROCESSED_DATA_DIR / "corpus_processed.csv"
FREQUENCY_DICT_TXT_PATH: Path = PROCESSED_DATA_DIR / "frequency_dict.txt"
STOPWORDS_TXT_PATH: Path = PROCESSED_DATA_DIR / "stopwords_extended.txt"
LINGUISTIC_EVAL_CSV_PATH: Path = PROCESSED_DATA_DIR / "linguistic_evaluation.csv"
TOKENIZER_COMP_TXT_PATH: Path = PROCESSED_DATA_DIR / "tokenizer_comparison.txt"
STATS_ANALYSIS_TXT_PATH: Path = PROCESSED_DATA_DIR / "statistical_analysis.txt"
ACADEMIC_REPORT_MD_PATH: Path = REPORTS_DIR / "academic_report.md"
ACADEMIC_REPORT_TXT_PATH: Path = REPORTS_DIR / "academic_report.txt"

# 3. HTTP Scraping & Ethical Rate-Limiting Settings
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}

REQUEST_DELAY_SECONDS: float = 0.6
REQUEST_DELAY_MIN_SECONDS: float = 0.5
REQUEST_DELAY_MAX_SECONDS: float = 1.0
REQUEST_TIMEOUT_SECONDS: int = 10
MAX_RETRIES: int = 3
RETRY_BACKOFF_FACTOR: float = 0.75

# 4. Corpus Validation Thresholds
MIN_CORPUS_ARTICLES: int = 100
MIN_CORPUS_WORDS: int = 30000
MIN_WORDS_PER_ARTICLE: int = 40

TARGET_SOURCES: tuple[str, ...] = (
    "https://lenta.ru",
    "https://ria.ru",
    "https://www.gazeta.ru",
    "https://tass.ru",
)

# 5. NLP & Preprocessing Settings
SPACY_MODEL_NAME: str = "ru_core_news_sm"
NORMALIZE_YO: bool = False
MIN_TOKEN_LENGTH: int = 3

CUSTOM_RUSSIAN_STOPWORDS: tuple[str, ...] = (
    "это",
    "который",
    "свой",
    "весь",
    "год",
    "также",
    "мочь",
    "очень",
    "еще",
    "ещё",
    "такой",
    "один",
    "другой",
    "наш",
    "ваш",
    "самый",
    "день",
    "время",
    "человек",
    "стать",
    "говорить",
    "сообщить",
    "заявить",
    "рассказать",
    "отметить",
    "риа",
    "тасс",
    "лента",
)

# 6. Visualization Settings
PLOT_DPI: int = 300
WORDCLOUD_WIDTH: int = 1200
WORDCLOUD_HEIGHT: int = 600
WORDCLOUD_MAX_WORDS: int = 100

# 7. Logging Configuration
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_FILE_PATH: Path = LOGS_DIR / "pipeline.log"