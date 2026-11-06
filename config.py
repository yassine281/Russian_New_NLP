"""Централизованная конфигурация проекта RussianNewsNLP."""

from pathlib import Path

# Базовые директории проекта
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOTS_DIR = DATA_DIR / "plots"
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

for _folder in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, REPORTS_DIR, CACHE_DIR, LOGS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# Пути к выходным файлам
RAW_CORPUS_JSON_PATH = RAW_DATA_DIR / "corpus.json"
PROCESSED_CORPUS_CSV_PATH = PROCESSED_DATA_DIR / "corpus_processed.csv"
FREQUENCY_DICT_TXT_PATH = PROCESSED_DATA_DIR / "frequency_dict.txt"
STOPWORDS_TXT_PATH = PROCESSED_DATA_DIR / "stopwords_extended.txt"
LINGUISTIC_EVAL_CSV_PATH = PROCESSED_DATA_DIR / "linguistic_evaluation.csv"
TOKENIZER_COMP_TXT_PATH = PROCESSED_DATA_DIR / "tokenizer_comparison.txt"
STATS_ANALYSIS_TXT_PATH = PROCESSED_DATA_DIR / "statistical_analysis.txt"
ACADEMIC_REPORT_MD_PATH = REPORTS_DIR / "academic_report.md"

# Настройки HTTP-запросов и вежливого парсинга
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3

# Требования к объему корпуса (Задание 1)
MIN_CORPUS_ARTICLES = 100
MIN_CORPUS_WORDS = 30000
MIN_WORDS_PER_ARTICLE = 30

# Настройки предобработки текста и NLP
SPACY_MODEL_NAME = "ru_core_news_sm"
CUSTOM_RUSSIAN_STOPWORDS = (
    "это", "который", "свой", "весь", "год", "также", "мочь",
    "очень", "еще", "ещё", "такой", "один", "другой", "наш",
    "самый", "день", "время", "человек", "стать", "говорить",
    "сообщить", "заявить", "рассказать", "отметить", "риа", "тасс", "лента",
)

# Настройки графиков и логирования
PLOT_DPI = 300
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_FILE_PATH = LOGS_DIR / "pipeline.log"