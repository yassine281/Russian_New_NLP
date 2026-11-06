"""Скрипт для создания правильной структуры папок и файлов проекта RussianNewsNLP."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 1. جميع المجلدات المطلوبة في المشروع
DIRECTORIES = [
    "models",
    "scraping",
    "preprocessing",
    "analysis",
    "visualization",
    "export",
    "pipeline",
    "tests",
    "data/raw",
    "data/processed",
    "data/plots",
    "data/reports",
    "data/cache",
    "logs",
]

# 2. جميع الملفات المطلوبة في المشروع
FILES = [
    "main.py",
    "config.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    # models
    "models/__init__.py",
    "models/article.py",
    "models/frequency.py",
    # scraping
    "scraping/__init__.py",
    "scraping/base_scraper.py",
    "scraping/lenta_scraper.py",
    "scraping/rss_scraper.py",
    "scraping/corpus_builder.py",
    "scraping/robots_checker.py",
    # preprocessing
    "preprocessing/__init__.py",
    "preprocessing/cleaner.py",
    "preprocessing/tokenizers.py",
    "preprocessing/lemmatizers.py",
    "preprocessing/stemmer.py",
    "preprocessing/stopwords.py",
    # analysis
    "analysis/__init__.py",
    "analysis/frequency.py",
    "analysis/evaluator.py",
    "analysis/law_fitter.py",
    "analysis/corpus_statistics.py",
    # visualization
    "visualization/__init__.py",
    "visualization/zipf.py",
    "visualization/heaps.py",
    "visualization/wordcloud_gen.py",
    # export
    "export/__init__.py",
    "export/exporter.py",
    "export/report_generator.py",
    # pipeline
    "pipeline/__init__.py",
    "pipeline/nlp_pipeline.py",
    "pipeline/pipeline_state.py",
    # tests
    "tests/__init__.py",
    "tests/test_models.py",
    "tests/test_cleaner.py",
    "tests/test_tokenizers.py",
    "tests/test_lemmatizers.py",
    "tests/test_frequency.py",
    "tests/test_pipeline.py",
]


def create_project_structure() -> None:
    """Создает все папки и файлы проекта и удаляет ошибочные файлы init.py."""
    print("=" * 65)
    print("Создание структуры проекта RussianNewsNLP...")
    print("=" * 65)

    # إنشاء المجلدات وإصلاح أي ملف init.py خاطئ
    for folder in DIRECTORIES:
        dir_path = PROJECT_ROOT / folder
        dir_path.mkdir(parents=True, exist_ok=True)
        wrong_init = dir_path / "init.py"
        if wrong_init.exists():
            wrong_init.unlink()
            print(f"[ИСПРАВЛЕНО] Удален неправильный файл: {folder}/init.py")
        print(f"[ПАПКА ГОТОВА] {folder}/")

    # إنشاء الملفات إذا لم تكن موجودة
    for rel_path in FILES:
        file_path = PROJECT_ROOT / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text("", encoding="utf-8")
            print(f"[ФАЙЛ СОЗДАН] {rel_path}")
        else:
            print(f"[ФАЙЛ СУЩЕСТВУЕТ] {rel_path}")

    print("=" * 65)
    print("Структура проекта успешно подготовлена!")
    print("=" * 65)


if __name__ == "__main__":
    create_project_structure()