"""Automated project structure builder and repair script for RussianNewsNLP.

Run this script from the project root directory:
    python creator.py

What this script does:
    1. Creates all required Python packages, data folders, and log directories.
    2. Automatically fixes accidental 'init.py' files into proper '__init__.py' files.
    3. Creates all required project files if they are missing (never overwrites
       files that already contain code).
    4. Scans for misplaced Python files and prints a complete verification report.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent

REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "models",
    "scraping",
    "preprocessing",
    "analysis",
    "visualization",
    "export",
    "pipeline",
    "tests",
    "data",
    "data/raw",
    "data/processed",
    "data/plots",
    "data/reports",
    "data/cache",
    "logs",
)

REQUIRED_FILES: dict[str, str] = {
    # Root files
    "main.py": '"""Command-line entry point for the Russian News NLP Pipeline."""\n',
    "dashboard.py": '"""Optional interactive dashboard for the Russian News NLP Pipeline."""\n',
    "config.py": '"""Centralized configuration for the RussianNewsNLP project."""\n',
    "requirements.txt": (
        "requests>=2.32.3\n"
        "beautifulsoup4>=4.12.3\n"
        "lxml>=5.3.0\n"
        "feedparser>=6.0.11\n"
        "pandas>=2.2.3\n"
        "numpy>=2.1.0\n"
        "matplotlib>=3.9.2\n"
        "scipy>=1.14.1\n"
        "pymorphy3>=2.0.2\n"
        "pymorphy3-dicts-ru>=2.4.417150.4580142\n"
        "spacy>=3.8.2\n"
        "nltk>=3.9.1\n"
        "wordcloud>=1.9.4\n"
        "pytest>=8.3.3\n"
    ),
    "README.md": "# RussianNewsNLP — Russian News Corpus & NLP Analysis System\n",
    ".gitignore": (
        ".venv/\n"
        ".venv312/\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        ".pytest_cache/\n"
        ".idea/\n"
        "logs/*.log\n"
        "data/cache/*\n"
    ),
    # models package
    "models/__init__.py": '"""Domain data models package."""\n',
    "models/article.py": '"""Article and ProcessedArticle data models."""\n',
    "models/frequency.py": '"""FrequencyEntry data model."""\n',
    # scraping package
    "scraping/__init__.py": '"""Web scraping and RSS collection package."""\n',
    "scraping/base_scraper.py": '"""Abstract base scraper class."""\n',
    "scraping/lenta_scraper.py": '"""Lenta.ru news scraper implementation."""\n',
    "scraping/rss_scraper.py": '"""Multi-source RSS and HTML news scraper."""\n',
    "scraping/corpus_builder.py": '"""Corpus builder and validator."""\n',
    "scraping/robots_checker.py": '"""Ethical robots.txt compliance checker."""\n',
    # preprocessing package
    "preprocessing/__init__.py": '"""Russian text preprocessing package."""\n',
    "preprocessing/cleaner.py": '"""Text cleaning and normalization module."""\n',
    "preprocessing/tokenizers.py": '"""Simple, NLTK, and spaCy tokenizers."""\n',
    "preprocessing/lemmatizers.py": '"""Pymorphy3, spaCy, and optional Mystem lemmatizers."""\n',
    "preprocessing/stemmer.py": '"""NLTK Russian SnowballStemmer wrapper."""\n',
    "preprocessing/stopwords.py": '"""Russian stop-word filtering module."""\n',
    # analysis package
    "analysis/__init__.py": '"""Statistical and linguistic analysis package."""\n',
    "analysis/frequency.py": '"""Counter-based word frequency analyzer."""\n',
    "analysis/evaluator.py": '"""Tokenizer and lemmatizer benchmark evaluator."""\n',
    "analysis/law_fitter.py": '"""Zipf\'s Law and Heaps\' Law regression fitter."""\n',
    "analysis/corpus_statistics.py": '"""Aggregate corpus statistics calculator."""\n',
    # visualization package
    "visualization/__init__.py": '"""Visualization package for plots and word clouds."""\n',
    "visualization/zipf.py": '"""Zipf\'s Law plotter (300 DPI)."""\n',
    "visualization/heaps.py": '"""Heaps\' Law plotter (300 DPI)."""\n',
    "visualization/wordcloud_gen.py": '"""Cyrillic-compatible Russian WordCloud generator."""\n',
    # export package
    "export/__init__.py": '"""Data export and academic report generation package."""\n',
    "export/exporter.py": '"""JSON, CSV, and TXT data exporter."""\n',
    "export/report_generator.py": '"""Russian academic report generator."""\n',
    # pipeline package
    "pipeline/__init__.py": '"""NLP pipeline orchestration package."""\n',
    "pipeline/nlp_pipeline.py": '"""Main NLPPipeline controller."""\n',
    "pipeline/pipeline_state.py": '"""Pipeline execution state container."""\n',
    # tests package
    "tests/__init__.py": '"""Automated pytest unit and integration test suite."""\n',
    "tests/test_models.py": '"""Unit tests for data models."""\n',
    "tests/test_cleaner.py": '"""Unit tests for TextCleaner."""\n',
    "tests/test_tokenizers.py": '"""Unit tests for tokenizers."""\n',
    "tests/test_lemmatizers.py": '"""Unit tests for lemmatizers, stemmer, and stopwords."""\n',
    "tests/test_frequency.py": '"""Unit tests for frequency analysis and Zipf/Heaps laws."""\n',
    "tests/test_pipeline.py": '"""End-to-end integration tests for NLPPipeline."""\n',
}

PYTHON_PACKAGES: tuple[str, ...] = (
    "models",
    "scraping",
    "preprocessing",
    "analysis",
    "visualization",
    "export",
    "pipeline",
    "tests",
)


def fix_mistaken_init_files(root: Path) -> list[str]:
    """Rename or remove mistaken 'init.py' files (without underscores) inside packages."""
    actions: list[str] = []
    for pkg in PYTHON_PACKAGES:
        pkg_dir = root / pkg
        wrong_init = pkg_dir / "init.py"
        correct_init = pkg_dir / "__init__.py"

        if wrong_init.exists():
            wrong_content = wrong_init.read_text(encoding="utf-8", errors="ignore").strip()
            if not correct_init.exists():
                wrong_init.rename(correct_init)
                actions.append(f"RENAMED mistaken '{pkg}/init.py' -> '{pkg}/__init__.py'")
            else:
                correct_content = correct_init.read_text(encoding="utf-8", errors="ignore").strip()
                if wrong_content and not correct_content:
                    correct_init.write_text(wrong_content + "\n", encoding="utf-8")
                wrong_init.unlink()
                actions.append(f"REMOVED duplicate mistaken '{pkg}/init.py'")
    return actions


def build_and_verify_structure(root: Path) -> None:
    """Create all missing directories and files safely and print a status report."""
    print("=" * 72)
    print(f"BUILDING & VERIFYING PROJECT STRUCTURE IN: {root}")
    print("=" * 72)

    # 1. Create all directories
    print("\n[1/4] Checking & Creating Directories:")
    for rel_dir in REQUIRED_DIRECTORIES:
        dir_path = root / rel_dir
        if dir_path.exists():
            print(f"  [OK - EXISTS]     {rel_dir}/")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED FOLDER]  {rel_dir}/")

    # 2. Fix mistaken init.py files if any
    print("\n[2/4] Checking for Mistaken 'init.py' Files:")
    fixed_inits = fix_mistaken_init_files(root)
    if fixed_inits:
        for msg in fixed_inits:
            print(f"  [FIXED]           {msg}")
    else:
        print("  [OK]              No mistaken 'init.py' files found.")

    # 3. Create missing files without overwriting existing code
    print("\n[3/4] Checking & Creating Project Files:")
    for rel_file, default_content in REQUIRED_FILES.items():
        file_path = root / rel_file
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.write_text(default_content, encoding="utf-8")
            print(f"  [CREATED FILE]    {rel_file}")
        else:
            size = file_path.stat().st_size
            if size == 0:
                file_path.write_text(default_content, encoding="utf-8")
                print(f"  [INITIALIZED]     {rel_file} (was 0 bytes)")
            else:
                print(f"  [OK - PRESERVED]  {rel_file} ({size} bytes)")

    # 4. Detect any misplaced .py files outside .venv
    print("\n[4/4] Scanning for Misplaced Python Files:")
    expected_py_paths = {
        (root / rel_path).resolve()
        for rel_path in REQUIRED_FILES
        if rel_path.endswith(".py")
    }
    expected_py_paths.add((root / "creator.py").resolve())

    stray_files: list[Path] = []
    for py_file in root.rglob("*.py"):
        parts = py_file.relative_to(root).parts
        if any(p.startswith(".venv") or p in ("venv", "__pycache__", ".pytest_cache") for p in parts):
            continue
        if py_file.resolve() not in expected_py_paths:
            stray_files.append(py_file.relative_to(root))

    if stray_files:
        print("  [WARNING] Found unexpected/misplaced .py files:")
        for stray in stray_files:
            print(f"    -> {stray}")
    else:
        print("  [OK]              Zero misplaced Python files detected!")

    print("\n" + "=" * 72)
    print("PROJECT STRUCTURE VERIFICATION COMPLETE — 100% READY!")
    print("=" * 72)


if __name__ == "__main__":
    build_and_verify_structure(PROJECT_ROOT)