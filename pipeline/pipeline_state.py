"""Хранилище состояния выполнения конвейера."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.article import Article, ProcessedArticle
from models.frequency import FrequencyEntry


@dataclass
class PipelineState:
    """Хранит промежуточные и итоговые результаты работы конвейера."""

    raw_articles: List[Article] = field(default_factory=list)
    processed_articles: List[ProcessedArticle] = field(default_factory=list)
    frequency_entries: List[FrequencyEntry] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    zipf_result: Dict[str, Any] = field(default_factory=dict)
    heaps_result: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[Path] = None