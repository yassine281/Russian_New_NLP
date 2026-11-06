"""Пакет управления конвейером обработки."""

from pipeline.nlp_pipeline import NLPPipeline
from pipeline.pipeline_state import PipelineState

__all__ = ["NLPPipeline", "PipelineState"]