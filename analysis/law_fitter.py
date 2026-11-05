"""Empirical linguistic law regression module for Zipf's Law and Heaps' Law.

Academic Control Question Reference (Q7 & Q8):
    Q7 (Formulate Zipf's law and explain its manifestation in the collected corpus):
        Zipf's Law states that in a natural language corpus, the frequency f(r) of a word is
        inversely proportional to its rank r in the frequency table: f(r) ≈ C / r^s (or
        log(f) = log(C) - s * log(r), with theoretical slope -s ≈ -1.0)[cite: 8, 10]. In a filtered
        Russian news corpus (where stop words are removed and words are lemmatized), the top
        ranks are slightly flatter than in raw text because ultra-frequent prepositions/particles
        were removed, while the middle and tail ranks follow a steep power-law decay close to -1.
    Q8 (Formulate Heaps' law and explain its manifestation in the collected corpus):
        Heaps' Law describes vocabulary growth V(N) as a power function of total processed
        tokens N: V(N) = k * N^β, where k is a scaling constant (typically 10–100) and
        β ∈ (0, 1) is the vocabulary growth exponent[cite: 8, 10]. As more news articles are added,
        N grows linearly while V(N) grows sub-linearly (diminishing marginal new lemmas per
        article) as core journalistic vocabulary repeats across stories.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import linregress

from models.frequency import FrequencyEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZipfFitResult:
    """Regression output for Zipf's law on log(frequency) vs log(rank)[cite: 8]."""

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    ranks: list[int]
    frequencies: list[int]
    fitted_frequencies: list[float]

    def format_summary(self) -> str:
        """Return formatted Zipf slope string required by specification."""
        return f"Zipf slope: {self.slope:.3f} (R² = {self.r_squared:.4f})"


@dataclass(frozen=True)
class HeapsFitResult:
    """Power-law estimation output for Heaps' law V(N) = k * N^beta[cite: 8]."""

    k: float
    beta: float
    r_squared: float
    p_value: float
    std_err: float
    cumulative_tokens: list[int]
    cumulative_vocab: list[int]
    fitted_vocab: list[float]

    def format_summary(self) -> str:
        """Return formatted Heaps parameters required by specification."""
        return f"Heaps k: {self.k:.3f}\nHeaps β: {self.beta:.3f} (R² = {self.r_squared:.4f})"


class LawFitter:
    """Fits Zipf's rank-frequency distribution and Heaps' vocabulary growth power law[cite: 8]."""

    @staticmethod
    def fit_zipf(entries: Sequence[FrequencyEntry]) -> ZipfFitResult:
        """Fit linear regression to `log(frequency)` against `log(rank)`[cite: 8].

        Args:
            entries: Sequence of ranked `FrequencyEntry` objects.

        Returns:
            Populated `ZipfFitResult`.

        Raises:
            ValueError: If fewer than 2 positive frequency observations are provided.
        """
        valid = [e for e in entries if e.rank > 0 and e.count > 0]
        if len(valid) < 2:
            raise ValueError(
                "At least 2 positive (rank, frequency) entries are required to fit Zipf's law."
            )

        ranks = [e.rank for e in valid]
        freqs = [e.count for e in valid]

        log_ranks = np.log(np.asarray(ranks, dtype=np.float64))
        log_freqs = np.log(np.asarray(freqs, dtype=np.float64))

        reg = linregress(log_ranks, log_freqs)
        slope = float(reg.slope)
        intercept = float(reg.intercept)
        r_squared = float(reg.rvalue ** 2)
        p_val = float(reg.pvalue) if not np.isnan(reg.pvalue) else 0.0
        std_err = float(reg.stderr) if not np.isnan(reg.stderr) else 0.0

        fitted_freqs = np.exp(intercept + slope * log_ranks).tolist()

        logger.info("Zipf regression complete — Zipf slope: %.3f (R²=%.4f)", slope, r_squared)
        return ZipfFitResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_val,
            std_err=std_err,
            ranks=ranks,
            frequencies=freqs,
            fitted_frequencies=[float(val) for val in fitted_freqs],
        )

    @staticmethod
    def fit_heaps(articles_tokens: Sequence[Sequence[str]]) -> HeapsFitResult:
        """Compute O(N) cumulative vocabulary growth and fit V(N) = k * N^beta[cite: 8].

        Args:
            articles_tokens: Sequence of token/lemma lists (one per article).

        Returns:
            Populated `HeapsFitResult`.

        Raises:
            ValueError: If fewer than 2 non-empty articles are available.
        """
        n_values: list[int] = []
        v_values: list[int] = []
        seen_vocab: set[str] = set()
        running_n = 0

        for doc_tokens in articles_tokens:
            if not doc_tokens:
                continue
            running_n += len(doc_tokens)
            for token in doc_tokens:
                seen_vocab.add(token)
            if running_n > 0 and len(seen_vocab) > 0:
                n_values.append(running_n)
                v_values.append(len(seen_vocab))

        if len(n_values) < 2:
            raise ValueError(
                "At least 2 articles with positive token counts are required to fit Heaps' law."
            )

        log_n = np.log(np.asarray(n_values, dtype=np.float64))
        log_v = np.log(np.asarray(v_values, dtype=np.float64))

        reg = linregress(log_n, log_v)
        beta = float(reg.slope)
        intercept = float(reg.intercept)
        k = float(math.exp(intercept))
        r_squared = float(reg.rvalue ** 2)
        p_val = float(reg.pvalue) if not np.isnan(reg.pvalue) else 0.0
        std_err = float(reg.stderr) if not np.isnan(reg.stderr) else 0.0

        fitted_v = (k * (np.asarray(n_values, dtype=np.float64) ** beta)).tolist()

        logger.info("Heaps estimation complete — Heaps k: %.3f, Heaps β: %.3f", k, beta)
        return HeapsFitResult(
            k=k,
            beta=beta,
            r_squared=r_squared,
            p_value=p_val,
            std_err=std_err,
            cumulative_tokens=n_values,
            cumulative_vocab=v_values,
            fitted_vocab=[float(val) for val in fitted_v],
        )