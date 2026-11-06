"""Математическая проверка законов Ципфа и Хипса (Задание 5, Шаги 2 и 3)."""

from typing import Dict, List, Sequence
import numpy as np
from scipy.stats import linregress

from models.frequency import FrequencyEntry


class LawFitter:
    """Вычисляет параметры законов Ципфа и Хипса на реальном корпусе."""

    @staticmethod
    def fit_zipf(entries: Sequence[FrequencyEntry]) -> Dict[str, Any]:
        """Аппроксимирует закон Ципфа в логарифмических координатах."""
        valid = [e for e in entries if e.rank > 0 and e.count > 0]
        if len(valid) < 2:
            raise ValueError("Недостаточно данных для расчета закона Ципфа.")

        ranks = [e.rank for e in valid]
        freqs = [e.count for e in valid]
        log_r = np.log(np.array(ranks, dtype=float))
        log_f = np.log(np.array(freqs, dtype=float))

        reg = linregress(log_r, log_f)
        slope = float(reg.slope)
        intercept = float(reg.intercept)
        r_squared = float(reg.rvalue ** 2)
        fitted = np.exp(intercept + slope * log_r).tolist()

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "ranks": ranks,
            "frequencies": freqs,
            "fitted_frequencies": fitted,
        }

    @staticmethod
    def fit_heaps(docs_tokens: Sequence[Sequence[str]]) -> Dict[str, Any]:
        """Вычисляет накопленный объем текста N и размер словаря V(N) для закона Хипса."""
        n_values: List[int] = []
        v_values: List[int] = []
        vocab = set()
        total_n = 0

        for tokens in docs_tokens:
            if not tokens:
                continue
            total_n += len(tokens)
            vocab.update(tokens)
            n_values.append(total_n)
            v_values.append(len(vocab))

        if len(n_values) < 2:
            raise ValueError("Недостаточно данных для расчета закона Хипса.")

        log_n = np.log(np.array(n_values, dtype=float))
        log_v = np.log(np.array(v_values, dtype=float))
        reg = linregress(log_n, log_v)

        beta = float(reg.slope)
        k = float(np.exp(reg.intercept))
        r_squared = float(reg.rvalue ** 2)
        fitted_v = (k * (np.array(n_values, dtype=float) ** beta)).tolist()

        return {
            "k": k,
            "beta": beta,
            "r_squared": r_squared,
            "n_values": n_values,
            "v_values": v_values,
            "fitted_vocab": fitted_v,
        }