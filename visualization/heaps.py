"""Heaps' Law vocabulary growth visualization module."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

import config
from analysis.law_fitter import HeapsFitResult

logger = logging.getLogger(__name__)


class HeapsPlotter:
    """Renders and saves the Heaps' law vocabulary growth curve at 300 DPI[cite: 8]."""

    def __init__(self, output_path: Path | None = None, dpi: int = 300) -> None:
        """Initialize the HeapsPlotter.

        Args:
            output_path: Destination PNG file path (defaults to `data/plots/heaps_law.png`).
            dpi: Resolution in dots per inch (default 300).
        """
        default_dir = getattr(config, "PLOTS_DIR", Path("data/plots"))
        self.output_path: Path = Path(output_path) if output_path else Path(default_dir) / "heaps_law.png"
        self.dpi: int = int(dpi)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, fit_result: HeapsFitResult, show_interactive: bool = False) -> Path:
        """Generate, save at 300 DPI, and optionally display the Heaps' law plot[cite: 8].

        Args:
            fit_result: Calculated `HeapsFitResult`.
            show_interactive: Whether to call `plt.show()` before closing the figure.

        Returns:
            Path to the saved PNG file.
        """
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Left subplot: Standard linear scale showing sub-linear saturation curve
        ax_lin.plot(
            fit_result.cumulative_tokens,
            fit_result.cumulative_vocab,
            color="#2ca02c",
            linewidth=2.0,
            label="Эмпирический словарь V(N)",
        )
        ax_lin.plot(
            fit_result.cumulative_tokens,
            fit_result.fitted_vocab,
            linestyle="--",
            color="#ff7f0e",
            linewidth=2.0,
            label=f"Модель: k={fit_result.k:.2f}, β={fit_result.beta:.3f}",
        )
        ax_lin.set_title("Закон Хипса (линейная шкала)", fontsize=12, fontweight="bold")
        ax_lin.set_xlabel("Объём текста N (число токенов)", fontsize=11)
        ax_lin.set_ylabel("Число уникальных слов V(N)", fontsize=11)
        ax_lin.grid(True, linestyle="--", alpha=0.6)
        ax_lin.legend(loc="lower right", fontsize=9)

        # Right subplot: Log-Log scale as shown in the laboratory manual
        ax_log.loglog(
            fit_result.cumulative_tokens,
            fit_result.cumulative_vocab,
            color="#2ca02c",
            linewidth=2.0,
            label="Эмпирический рост (log-log)",
        )
        ax_log.loglog(
            fit_result.cumulative_tokens,
            fit_result.fitted_vocab,
            linestyle="--",
            color="#ff7f0e",
            linewidth=2.0,
            label=f"Аппроксимация (R² = {fit_result.r_squared:.3f})",
        )
        ax_log.set_title("Закон Хипса (логарифмическая шкала)", fontsize=12, fontweight="bold")
        ax_log.set_xlabel("Объём текста N (log)", fontsize=11)
        ax_log.set_ylabel("Число уникальных слов V (log)", fontsize=11)
        ax_log.grid(True, which="both", linestyle="--", alpha=0.6)
        ax_log.legend(loc="lower right", fontsize=9)

        fig.suptitle(
            f"Закон Хипса для новостного корпуса (k = {fit_result.k:.3f}, β = {fit_result.beta:.3f})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=self.dpi, bbox_inches="tight")
        logger.info("Saved Heaps' law plot (300 DPI) to %s", self.output_path)

        if show_interactive:
            plt.show()
        plt.close(fig)
        return self.output_path