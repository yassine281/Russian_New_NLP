"""Zipf's Law visualization module."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

import config
from analysis.law_fitter import ZipfFitResult

logger = logging.getLogger(__name__)


class ZipfPlotter:
    """Renders and exports the log-log Zipf rank-frequency distribution graph[cite: 8]."""

    def __init__(self, output_path: Path | None = None, dpi: int = 300) -> None:
        """Initialize the ZipfPlotter.

        Args:
            output_path: Destination PNG file path (defaults to `data/plots/zipf_law.png`).
            dpi: Image resolution in dots per inch (default 300).
        """
        default_dir = getattr(config, "PLOTS_DIR", Path("data/plots"))
        self.output_path: Path = Path(output_path) if output_path else Path(default_dir) / "zipf_law.png"
        self.dpi: int = int(dpi)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, fit_result: ZipfFitResult, show_interactive: bool = False) -> Path:
        """Generate, save at 300 DPI, and optionally display the Zipf distribution plot[cite: 8].

        Args:
            fit_result: Calculated `ZipfFitResult`.
            show_interactive: Whether to call `plt.show()` before closing.

        Returns:
            Path to the saved PNG file.
        """
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(
            fit_result.ranks,
            fit_result.frequencies,
            label="Эмпирическое распределение корпуса",
            color="#1f77b4",
            linewidth=2.0,
        )
        ax.loglog(
            fit_result.ranks,
            fit_result.fitted_frequencies,
            linestyle="--",
            color="#d62728",
            linewidth=2.0,
            label=f"Аппроксимация Ципфа (наклон = {fit_result.slope:.3f}, R² = {fit_result.r_squared:.3f})",
        )

        ax.set_title("Закон Ципфа для русскоязычного новостного корпуса", fontsize=14, fontweight="bold")
        ax.set_xlabel("Ранг слова r (логарифмическая шкала)", fontsize=12)
        ax.set_ylabel("Частота слова f(r) (логарифмическая шкала)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)

        fig.tight_layout()
        fig.savefig(self.output_path, dpi=self.dpi, bbox_inches="tight")
        logger.info("Saved Zipf distribution plot (300 DPI) to %s", self.output_path)

        if show_interactive:
            plt.show()
        plt.close(fig)
        return self.output_path