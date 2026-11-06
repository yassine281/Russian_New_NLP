"""Построение графика распределения Ципфа (Задание 5, Шаг 2)."""

from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

import config


class ZipfPlotter:
    """Строит и сохраняет график закона Ципфа в двойной логарифмической шкале."""

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = Path(output_path) if output_path else (config.PLOTS_DIR / "zipf_law.png")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, zipf_data: Dict[str, Any], show: bool = False) -> Path:
        plt.figure(figsize=(10, 6))
        plt.loglog(
            zipf_data["ranks"],
            zipf_data["frequencies"],
            label="Эмпирическая частота",
            linewidth=2,
        )
        plt.loglog(
            zipf_data["ranks"],
            zipf_data["fitted_frequencies"],
            linestyle="--",
            color="red",
            label=f"Линия регрессии (наклон = {zipf_data['slope']:.2f})",
        )
        plt.xlabel("Ранг")
        plt.ylabel("Частота")
        plt.title("Закон Ципфа для новостного корпуса")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=config.PLOT_DPI)
        if show:
            plt.show()
        plt.close()
        return self.output_path