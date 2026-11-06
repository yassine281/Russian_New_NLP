"""Построение графика закона Хипса (Задание 5, Шаг 3)."""

from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

import config


class HeapsPlotter:
    """Строит и сохраняет график роста словаря по закону Хипса."""

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = Path(output_path) if output_path else (config.PLOTS_DIR / "heaps_law.png")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, heaps_data: Dict[str, Any], show: bool = False) -> Path:
        plt.figure(figsize=(10, 6))
        plt.loglog(
            heaps_data["n_values"],
            heaps_data["v_values"],
            label="Эмпирический словарь V(N)",
            color="green",
            linewidth=2,
        )
        plt.loglog(
            heaps_data["n_values"],
            heaps_data["fitted_vocab"],
            linestyle="--",
            color="orange",
            label=f"Модель Хипса (k={heaps_data['k']:.1f}, beta={heaps_data['beta']:.2f})",
        )
        plt.xlabel("Объём текста N")
        plt.ylabel("Число уникальных слов V")
        plt.title("Закон Хипса")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=config.PLOT_DPI)
        if show:
            plt.show()
        plt.close()
        return self.output_path