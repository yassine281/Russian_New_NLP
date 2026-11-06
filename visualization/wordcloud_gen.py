"""Построение облака слов для русского корпуса (Задание 5, Шаг 4)."""

from pathlib import Path
from typing import Mapping, Optional
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import config


class WordCloudGenerator:
    """Генерирует облако 100 наиболее частотных русских слов."""

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = Path(output_path) if output_path else (config.PLOTS_DIR / "wordcloud.png")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_cyrillic_font() -> Optional[str]:
        candidates = [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf",
        ]
        for font in candidates:
            if font.exists():
                return str(font)
        return None

    def generate(self, frequencies: Mapping[str, int], show: bool = False) -> Path:
        if not frequencies:
            raise ValueError("Словарь частот пуст.")

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            font_path=self._find_cyrillic_font(),
        ).generate_from_frequencies(dict(frequencies))

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Облако наиболее частотных слов корпуса")
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=config.PLOT_DPI)
        if show:
            plt.show()
        plt.close()
        return self.output_path