"""Russian Cyrillic-compatible WordCloud visualization generator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import config

logger = logging.getLogger(__name__)


class WordCloudGenerator:
    """Generates and exports a 300 DPI Russian word cloud image[cite: 9]."""

    WINDOWS_FONT_CANDIDATES: tuple[str, ...] = (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    )

    def __init__(
        self,
        output_path: Path | None = None,
        dpi: int = 300,
        width: int = 1200,
        height: int = 600,
        max_words: int = 100,
    ) -> None:
        """Initialize the WordCloudGenerator[cite: 9].

        Args:
            output_path: Destination PNG path (defaults to `data/plots/wordcloud.png`).
            dpi: Output image resolution (default 300).
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            max_words: Maximum number of top Russian words to display (default 100)[cite: 9].
        """
        default_dir = getattr(config, "PLOTS_DIR", Path("data/plots"))
        self.output_path: Path = (
            Path(output_path) if output_path else Path(default_dir) / "wordcloud.png"
        )
        self.dpi: int = int(dpi)
        self.width: int = int(width)
        self.height: int = int(height)
        self.max_words: int = int(max_words)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resolve_cyrillic_font(cls) -> str | None:
        """Locate an installed TrueType font that supports Russian Cyrillic glyphs."""
        for candidate in cls.WINDOWS_FONT_CANDIDATES:
            path = Path(candidate)
            if path.exists():
                return str(path)

        # Fallback to Matplotlib's bundled DejaVuSans.ttf (full Cyrillic coverage)
        mpl_font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if mpl_font.exists():
            return str(mpl_font)
        return None

    def generate(
        self,
        frequencies: Mapping[str, int],
        show_interactive: bool = False,
    ) -> Path:
        """Generate and save the Russian word cloud at 300 DPI[cite: 9].

        Args:
            frequencies: Mapping of Russian lemma -> frequency count.
            show_interactive: Whether to display the figure interactively via `plt.show()`.

        Returns:
            Path to the saved PNG file.

        Raises:
            ValueError: If `frequencies` is empty.
        """
        if not frequencies:
            raise ValueError("Cannot generate a word cloud from an empty frequency mapping.")

        font_path = self.resolve_cyrillic_font()
        wc = WordCloud(
            width=self.width,
            height=self.height,
            background_color="white",
            max_words=self.max_words,
            font_path=font_path,
            colormap="viridis",
            random_state=42,
        ).generate_from_frequencies(dict(frequencies))

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(
            "Облако наиболее частотных слов русскоязычного новостного корпуса",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(self.output_path, dpi=self.dpi, bbox_inches="tight")
        logger.info("Saved Russian WordCloud (300 DPI) to %s", self.output_path)

        if show_interactive:
            plt.show()
        plt.close(fig)
        return self.output_path