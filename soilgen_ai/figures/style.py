"""One place for figure size, type, and color.

Nature figure practice, as used by this repository:

- Widths are 89 mm, 120 mm, or 183 mm (one, one-and-a-half, and two columns).
- The face is sans serif. Arial and Helvetica are not installed here, so the
  face is DejaVu Sans. PDF text is TrueType (fonttype 42), which keeps the
  file free of Type 3 fonts.
- Tick labels are 7 pt and axis labels are 8 pt. Both are above the 5 pt
  minimum Nature asks for at final size.
- Panel letters are lowercase, 8 pt, bold, outside the upper left of the axes.
- The palette is the Nature Publishing Group set. Real data is navy, the flow
  is vermillion, and the Gaussian baseline is cyan. Red is not paired with green.
- Only the left and bottom spines are drawn. There is no grid.
- A figure has no title. The caption carries the description.
- Each figure is written as a 300 dpi PNG and as a PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

_COLUMN_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}


@dataclass(frozen=True)
class FigureStyle:
    """Colors, type sizes, and save settings for every figure."""

    real: str = "#3C5488"
    flow: str = "#E64B35"
    baseline: str = "#4DBBD5"
    ink: str = "#222222"
    muted: str = "#666666"
    font: str = "DejaVu Sans"
    label_pt: float = 8.0
    tick_pt: float = 7.0
    panel_pt: float = 8.0
    legend_pt: float = 7.0
    spine_pt: float = 0.6
    dpi: int = 300

    def size(
        self, columns: str = "double", aspect: float = 0.45
    ) -> tuple[float, float]:
        """Figure size in inches for a Nature column width."""
        if columns not in _COLUMN_MM:
            raise ValueError(f"columns must be one of {sorted(_COLUMN_MM)}")
        width = _COLUMN_MM[columns] / 25.4
        return (width, width * aspect)

    def apply(self) -> None:
        """Set matplotlib defaults. Call once before drawing."""
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [self.font, "DejaVu Sans"],
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": self.ink,
                "axes.labelcolor": self.ink,
                "axes.linewidth": self.spine_pt,
                "axes.labelsize": self.label_pt,
                "axes.titlesize": self.label_pt,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
                "text.color": self.ink,
                "xtick.color": self.ink,
                "ytick.color": self.ink,
                "xtick.labelsize": self.tick_pt,
                "ytick.labelsize": self.tick_pt,
                "xtick.major.width": self.spine_pt,
                "ytick.major.width": self.spine_pt,
                "xtick.major.size": 3.0,
                "ytick.major.size": 3.0,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "legend.fontsize": self.legend_pt,
                "legend.frameon": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.dpi": self.dpi,
                "savefig.facecolor": "white",
            }
        )

    def panel(self, ax: Axes, letter: str) -> None:
        """Lowercase panel letter, outside the upper left."""
        ax.text(
            -0.12,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=self.panel_pt,
            fontweight="bold",
            va="bottom",
            ha="left",
            color=self.ink,
        )

    def save(self, fig: plt.Figure, path: Path) -> None:
        """Write a 300 dpi PNG and a PDF, then close the figure."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.with_suffix("")
        fig.savefig(stem.with_suffix(".png"), dpi=self.dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem.with_suffix('.png')}")


STYLE = FigureStyle()
