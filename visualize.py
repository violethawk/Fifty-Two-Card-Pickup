"""
visualize.py
============

Generate PNG visualizations of benchmark scatter patterns.

Usage::

    python visualize.py              # save to images/
    python visualize.py --show       # display interactively
"""

from __future__ import annotations

import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from benchmarks import PATTERNS


# Card suit colors and markers
SUIT_STYLE = {
    "hearts":   {"color": "#e74c3c", "marker": "h"},
    "diamonds": {"color": "#e67e22", "marker": "D"},
    "clubs":    {"color": "#2c3e50", "marker": "s"},
    "spades":   {"color": "#34495e", "marker": "^"},
}


def render_pattern(name: str, cards: list, ax: plt.Axes) -> None:
    """Plot cards on a 10x10 grid axis."""
    for suit, style in SUIT_STYLE.items():
        suit_cards = [c for c in cards if c["suit"] == suit]
        xs = [c["x"] for c in suit_cards]
        ys = [c["y"] for c in suit_cards]
        ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                   s=60, alpha=0.85, edgecolors="white", linewidths=0.5,
                   label=suit.capitalize(), zorder=3)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.set_title(name.replace("_", " ").title(), fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.tick_params(labelsize=8)


def generate_all(output_dir: str = "images", show: bool = False) -> None:
    """Generate individual PNGs and a combined overview."""
    os.makedirs(output_dir, exist_ok=True)

    # Combined figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("52 Card Pickup — Benchmark Scatter Patterns",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, (name, pattern_fn) in enumerate(PATTERNS.items()):
        cards = pattern_fn()
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        render_pattern(name, cards, ax)

    # Shared legend
    handles = [mpatches.Patch(color=s["color"], label=suit.capitalize())
               for suit, s in SUIT_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    combined_path = os.path.join(output_dir, "patterns_overview.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {combined_path}")

    if show:
        plt.show()
    plt.close(fig)

    # Individual figures
    for name, pattern_fn in PATTERNS.items():
        cards = pattern_fn()
        fig_single, ax_single = plt.subplots(figsize=(6, 6))
        render_pattern(name, cards, ax_single)
        handles = [mpatches.Patch(color=s["color"], label=suit.capitalize())
                   for suit, s in SUIT_STYLE.items()]
        ax_single.legend(handles=handles, loc="upper right", fontsize=9)
        path = os.path.join(output_dir, f"pattern_{name}.png")
        fig_single.savefig(path, dpi=150, bbox_inches="tight",
                           facecolor="white", edgecolor="none")
        print(f"Saved {path}")
        plt.close(fig_single)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark scatter patterns")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--output", default="images", help="Output directory (default: images)")
    args = parser.parse_args()
    generate_all(output_dir=args.output, show=args.show)
