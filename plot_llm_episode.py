"""
Plot within-episode LLM adaptation from `llm_episode_curve.json`.

Usage:
  python3 plot_llm_episode.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main(in_json: str = "llm_episode_curve.json", out_png: str = "llm_episode_curve.png") -> None:
    curve = json.loads(Path(in_json).read_text())

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        from ace_plot import plot_accuracy_curve

        plot_accuracy_curve(curve, out_png)
        print(f"Saved: {out_png}")
        return

    xs = list(range(1, len(curve) + 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, curve, linewidth=2)
    ax.set_title("LLM Within-Episode Adaptation")
    ax.set_xlabel("Round")
    ax.set_ylabel("Rolling Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
