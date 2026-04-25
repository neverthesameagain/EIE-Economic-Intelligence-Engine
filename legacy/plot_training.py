"""
Plot ACE++ deterministic training curves from `training_logs.json`.

Usage:
  python3 plot_training.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main(in_json: str = "training_logs.json", out_png: str = "training_curves.png") -> None:
    data = json.loads(Path(in_json).read_text())
    episodes = data["episodes"]

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        from ace_plot import plot_training_curves

        plot_training_curves(episodes, out_png)
        return

    xs = [e["episode"] for e in episodes]
    rewards = [e["total_reward"] for e in episodes]
    accs = [e["inference_accuracy"] for e in episodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("ACE++ Learnability Simulation (Deterministic)")

    ax1.plot(xs, rewards, linewidth=2)
    ax1.set_ylabel("Learning Progress (Reward)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, accs, linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Hidden State Inference Accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
