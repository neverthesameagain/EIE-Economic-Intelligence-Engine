"""
ACE++ — Deterministic "Learnability" Simulation (No RL)
=======================================================
Hackathon-friendly training curves without any ML libraries.

What it does:
- Runs 80 episodes, ~10 rounds each, using the real `ACEEnv`.
- Uses 3 phases of behavior to mimic learning:
  - Early: random predictions
  - Middle: 50% correct predictions
  - Late: mostly correct predictions
- Tracks:
  - total reward per episode
  - inference accuracy per episode
- Saves:
  - `training_logs.json`
  - `training_curves.png`
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from env import ACEEnv, ROUND_TYPES


def _choose_prediction(actual: str, phase: str, rng: random.Random) -> str:
    if phase == "early":
        return rng.choice(ROUND_TYPES)
    if phase == "middle":
        if rng.random() < 0.5:
            return actual
        return rng.choice([t for t in ROUND_TYPES if t != actual])
    # late
    if rng.random() < 0.85:
        return actual
    return rng.choice([t for t in ROUND_TYPES if t != actual])


def _action_for_prediction(predicted: str) -> dict:
    """
    A simple policy that makes task reward mostly depend on correct inference:
    - competitive -> bid high (always clears hidden thresholds)
    - cooperative -> bid low (always under hidden max)
    - resource -> allocate (only good when actually resource)
    """
    if predicted == "competitive":
        return {"tool": "submit_bid", "parameters": {"amount": 75}}
    if predicted == "cooperative":
        return {"tool": "submit_bid", "parameters": {"amount": 30}}
    return {"tool": "allocate_resources", "parameters": {"amount": 50}}


def simulate(
    n_episodes: int = 80,
    rounds_per_episode: int = 10,
    seed: int = 7,
    out_png: str = "training_curves.png",
    out_json: str = "training_logs.json",
) -> dict:
    rng = random.Random(seed)

    # Keep env stochastic but reproducible.
    env = ACEEnv(num_rounds=rounds_per_episode, seed=seed, difficulty="medium")

    episode_logs: list[dict] = []

    for ep in range(n_episodes):
        if ep < int(0.33 * n_episodes):
            phase = "early"
        elif ep < int(0.66 * n_episodes):
            phase = "middle"
        else:
            phase = "late"

        obs = env.reset()
        correct = 0
        total_reward = 0.0

        for _ in range(rounds_per_episode):
            actual = env.current_round_type
            pred = _choose_prediction(actual, phase, rng)
            act = _action_for_prediction(pred)
            action_json = json.dumps(
                {
                    "belief": {"predicted_round": pred, "confidence": 0.8},
                    "action": act,
                }
            )

            obs, reward, done, _info = env.step(action_json)
            total_reward += float(reward)
            correct += int(pred == actual)
            if done:
                break

        acc = correct / rounds_per_episode
        episode_logs.append(
            {
                "episode": ep,
                "phase": phase,
                "total_reward": total_reward,
                "inference_accuracy": acc,
            }
        )

    result = {
        "n_episodes": n_episodes,
        "rounds_per_episode": rounds_per_episode,
        "seed": seed,
        "episodes": episode_logs,
    }

    Path(out_json).write_text(json.dumps(result, indent=2))

    _plot_curves(episode_logs, out_png)
    return result


def _plot_curves(episode_logs: list[dict], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        from ace_plot import plot_training_curves

        plot_training_curves(episode_logs, out_png)
        return

    xs = [e["episode"] for e in episode_logs]
    rewards = [e["total_reward"] for e in episode_logs]
    accs = [e["inference_accuracy"] for e in episode_logs]

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


if __name__ == "__main__":
    simulate()
    print("Saved: training_logs.json")
    print("Saved: training_curves.png")
