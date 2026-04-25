"""
ACE++ — LLM Agent Evaluation (Evidence Script)
==============================================
Runs multiple episodes of the `llm_agent` loop and writes `llm_eval.json`.

This is NOT RL training. It’s repeatable measurement of online adaptation.

Run:
  python3 llm_eval.py
"""

from __future__ import annotations

import json
import statistics

from llm_agent import compute_improvement_metrics, run_episode


def main(episodes: int = 20, num_rounds: int = 10, seed: int = 11) -> None:
    runs = []
    for i in range(episodes):
        acc, total_reward, curve = run_episode(
            num_rounds=num_rounds,
            seed=seed + i,
            verbose=False,
            return_metrics=True,
            curve_path=None,
        )
        improvement = compute_improvement_metrics(curve)
        runs.append(
            {
                "episode": i,
                "seed": seed + i,
                "accuracy": acc,
                "total_reward": total_reward,
                "accuracy_curve": curve,
                "start_accuracy": improvement["start_accuracy"],
                "end_accuracy": improvement["end_accuracy"],
                "improvement": improvement["improvement"],
                "improved": improvement["improved"],
            }
        )

    accs = [r["accuracy"] for r in runs]
    rews = [r["total_reward"] for r in runs]
    improvements = [r["improvement"] for r in runs]
    improved_count = sum(1 for r in runs if r["improved"])

    summary = {
        "episodes": episodes,
        "num_rounds": num_rounds,
        "accuracy_mean": sum(accs) / len(accs),
        "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "reward_mean": sum(rews) / len(rews),
        "reward_std": statistics.pstdev(rews) if len(rews) > 1 else 0.0,
        "within_episode_start_acc_mean": sum(r["start_accuracy"] for r in runs) / len(runs),
        "within_episode_end_acc_mean": sum(r["end_accuracy"] for r in runs) / len(runs),
        "avg_improvement_per_episode": sum(improvements) / len(improvements),
        "episodes_with_improvement": improved_count,
        "episodes_with_improvement_pct": improved_count / len(runs),
    }

    out = {"runs": runs, "summary": summary}
    with open("llm_eval.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== LLM EVAL SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"Episodes with improvement: {improved_count} / {episodes}")
    print(f"Avg improvement: {summary['avg_improvement_per_episode']:+.4f}")
    print("Saved: llm_eval.json")


if __name__ == "__main__":
    main()
