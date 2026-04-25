"""
ACE++ — Baseline Comparison
==========================
Compares:
- Random agent
- Rule-based (signal heuristic) agent
- LLM-style adaptive agent (`llm_agent.run_episode`)

Run:
  python3 compare_agents.py
"""

from __future__ import annotations

import json
import random
from typing import Callable

from env import ACEEnv, ROUND_TYPES
from llm_agent import compute_improvement_metrics, run_episode as run_llm_episode


def _action_for_prediction(predicted: str) -> dict:
    if predicted == "competitive":
        return {"belief": {"predicted_round": predicted, "confidence": 0.8}, "action": {"tool": "submit_bid", "parameters": {"amount": 75}}}
    if predicted == "cooperative":
        return {"belief": {"predicted_round": predicted, "confidence": 0.8}, "action": {"tool": "submit_bid", "parameters": {"amount": 30}}}
    return {"belief": {"predicted_round": predicted, "confidence": 0.8}, "action": {"tool": "allocate_resources", "parameters": {"amount": 50}}}


def random_policy(_obs: dict, rng: random.Random) -> str:
    pred = rng.choice(ROUND_TYPES)
    return json.dumps(_action_for_prediction(pred))


def rule_policy(obs: dict, _rng: random.Random) -> str:
    sig = obs["market_state"]["competition_signal"]
    if sig == "high":
        pred = "competitive"
    elif sig == "low":
        pred = "cooperative"
    else:
        pred = "resource"
    return json.dumps(_action_for_prediction(pred))


def run_policy_episode(
    policy: Callable[[dict, random.Random], str],
    *,
    num_rounds: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    env = ACEEnv(num_rounds=num_rounds, seed=seed, difficulty="medium")
    obs = env.reset()
    correct = 0
    total_reward = 0.0

    for _ in range(num_rounds):
        actual = env.current_round_type
        action_json = policy(obs, rng)
        try:
            pred = (json.loads(action_json).get("belief") or {}).get("predicted_round")
        except Exception:
            pred = None
        if pred == actual:
            correct += 1

        obs, reward, done, _ = env.step(action_json)
        total_reward += float(reward)
        if done:
            break

    return correct / num_rounds, total_reward


def main(episodes: int = 20, num_rounds: int = 10, seed: int = 21) -> None:
    random_accs, random_rews = [], []
    rule_accs, rule_rews = [], []
    llm_accs, llm_rews, llm_start, llm_end, llm_improvement = [], [], [], [], []

    for i in range(episodes):
        s = seed + i

        a, r = run_policy_episode(random_policy, num_rounds=num_rounds, seed=s)
        random_accs.append(a)
        random_rews.append(r)

        a, r = run_policy_episode(rule_policy, num_rounds=num_rounds, seed=s)
        rule_accs.append(a)
        rule_rews.append(r)

        acc, rew, curve = run_llm_episode(
            num_rounds=num_rounds,
            seed=s,
            verbose=False,
            return_metrics=True,
            curve_path=None,
        )
        improvement = compute_improvement_metrics(curve)
        llm_accs.append(acc)
        llm_rews.append(rew)
        llm_start.append(improvement["start_accuracy"])
        llm_end.append(improvement["end_accuracy"])
        llm_improvement.append(improvement["improvement"])

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("Agent                         Accuracy (mean)    Reward (mean)")
    print("--------------------------------------------------------------")
    print(f"Random                        {mean(random_accs):.2f}             {mean(random_rews):.2f}")
    print(f"Rule-based (signal heuristic) {mean(rule_accs):.2f}             {mean(rule_rews):.2f}")
    print(f"LLM-style (adaptive)          {mean(llm_accs):.2f}             {mean(llm_rews):.2f}")
    print("")
    print("LLM Agent:")
    print(f"Start Accuracy: {mean(llm_start):.2f}")
    print(f"End Accuracy: {mean(llm_end):.2f}")
    print(f"Improvement: {mean(llm_improvement):+.2f}")
    print("")
    print("Interpretation:")
    print("Rule-based is an oracle-style signal baseline for this environment.")
    print("LLM results emphasize within-episode belief adjustment under feedback.")


if __name__ == "__main__":
    main()
