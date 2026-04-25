"""
ACE++ — Judge-Friendly Terminal Demo
===================================
Runs a short episode and prints a clean, readable trace.
"""

from __future__ import annotations

import json

from env import ACEEnv


def infer_round_from_signal(signal: str) -> str:
    if signal == "high":
        return "competitive"
    if signal == "low":
        return "cooperative"
    return "resource"


def action_for_prediction(predicted_round: str) -> dict:
    if predicted_round == "competitive":
        return {"tool": "submit_bid", "parameters": {"amount": 75}}
    if predicted_round == "cooperative":
        return {"tool": "submit_bid", "parameters": {"amount": 30}}
    return {"tool": "allocate_resources", "parameters": {"amount": 50}}


def run_demo(num_rounds: int = 8, seed: int = 1) -> None:
    env = ACEEnv(num_rounds=num_rounds, seed=seed, difficulty="medium")
    obs = env.reset()
    correct = 0
    total = 0

    for i in range(num_rounds):
        market = obs["market_state"]
        signal = market["competition_signal"]

        predicted = infer_round_from_signal(signal)
        action = action_for_prediction(predicted)

        action_json = json.dumps(
            {
                "belief": {"predicted_round": predicted, "confidence": 0.8},
                "action": action,
            }
        )

        obs, reward, done, info = env.step(action_json)
        actual = info.get("debug_round_type", env.current_round_type)

        print(f"\n--- ROUND {i + 1} ---")
        print(f"Signal      : {signal}")
        print(f"Prediction  : {predicted}")
        print(f"Actual      : {actual}")

        is_correct = predicted == actual
        if is_correct:
            print("Result      : CORRECT")
            correct += 1
        else:
            print("Result      : WRONG")

        total += 1
        print(f"Reward      : {reward:.2f}")

        if done:
            break

    if total > 0:
        print("\n=== SUMMARY ===")
        print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")


if __name__ == "__main__":
    run_demo()
