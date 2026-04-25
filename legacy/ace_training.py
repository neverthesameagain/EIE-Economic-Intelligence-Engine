"""
ACE++ — Prompting + Dataset + Verifiable Reward (for TRL/GRPO)
==============================================================
This module is intentionally env-state-free for reward scoring:
we embed ground truth into the prompt and re-compute reward locally.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

from env import ACEEnv, ROUND_TYPES, PAYOFF_SALT


SYSTEM_PROMPT = """You are an economic AI agent competing in a hidden-market environment.

Each round, you observe a market state (JSON). Your job:
1) INFER the hidden round type from the signals.
2) CALL the correct tool with correct parameters.

Round types:
- competitive: high demand/volatility/competition → bid aggressively (amount > 60)
- cooperative: low demand/volatility/competition → bid conservatively (amount < 40)
- resource: medium signals → allocate resources (amount 35–65)

You MUST respond with ONLY valid JSON. No extra text.

Required output format:
{
  "belief": {
    "predicted_round": "cooperative|competitive|resource",
    "confidence": 0.0
  },
  "action": {
    "tool": "submit_bid|allocate_resources|execute_contract",
    "parameters": { "amount": 50.0 }
  }
}
"""


def build_prompt(observation: dict) -> str:
    market = observation["market_state"]
    history = observation.get("history", [])
    last_error = observation.get("last_error")

    history_str = ""
    if history:
        recent = history[-2:]  # keep context short
        history_str = "\nRecent history:\n"
        for h in recent:
            correct_str = "✓" if h.get("correct") else "✗"
            history_str += (
                f"  Round {h['round']}: predicted={h.get('predicted_round')} "
                f"actual={h.get('actual_round_type')} {correct_str} "
                f"reward={h.get('r_total', 0.0):.1f}\n"
            )

    error_str = ""
    if last_error:
        error_str = "\nLast error:\n" + json.dumps(last_error, indent=2)

    prompt = f"""Current market state:
{json.dumps(market, indent=2)}
{history_str}{error_str}
Round {observation['round']} of the episode. What is your action?"""
    return prompt


def ace_reward_function(completions: list[str], prompts: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, prompt in zip(completions, prompts):
        rewards.append(_score_single_completion(completion, prompt))
    return rewards


def _score_single_completion(completion: str, prompt: str) -> float:
    parsed = _extract_first_json_object(completion)
    if parsed is None:
        return -1.5

    round_type = _ground_truth_from_prompt(prompt)
    if round_type is None:
        return 0.0

    payoff_seed = _payoff_seed_from_prompt(prompt)
    payoff = _payoff_from_seed(payoff_seed) if payoff_seed is not None else None

    belief = parsed.get("belief") if isinstance(parsed.get("belief"), dict) else {}
    predicted_round = belief.get("predicted_round", parsed.get("predicted_round"))
    confidence = belief.get("confidence", parsed.get("confidence", 1.0))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    if predicted_round not in ROUND_TYPES:
        return -1.0

    # Inference reward (confidence-scaled)
    r_inference = (1.0 if predicted_round == round_type else -0.5) * confidence

    tool, parameters = _normalize_action_tool(parsed)
    r_task = _tool_reward(tool, parameters, round_type, payoff)

    return r_task + 1.2 * r_inference


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Robustly extract the first JSON object from arbitrary model output.
    Uses JSONDecoder.raw_decode so nested braces don't break extraction.
    """
    start = text.find("{")
    if start == -1:
        return None

    decoder = json.JSONDecoder()
    for i in range(start, len(text)):
        if text[i] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _ground_truth_from_prompt(prompt: str) -> Optional[str]:
    match = re.search(r"GROUND_TRUTH:(\w+)", prompt)
    if not match:
        return None
    gt = match.group(1)
    return gt if gt in ROUND_TYPES else None


def _payoff_seed_from_prompt(prompt: str) -> Optional[int]:
    match = re.search(r"PAYOFF_SEED:(\d+)", prompt)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _payoff_from_seed(payoff_seed: int) -> dict[str, float]:
    import random as _random

    r = _random.Random(int(payoff_seed) ^ PAYOFF_SALT)
    competitive_thr = r.uniform(60.0, 70.0)
    cooperative_max = r.uniform(35.0, 45.0)
    resource_min = r.uniform(35.0, 45.0)
    resource_max = r.uniform(55.0, 65.0)
    if resource_max < resource_min:
        resource_min, resource_max = resource_max, resource_min
    return {
        "competitive_bid_threshold": competitive_thr,
        "cooperative_bid_max": cooperative_max,
        "resource_bid_min": resource_min,
        "resource_bid_max": resource_max,
    }


def _normalize_action_tool(parsed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    action_field = parsed.get("action")
    if isinstance(action_field, dict):
        tool = action_field.get("tool") or action_field.get("name") or "execute_contract"
        parameters = action_field.get("parameters") or action_field.get("args") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        return str(tool), parameters

    if isinstance(action_field, str):
        # v0 compatibility
        if action_field == "bid":
            return "submit_bid", {"amount": parsed.get("amount")}
        if action_field == "allocate":
            return "allocate_resources", {"amount": parsed.get("amount")}
        if action_field == "solo":
            return "execute_contract", {}
        return action_field, {}

    # best-effort default
    return "execute_contract", {}


def _tool_reward(
    tool: str,
    parameters: dict[str, Any],
    round_type: str,
    payoff: Optional[dict[str, float]],
) -> float:
    if tool == "submit_bid":
        try:
            amount = float(parameters.get("amount", 50))
        except (TypeError, ValueError):
            return -1.0
        if payoff is None:
            # fallback
            if round_type == "competitive":
                return 2.0 if amount >= 60 else -1.0
            if round_type == "cooperative":
                return 2.0 if amount <= 40 else -1.0
            return 2.0 if 35 <= amount <= 65 else -1.0

        if round_type == "competitive":
            return 2.0 if amount >= payoff["competitive_bid_threshold"] else -1.0
        if round_type == "cooperative":
            return 2.0 if amount <= payoff["cooperative_bid_max"] else -1.0
        return 2.0 if payoff["resource_bid_min"] <= amount <= payoff["resource_bid_max"] else -1.0

    if tool == "allocate_resources":
        return 1.0 if round_type == "resource" else 0.0

    if tool == "execute_contract":
        return 0.5

    return 0.0


def generate_ace_dataset(n_samples: int = 500, num_rounds: int = 10):
    """
    Generate training examples by rolling out the env with a random policy.
    Each example = one step = (prompt, actual_round_type).
    """
    try:
        from datasets import Dataset  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from e

    env = ACEEnv(num_rounds=num_rounds)
    records: list[dict[str, Any]] = []

    while len(records) < n_samples:
        obs = env.reset()
        for _ in range(env.num_rounds):
            actual_round_type = env.current_round_type

            payoff_seed = int(getattr(env, "current_payoff_seed", 0) or 0)
            prompt = build_prompt(obs) + f"\n<!-- GROUND_TRUTH:{actual_round_type} PAYOFF_SEED:{payoff_seed} -->"
            records.append(
                {
                    "prompt": (
                        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                        f"<|im_start|>user\n{prompt}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    ),
                    "actual_round_type": actual_round_type,
                    "payoff_seed": payoff_seed,
                }
            )

            random_action = json.dumps(
                {
                    "belief": {
                        "predicted_round": random.choice(ROUND_TYPES),
                        "confidence": random.random(),
                    },
                    "action": {
                        "tool": "submit_bid",
                        "parameters": {"amount": random.uniform(20, 80)},
                    },
                }
            )
            obs, _, done, _ = env.step(random_action)
            if done:
                break

    return Dataset.from_list(records[:n_samples])
