"""
ACE++ — LLM-Driven Agent Loop (Online Adaptation Demo)
=====================================================
This is *not* RL training. It's a step-by-step agent loop that:
- observes market_state
- predicts hidden round type
- takes a tool action (JSON)
- receives reward + ground truth
- adapts within the same episode via history + feedback

Backends (auto):
1) OpenAI (if `OPENAI_API_KEY` set)
2) HuggingFace inference (if `HUGGINGFACE_API_TOKEN` set)
3) Fallback mock learner (always available, shows adaptation deterministically)

Run:
  python3 llm_agent.py
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from env_config import load_local_env
from env import ACEEnv, ROUND_TYPES


load_local_env()


SYSTEM_INSTRUCTION = (
    "You are an economic agent in a hidden-market environment.\n"
    "Each round you see a market_state JSON and must infer the hidden round type.\n"
    "You will receive feedback (Correct/Wrong) and the actual type after each round.\n"
    "Your goal is to improve within the episode.\n\n"
    "IMPORTANT: Output ONLY valid JSON, no extra text.\n"
    "Required format:\n"
    "{\n"
    '  "predicted_round": "cooperative|competitive|resource",\n'
    '  "action": "submit_bid|allocate_resources|execute_contract",\n'
    '  "parameters": { "amount": 50 }\n'
    "}\n"
)


def build_prompt(obs: dict, history: list[dict]) -> str:
    market = obs["market_state"]
    lines = []
    if history:
        lines.append("Recent feedback (last rounds):")
        for h in history[-3:]:
            lines.append(
                f"- Round {h['round']}: you predicted={h['prediction']} actual={h['actual']} "
                f"signal={h['signal']} result={'CORRECT' if h['correct'] else 'WRONG'} reward={h['reward']:.2f}"
            )
        lines.append("")

    lines.append("Current market_state:")
    lines.append(json.dumps(market, indent=2))
    lines.append("")
    lines.append("Output ONLY the JSON object in the required format.")
    return "\n".join(lines)


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    start = text.find("{")
    if start == -1:
        return None
    dec = json.JSONDecoder()
    for i in range(start, len(text)):
        if text[i] != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _safe_amount(v: Any) -> float:
    try:
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")):
            raise ValueError("non-finite")
        return x
    except (TypeError, ValueError):
        return 50.0


@dataclass
class ParsedAction:
    predicted_round: str
    tool: str
    amount: float
    format_valid: bool
    parse_error: str | None


def estimate_confidence(history: list[dict], signal: str, prediction: str) -> float:
    """
    Lightweight belief proxy for demos.
    Confidence rises when the current prediction aligns with prior feedback for
    the same signal and with recent correct predictions.
    """
    same_signal = [h for h in history if h["signal"] == signal]
    confidence = 0.45

    if same_signal:
        matches = sum(1 for h in same_signal if h["actual"] == prediction)
        confidence += 0.3 * (matches / len(same_signal))
        confidence += min(0.1, 0.03 * len(same_signal))

    streak = 0
    for item in reversed(history):
        if item["correct"] and item["prediction"] == prediction:
            streak += 1
        else:
            break
    confidence += min(0.1, 0.05 * streak)

    return max(0.35, min(0.95, confidence))


def compute_improvement_metrics(rolling_acc: list[float]) -> dict[str, Any]:
    start_acc = rolling_acc[0] if rolling_acc else 0.0
    end_acc = rolling_acc[-1] if rolling_acc else 0.0
    improvement = end_acc - start_acc
    return {
        "start_accuracy": start_acc,
        "end_accuracy": end_acc,
        "improvement": improvement,
        "improved": improvement > 1e-9,
    }


def parse_llm_action(text: str, rng: random.Random) -> ParsedAction:
    obj = _extract_first_json_object(text)
    format_valid = True
    parse_error: str | None = None

    if obj is None:
        obj = {}
        format_valid = False
        parse_error = (
            "Invalid JSON output. Expected "
            '{"predicted_round":"...","action":"...","parameters":{"amount":50}}'
        )

    predicted = obj.get("predicted_round") or obj.get("belief", {}).get("predicted_round")
    if predicted not in ROUND_TYPES:
        format_valid = False
        if parse_error is None:
            parse_error = f"Invalid or missing predicted_round. Expected one of {ROUND_TYPES}."
        predicted = rng.choice(ROUND_TYPES)

    tool = obj.get("action")
    params = obj.get("parameters") or {}
    if isinstance(obj.get("action"), dict):
        tool = obj["action"].get("tool")
        params = obj["action"].get("parameters") or {}

    if tool not in {"submit_bid", "allocate_resources", "execute_contract"}:
        format_valid = False
        if parse_error is None:
            parse_error = "Invalid or missing action. Expected submit_bid, allocate_resources, or execute_contract."
        tool = "submit_bid"

    amount = _safe_amount(params.get("amount", obj.get("amount", 50)))
    original_amount = params.get("amount", obj.get("amount", 50))
    try:
        float(original_amount)
    except (TypeError, ValueError):
        format_valid = False
        if parse_error is None:
            parse_error = "Invalid amount. Expected numeric parameters.amount."

    return ParsedAction(
        predicted_round=predicted,
        tool=tool,
        amount=amount,
        format_valid=format_valid,
        parse_error=parse_error,
    )


def to_env_action(pa: ParsedAction, confidence: float) -> str:
    return json.dumps(
        {
            "belief": {"predicted_round": pa.predicted_round, "confidence": confidence},
            "action": {"tool": pa.tool, "parameters": {"amount": pa.amount}},
        }
    )


# ----------------------------
# LLM backends
# ----------------------------


def get_llm_response(prompt: str) -> str:
    if os.getenv("OPENAI_API_KEY"):
        return _openai_chat_completion(prompt)
    if os.getenv("HUGGINGFACE_API_TOKEN"):
        return _hf_inference(prompt)
    return _mock_adaptive_response(prompt)


def _openai_chat_completion(prompt: str) -> str:
    """
    Minimal OpenAI Chat Completions call via stdlib (no extra deps).
    Env vars:
      - OPENAI_API_KEY (required)
      - OPENAI_MODEL (default: gpt-4o-mini)
      - OPENAI_BASE_URL (default: https://api.openai.com)
    """
    api_key = os.environ["OPENAI_API_KEY"]
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    j = json.loads(body)
    return j["choices"][0]["message"]["content"]


def _hf_inference(prompt: str) -> str:
    """
    Minimal HuggingFace Inference API call via stdlib.
    Env vars:
      - HUGGINGFACE_API_TOKEN (required)
      - HUGGINGFACE_MODEL (default: google/gemma-2-2b-it)
    """
    token = os.environ["HUGGINGFACE_API_TOKEN"]
    model = os.getenv("HUGGINGFACE_MODEL", "google/gemma-2-2b-it")
    url = f"https://api-inference.huggingface.co/models/{model}"

    payload = {
        "inputs": SYSTEM_INSTRUCTION + "\n\n" + prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.2},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    j = json.loads(body)
    # HF may return a list of {generated_text: ...}
    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
        return j[0]["generated_text"]
    if isinstance(j, dict) and "generated_text" in j:
        return j["generated_text"]
    return json.dumps(j)


# ----------------------------
# Fallback mock learner (always runs)
# ----------------------------


_MOCK_STATE: dict[str, Any] = {"by_signal": {}, "t": 0, "seen": set(), "rng": random.Random(0)}


def _mock_adaptive_response(prompt: str) -> str:
    """
    Learns a mapping from competition_signal -> round_type online, from feedback.
    Also emits occasional invalid JSON early to demonstrate robustness.
    """
    _MOCK_STATE["t"] += 1
    t = _MOCK_STATE["t"]

    signal = _extract_signal(prompt) or "medium"
    by_signal: dict[str, dict[str, int]] = _MOCK_STATE["by_signal"]
    seen: set[int] = _MOCK_STATE["seen"]

    # Parse feedback lines and update counts per-signal, but only once per round.
    for m in re.finditer(
        r"- Round (\d+): .*?actual=(cooperative|competitive|resource) .*?signal=(high|low|medium)",
        prompt,
    ):
        round_num = int(m.group(1))
        actual = m.group(2)
        sig = m.group(3)
        if round_num in seen:
            continue
        seen.add(round_num)
        counts = by_signal.setdefault(sig, {rt: 0 for rt in ROUND_TYPES})
        counts[actual] += 1

    counts = by_signal.setdefault(signal, {rt: 0 for rt in ROUND_TYPES})

    # Exploration decays with time to show adaptation.
    explore_p = 0.55 if t < 4 else 0.25 if t < 7 else 0.05
    r = _MOCK_STATE["rng"]
    if r.random() < explore_p:
        pred = r.choice(ROUND_TYPES)
    else:
        pred = max(counts, key=lambda k: counts[k])

    # Optional: intentionally bad output once early (robust parsing demo).
    if os.getenv("ACE_NOISY_LLM") and t == 2:
        return '{"predicted_round": '  # invalid JSON on purpose

    if pred == "competitive":
        action = {"action": "submit_bid", "parameters": {"amount": 75}}
    elif pred == "cooperative":
        action = {"action": "submit_bid", "parameters": {"amount": 30}}
    else:
        action = {"action": "allocate_resources", "parameters": {"amount": 50}}

    return json.dumps({"predicted_round": pred, **action})


def _extract_signal(prompt: str) -> Optional[str]:
    m = re.search(r'"competition_signal"\s*:\s*"(\w+)"', prompt)
    return m.group(1) if m else None


# ----------------------------
# Main loop
# ----------------------------


def run_episode(
    *,
    num_rounds: int = 10,
    seed: int = 3,
    verbose: bool = True,
    return_metrics: bool = False,
    curve_path: Optional[str] = "llm_episode_curve.json",
) -> Any:
    rng = random.Random(seed)
    env = ACEEnv(num_rounds=num_rounds, seed=seed, difficulty="medium")
    obs = env.reset()

    # If we're using the fallback backend, reset its state so each episode
    # demonstrates adaptation from scratch.
    using_fallback = not os.getenv("OPENAI_API_KEY") and not os.getenv("HUGGINGFACE_API_TOKEN")
    if using_fallback:
        _MOCK_STATE["by_signal"] = {}
        _MOCK_STATE["t"] = 0
        _MOCK_STATE["seen"] = set()
        _MOCK_STATE["rng"] = random.Random(seed)

    history: list[dict] = []
    rolling_acc: list[float] = []

    correct = 0
    total = 0
    total_reward = 0.0
    adaptation_count = 0
    invalid_outputs = 0

    for i in range(num_rounds):
        market = obs["market_state"]
        signal = market["competition_signal"]

        prompt = build_prompt(obs, history)

        try:
            raw = get_llm_response(prompt)
        except Exception:
            raw = ""  # hard fallback: parsing will default safely

        parsed = parse_llm_action(raw, rng)
        confidence = estimate_confidence(history, signal, parsed.predicted_round)
        action_json = to_env_action(parsed, confidence)

        obs, reward, done, info = env.step(action_json)
        actual = info.get("debug_round_type", env.current_round_type)
        is_correct = parsed.predicted_round == actual
        adapted = bool(history) and (not history[-1]["correct"]) and is_correct
        validation_penalty = -0.5 if not parsed.format_valid else 0.0

        total += 1
        if is_correct:
            correct += 1
        rolling_acc.append(correct / total)
        total_reward += float(reward) + validation_penalty
        if adapted:
            adaptation_count += 1
        if not parsed.format_valid:
            invalid_outputs += 1

        history.append(
            {
                "round": i + 1,
                "signal": signal,
                "prediction": parsed.predicted_round,
                "confidence": confidence,
                "actual": actual,
                "correct": is_correct,
                "reward": float(reward) + validation_penalty,
                "format_valid": parsed.format_valid,
            }
        )

        if verbose:
            result_icon = "✅" if is_correct else "❌"
            print("\n--------------------------------")
            print(f"ROUND {i + 1}")
            print(f"Prediction: {parsed.predicted_round} (confidence: {confidence:.2f})")
            print(f"Actual: {actual}")
            print(f"Result: {result_icon} {'CORRECT' if is_correct else 'WRONG'}")
            if is_correct:
                print("Prediction matched hidden state")
            if adapted:
                print("Adaptation Detected")
                print("Belief Update: adjusting strategy based on feedback")
            if not parsed.format_valid:
                print("Validation Error:")
                print(parsed.parse_error)
                print("Penalty Applied: -0.50")
            print(f"Accuracy so far: {rolling_acc[-1]:.2f}")
            print(f"Reward: {float(reward) + validation_penalty:.2f}")
            print("--------------------------------")

        if done:
            break

        # tiny pause makes it feel interactive in live demos (optional)
        if os.getenv("ACE_DEMO_SLEEP"):
            time.sleep(float(os.getenv("ACE_DEMO_SLEEP")))

    if curve_path:
        try:
            with open(curve_path, "w") as f:
                json.dump(rolling_acc, f, indent=2)
        except OSError:
            pass

    if verbose:
        episode_metrics = compute_improvement_metrics(rolling_acc)
        print("\n=== SUMMARY ===")
        print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")
        print(f"Accuracy over time: {[round(x, 2) for x in rolling_acc]}")
        print(f"Start accuracy: {episode_metrics['start_accuracy']:.2f}")
        print(f"End accuracy: {episode_metrics['end_accuracy']:.2f}")
        print(f"Improvement: {episode_metrics['improvement']:+.2f}")
        print(f"Adaptation events: {adaptation_count}")
        print(f"Invalid outputs: {invalid_outputs}")
        print("\n=== KEY RESULT ===")
        if episode_metrics["improvement"] >= 0:
            print(
                f"Agent improved from {episode_metrics['start_accuracy']:.2f} -> "
                f"{episode_metrics['end_accuracy']:.2f} within a single episode"
            )
        else:
            print(
                f"Agent changed from {episode_metrics['start_accuracy']:.2f} -> "
                f"{episode_metrics['end_accuracy']:.2f} within a single episode "
                f"({episode_metrics['improvement']:+.2f})"
            )
        if curve_path:
            print(f"Saved: {curve_path}")

    acc = (correct / total) if total else 0.0
    if return_metrics:
        return acc, total_reward, rolling_acc
    return None


def run_agent(num_rounds: int = 10, seed: int = 3) -> None:
    run_episode(num_rounds=num_rounds, seed=seed, verbose=True, return_metrics=False, curve_path="llm_episode_curve.json")


if __name__ == "__main__":
    run_agent()
