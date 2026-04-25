from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache
from typing import Any

from env_config import load_local_env


load_local_env()


EVENT_SYSTEM_PROMPT = """You convert economic event descriptions into strict JSON.
You are powering the ACE++ live demo. Prefer rich, causal interpretation.
Return ONLY valid JSON with this schema:
{
  "resource_multiplier": 1.0,
  "demand_multiplier": 1.0,
  "volatility": 0.0,
  "uncertainty": 0.0,
  "policy_constraints": [],
  "resource_shock": {},
  "hidden_round_type": "cooperative|competitive|resource",
  "alliance_pressure": 0.0,
  "stakes_multiplier": 1.0,
  "narrative": "one sentence",
  "confidence": 0.0
}

Few-shot examples:
Input: "global oil crisis and strict government regulation"
Output: {"resource_multiplier": 1.4, "demand_multiplier": 0.85, "volatility": 0.35, "uncertainty": 0.25, "policy_constraints": ["price_cap", "restricted_trade"], "resource_shock": {"oil": -0.5}, "hidden_round_type": "resource", "alliance_pressure": 0.78, "stakes_multiplier": 1.35, "narrative": "Energy scarcity forces agents to decide whether to pool supply or hoard it.", "confidence": 0.85}

Input: "AI boom increases productivity"
Output: {"resource_multiplier": 0.9, "demand_multiplier": 1.2, "volatility": 0.08, "uncertainty": 0.02, "policy_constraints": [], "resource_shock": {"compute": 0.3}, "hidden_round_type": "competitive", "alliance_pressure": 0.28, "stakes_multiplier": 1.15, "narrative": "A productivity boom raises demand and makes contract capture more competitive.", "confidence": 0.82}
"""

AGENT_SYSTEM_PROMPT = """You are a strategic ACE++ economic agent using Llama-style reasoning.
You see only public market signals, agent state, and peer summaries. Infer the hidden round type.
Favor decisions that react causally to the event, not arbitrary signaling.
Return ONLY valid JSON:
{
  "action": "string",
  "predicted_round": "cooperative|competitive|resource",
  "confidence": 0.0,
  "alliance": "solo|seeking coalition|temporary truce|challenge",
  "trust_delta": 0.0,
  "stake_shift": 0.0,
  "reason": "short explanation",
  "capital_delta_shift": 0.0
}
"""

EXPLANATION_SYSTEM_PROMPT = """You explain the current simulation state in 2 concise sentences.
Focus on what changed, how agents reacted, and what trend is emerging.
"""


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    for idx in range(start, len(text)):
        if text[idx] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _llm_available() -> bool:
    return bool(
        (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
        or os.getenv("HUGGINGFACE_API_TOKEN")
    )


def _chat_completion(system_prompt: str, user_prompt: str, *, temperature: float = 0.25, max_tokens: int = 220) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    if not api_key or not model:
        return _hf_completion(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)

    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _hf_completion(system_prompt: str, user_prompt: str, *, temperature: float, max_tokens: int) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("LLM not configured.")

    model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
    url = f"https://api-inference.huggingface.co/models/{model}"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if isinstance(body, list) and body and isinstance(body[0], dict):
        return str(body[0].get("generated_text", ""))
    if isinstance(body, dict):
        return str(body.get("generated_text", body))
    return str(body)


def _fallback_event_parser(text: str) -> dict[str, Any]:
    lowered = text.lower()
    result = {
        "resource_multiplier": 1.0,
        "demand_multiplier": 1.0,
        "volatility": 0.0,
        "uncertainty": 0.0,
        "policy_constraints": [],
        "resource_shock": {},
        "hidden_round_type": "resource",
        "alliance_pressure": 0.35,
        "stakes_multiplier": 1.0,
        "narrative": text or "No external event provided.",
        "confidence": 0.55,
    }

    if any(word in lowered for word in ["recession", "slowdown", "crash"]):
        result["demand_multiplier"] = 0.72
        result["volatility"] = 0.28
        result["uncertainty"] = 0.18
        result["hidden_round_type"] = "cooperative"
        result["alliance_pressure"] = 0.72
        result["stakes_multiplier"] = 1.2
        result["narrative"] = "Demand shock pushes agents toward defensive coalitions and liquidity sharing."
    if any(word in lowered for word in ["boom", "surge", "growth", "rally"]):
        result["demand_multiplier"] = max(result["demand_multiplier"], 1.18)
        result["volatility"] += 0.08
        result["hidden_round_type"] = "competitive"
        result["alliance_pressure"] = min(result["alliance_pressure"], 0.32)
        result["stakes_multiplier"] = max(result["stakes_multiplier"], 1.12)
        result["narrative"] = "Demand surge turns market share capture into the dominant opportunity."
    if any(word in lowered for word in ["oil", "energy", "shortage", "supply shock"]):
        result["resource_multiplier"] = 1.35
        result["resource_shock"]["oil"] = -0.4
        result["volatility"] += 0.18
        result["hidden_round_type"] = "resource"
        result["alliance_pressure"] = 0.78
        result["stakes_multiplier"] = max(result["stakes_multiplier"], 1.35)
        result["narrative"] = "Resource scarcity makes supply alliances and hoarding strategies collide."
    if any(word in lowered for word in ["regulation", "policy", "government", "sanction"]):
        result["policy_constraints"].append("restricted_trade")
        result["volatility"] += 0.10
        result["alliance_pressure"] = max(result["alliance_pressure"], 0.65)
    if any(word in lowered for word in ["price cap", "price-control", "price control"]):
        result["policy_constraints"].append("price_cap")
    if any(word in lowered for word in ["uncertainty", "panic", "war", "conflict"]):
        result["uncertainty"] += 0.22
        result["volatility"] += 0.20
        result["hidden_round_type"] = "competitive"
        result["stakes_multiplier"] = max(result["stakes_multiplier"], 1.45)
    if any(word in lowered for word in ["ai", "productivity", "automation"]):
        result["demand_multiplier"] = max(result["demand_multiplier"], 1.12)
        result["resource_multiplier"] = min(result["resource_multiplier"], 0.92)
        result["resource_shock"]["compute"] = 0.3
        result["hidden_round_type"] = "competitive"
        result["narrative"] = "Automation expands capacity, but agents compete to capture the new productivity premium."

    result["volatility"] = min(0.9, result["volatility"])
    result["uncertainty"] = min(0.9, result["uncertainty"])
    return result


@lru_cache(maxsize=128)
def process_event_text(text: str) -> dict[str, Any]:
    return _fallback_event_parser(text)


@lru_cache(maxsize=128)
def llm_parse_event(text: str) -> dict[str, Any]:
    if not text.strip():
        return process_event_text(text)
    if not _llm_available():
        return process_event_text(text)
    try:
        raw = _chat_completion(EVENT_SYSTEM_PROMPT, f'Input: "{text}"', temperature=0.2, max_tokens=180)
        parsed = _extract_first_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("No JSON object returned.")
        fallback = process_event_text(text)
        return {
            "resource_multiplier": float(parsed.get("resource_multiplier", fallback["resource_multiplier"])),
            "demand_multiplier": float(parsed.get("demand_multiplier", fallback["demand_multiplier"])),
            "volatility": float(parsed.get("volatility", fallback["volatility"])),
            "uncertainty": float(parsed.get("uncertainty", fallback["uncertainty"])),
            "policy_constraints": list(parsed.get("policy_constraints", fallback["policy_constraints"])),
            "resource_shock": dict(parsed.get("resource_shock", fallback["resource_shock"])),
            "hidden_round_type": str(parsed.get("hidden_round_type", fallback["hidden_round_type"])),
            "alliance_pressure": float(parsed.get("alliance_pressure", fallback["alliance_pressure"])),
            "stakes_multiplier": float(parsed.get("stakes_multiplier", fallback["stakes_multiplier"])),
            "narrative": str(parsed.get("narrative", fallback["narrative"])),
            "confidence": float(parsed.get("confidence", 0.75)),
        }
    except Exception:
        return process_event_text(text)


@lru_cache(maxsize=256)
def agent_llm_decide(agent_state_json: str, environment_state_json: str) -> dict[str, Any]:
    if not _llm_available():
        return {}
    try:
        raw = _chat_completion(
            AGENT_SYSTEM_PROMPT,
            f"Agent state:\n{agent_state_json}\n\nEnvironment state:\n{environment_state_json}",
            temperature=0.25,
            max_tokens=140,
        )
        parsed = _extract_first_json_object(raw)
        if not isinstance(parsed, dict):
            return {}
        return {
            "action": str(parsed.get("action", "")),
            "predicted_round": str(parsed.get("predicted_round", "")),
            "confidence": float(parsed.get("confidence", 0.0)),
            "alliance": str(parsed.get("alliance", "")),
            "trust_delta": float(parsed.get("trust_delta", 0.0)),
            "stake_shift": float(parsed.get("stake_shift", 0.0)),
            "reason": str(parsed.get("reason", "")),
            "capital_delta_shift": float(parsed.get("capital_delta_shift", 0.0)),
        }
    except Exception:
        return {}


@lru_cache(maxsize=256)
def generate_system_explanation(state_json: str) -> str:
    state = json.loads(state_json)
    env = state["environment"]
    agents = state["agents"]
    fallback = (
        f"After '{env['last_event']}', demand is at {env['demand']:.1f} and volatility is {env['volatility']:.2f}. "
        f"{sum(1 for agent in agents if agent['state'] == 'active')} active agents are adapting, with the latest actions reflecting current policy constraints and macro stress."
    )
    if not _llm_available():
        return fallback
    try:
        return _chat_completion(EXPLANATION_SYSTEM_PROMPT, state_json, temperature=0.3, max_tokens=120).strip()
    except Exception:
        return fallback
