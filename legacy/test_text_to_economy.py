"""Smoke tests for the deterministic text-to-economy pipeline.

Run:
    python test_text_to_economy.py
"""

from __future__ import annotations

import os

import ace_text_inject
from ace_text_inject import DELTA_FIELDS, parse_event_payload
from ace_world_env import CLAMP_RANGES, WorldState


def assert_valid_payload(payload: dict) -> None:
    assert set(DELTA_FIELDS).issubset(payload["deltas"])
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["reasoning"], str) and payload["reasoning"]
    assert isinstance(payload["affected_sectors"], list)
    for key in DELTA_FIELDS:
        value = payload["deltas"][key]
        assert isinstance(value, float)
        assert abs(value) <= 0.4


def test_oil_crisis() -> None:
    payload = parse_event_payload("oil crisis", provider="fallback")
    assert_valid_payload(payload)
    assert payload["deltas"]["oil_price"] > 0
    assert payload["deltas"]["market_volatility"] > 0
    assert payload["deltas"]["inflation"] > 0
    assert "energy" in payload["affected_sectors"]


def test_peace_deal() -> None:
    payload = parse_event_payload("peace deal", provider="fallback")
    assert_valid_payload(payload)
    assert payload["deltas"]["cooperation_index"] > 0
    assert payload["deltas"]["market_volatility"] < 0


def test_repeated_same_input_consistent() -> None:
    first = parse_event_payload("major food supply chain disruption", provider="fallback")
    second = parse_event_payload("  major   food supply chain disruption ", provider="fallback")
    assert first == second


def test_invalid_llm_output_falls_back() -> None:
    original_key = os.environ.get("GROQ_API_KEY")
    original_call = ace_text_inject.call_groq_chat_completion
    try:
        os.environ["GROQ_API_KEY"] = "dummy"

        def bad_call(*args, **kwargs):
            return "not json"

        ace_text_inject.call_groq_chat_completion = bad_call
        payload = parse_event_payload("oil crisis invalid llm", provider="groq")
    finally:
        ace_text_inject.call_groq_chat_completion = original_call
        if original_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = original_key

    assert_valid_payload(payload)
    assert payload["deltas"]["oil_price"] > 0


def test_world_update_clamps_bounds() -> None:
    world = WorldState(oil_price=2.45)
    world.apply_deltas({"oil_price": 0.4})
    low, high = CLAMP_RANGES["oil_price"]
    assert low <= world.oil_price <= high
    assert world.oil_price == high


if __name__ == "__main__":
    test_oil_crisis()
    test_peace_deal()
    test_repeated_same_input_consistent()
    test_invalid_llm_output_falls_back()
    test_world_update_clamps_bounds()
    print("text-to-economy tests passed")
