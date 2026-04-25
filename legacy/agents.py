from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


ROUND_TYPES = {"cooperative", "competitive", "resource"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class EconomicAgent:
    agent_id: int
    capital: float
    strategy: str
    risk_appetite: float
    stake: float = 1.0
    active: bool = True
    last_action: str = "idle"
    last_reasoning: str = "No action taken yet."
    predicted_round: str = "unknown"
    confidence: float = 0.0
    trust_score: float = 0.5
    alliance_state: str = "solo"
    last_reward: float = 0.0

    def act(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        if not self.active:
            self.last_action = "inactive"
            self.last_reasoning = "Agent is inactive."
            return {"action": "inactive", "capital_delta": 0.0, "resource_delta": 0.0, "reason": self.last_reasoning}

        demand = float(environment_state["demand"])
        volatility = float(environment_state["volatility"])
        uncertainty = float(environment_state["uncertainty"])
        constraints = set(environment_state.get("policy_constraints", []))
        resources = float(environment_state["resources"])
        alliance_pressure = float(environment_state.get("alliance_pressure", 0.35))
        market_pressure = float(environment_state.get("market_pressure", 0.2))

        base_action = "hold"
        capital_delta = 0.0
        resource_delta = 0.0
        reason = "Maintaining current position."
        predicted_round, confidence = self._predict_round(environment_state)
        alliance_state = self.alliance_state
        trust_delta = 0.0

        if self.strategy == "greedy":
            if predicted_round == "competitive" and demand > 100:
                base_action = "aggressive_bid"
                capital_delta = 7.0 + 5.0 * self.risk_appetite
                resource_delta = -4.0
                alliance_state = "solo challenger"
                trust_delta = -0.03
                reason = "Signals look competitive, so the agent raises exposure to win scarce contracts."
            elif volatility > 0.6:
                base_action = "speculate"
                capital_delta = 2.0 if self.risk_appetite > 0.6 else -2.0
                trust_delta = -0.02
                reason = "Volatility creates asymmetric upside for a greedy strategy."
            else:
                base_action = "hold_cash"
                capital_delta = 1.0
                reason = "Demand is not strong enough to justify a larger move."

        elif self.strategy == "cooperative":
            if predicted_round in {"cooperative", "resource"} or alliance_pressure > 0.55:
                base_action = "form_coalition"
                capital_delta = 3.0 + 2.0 * alliance_pressure
                resource_delta = -1.5
                alliance_state = "seeking coalition"
                trust_delta = 0.05
                reason = "Observable signals favor pooling risk through a coalition."
            elif "restricted_trade" in constraints or "price_cap" in constraints:
                base_action = "stabilize"
                capital_delta = 1.5
                trust_delta = 0.03
                reason = "Policy constraints favor smaller, coordinated moves."
            elif demand > 100:
                base_action = "steady_trade"
                capital_delta = 4.5
                resource_delta = -2.0
                reason = "Moderate demand supports steady cooperation."
            else:
                base_action = "share_liquidity"
                capital_delta = 2.0
                reason = "Low demand suggests preserving stability across the system."

        elif self.strategy == "adversarial":
            if predicted_round == "competitive" or volatility > 0.5 or uncertainty > 0.4:
                base_action = "challenge_rival"
                capital_delta = 5.5 * max(0.5, self.risk_appetite)
                alliance_state = "opportunistic"
                trust_delta = -0.06
                reason = "Disorder creates opportunities for adversarial positioning."
            elif self.trust_score > 0.65 and market_pressure < 0.35:
                base_action = "temporary_truce"
                capital_delta = 2.0
                alliance_state = "temporary truce"
                trust_delta = 0.02
                reason = "Low pressure makes a short-term truce more valuable than immediate conflict."
            else:
                base_action = "probe_market"
                capital_delta = 1.5
                reason = "Conditions are too calm for larger disruption plays."

        else:  # conservative
            if predicted_round == "resource" or resources < 85:
                base_action = "secure_supply"
                capital_delta = 2.0
                resource_delta = -0.5
                alliance_state = "supply pact"
                trust_delta = 0.03
                reason = "Resource stress makes supply protection more important than expansion."
            elif demand < 95 or volatility > 0.4:
                base_action = "hedge"
                capital_delta = 2.5
                reason = "Risk is elevated, so capital preservation dominates."
            else:
                base_action = "allocate_carefully"
                capital_delta = 3.0
                resource_delta = -1.0
                reason = "Environment is stable enough for measured deployment."

        capital_delta *= max(0.6, min(1.8, self.stake))
        self.predicted_round = predicted_round
        self.confidence = confidence
        self.alliance_state = alliance_state
        self.trust_score = _clamp(self.trust_score + trust_delta)
        self.last_action = base_action
        self.last_reasoning = reason
        return {
            "action": base_action,
            "capital_delta": capital_delta,
            "resource_delta": resource_delta,
            "reason": reason,
            "predicted_round": predicted_round,
            "confidence": confidence,
            "alliance": alliance_state,
        }

    def llm_adjust(self, action_payload: dict[str, Any], llm_adjustment: dict[str, Any] | None) -> dict[str, Any]:
        if not llm_adjustment:
            return action_payload

        action = llm_adjustment.get("action")
        reason = llm_adjustment.get("reason")
        delta_shift = float(llm_adjustment.get("capital_delta_shift", 0.0))
        predicted_round = llm_adjustment.get("predicted_round")
        confidence = llm_adjustment.get("confidence")
        alliance = llm_adjustment.get("alliance")
        trust_delta = float(llm_adjustment.get("trust_delta", 0.0))
        stake_shift = float(llm_adjustment.get("stake_shift", 0.0))

        if isinstance(action, str) and action:
            action_payload["action"] = action
            self.last_action = action
        if isinstance(reason, str) and reason:
            action_payload["reason"] = reason
            self.last_reasoning = reason
        if predicted_round in ROUND_TYPES:
            self.predicted_round = predicted_round
            action_payload["predicted_round"] = predicted_round
        if confidence is not None:
            self.confidence = _clamp(float(confidence))
            action_payload["confidence"] = self.confidence
        if isinstance(alliance, str) and alliance:
            self.alliance_state = alliance[:48]
            action_payload["alliance"] = self.alliance_state
        self.trust_score = _clamp(self.trust_score + trust_delta)
        self.stake = max(0.2, min(2.0, self.stake + stake_shift))
        action_payload["capital_delta"] = float(action_payload.get("capital_delta", 0.0)) + delta_shift
        return action_payload

    def apply_result(self, action_payload: dict[str, Any], environment_state: dict[str, Any]) -> None:
        demand_factor = float(environment_state["demand"]) / 100.0
        volatility_penalty = max(0.0, float(environment_state["volatility"]) - self.risk_appetite * 0.5)
        stakes_multiplier = float(environment_state.get("stakes_multiplier", 1.0))
        exposure = max(0.5, min(2.2, self.stake * stakes_multiplier))
        net = float(action_payload.get("capital_delta", 0.0)) * demand_factor - volatility_penalty * 3.0 * exposure
        actual_round = environment_state.get("hidden_round_type")
        if self.predicted_round == actual_round:
            net += 1.5 * self.confidence
        elif actual_round in ROUND_TYPES:
            net -= 1.0 * max(0.2, self.confidence)
        self.last_reward = net
        self.capital = max(0.0, self.capital + net)
        if self.capital <= 0.0:
            self.active = False
            self.last_reasoning = "Capital exhausted; agent became inactive."

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "capital": round(self.capital, 2),
            "strategy": self.strategy,
            "risk_appetite": round(self.risk_appetite, 2),
            "stake": round(self.stake, 2),
            "trust": round(self.trust_score, 2),
            "predicted_round": self.predicted_round,
            "confidence": round(self.confidence, 2),
            "alliance": self.alliance_state,
            "last_reward": round(self.last_reward, 2),
            "state": "active" if self.active else "inactive",
            "last_action": self.last_action,
            "reasoning": self.last_reasoning,
        }

    def _predict_round(self, environment_state: dict[str, Any]) -> tuple[str, float]:
        demand = float(environment_state["demand"])
        resources = float(environment_state["resources"])
        volatility = float(environment_state["volatility"])
        uncertainty = float(environment_state["uncertainty"])
        constraints = set(environment_state.get("policy_constraints", []))
        resource_shock = environment_state.get("last_event_structured", {}).get("resource_shock", {})

        scores = {
            "competitive": volatility * 0.45 + uncertainty * 0.30 + max(0.0, demand - 100.0) / 70.0,
            "cooperative": len(constraints) * 0.22 + max(0.0, 100.0 - demand) / 80.0,
            "resource": max(0.0, 100.0 - resources) / 70.0 + (0.35 if resource_shock else 0.0),
        }
        predicted = max(scores, key=scores.get)
        confidence = 0.42 + min(0.48, scores[predicted])
        return predicted, _clamp(confidence, 0.35, 0.95)
