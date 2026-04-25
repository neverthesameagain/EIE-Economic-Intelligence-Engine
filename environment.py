from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


ROUND_TYPES = {"cooperative", "competitive", "resource"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class MarketEnvironment:
    demand: float = 100.0
    resources: float = 100.0
    volatility: float = 0.2
    resource_multiplier: float = 1.0
    demand_multiplier: float = 1.0
    policy_constraints: list[str] = field(default_factory=list)
    uncertainty: float = 0.1
    timestep: int = 0
    last_event: str = "No external event applied yet."
    last_event_structured: dict[str, Any] = field(default_factory=dict)
    hidden_round_type: str = "resource"
    market_pressure: float = 0.2
    alliance_pressure: float = 0.35
    stakes_multiplier: float = 1.0
    narrative: str = "Stable market with no active shock."

    def apply_event(self, event_text: str, structured: dict[str, Any]) -> None:
        self.last_event = event_text or "No external event applied yet."
        self.last_event_structured = structured

        self.resource_multiplier *= float(structured.get("resource_multiplier", 1.0))
        self.demand_multiplier *= float(structured.get("demand_multiplier", 1.0))
        self.volatility = max(0.0, min(1.0, self.volatility + float(structured.get("volatility", 0.0))))
        self.uncertainty = max(0.0, min(1.0, self.uncertainty + float(structured.get("uncertainty", 0.0))))

        for constraint in structured.get("policy_constraints", []):
            if constraint not in self.policy_constraints:
                self.policy_constraints.append(constraint)

        requested_round = structured.get("hidden_round_type")
        self.hidden_round_type = (
            requested_round if requested_round in ROUND_TYPES else self._infer_hidden_round_type(structured)
        )
        self.market_pressure = _clamp(
            abs(float(structured.get("demand_multiplier", 1.0)) - 1.0)
            + float(structured.get("volatility", 0.0))
            + float(structured.get("uncertainty", 0.0))
        )
        self.alliance_pressure = _clamp(
            structured.get("alliance_pressure", 0.65 if self.hidden_round_type in {"cooperative", "resource"} else 0.25)
        )
        self.stakes_multiplier = max(0.5, min(2.0, float(structured.get("stakes_multiplier", 1.0))))
        self.narrative = str(structured.get("narrative", self.last_event))

        # Immediate first-order effects on the macro state.
        self.resources = max(10.0, self.resources * (2.0 - self.resource_multiplier))
        self.demand = max(10.0, self.demand * self.demand_multiplier)

    def advance(self) -> None:
        self.timestep += 1

        # Gradual pull toward equilibrium so the simulation stays responsive.
        self.demand += (100.0 - self.demand) * 0.12
        self.resources += (100.0 - self.resources) * 0.10
        self.resource_multiplier += (1.0 - self.resource_multiplier) * 0.18
        self.demand_multiplier += (1.0 - self.demand_multiplier) * 0.18
        self.volatility += (0.2 - self.volatility) * 0.10
        self.uncertainty += (0.1 - self.uncertainty) * 0.10
        self.market_pressure += (0.2 - self.market_pressure) * 0.08
        self.alliance_pressure += (0.35 - self.alliance_pressure) * 0.05
        self.stakes_multiplier += (1.0 - self.stakes_multiplier) * 0.08

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    def _infer_hidden_round_type(self, structured: dict[str, Any]) -> str:
        resource_shock = structured.get("resource_shock", {})
        demand_multiplier = float(structured.get("demand_multiplier", 1.0))
        volatility = float(structured.get("volatility", 0.0))
        uncertainty = float(structured.get("uncertainty", 0.0))
        constraints = set(structured.get("policy_constraints", []))

        if resource_shock or float(structured.get("resource_multiplier", 1.0)) > 1.15:
            return "resource"
        if volatility + uncertainty > 0.45 or demand_multiplier > 1.15:
            return "competitive"
        if constraints or demand_multiplier < 0.9:
            return "cooperative"
        return "resource"
