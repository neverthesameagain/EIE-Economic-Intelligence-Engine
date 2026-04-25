from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents import EconomicAgent
from environment import MarketEnvironment
from llm_engine import agent_llm_decide, generate_system_explanation, llm_parse_event, process_event_text


@dataclass
class SimulationManager:
    environment: MarketEnvironment = field(default_factory=MarketEnvironment)
    agents: list[EconomicAgent] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    next_agent_id: int = 1
    running: bool = False
    insight: str = "Simulation ready."
    last_event_summary: dict[str, Any] = field(default_factory=dict)

    def add_agent(self, strategy: str, capital: float, risk_appetite: float, stake: float = 1.0) -> None:
        self.agents.append(
            EconomicAgent(
                agent_id=self.next_agent_id,
                capital=float(capital),
                strategy=strategy,
                risk_appetite=float(risk_appetite),
                stake=float(stake),
            )
        )
        self.next_agent_id += 1
        self._log_state("Added agent")

    def remove_agent(self, agent_id: int) -> None:
        self.agents = [agent for agent in self.agents if agent.agent_id != agent_id]
        self._log_state(f"Removed agent {agent_id}")

    def apply_event(self, text: str, use_llm: bool) -> None:
        structured = llm_parse_event(text) if use_llm else process_event_text(text)
        self.environment.apply_event(text, structured)
        self.last_event_summary = structured
        self.insight = self._explain_state()
        self._log_state(f"Applied event: {text}")

    def step(self, use_llm: bool) -> None:
        env_state = self.environment.snapshot()
        significant_change = (
            abs(env_state["demand"] - 100.0) > 8.0
            or env_state["volatility"] > 0.35
            or bool(env_state["policy_constraints"])
            or env_state["market_pressure"] > 0.35
        )

        for idx, agent in enumerate(self.agents):
            action_payload = agent.act(env_state)
            if use_llm and significant_change:
                llm_adjustment = agent_llm_decide(
                    json.dumps(agent.snapshot(), sort_keys=True),
                    json.dumps(self._public_environment_state(env_state), sort_keys=True),
                )
                action_payload = agent.llm_adjust(action_payload, llm_adjustment)
            agent.apply_result(action_payload, env_state)

        if self.agents:
            avg_resource_draw = sum(
                float(agent.last_action not in {"inactive", "hold", "hold_cash"}) for agent in self.agents
            )
            self.environment.resources = max(5.0, self.environment.resources - avg_resource_draw)

        self.environment.advance()
        self.insight = self._explain_state()
        self._log_state("Advanced simulation")

    def run_steps(self, count: int, use_llm: bool) -> None:
        self.running = True
        for _ in range(count):
            self.step(use_llm=use_llm)
        self.running = False

    def pause(self) -> None:
        self.running = False
        self.insight = "Simulation paused."

    def snapshot(self) -> dict[str, Any]:
        return {
            "environment": self.environment.snapshot(),
            "agents": [agent.snapshot() for agent in self.agents],
            "history": self.history,
            "insight": self.insight,
            "last_event_summary": self.last_event_summary,
            "running": self.running,
        }

    def agent_choices(self) -> list[tuple[str, int]]:
        return [(f"Agent {agent.agent_id} ({agent.strategy})", agent.agent_id) for agent in self.agents]

    def _log_state(self, label: str) -> None:
        self.history.append(
            {
                "label": label,
                "timestep": self.environment.timestep,
                "environment": self.environment.snapshot(),
                "agents": [agent.snapshot() for agent in self.agents],
            }
        )
        self.history = self.history[-40:]

    def _explain_state(self) -> str:
        state_json = json.dumps(
            {
                "environment": self.environment.snapshot(),
                "agents": [agent.snapshot() for agent in self.agents],
            },
            sort_keys=True,
        )
        return generate_system_explanation(state_json)

    def _public_environment_state(self, env_state: dict[str, Any]) -> dict[str, Any]:
        """
        Agents get observable signals only. The UI may reveal hidden fields in
        God Mode, but LLM decisions should still infer them from public evidence.
        """
        hidden_keys = {"hidden_round_type"}
        public_state = {key: value for key, value in env_state.items() if key not in hidden_keys}
        if isinstance(public_state.get("last_event_structured"), dict):
            public_state["last_event_structured"] = {
                key: value
                for key, value in public_state["last_event_structured"].items()
                if key not in hidden_keys
            }
        return public_state
