"""Judge-facing ACE++ Option B demo.

Run:
    LLM_PROVIDER=groq GROQ_API_KEY=... python demo_gradio.py
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... python demo_gradio.py

The app also works without an API key by using deterministic adaptive fallback
agents, so the demo never crashes during judging.
"""

from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover
    gr = None


def load_local_env(path: str = ".env") -> None:
    """Load local env vars for development without overriding Space secrets."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

from ace_agents import AgentProfile
from ace_text_inject import call_groq_chat_completion, describe_impact
from ace_world_env import ACEWorldEnv


ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PROVIDERS = {"fallback", "groq", "anthropic"}
PERFECT_DEMO_EVENT = "oil crisis hits Middle East"
PRESET_EVENTS = [
    PERFECT_DEMO_EVENT,
    "global cooperation agreement signed",
    "major food supply chain disruption",
]
APP_CSS = """
:root {
  --ace-bg: #070b1a;
  --ace-card: rgba(255, 255, 255, 0.08);
  --ace-card-strong: rgba(255, 255, 255, 0.14);
  --ace-border: rgba(255, 255, 255, 0.16);
  --ace-text: #e5e7eb;
  --ace-muted: #94a3b8;
  --ace-cyan: #22d3ee;
  --ace-purple: #a78bfa;
  --ace-pink: #fb7185;
}

body, .gradio-container {
  background:
    radial-gradient(circle at 8% 8%, rgba(34, 211, 238, 0.24), transparent 28%),
    radial-gradient(circle at 88% 12%, rgba(167, 139, 250, 0.28), transparent 28%),
    radial-gradient(circle at 50% 95%, rgba(251, 113, 133, 0.16), transparent 30%),
    linear-gradient(180deg, #070b1a 0%, #0f172a 48%, #111827 100%) !important;
  color: var(--ace-text) !important;
}

.gradio-container {
  max-width: 1500px !important;
  margin: auto;
}

.block, .form, .panel, .gr-box, .gr-form, .gr-panel {
  border-color: var(--ace-border) !important;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 34px 36px;
  border-radius: 30px;
  color: white;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(37, 99, 235, 0.86) 50%, rgba(124, 58, 237, 0.88) 100%);
  border: 1px solid rgba(255,255,255,0.22);
  box-shadow: 0 24px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.22);
  backdrop-filter: blur(18px);
}

.hero::after {
  content: "";
  position: absolute;
  width: 320px;
  height: 320px;
  right: -110px;
  top: -130px;
  background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 92%);
}

.hero p {
  margin: 0;
  max-width: 980px;
  color: rgba(255,255,255,0.88);
}

.hero h1 {
  font-size: 42px;
  font-weight: 800;
  margin: 0 0 8px 0;
  letter-spacing: -0.045em;
}

.demo-hint {
  margin: 16px 0 6px;
  padding: 15px 18px;
  border-radius: 20px !important;
  border: 1px solid rgba(34, 211, 238, 0.32);
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.18) 0%, rgba(124, 58, 237, 0.18) 100%);
  color: #e0f2fe;
  font-weight: 700;
  box-shadow: 0 14px 35px rgba(2, 6, 23, 0.22);
}

.flow-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0 10px;
}

.flow-step {
  background: linear-gradient(180deg, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.07) 100%);
  border-radius: 20px;
  border: 1px solid var(--ace-border);
  padding: 16px 18px;
  box-shadow: 0 16px 38px rgba(2, 6, 23, 0.24);
  backdrop-filter: blur(14px);
  transition: all 0.25s ease;
}

.flow-step b {
  display: block;
  margin-bottom: 6px;
  color: #f8fafc;
}

.flow-step span {
  color: #cbd5e1;
  font-size: 13px;
}

.flow-step:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

.agent-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.075) 100%);
  padding: 18px;
  border-radius: 22px;
  border: 1px solid var(--ace-border);
  margin-bottom: 14px;
  box-shadow: 0 18px 42px rgba(2, 6, 23, 0.24);
  backdrop-filter: blur(14px);
}

.agent-card-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 12px;
}

.agent-card h3 {
  margin:0;
  color: #f8fafc;
}

.agent-card small, .agent-muted {
  color: #cbd5e1;
}

.agent-resources {
  font-weight: 800;
  color:#86efac;
}

.belief-pre {
  margin: 6px 0 0;
  padding: 10px;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.55);
  color: #e0f2fe;
  white-space: pre-wrap;
  border: 1px solid rgba(148, 163, 184, 0.22);
}

button {
  border-radius: 16px !important;
  font-weight: 600;
  transition: all 0.2s ease;
  box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
}

button:hover {
  transform: scale(1.04);
}

textarea {
  font-family: monospace;
  font-size: 13px;
  background: rgba(15, 23, 42, 0.72) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(148, 163, 184, 0.28) !important;
}

.section-title {
  margin: 20px 0 10px;
  padding: 12px 16px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.16) 0%, rgba(167, 139, 250, 0.16) 100%);
  color: #f8fafc;
  font-weight: 700;
  letter-spacing: 0.3px;
  border: 1px solid var(--ace-border);
  box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

.flow-strip, .agent-card {
  animation: fadeInUp 0.4s ease;
}

.why-matters {
  margin: 18px 0 8px;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), rgba(251, 113, 133, 0.15));
  color: #e2e8f0;
  font-weight: 700;
  border: 1px solid var(--ace-border);
}

.small-note { color: #cbd5e1; font-size: 13px; margin: 8px 0 2px; }
textarea, input { border-radius: 14px !important; }

label, .wrap, .prose, .markdown, .gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: var(--ace-text) !important;
}

input, select {
  background: rgba(15, 23, 42, 0.72) !important;
  color: #e5e7eb !important;
  border-color: rgba(148, 163, 184, 0.28) !important;
}
"""
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    go = None


def make_fresh_env() -> ACEWorldEnv:
    return ACEWorldEnv()


def parse_agent_json(raw: str) -> dict[str, Any] | None:
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    decoder = json.JSONDecoder()
    for idx, char in enumerate(clean):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(clean[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_provider(provider: str | None) -> str:
    value = (provider or os.getenv("LLM_PROVIDER", "fallback")).lower().strip()
    return value if value in PROVIDERS else "fallback"


def llm_or_fallback_decision(
    env: ACEWorldEnv,
    agent: AgentProfile,
    provider: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    available = [item.agent_id for item in env.agents]
    fallback = agent.choose_fallback_action(
        env.world.derive_round_probabilities(),
        env.round_number + 1,
        available,
    )
    provider = normalize_provider(provider)

    if provider == "fallback":
        return fallback

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()

            user_prompt = "\n".join([
                f"Upcoming round: {env.round_number + 1}",
                f"Visible alliances: {sorted([list(pair) for pair in env.alliances])}",
                "Recent global round history:",
                json.dumps(env.round_history[-3:], indent=2),
                "Return ONLY valid JSON.",
            ])

            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=260,
                system=agent.system_prompt(env.world.to_prompt_str()),
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = response.content[0].text.strip()
            print_raw_llm(agent.name, raw, debug)
            parsed = parse_agent_json(raw)

            return merge_decision(parsed, fallback)

        except Exception:
            return fallback

    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        try:
            messages = [
                {"role": "system", "content": agent.system_prompt(env.world.to_prompt_str())},
                {"role": "user", "content": "\n".join([
                    f"Upcoming round: {env.round_number + 1}",
                    f"Visible alliances: {sorted([list(pair) for pair in env.alliances])}",
                    "Recent global round history:",
                    json.dumps(env.round_history[-3:], indent=2),
                    "Return ONLY valid JSON. No explanation.",
                ])}
            ]

            raw = call_groq_chat_completion(
                messages=messages,
                model=GROQ_MODEL,
                temperature=0.35,
                max_tokens=320,
            )
            print_raw_llm(agent.name, raw, debug)

            return merge_decision(parse_agent_json(raw or ""), fallback)

        except Exception:
            return fallback

    return fallback


def print_raw_llm(label: str, raw: str, debug: bool) -> None:
    if not debug:
        return
    print(f"\n[LLM RAW:{label}]")
    print(raw)
    print(f"[/LLM RAW:{label}]\n")


def merge_decision(parsed: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return fallback
    return {
        "predicted_round": parsed.get("predicted_round", fallback["predicted_round"]),
        "action": parsed.get("action", fallback["action"]),
        "parameters": parsed.get("parameters", fallback["parameters"]),
        "beliefs": parsed.get("beliefs", fallback.get("beliefs", {})),
        "factors": parsed.get("factors", fallback.get("factors", {})),
        "reasoning": parsed.get("reasoning", fallback["reasoning"]),
    }

def render_world(env: ACEWorldEnv) -> tuple[Any, ...]:
    world = env.world
    probs = world.derive_round_probabilities()
    return (
        round(world.oil_price, 3),
        round(world.gold_price, 3),
        round(world.food_index, 3),
        round(world.energy_cost, 3),
        round(world.interest_rate, 4),
        round(world.inflation, 4),
        round(world.gdp_growth, 4),
        round(world.liquidity_index, 3),
        round(world.credit_spread, 3),
        round(world.trade_tension, 3),
        round(world.market_volatility, 3),
        round(world.cooperation_index, 3),
        round(world.resource_scarcity, 3),
        round(world.geopolitical_risk, 3),
        round(world.supply_chain_stability, 3),
        world.economic_regime(),
        world.sector_health,
        (
            f"Competitive: {probs['competitive']:.1%} | "
            f"Cooperative: {probs['cooperative']:.1%} | "
            f"Resource: {probs['resource']:.1%}"
        ),
        render_causal_log(env),
    )


def bar(label: str, value: float, width: int = 12, suffix: str = "") -> str:
    clipped = max(0.0, min(1.0, float(value)))
    filled = int(round(clipped * width))
    return f"{label:<16} {'█' * filled}{'░' * (width - filled)} {value:.2f}{suffix}"


def render_world_gauges(env: ACEWorldEnv) -> str:
    world = env.world
    return "\n".join(
        [
            bar("Oil Price", min(world.oil_price / 2.0, 1.0), suffix=f" ({world.oil_price:.2f}x)"),
            bar("Volatility", world.market_volatility),
            bar("Cooperation", world.cooperation_index),
            bar("Scarcity", world.resource_scarcity),
            bar("Liquidity", world.liquidity_index),
            f"Regime: {world.economic_regime().upper()}",
        ]
    )


def render_probability_bars(env: ACEWorldEnv) -> str:
    probs = env.world.derive_round_probabilities()
    return "\n".join(
        [
            bar("Competitive", probs["competitive"], suffix=f" ({probs['competitive']:.0%})"),
            bar("Cooperative", probs["cooperative"], suffix=f" ({probs['cooperative']:.0%})"),
            bar("Resource", probs["resource"], suffix=f" ({probs['resource']:.0%})"),
        ]
    )


def render_flow_strip(env: ACEWorldEnv) -> str:
    event = escape(env.world.event_log[-1] if env.world.event_log else "Waiting for event")
    probs = env.world.derive_round_probabilities()
    likely = max(probs, key=probs.get)
    rounds = env.round_number
    return f"""
<div class="flow-strip">
  <div class="flow-step">
    <b>🌍 Event</b>
    <span>{event}</span>
  </div>
  <div class="flow-step">
    <b>📊 Economy</b>
    <span>{escape(env.world.economic_regime().title())} | Vol {env.world.market_volatility:.2f}</span>
  </div>
  <div class="flow-step">
    <b>🎯 Incentives</b>
    <span>{escape(likely.title())} ({probs[likely]:.0%})</span>
  </div>
  <div class="flow-step">
    <b>🤖 Agents</b>
    <span>Round {rounds}</span>
  </div>
</div>
"""


def render_economic_flow(env: ACEWorldEnv) -> str:
    if not env.world.causal_log:
        return "Event\n  ↓\nNo shock injected yet\n  ↓\nRound probabilities remain near baseline"
    item = env.world.causal_log[-1]
    deltas = item["deltas"]
    changed = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    first_line = " + ".join(f"{key} {'↑' if value > 0 else '↓'}" for key, value in changed) or "limited direct impact"
    probs = env.world.derive_round_probabilities()
    likely = max(probs, key=probs.get)
    return "\n".join(
        [
            f"Event: {item['event']}",
            "  ↓",
            first_line,
            "  ↓",
            item.get("reasoning", "Economic pressure changes incentives."),
            "  ↓",
            f"{likely.upper()} rounds now most likely ({probs[likely]:.0%})",
        ]
    )


def render_causal_log(env: ACEWorldEnv) -> str:
    if not env.world.causal_log:
        return "No event injected yet."
    lines = []
    for item in env.world.causal_log[-3:]:
        deltas = ", ".join(
            f"{key}: {value:+.2f}"
            for key, value in item["deltas"].items()
            if abs(float(value)) > 1e-9
        )
        sectors = ", ".join(item.get("affected_sectors", [])) or "none"
        lines.append(
            f"Event: {item['event']}\n"
            f"Type: {item.get('event_type', 'unknown')}\n"
            f"Affected sectors: {sectors}\n"
            f"Deltas: {deltas or 'none'}\n"
            f"Reasoning: {item['reasoning']}\n"
            f"Confidence: {item['confidence']:.2f}"
        )
    return "\n\n".join(lines)


def render_agent_rows(env: ACEWorldEnv, round_result: dict[str, Any] | None = None) -> list[list[Any]]:
    result_by_id = {}
    if round_result:
        result_by_id = {item["agent"].agent_id: item for item in round_result["results"]}

    rows = []
    for agent in env.agents:
        item = result_by_id.get(agent.agent_id)
        if item:
            action = item["action"]
            reward = item["reward"]
            rows.append(
                [
                    agent.name,
                    agent.company_type,
                    round(agent.resources, 1),
                    action["predicted_round"],
                    action["action"],
                    json.dumps(action.get("parameters", {})),
                    _belief_text(item.get("beliefs", {})),
                    "correct" if item["correct"] else "wrong",
                    round(reward["total"], 3),
                    round(reward["inference"], 2),
                    round(reward["action"], 2),
                    round(reward.get("market_return", 0.0), 3),
                    action.get("reasoning", ""),
                ]
            )
        else:
            rows.append(
                [
                    agent.name,
                    agent.company_type,
                    round(agent.resources, 1),
                    "-",
                    "-",
                    "-",
                    _belief_text(agent.beliefs),
                    "-",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    agent.memory_summary(),
                ]
            )
    return rows


def render_agent_cards(env: ACEWorldEnv, round_result: dict[str, Any] | None = None) -> str:
    result_by_id = {}
    if round_result:
        result_by_id = {item["agent"].agent_id: item for item in round_result["results"]}
    cards = []
    previous_resources = _previous_resources(env)
    for agent in env.agents:
        item = result_by_id.get(agent.agent_id)
        resources = round(agent.resources, 1)
        delta = resources - previous_resources.get(agent.name, resources)
        if item:
            action = item["action"]
            reward = item["reward"]
            beliefs = item.get("beliefs", agent.beliefs)
            factors = item.get("factors", action.get("factors", {}))
            action_line = f"{action['action'].upper()} {json.dumps(action.get('parameters', {}))}"
            correct = "Correct" if item["correct"] else "Wrong"
            reasoning = action.get("reasoning", "")
            reward_line = f"{reward['total']:+.2f}"
        else:
            beliefs = agent.beliefs
            factors = {"past_success": "No rounds yet", "trust_target": "-"}
            action_line = "Waiting"
            correct = "-"
            reasoning = agent.primary_objective
            reward_line = "+0.00"
        q_line = _best_q_line(agent)
        opponent_line = _opponent_model_line(agent)
        mode = "explore" if factors.get("exploration") else "exploit"
        target_trust = factors.get("trust_target", "-")
        cards.append(f"""
<div class="agent-card">
  <div class="agent-card-header">
    <div>
      <h3>{escape(agent.emoji)} {escape(agent.name)}</h3>
      <small>{escape(agent.company_type)}</small>
    </div>
    <div class="agent-resources">{resources:.1f} ({delta:+.1f})</div>
  </div>
  <div style="margin-top:10px;">
    <b>Beliefs</b>
    <pre class="belief-pre">{escape(_belief_bars(beliefs))}</pre>
  </div>
  <div style="margin-top:10px;"><b>Action:</b> {escape(action_line)}</div>
  <div style="margin-top:6px;"><b>Reward:</b> {escape(reward_line)}</div>
  <div style="margin-top:6px;"><b>Learning:</b> {escape(q_line)} | mode={escape(str(mode))} | target trust={escape(str(target_trust))}</div>
  <div style="margin-top:6px;"><b>Opponent model:</b> {escape(opponent_line)}</div>
  <div class="agent-muted" style="margin-top:6px; font-size:13px;">{escape(reasoning)}</div>
</div>
""")
    return "\n".join(cards)


def _belief_text(beliefs: dict[str, float]) -> str:
    if not beliefs:
        return "-"
    return " | ".join(f"{key[:4]} {value:.0%}" for key, value in beliefs.items())


def _best_q_line(agent: AgentProfile) -> str:
    best: tuple[float, str, str] | None = None
    for action, regimes in agent.q_values.items():
        if not isinstance(regimes, dict):
            continue
        for regime, value in regimes.items():
            candidate = (float(value), action, regime)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or abs(best[0]) < 1e-9:
        return "no learned preference yet"
    return f"best Q={best[1]} in {best[2]} ({best[0]:+.2f})"


def _opponent_model_line(agent: AgentProfile) -> str:
    if not agent.opponent_memory:
        return "no opponent observations yet"
    riskiest_id, model = max(
        agent.opponent_memory.items(),
        key=lambda item: item[1].get("betrayal_rate", 0.0) + item[1].get("aggression", 0.0),
    )
    return (
        f"Agent {riskiest_id}: aggression={model.get('aggression', 0.0):.2f}, "
        f"cooperation={model.get('cooperation', 0.0):.2f}, "
        f"betrayal={model.get('betrayal_rate', 0.0):.2f}"
    )


def _belief_bars(beliefs: dict[str, float]) -> str:
    return "\n".join(bar(key.title(), value, width=10, suffix=f" ({value:.0%})") for key, value in beliefs.items())


def _previous_resources(env: ACEWorldEnv) -> dict[str, float]:
    if not env.round_history:
        return {agent.name: agent.resources for agent in env.agents}
    # Approximate previous resources from current minus latest displayed reward.
    previous = {agent.name: agent.resources for agent in env.agents}
    for item in env.round_history[-1].get("results", []):
        previous[item["name"]] = previous.get(item["name"], 100.0) - float(item.get("reward", 0.0)) * 8.0
    return previous


def render_history(env: ACEWorldEnv) -> str:
    if not env.round_history:
        return "No rounds played yet."
    lines = []
    for entry in reversed(env.round_history[-10:]):
        correct = [item["name"] for item in entry["results"] if item["correct"]]
        lines.append(
            f"Round {entry['round']} | Event: {entry['event']} | Actual: {entry['ground_truth'].upper()} | "
            f"Correct: {', '.join(correct) or 'none'}"
        )
    return "\n".join(lines)


def render_interactions(env: ACEWorldEnv) -> str:
    if not env.interaction_log:
        return "No direct interactions yet. Run a round to see alliances, challenges, and betrayals."
    return "\n".join(f"- {item}" for item in env.interaction_log[-10:])


def render_behavior_evolution(env: ACEWorldEnv) -> str:
    lines = []
    for agent in env.agents:
        actions = [item["action"] for item in agent.self_memory[-5:]]
        if not actions:
            trajectory = "No rounds yet"
        else:
            trajectory = " → ".join(_behavior_label(action) for action in actions)
        latest = agent.self_memory[-1]["reward"] if agent.self_memory else 0.0
        lines.append(f"{agent.name}: {trajectory} (latest reward {latest:+.2f})")
    return "\n".join(lines)


def _behavior_label(action: str) -> str:
    if action in {"challenge", "betray", "submit_bid"}:
        return "Aggressive"
    if action in {"propose_alliance", "accept_alliance", "execute_contract"}:
        return "Cooperative"
    if action == "allocate_resources":
        return "Defensive"
    return "Neutral"


def render_optimal_comparison(round_result: dict[str, Any] | None) -> str:
    if not round_result:
        return "Run a round to compare actions against the hidden optimal behavior."
    ground_truth = round_result["ground_truth"]
    optimal = {
        "competitive": {"challenge", "betray", "submit_bid"},
        "cooperative": {"propose_alliance", "accept_alliance", "execute_contract", "submit_bid"},
        "resource": {"allocate_resources", "execute_contract"},
    }[ground_truth]
    lines = [f"Actual round: {ground_truth.upper()}", f"Optimal actions: {', '.join(sorted(optimal))}"]
    for item in round_result["results"]:
        action = item["action"]["action"]
        mark = "OK" if action in optimal else "MISS"
        lines.append(f"- {item['agent'].name}: {action} {mark}")
    return "\n".join(lines)


def resource_plot(env: ACEWorldEnv):
    if go is None:
        return None
    fig = go.Figure()
    for agent in env.agents:
        xs = [0]
        ys = [100.0]
        running = 100.0
        for memory in agent.self_memory:
            running += float(memory.get("reward", 0.0)) * 8.0
            xs.append(memory["round"])
            ys.append(running)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=agent.name))
    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0"),
    height=300,
    margin=dict(l=20, r=20, t=30, b=20),
)
    return fig


def world_plot(env: ACEWorldEnv):
    if go is None:
        return None
    history = env.world_history or [
        {
            "round": 0,
            "oil_price": env.world.oil_price,
            "market_volatility": env.world.market_volatility,
            "cooperation_index": env.world.cooperation_index,
        }
    ]
    xs = [item["round"] for item in history]
    oil = [item["oil_price"] for item in history]
    volatility = [item["market_volatility"] for item in history]
    cooperation = [item["cooperation_index"] for item in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=oil, mode="lines+markers", name="Oil"))
    fig.add_trace(go.Scatter(x=xs, y=volatility, mode="lines+markers", name="Volatility"))
    fig.add_trace(go.Scatter(x=xs, y=cooperation, mode="lines+markers", name="Cooperation"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), title="World Dynamics")
    return fig


def render_trust(env: ACEWorldEnv) -> dict[str, float]:
    trust = {}
    for agent in env.agents:
        for other_id, value in agent.trust_scores.items():
            trust[f"{agent.agent_id}->{other_id}"] = round(value, 2)
    return trust


def inject_event(event_text: str, provider: str, debug_llm: bool, env: ACEWorldEnv | None):
    env = env or make_fresh_env()
    text = event_text.strip() or "No event provided."
    trace = env.apply_event(text, provider=normalize_provider(provider), debug=bool(debug_llm))
    impact = describe_impact(trace["deltas"], text, trace["reasoning"])
    return (
        env,
        impact,
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env),
        render_agent_cards(env),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(None),
        resource_plot(env),
        world_plot(env),
        render_history(env),
        "Inject an event, then run a round.",
    )


def select_preset_event(preset_event: str) -> str:
    return preset_event or PERFECT_DEMO_EVENT


def run_round(env: ACEWorldEnv | None, provider: str, debug_llm: bool):
    env = env or make_fresh_env()
    provider = normalize_provider(provider)

    if provider != "fallback":
        actions = [llm_or_fallback_decision(env, agent, provider, bool(debug_llm)) for agent in env.agents]
        result = env.step(actions)
    else:
        result = env.step()

    ground_truth = result["ground_truth"]
    correct = [item["agent"].name for item in result["results"] if item["correct"]]
    god_mode = (
        f"Actual hidden round: {ground_truth.upper()}\n"
        f"Correct agents: {', '.join(correct) or 'none'}\n"
        f"Alliances: {sorted([list(pair) for pair in env.alliances])}\n"
        f"Regime: {env.world.economic_regime()}"
    )
    return (
        env,
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env, result),
        render_agent_cards(env, result),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(result),
        resource_plot(env),
        world_plot(env),
        render_history(env),
        god_mode,
    )


def run_five_rounds(env: ACEWorldEnv | None, provider: str, debug_llm: bool):
    env = env or make_fresh_env()
    provider = normalize_provider(provider)
    last = None
    for _ in range(5):
        if provider != "fallback":
            actions = [llm_or_fallback_decision(env, agent, provider, bool(debug_llm)) for agent in env.agents]
            last = env.step(actions)
        else:
            last = env.step()
    ground_truth = last["ground_truth"] if last else "none"
    god_mode = f"Ran 5 rounds. Last hidden round: {ground_truth.upper()}."
    return (
        env,
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env, last),
        render_agent_cards(env, last),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(last),
        resource_plot(env),
        world_plot(env),
        render_history(env),
        god_mode,
    )


def run_full_demo(provider: str, debug_llm: bool, env: ACEWorldEnv | None):
    env = make_fresh_env()
    injected = inject_event(PERFECT_DEMO_EVENT, provider, debug_llm, env)
    env = injected[0]
    impact = injected[1]
    round_result = run_round(env, provider, debug_llm)
    env = round_result[0]
    final_result = run_five_rounds(env, provider, debug_llm)
    return (
        final_result[0],
        impact,
        *final_result[1:],
    )


def reset_demo():
    env = make_fresh_env()
    return (
        env,
        "World reset.",
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env),
        render_agent_cards(env),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(None),
        resource_plot(env),
        world_plot(env),
        render_history(env),
        "God Mode ready.",
    )


def build_ui():
    if gr is None:
        raise ModuleNotFoundError("Install demo dependencies with: pip install -r requirements_demo.txt")

    world_outputs = []
    with gr.Blocks(title="ACE++ Option B") as demo:
        env_state = gr.State(make_fresh_env())

        gr.HTML(
            """
<div class="hero">
  <h1>ACE++ Adaptive Coalition Economy</h1>
  <p>Type an event, watch the economy shift, then see agents form beliefs, interact, betray, adapt, and get scored against the hidden optimal behavior.</p>
</div>
"""
        )
        gr.HTML('<div class="demo-hint">Winning demo path: try <b>oil crisis</b> → <b>Inject Event</b> → <b>Run Round</b> → <b>Run 5 Rounds</b>. Or press <b>Run Full Demo</b>.</div>')
        flow_strip = gr.HTML(render_flow_strip(make_fresh_env()))

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">🌍 Scenario Setup</div>')
                preset_events = gr.Dropdown(
                    choices=PRESET_EVENTS,
                    value=PERFECT_DEMO_EVENT,
                    label="Quick Scenarios",
                )
                event_input = gr.Textbox(
                    label="World Event",
                    value=PERFECT_DEMO_EVENT,
                    lines=2,
                    placeholder="Example: OPEC cuts production by 20%",
                )
                with gr.Row():
                    use_llm = gr.Dropdown(
                        choices=["fallback", "groq", "anthropic"],
                        value=os.getenv("LLM_PROVIDER", "fallback"),
                        label="LLM Provider",
                        scale=2,
                    )
                    debug_llm = gr.Checkbox(
                        label="Print raw model responses",
                        value=False,
                        scale=1,
                    )
                gr.HTML('<div class="small-note">Enable raw responses when you want to inspect exactly what the model returned in the terminal.</div>')
                with gr.Row():
                    auto_demo_btn = gr.Button("🚀 Run Full Demo", variant="primary")
                    inject_btn = gr.Button("Inject Event", variant="primary")
                    round_btn = gr.Button("Run Round", variant="secondary")
                    burst_btn = gr.Button("Run 5 Rounds")
                    reset_btn = gr.Button("Reset", variant="stop")
                impact_box = gr.Textbox(label="Immediate Economic Impact", lines=7, interactive=False)

            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">📊 Market Reaction</div>')
                world_gauges = gr.Textbox(label="World Gauges", lines=7, interactive=False)
                round_prob_bars = gr.Textbox(label="Hidden Round Pressure", lines=4, interactive=False)

        gr.HTML('<div class="section-title">🔁 Causal Chain</div>')
        economic_flow = gr.Textbox(label="Event → Economy → Incentives", lines=7, interactive=False)

        gr.HTML('<div class="section-title">🧠 Agents Think and Act</div>')
        agent_cards = gr.HTML()

        gr.HTML('<div class="section-title">⚔️ Strategic Interactions</div>')
        with gr.Row():
            interactions = gr.Textbox(label="Live Interaction Story", lines=8, interactive=False)
            behavior = gr.Textbox(label="Behavior Evolution", lines=8, interactive=False)
            optimal = gr.Textbox(label="Actions vs Optimal", lines=8, interactive=False)

        with gr.Row():
            resource_chart = gr.Plot(label="Agent Resources Over Time")
            world_chart = gr.Plot(label="World Dynamics")

        gr.HTML('<div class="why-matters">Why this matters: this simulation shows how economic shocks reshape incentives and drive strategic behavior among agents.</div>')

        with gr.Accordion("Details: raw state, trust matrix, history, and rewards table", open=False):
            causal_box = gr.Textbox(label="Causal Trace", lines=8, interactive=False)
            with gr.Row():
                trust_json = gr.JSON(label="Trust Matrix")
                god_mode = gr.Textbox(label="God Mode Reveal", lines=5, interactive=False)
                history = gr.Textbox(label="Round History", lines=10, interactive=False)
            with gr.Row():
                oil = gr.Number(label="Oil x", precision=3)
                gold = gr.Number(label="Gold x", precision=3)
                food = gr.Number(label="Food x", precision=3)
                energy = gr.Number(label="Energy x", precision=3)
            with gr.Row():
                interest = gr.Number(label="Interest", precision=4)
                inflation = gr.Number(label="Inflation", precision=4)
                gdp = gr.Number(label="GDP Growth", precision=4)
                regime = gr.Textbox(label="Regime", interactive=False)
            with gr.Row():
                liquidity = gr.Slider(0, 1, label="Liquidity", interactive=False)
                credit = gr.Slider(0, 1, label="Credit Spread", interactive=False)
                tension = gr.Slider(0, 1, label="Trade Tension", interactive=False)
            with gr.Row():
                volatility = gr.Slider(0, 1, label="Volatility", interactive=False)
                cooperation = gr.Slider(0, 1, label="Cooperation", interactive=False)
                scarcity = gr.Slider(0, 1, label="Scarcity", interactive=False)
            with gr.Row():
                geopolitics = gr.Slider(0, 1, label="Geopolitical Risk", interactive=False)
                supply_chain = gr.Slider(0, 1, label="Supply Chain Stability", interactive=False)
            sectors = gr.JSON(label="Sector Health")
            probabilities = gr.Textbox(label="Round Probabilities", interactive=False)
            agents_table = gr.Dataframe(
                headers=[
                    "Agent",
                    "Type",
                    "Resources",
                    "Predicted",
                    "Action",
                    "Params",
                    "Beliefs",
                    "Correct",
                    "Total Reward",
                    "Inference",
                    "Action Reward",
                    "Market Return",
                    "Reasoning / Memory",
                ],
                row_count=(7, "dynamic"),
                column_count=(13, "fixed"),
                label="Detailed reward table",
            )

        world_outputs = [
            oil,
            gold,
            food,
            energy,
            interest,
            inflation,
            gdp,
            liquidity,
            credit,
            tension,
            volatility,
            cooperation,
            scarcity,
            geopolitics,
            supply_chain,
            regime,
            sectors,
            probabilities,
            causal_box,
        ]
        story_outputs = [flow_strip, world_gauges, economic_flow, round_prob_bars]
        sim_outputs = [
            agents_table,
            agent_cards,
            trust_json,
            interactions,
            behavior,
            optimal,
            resource_chart,
            world_chart,
            history,
            god_mode,
        ]

        preset_events.change(
            select_preset_event,
            inputs=[preset_events],
            outputs=[event_input],
        )
        auto_demo_btn.click(
            run_full_demo,
            inputs=[use_llm, debug_llm, env_state],
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
        inject_btn.click(
            inject_event,
            inputs=[event_input, use_llm, debug_llm, env_state],
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
        round_btn.click(
            run_round,
            inputs=[env_state, use_llm, debug_llm],
            outputs=[env_state, *world_outputs, *story_outputs, *sim_outputs],
        )
        burst_btn.click(
            run_five_rounds,
            inputs=[env_state, use_llm, debug_llm],
            outputs=[env_state, *world_outputs, *story_outputs, *sim_outputs],
        )
        reset_btn.click(
            reset_demo,
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
        demo.load(
            reset_demo,
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True, css=APP_CSS)
