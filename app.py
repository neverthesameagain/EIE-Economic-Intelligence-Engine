"""
Hugging Face Space app for ACE++.

This app adds an interactive economic simulation layer on top of the existing
ACE++ repository without modifying the RL/training backend.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None

from env_config import load_local_env, llm_status_label
from simulation import SimulationManager


load_local_env()


ROOT = Path(__file__).resolve().parent
LLM_CURVE_PATH = ROOT / "llm_episode_curve.png"
OPENENV_CURVE_PATH = ROOT / "openenv_training_curves.png"
OPENENV_MULTI_CURVE_PATH = ROOT / "openenv_multiagent_training_curves.png"


def _run_script(script_name: str, timeout: int = 90) -> str:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    try:
        result = subprocess.run(
            ["python3", script_name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"{script_name} timed out after {timeout} seconds."
    except Exception as exc:
        return f"Failed to run {script_name}: {exc}"

    output = result.stdout.strip()
    if result.stderr.strip():
        output = f"{output}\n\n[stderr]\n{result.stderr.strip()}".strip()
    if not output:
        output = f"{script_name} finished with exit code {result.returncode}."
    return output


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _build_llm_summary(output: str) -> str:
    start = _extract_float(r"Start accuracy:\s*([+-]?\d+(?:\.\d+)?)", output)
    end = _extract_float(r"End accuracy:\s*([+-]?\d+(?:\.\d+)?)", output)
    improvement = _extract_float(r"Improvement:\s*([+-]?\d+(?:\.\d+)?)", output)
    events = _extract_int(r"Adaptation events:\s*(\d+)", output)
    invalid = _extract_int(r"Invalid outputs:\s*(\d+)", output)

    lines = ["### Summary"]
    if start is not None:
        lines.append(f"Start Accuracy: {start:.2f}  ")
    if end is not None:
        lines.append(f"End Accuracy: {end:.2f}  ")
    if improvement is not None:
        lines.append(f"Improvement: {improvement:+.2f}  ")
    if events is not None:
        lines.append(f"Adaptation Events: {events}  ")
    if invalid is not None:
        lines.append(f"Invalid Outputs: {invalid}  ")
    return "\n".join(lines)


def _build_training_summary(output: str) -> str:
    lines = [
        "### Training Summary",
        "Training completed. Learning curves shown below.",
    ]
    if "Saved: openenv_training_logs.json" in output:
        lines.append("- Logs saved to `openenv_training_logs.json`")
    if "Saved: openenv_training_curves.png" in output:
        lines.append("- Plot saved to `openenv_training_curves.png`")
    if "Saved: openenv_multiagent_training_logs.json" in output:
        lines.append("- Multi-agent logs saved to `openenv_multiagent_training_logs.json`")
    if "Saved: openenv_multiagent_training_curves.png" in output:
        lines.append("- Multi-agent plot saved to `openenv_multiagent_training_curves.png`")
    return "\n".join(lines)


def _build_comparison_summary(output: str) -> str:
    random_acc = _extract_float(r"Random\s+([0-9.]+)\s+", output)
    rule_acc = _extract_float(r"Rule-based \(signal heuristic\)\s+([0-9.]+)\s+", output)
    llm_acc = _extract_float(r"LLM-style \(adaptive\)\s+([0-9.]+)\s+", output)
    llm_start = _extract_float(r"Start Accuracy:\s*([+-]?\d+(?:\.\d+)?)", output)
    llm_end = _extract_float(r"End Accuracy:\s*([+-]?\d+(?:\.\d+)?)", output)
    llm_improvement = _extract_float(r"Improvement:\s*([+-]?\d+(?:\.\d+)?)", output)

    rows = ["| Agent | Metric |", "|---|---|"]
    if random_acc is not None:
        rows.append(f"| Random | {random_acc:.2f} accuracy |")
    if rule_acc is not None:
        rows.append(f"| Rule-based | {rule_acc:.2f} accuracy |")
    if llm_acc is not None:
        rows.append(f"| LLM Agent | {llm_acc:.2f} mean accuracy |")
    if llm_start is not None:
        rows.append(f"| LLM Start | {llm_start:.2f} |")
    if llm_end is not None:
        rows.append(f"| LLM End | {llm_end:.2f} |")
    if llm_improvement is not None:
        rows.append(f"| LLM Improvement | {llm_improvement:+.2f} |")
    return "\n".join(rows)


def run_llm_demo() -> tuple[str, str, str | None]:
    output = _run_script("llm_agent.py", timeout=90)
    summary = _build_llm_summary(output)
    curve_path = str(LLM_CURVE_PATH) if LLM_CURVE_PATH.exists() else None
    return output, summary, curve_path


def run_openenv_training() -> tuple[str, str, str | None, str | None]:
    output = _run_script("train_openenv.py", timeout=90)
    summary = _build_training_summary(output)
    curve_path = str(OPENENV_CURVE_PATH) if OPENENV_CURVE_PATH.exists() else None
    multi_curve_path = str(OPENENV_MULTI_CURVE_PATH) if OPENENV_MULTI_CURVE_PATH.exists() else None
    return output, summary, curve_path, multi_curve_path


def run_compare_agents() -> tuple[str, str]:
    output = _run_script("compare_agents.py", timeout=90)
    summary = _build_comparison_summary(output)
    return output, summary


def _render_agent_table(sim: SimulationManager) -> list[list[object]]:
    rows = []
    for agent in sim.snapshot()["agents"]:
        rows.append([
            agent["id"],
            agent["state"],
            agent["strategy"],
            agent["capital"],
            agent["stake"],
            agent["trust"],
            agent["predicted_round"],
            agent["confidence"],
            agent["alliance"],
            agent["last_action"],
            agent["last_reward"],
            agent["reasoning"],
        ])
    return rows


def _render_environment_panel(sim: SimulationManager) -> dict:
    state = sim.snapshot()
    env = state["environment"]
    return {
        "demand": round(env["demand"], 2),
        "resources": round(env["resources"], 2),
        "volatility": round(env["volatility"], 2),
        "uncertainty": round(env["uncertainty"], 2),
        "hidden_round_type": env["hidden_round_type"],
        "market_pressure": round(env["market_pressure"], 2),
        "alliance_pressure": round(env["alliance_pressure"], 2),
        "stakes_multiplier": round(env["stakes_multiplier"], 2),
        "narrative": env["narrative"],
        "policy_constraints": env["policy_constraints"],
        "last_event": env["last_event"],
        "last_event_structured": state["last_event_summary"],
        "timestep": env["timestep"],
    }


def _render_history_plot(sim: SimulationManager):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.5))
    agent_series: dict[int, list[tuple[int, float]]] = {}
    for item in sim.history:
        timestep = item["timestep"]
        for agent in item["agents"]:
            agent_series.setdefault(agent["id"], []).append((timestep, agent["capital"]))

    if not agent_series:
        ax.text(0.5, 0.5, "Add agents to start the simulation", ha="center", va="center")
        ax.set_axis_off()
        return fig

    for agent_id, points in agent_series.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"Agent {agent_id}")

    ax.set_title("Capital Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _render_event_summary(sim: SimulationManager) -> str:
    summary = sim.last_event_summary or {}
    if not summary:
        return "### Event Parsing\nNo external event has been applied yet."
    return (
        "### Event Parsing\n"
        f"- Hidden round type: `{summary.get('hidden_round_type', 'resource')}`\n"
        f"- Demand multiplier: {summary.get('demand_multiplier', 1.0):.2f}\n"
        f"- Resource multiplier: {summary.get('resource_multiplier', 1.0):.2f}\n"
        f"- Volatility delta: {summary.get('volatility', 0.0):.2f}\n"
        f"- Uncertainty delta: {summary.get('uncertainty', 0.0):.2f}\n"
        f"- Alliance pressure: {summary.get('alliance_pressure', 0.35):.2f}\n"
        f"- Stakes multiplier: {summary.get('stakes_multiplier', 1.0):.2f}\n"
        f"- Constraints: {', '.join(summary.get('policy_constraints', [])) or 'none'}\n"
        f"- Narrative: {summary.get('narrative', 'No narrative generated.')}\n"
        f"- Confidence: {summary.get('confidence', 0.0):.2f}"
    )


def _agent_choices(sim: SimulationManager):
    choices = sim.agent_choices()
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def _simulation_outputs(sim: SimulationManager):
    choices = sim.agent_choices()
    dropdown_update = (
        gr.update(choices=choices, value=choices[0][1] if choices else None)
        if gr is not None
        else {"choices": choices, "value": choices[0][1] if choices else None}
    )
    return (
        _render_agent_table(sim),
        _render_environment_panel(sim),
        sim.insight,
        _render_event_summary(sim),
        _render_history_plot(sim),
        dropdown_update,
    )


def initialize_simulation() -> SimulationManager:
    sim = SimulationManager()
    sim.add_agent("greedy", 120.0, 0.75, 1.25)
    sim.add_agent("cooperative", 100.0, 0.45, 0.9)
    sim.insight = "Simulation initialized with two representative agents."
    return sim


def apply_event(sim: SimulationManager | None, text: str, use_llm: bool):
    sim = sim or initialize_simulation()
    sim.apply_event(text or "No event provided.", use_llm=use_llm)
    return (sim, *_simulation_outputs(sim))


def add_agent(sim: SimulationManager | None, strategy: str, capital: float, risk_appetite: float, stake: float):
    sim = sim or initialize_simulation()
    sim.add_agent(strategy, capital, risk_appetite, stake)
    sim.insight = f"Added a {strategy} agent with capital {capital:.1f} and stake {stake:.2f}."
    return (sim, *_simulation_outputs(sim))


def remove_agent(sim: SimulationManager | None, agent_id: int | None):
    sim = sim or initialize_simulation()
    if agent_id is not None:
        sim.remove_agent(int(agent_id))
        sim.insight = f"Removed agent {agent_id}."
    return (sim, *_simulation_outputs(sim))


def step_simulation(sim: SimulationManager | None, use_llm: bool):
    sim = sim or initialize_simulation()
    sim.step(use_llm=use_llm)
    return (sim, *_simulation_outputs(sim))


def run_simulation(sim: SimulationManager | None, use_llm: bool):
    sim = sim or initialize_simulation()
    sim.run_steps(5, use_llm=use_llm)
    sim.insight = "Ran a short multi-step burst so you can see adaptation in real time."
    return (sim, *_simulation_outputs(sim))


def pause_simulation(sim: SimulationManager | None):
    sim = sim or initialize_simulation()
    sim.pause()
    return (sim, *_simulation_outputs(sim))


def load_simulation(sim: SimulationManager | None):
    sim = sim or initialize_simulation()
    return (sim, *_simulation_outputs(sim))


if gr is not None:
    with gr.Blocks(title="ACE++ Interactive Demo") as demo:
        sim_state = gr.State(value=initialize_simulation())

        gr.Markdown("# ACE++ Interactive Economic Simulation")
        gr.Markdown(
            "An interactive multi-agent economic demo layered on top of the ACE++ project. "
            "Inject macro events, add or remove agents, and watch beliefs, stakes, trust, coalitions, and capital update live."
        )
        gr.Markdown(f"**{llm_status_label()}**")

        with gr.Tab("Interactive Simulation"):
            with gr.Row():
                with gr.Column(scale=2):
                    event_text = gr.Textbox(
                        label="Describe an Event",
                        placeholder="Examples: global recession hits, oil crisis, AI boom increases productivity",
                    )
                    use_llm = gr.Checkbox(
                        label="Use LLM Reasoning (if configured)",
                        value=bool(
                            (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
                            or os.getenv("HUGGINGFACE_API_TOKEN")
                        ),
                    )
                    apply_btn = gr.Button("Apply Event", variant="primary")

                    gr.Markdown("### Agent Control Panel")
                    strategy = gr.Dropdown(
                        choices=["greedy", "cooperative", "adversarial", "conservative"],
                        value="greedy",
                        label="Strategy",
                    )
                    capital = gr.Number(value=100.0, label="Initial Capital")
                    risk = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Risk Appetite")
                    stake = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="Stake / Market Exposure")
                    add_btn = gr.Button("Add Agent")
                    remove_dropdown = gr.Dropdown(label="Remove Agent", choices=[])
                    remove_btn = gr.Button("Remove Agent")

                    gr.Markdown("### Simulation Control")
                    with gr.Row():
                        step_btn = gr.Button("Step")
                        run_btn = gr.Button("Start / Run")
                        pause_btn = gr.Button("Pause")

                with gr.Column(scale=3):
                    ai_insight = gr.Markdown(label="AI Insight Panel")
                    event_summary = gr.Markdown()
                    environment_panel = gr.JSON(label="God Mode Environment State")
                    agents_table = gr.Dataframe(
                        headers=[
                            "Agent ID",
                            "State",
                            "Strategy",
                            "Capital",
                            "Stake",
                            "Trust",
                            "Belief",
                            "Confidence",
                            "Alliance",
                            "Last Action",
                            "Last Reward",
                            "Reasoning",
                        ],
                        row_count=(2, "dynamic"),
                        col_count=(12, "fixed"),
                        label="Live Agent State",
                    )
                    capital_plot = gr.Plot(label="Capital Over Time")

            apply_btn.click(
                fn=apply_event,
                inputs=[sim_state, event_text, use_llm],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )
            add_btn.click(
                fn=add_agent,
                inputs=[sim_state, strategy, capital, risk, stake],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )
            remove_btn.click(
                fn=remove_agent,
                inputs=[sim_state, remove_dropdown],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )
            step_btn.click(
                fn=step_simulation,
                inputs=[sim_state, use_llm],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )
            run_btn.click(
                fn=run_simulation,
                inputs=[sim_state, use_llm],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )
            pause_btn.click(
                fn=pause_simulation,
                inputs=[sim_state],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )
            demo.load(
                fn=load_simulation,
                inputs=[sim_state],
                outputs=[sim_state, agents_table, environment_panel, ai_insight, event_summary, capital_plot, remove_dropdown],
            )

        with gr.Tab("ACE++ Agent Demo"):
            gr.Markdown("Runs the existing adaptive fallback demo and displays within-episode improvement.")
            live_output = gr.Textbox(label="Terminal Output", lines=24, max_lines=30)
            live_summary = gr.Markdown()
            live_curve = gr.Image(label="Within-Episode Adaptation Curve", type="filepath")
            live_btn = gr.Button("Run LLM Agent Demo", variant="primary")
            live_btn.click(fn=run_llm_demo, outputs=[live_output, live_summary, live_curve])

        with gr.Tab("Training Evidence"):
            gr.Markdown("Runs the existing OpenEnv training evidence script and shows the saved learning curves.")
            train_output = gr.Textbox(label="Training Output", lines=18, max_lines=24)
            train_summary = gr.Markdown()
            train_curve = gr.Image(label="OpenEnv Training Curves", type="filepath")
            train_multi_curve = gr.Image(label="OpenEnv Multi-Agent Curves", type="filepath")
            train_btn = gr.Button("Run Training Evidence", variant="primary")
            train_btn.click(fn=run_openenv_training, outputs=[train_output, train_summary, train_curve, train_multi_curve])

        with gr.Tab("Agent Comparison"):
            gr.Markdown("Runs the existing comparison script to show random, rule-based, and adaptive performance.")
            compare_output = gr.Textbox(label="Comparison Output", lines=14, max_lines=20)
            compare_summary = gr.Markdown()
            compare_btn = gr.Button("Compare Agents", variant="primary")
            compare_btn.click(fn=run_compare_agents, outputs=[compare_output, compare_summary])
else:
    demo = None


if __name__ == "__main__":
    if demo is None:
        raise ModuleNotFoundError("gradio is required to run app.py. Install with: pip install -r requirements.txt")
    demo.launch()
