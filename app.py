"""
Hugging Face Space entrypoint for ACE++.

This app keeps the UI intentionally simple:
- one tab for the live adaptive-agent demo
- one tab for OpenEnv training evidence

Both actions run the existing project scripts so the Space reflects the real
repository behavior rather than a separate toy interface.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None

from env import ACEEnv


ROOT = Path(__file__).resolve().parent
LLM_CURVE_PATH = ROOT / "llm_episode_curve.png"
OPENENV_CURVE_PATH = ROOT / "openenv_training_curves.png"


def _run_script(script_name: str, timeout: int = 90) -> str:
    env = os.environ.copy()
    # Force deterministic fallback behavior so the Space demo is stable.
    env.pop("OPENAI_API_KEY", None)
    env.pop("HUGGINGFACE_API_TOKEN", None)
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_MODEL", None)
    env.pop("HUGGINGFACE_MODEL", None)
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
    return "\n".join(lines)


def _build_comparison_summary(output: str) -> str:
    random_acc = _extract_float(r"Random\s+([0-9.]+)\s+", output)
    rule_acc = _extract_float(r"Rule-based \(signal heuristic\)\s+([0-9.]+)\s+", output)
    llm_acc = _extract_float(r"LLM-style \(adaptive\)\s+([0-9.]+)\s+", output)
    llm_start = _extract_float(r"Start Accuracy:\s*([+-]?\d+(?:\.\d+)?)", output)
    llm_end = _extract_float(r"End Accuracy:\s*([+-]?\d+(?:\.\d+)?)", output)
    llm_improvement = _extract_float(r"Improvement:\s*([+-]?\d+(?:\.\d+)?)", output)

    rows = [
        "| Agent | Metric |",
        "|---|---|",
    ]
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


def _action_for_prediction(predicted: str) -> dict:
    if predicted == "competitive":
        return {"tool": "submit_bid", "parameters": {"amount": 75}}
    if predicted == "cooperative":
        return {"tool": "submit_bid", "parameters": {"amount": 30}}
    return {"tool": "allocate_resources", "parameters": {"amount": 50}}


def _run_god_mode_snapshot() -> tuple[dict, dict, list[list[float | str]], str]:
    env = ACEEnv(
        num_rounds=1,
        seed=7,
        difficulty="easy",
        round_type_schedule=["competitive"],
    )
    obs = env.reset()
    state_before = env.state()
    signal = obs["market_state"]["competition_signal"]

    if signal == "high":
        predicted = "competitive"
        confidence = 0.9
    elif signal == "low":
        predicted = "cooperative"
        confidence = 0.9
    else:
        predicted = "resource"
        confidence = 0.85

    action = _action_for_prediction(predicted)
    action_json = json.dumps(
        {
            "belief": {"predicted_round": predicted, "confidence": confidence},
            "action": action,
        }
    )
    _, reward, _, info = env.step(action_json)
    step_log = info["step_log"]
    state_after = env.state()

    agent_view = {
        "market_state": obs["market_state"],
        "belief_log": {
            "predicted_round": predicted,
            "confidence": confidence,
        },
        "chosen_action": action,
        "pre_step_state": {
            "current_round_type_hidden": state_before["current_round_type"],
            "current_payoff_seed": state_before["current_payoff_seed"],
        },
    }
    ground_truth = {
        "actual_round_type": info["debug_round_type"],
        "correct_inference": info["correct_inference"],
        "payoff_structure": step_log["actual_payoff"],
        "reward_total": round(float(reward), 2),
        "post_step_state": {
            "current_round": state_after["current_round"],
            "inference_accuracy": round(float(state_after["inference_accuracy"]), 2),
        },
    }
    reward_rows = [[
        round(float(step_log["r_task"]), 2),
        round(float(step_log["r_inference"]), 2),
        round(float(step_log["anti_collusion_penalty"]), 2),
        round(float(step_log["parse_penalty"]), 2),
        round(float(step_log["r_total"]), 2),
    ]]
    status = "🟢 Prediction matched hidden state" if info["correct_inference"] else "🔴 Prediction did not match hidden state"
    return agent_view, ground_truth, reward_rows, status


def run_llm_demo() -> tuple[str, str, str | None, dict, dict, list[list[float | str]], str]:
    output = _run_script("llm_agent.py", timeout=90)
    summary = _build_llm_summary(output)
    curve_path = str(LLM_CURVE_PATH) if LLM_CURVE_PATH.exists() else None
    agent_view, ground_truth, reward_rows, status = _run_god_mode_snapshot()
    return output, summary, curve_path, agent_view, ground_truth, reward_rows, status


def run_openenv_training() -> tuple[str, str, str | None]:
    output = _run_script("train_openenv.py", timeout=90)
    summary = _build_training_summary(output)
    curve_path = str(OPENENV_CURVE_PATH) if OPENENV_CURVE_PATH.exists() else None
    return output, summary, curve_path


def run_compare_agents() -> tuple[str, str]:
    output = _run_script("compare_agents.py", timeout=90)
    summary = _build_comparison_summary(output)
    return output, summary


if gr is not None:
    with gr.Blocks(title="ACE++ Demo") as demo:
        gr.Markdown("# ACE++: Adaptive Coalition Economy")
        gr.Markdown(
            "A partially observable economic environment where agents infer hidden states, "
            "act with structured JSON tools, and adapt using feedback."
        )
        gr.Markdown(
            "This Space forces the deterministic fallback agent and ignores external API "
            "credentials so judges always see the same reproducible behavior."
        )

        with gr.Tab("Live Agent Demo"):
            gr.Markdown(
                "Runs the adaptive fallback demo and a God Mode snapshot so judges can "
                "see both the agent view and the hidden ground truth."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Agent View")
                    agent_view = gr.JSON(label="Market State, Belief Log, Chosen Action")
                with gr.Column():
                    gr.Markdown("### Ground Truth (God Mode)")
                    ground_truth = gr.JSON(label="Actual Hidden State and Payoff")
            god_mode_status = gr.Markdown()
            reward_breakdown = gr.Dataframe(
                headers=["R_task", "R_inference", "anti_collusion_penalty", "parse_penalty", "R_total"],
                row_count=1,
                col_count=(5, "fixed"),
                label="Reward Breakdown",
            )
            live_output = gr.Textbox(label="Terminal Output", lines=24, max_lines=30)
            live_summary = gr.Markdown()
            live_curve = gr.Image(label="Within-Episode Adaptation Curve", type="filepath")
            live_btn = gr.Button("Run God Mode Demo", variant="primary")
            live_btn.click(
                fn=run_llm_demo,
                outputs=[
                    live_output,
                    live_summary,
                    live_curve,
                    agent_view,
                    ground_truth,
                    reward_breakdown,
                    god_mode_status,
                ],
            )

        with gr.Tab("OpenEnv Training"):
            gr.Markdown(
                "Runs `train_openenv.py` to generate OpenEnv-style training evidence and "
                "reward/accuracy curves."
            )
            train_output = gr.Textbox(label="Training Output", lines=18, max_lines=24)
            train_summary = gr.Markdown()
            train_curve = gr.Image(label="OpenEnv Training Curves", type="filepath")
            train_btn = gr.Button("Run OpenEnv Training", variant="primary")
            train_btn.click(fn=run_openenv_training, outputs=[train_output, train_summary, train_curve])

        with gr.Tab("Agent Comparison"):
            gr.Markdown(
                "Runs `compare_agents.py` to show random, rule-based, and adaptive-agent "
                "behavior side by side."
            )
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
