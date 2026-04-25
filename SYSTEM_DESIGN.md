# ACE++ System Design

## 1. System Overview

ACE++ is a partially observable decision-making environment for LLM agents. The agent does not directly observe the true world state. Instead, it sees market-like signals, must infer a hidden round type, choose a structured JSON action, and then learn from reward plus revealed ground truth.

The system is designed to answer a specific question:

How do language-model agents behave when they must reason under hidden state, act with structured tools, and adapt using feedback?

This matters because many real-world agent problems are not static question-answering tasks. They require:
- incomplete information
- structured action selection
- feedback loops
- repeated correction over time

ACE++ turns that into a compact, judge-readable benchmark and demo.

## 2. Core Components

### Environment (`ACEEnv`)

The environment contains a hidden round type:
- `cooperative`
- `competitive`
- `resource`

Each round type generates correlated but non-explicit market signals. The agent sees the signals, not the hidden label.

The environment is partially observable because:
- the hidden state is sampled internally
- the agent only sees `market_state`
- the true round type is revealed after the action is scored

### Agent

The agent produces a structured JSON action with two conceptual parts:
- belief: predicted hidden state
- action: tool choice plus parameters

The single-agent demo path supports:
- rule-based inference (`demo.py`)
- adaptive LLM inference (`llm_agent.py`)
- fallback deterministic mock learning (`llm_agent.py`)

### Reward System

Reward combines:
- task reward: was the chosen tool/amount appropriate?
- inference reward: did the predicted hidden state match the truth?
- penalties: parsing mistakes, invalid fields, anti-collusion heuristics

This makes the task not just “guess the label,” but “infer the label and act correctly.”

## 3. Action Space

The recommended action schema is:

```json
{
  "belief": {
    "predicted_round": "competitive",
    "confidence": 0.8
  },
  "action": {
    "tool": "submit_bid",
    "parameters": {
      "amount": 75
    }
  }
}
```

Main fields:
- `predicted_round`: the agent’s inferred hidden state
- `confidence`: scalar belief strength in `[0, 1]`
- `tool`: one of `submit_bid`, `allocate_resources`, `execute_contract`
- `parameters.amount`: numeric control value for bid/allocation actions

Backward-compatible legacy forms are still accepted by validation logic, but the schema above is the intended interface.

## 4. Observation Space

Each observation contains:
- `round`
- `market_state`
- `history`

Example:

```json
{
  "round": 0,
  "market_state": {
    "demand_index": 0.91,
    "volatility": 0.82,
    "competition_signal": "high",
    "cooperation_signal": "low"
  },
  "history": []
}
```

`market_state` is the key partial-observability channel. It is correlated with the hidden state but does not explicitly reveal it.

`history` contains recent feedback so the agent can adapt over time.

## 5. Environment Flow

Core loop:

```text
Agent
  ↓
Build JSON action
  ↓
Environment validates action
  ↓
Environment scores inference + task reward
  ↓
Environment reveals actual hidden state
  ↓
Environment emits next observation
  ↓
Agent updates belief on next round
```

Step-by-step:
1. `reset()` samples the hidden round type and payoff structure.
2. The agent receives `market_state`.
3. The agent predicts the hidden round type and chooses a tool action.
4. `step()` validates the JSON.
5. The environment computes:
   - inference reward
   - task reward
   - penalties if needed
6. The actual round type is revealed in debug/info outputs.
7. A new round is sampled and the next observation is returned.

## 6. Training Flow

ACE++ has two training/evidence paths.

### A. Simulated / staged learnability

Files:
- `train_sim.py`
- `train_openenv.py`

Flow:

```text
episode
  ↓
policy phase (random → mixed → grounded)
  ↓
env.step(...)
  ↓
reward + accuracy logs
  ↓
plots
```

This demonstrates that the environment is learnable and that better policies produce better reward and inference accuracy.

### B. RL training path

File:
- `ACE_OpenEnv_GRPO_Training.ipynb`

Flow:

```text
prompt
  ↓
model completion
  ↓
parse JSON action
  ↓
run ACE environment step
  ↓
compute scalar reward
  ↓
GRPO update
```

This is the real RL bridge for TRL + Unsloth training.

## 7. LLM Adaptation Logic

The adaptive agent in `llm_agent.py` is not doing gradient-based learning during a live episode.

Instead, it adapts through prompt state:
- current `market_state`
- recent history of predictions
- actual hidden state from prior rounds
- correctness and reward feedback

Prompt construction:

```text
current observation
+ recent prediction/actual history
+ instruction to output JSON
```

Feedback loop:
1. The model predicts.
2. The environment reveals truth.
3. That truth is stored in prompt history.
4. The next prompt contains the previous mistake/correction signal.
5. The model can revise its next prediction.

This is in-context belief updating rather than parameter updating.

## 8. Demo Flow

### Run Demo

UI tab: `Live Agent Demo`

Flow:
- `app.py` runs `llm_agent.py`
- output is captured
- round-by-round logs are shown
- summary metrics are extracted
- `llm_episode_curve.png` is shown inline

What judges see:
- wrong predictions
- corrected predictions
- prediction vs actual
- improvement summary

### Run Training

UI tab: `OpenEnv Training`

Flow:
- `app.py` runs `train_openenv.py`
- logs are captured
- `openenv_training_curves.png` is shown inline

What judges see:
- training completed
- learning curves generated

### Compare Agents

UI tab: `Agent Comparison`

Flow:
- `app.py` runs `compare_agents.py`
- raw output is shown
- summary metrics are parsed into a markdown table

What judges see:
- random baseline
- rule-based heuristic
- adaptive LLM start/end/improvement

## 9. System Architecture Diagram

```text
Gradio UI (app.py)
   ↓
Script Runner
   ↓
LLM Demo / Training / Comparison Scripts
   ↓
ACE Environment (ACEEnv / ACEOpenEnv)
   ↓
Reward + Validation + Feedback
   ↓
Logs / JSON Artifacts / PNG Plots
   ↓
UI Summaries + Inline Visuals
```

Expanded view:

```text
UI (Gradio)
   ↓
app.py
   ↓
--------------------------------
| llm_agent.py                |
| train_openenv.py           |
| compare_agents.py          |
--------------------------------
   ↓
ACEEnv / ACEOpenEnv
   ↓
Validation + Reward Logic
   ↓
history / metrics / plots
```

## 10. Learning Mechanism

We demonstrate both:
- environment learnability via RL training
- in-context adaptation via LLM interaction

These are related but distinct:
- RL learnability asks whether reward supports policy improvement over time.
- In-context adaptation asks whether an agent can revise behavior within the same episode using feedback.

ACE++ intentionally supports both views.

## 11. Key Insight

ACE++ is different from:

### Static tasks

Static tasks do not require sequential correction. ACE++ does.

### Simple RL environments

Many simple RL environments expose a direct mapping from state to action. ACE++ adds:
- hidden state
- structured tool output
- parsing and validation
- belief reporting
- interpretable feedback

This makes the environment closer to realistic agent systems where the model must both infer and act.

## 12. Limitations

The current system is intentionally simplified.

Main limitations:
- single-agent path is the primary polished demo
- multi-agent logic exists but is not the main submission path
- the environment is synthetic, not tied to real-world external data
- demo behavior is deterministic in the Space for stability
- RL notebook exists, but full training credibility still depends on executing it on a real GPU runtime

These are acceptable tradeoffs for a hackathon submission focused on clarity and reproducibility.

## 13. Future Extensions

Logical next steps:
- integrate live economic/news/API signals into the observation stream
- scale the multi-agent coalition environment into a full self-play setting
- add enterprise-style tool workflows with contracts, approvals, and multi-step execution
- strengthen anti-collusion and schema validation
- add richer dashboards for belief trajectories, reward decomposition, and trust graphs

## Bottom Line

ACE++ is a compact system for studying hidden-state reasoning, structured action generation, reward-driven behavior, and visible adaptation. It is designed so a judge can understand both the environment and the learning story quickly, without needing to inspect deep infrastructure.
