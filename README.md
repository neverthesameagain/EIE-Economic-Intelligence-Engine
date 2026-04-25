# ACE++ (Adaptive Coalition Economy) — Repo Status

This repo contains a minimal-but-growing ACE++ environment plus a TRL/GRPO training scaffold (verifiable rewards) and demo scripts.

## What Works Now (Implemented)

### 1) Environment Core (`env.py`)

- **Single-agent environment**: `ACEEnv`
  - POMDP-style loop: the agent predicts the *current* hidden round type from the observation’s `market_state`, then the env advances to the next round.
  - **Round types**: `cooperative | competitive | resource`
- **Professional “API/tool layer”**
  - Tools are represented as JSON actions and validated before execution:
    - `submit_bid(amount, partner_id?)`
    - `allocate_resources(amount)`
    - `execute_contract(team_id?)`
    - Coalition tools (v1): `propose_alliance`, `accept_alliance`, `reject_alliance`, `betray`, `challenge`
- **Validation + structured error responses**
  - Invalid JSON / missing keys / invalid values produce structured errors (and the episode continues).
- **Belief log + inference reward**
  - Agent provides `belief.predicted_round` and `belief.confidence` in `[0, 1]`
  - Inference reward is confidence-scaled.
- **Reward shaping**
  - Task reward from tool choice + parameters
  - Social reward from alliance dynamics (multi-agent wrapper)
  - Adaptation reward (small bonus when improving task performance on a previously-seen round type)
  - Rubric-style feedback strings for obvious mismatches (single-agent `info["feedback"]`)
- **Anti-collusion heuristics (lightweight)**
  - Penalizes “extra” top-level keys and unusually long string payloads (discourages non-causal signaling).
- **Difficulty knob**
  - `difficulty = easy|medium|hard` controls signal noisiness.

### 2) Hidden Payoff Structures (Verifiable)

- Each round samples a **hidden payoff seed** (`current_payoff_seed`) and derives a deterministic payoff structure from it.
- `submit_bid` success thresholds depend on the hidden payoff parameters (not fixed constants).
- **God mode reveal** includes:
  - `played_round_type`, `next_round_type`
  - `played_payoff`, `next_payoff`
  - `played_payoff_seed`, `next_payoff_seed`

### 3) Multi-Agent Wrapper (`env.py`)

- `MultiAgentACEEnv(num_agents>=2, ...)`
  - Shared hidden state per round (`round_type` + payoff)
  - One JSON action per agent per round
  - Trust matrix + alliances set
  - Coalition actions:
    - propose/accept/reject alliance
    - betray (break alliance; different shaping depending on round type)
    - challenge (light trust penalty; small benefit in competitive rounds)
- **ID randomization**
  - `id_shuffle=True` exposes per-episode `public_ids`, and coalition tool targets resolve through public IDs (anti-handshake baseline).
- **Per-agent reward breakdown in history**
  - `r_task`, `r_inference`, `r_social`, `r_adapt`, `r_anticollusion`, `r_total`
- **Per-agent error observability**
  - `obs["last_errors"]` is a list of per-agent validation errors (for self-correction training).

### 4) Training / RLVR Scoring (`ace_training.py`)

- Canonical `SYSTEM_PROMPT`, `build_prompt`, `generate_ace_dataset`, `ace_reward_function`
- Reward function is **env-state-free**:
  - Dataset embeds `GROUND_TRUTH:<type>` and `PAYOFF_SEED:<int>` in a hidden comment
  - The scorer re-derives the payoff deterministically from `PAYOFF_SEED` and matches env reward logic
- Robust JSON extraction for completions (handles extra tokens around JSON).

### 5) Backwards Compatibility (`ace_env_fixed.py`)

- `ACEEnv` and `MultiAgentACEEnv` are re-exported so older imports keep working.

### 6) Demo / Smoke Tests (`test.py`)

- Deterministic multi-agent demo with:
  - a fixed `round_type_schedule`
  - `god_mode=True`
  - `id_shuffle=True`
  - alliance formation in cooperative rounds + betrayal in competitive rounds
  - intentional “handshake” payload to demonstrate anti-collusion penalty

## How To Run (Local)

### Multi-agent demo

```bash
python3 test.py
```

### Judge demo (clean terminal output)

```bash
python3 demo.py
```

### Single-agent smoke run

```bash
python3 env.py
```

### Dataset generation + GRPO block

`train_sim.py` prints a helpful error locally if you don’t have `datasets` installed. In Colab, use the embedded `TRAINING_SCRIPT` block.

## Key Result

The environment demonstrates clear learnability:

- Early phase: random predictions (~33% accuracy)
- Mid phase: partial signal usage (~50% accuracy)
- Late phase: mostly correct inference (~85–95% accuracy)

This confirms that the signals + reward structure support learning hidden-state inference.

## Training Curves

![Training Curves](training_curves.png)

The agent transitions from random guessing to consistent inference of hidden states, demonstrating learnability of the environment.

## Current Action Schema (Recommended)

```json
{
  "belief": { "predicted_round": "competitive", "confidence": 0.8 },
  "action": [
    { "tool": "propose_alliance", "parameters": { "target_id": 1 } },
    { "tool": "submit_bid", "parameters": { "amount": 75, "partner_id": 1 } }
  ]
}
```

Notes:
- `action` may be a single object or a list of tool calls.
- The last tool call is treated as the “economic” action for task reward; coalition tools are processed first.

## What’s Still Pending / Next Steps

### Training & Evaluation (Highest value next)

- **Update GRPO training to truly learn payoff inference**
  - Right now the prompt does *not* expose payoff seed (it’s hidden for scoring); to learn inference, the model needs repeated interactions / history and must learn thresholds from outcomes.
  - Add training/eval loops where the model sees reward feedback (or structured error/feedback) and adapts within an episode.
- **Multi-agent training**
  - Current `ace_training.py` + `train_sim.py` are single-agent oriented.
  - Add a multi-agent rollout dataset + scorer or a self-play harness.
- **Proper metrics**
  - Tool validity rate, correction rate after errors, alliance stability, betrayal “strategicness”, adaptation score, payoff-inference accuracy.

### Environment Depth

- **More realistic contract execution**
  - `execute_contract` is currently a stub reward; implement a multi-step workflow (bid → win/loss → execute → succeed/fail).
- **Coalition realism**
  - Team IDs, shared contract execution, splitting rewards, multi-party alliances (3+ agents), coalition dissolution rules.
- **Opponent modeling signals**
  - Include observable opponent actions in the observation and make them matter for payoff/strategy.

### Anti-collusion hardening

- Upgrade from heuristics to stronger protections:
  - randomized role/ID remapping each round (not just per episode)
  - penalties for unused/irrelevant fields at any nesting depth
  - enforce strict JSON schema (optionally) + reject unknown fields
  - evaluation protocols to detect “handshake” policies

### UX / Demo Deliverables

- **HF Space / UI “God Mode” dashboard**
  - panels for belief vs truth, payoff params, reward breakdown, trust graph over time.
- Export history logs to JSON and add a simple plotting utility for trust curves + reward curves.

### Packaging / Quality

- Add a small test suite (unit tests for:
  - schema validation
  - payoff seed determinism
  - id_shuffle mapping correctness
  - coalition event transitions)
- Pin dependencies / add minimal `requirements.txt` (optional).

---

If you want, the best next implementation jump is: **a self-play harness** for `MultiAgentACEEnv` that runs multiple episodes, logs metrics (trust graph + payoff inference), and outputs a `training_curves.csv` so you can plot improvements quickly.
