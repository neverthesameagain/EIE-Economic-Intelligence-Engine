# ACE++ Report

## Summary

ACE++ is a partially observable decision environment where an agent must infer a hidden round type from noisy observable signals, act in JSON tool form, and use feedback to improve its reasoning over time.

Unlike standard RL approaches that update model weights, ACE++ demonstrates in-context belief updating, where the agent can revise its internal reasoning within a single episode using structured feedback from the environment.

## Core System

- Hidden state: `cooperative`, `competitive`, or `resource`
- Observation: `market_state` with partial signals
- Action: structured JSON belief plus tool call
- Feedback: reward plus revealed ground truth
- Memory: prompt history reused on the next round

This creates a clean concept-to-execution loop:

`observe -> infer -> act -> receive feedback -> update belief -> act again`

## What The Demo Shows

The demo is designed to make learning visible, not implicit.

- Per-round predictions are printed with a belief confidence proxy
- Wrong-to-correct transitions are explicitly labeled
- Rolling accuracy is shown round by round
- Within-episode improvement is measured numerically
- A dedicated plot (`llm_episode_curve.png`) visualizes adaptation across rounds

## How To Interpret Baselines

- `Random` is the floor.
- `Rule-based` is a signal-heuristic ceiling for the current environment design.
- `LLM-style adaptive` is the reasoning agent under partial observability and feedback.

The purpose of the rule-based policy is not to prove intelligence; it shows that the environment is learnable. The purpose of the LLM agent is to show feedback-driven belief adjustment inside an episode.

Rule-based agents overfit deterministic signals, while LLM agents operate under uncertainty and require feedback to adapt.

## Learning Mechanism

ACE++ converts environment feedback into structured belief updates.

- The agent predicts a hidden state from the current signal
- The environment returns reward and the actual state
- That information is stored in history
- The next prompt includes that history
- The agent can revise its belief on the next round

This means adaptation happens without retraining, gradient updates, or parameter changes.

Confidence in the LLM demo is a heuristic proxy based on recent prediction consistency, used only to visualize belief stabilization over time.

## Recommended Demo Flow

1. Run `python3 demo.py` to show the environment and clean deterministic behavior.
2. Run `python3 llm_agent.py` to show visible failure, correction, and belief updates.
3. Run `python3 plot_llm_episode.py` to show within-episode adaptation visually.
4. Run `python3 compare_agents.py` to frame random, rule-based, and adaptive reasoning side by side.

## Deliverables

- `training_curves.png`: simulated learnability across episodes
- `llm_episode_curve.png`: within-episode LLM adaptation
- `llm_eval.json`: aggregate evaluation statistics
- `README.md`: project overview and learning narrative

## Bottom Line

ACE++ is not just an environment. It is a judge-readable demonstration of hidden-state inference, structured action generation, feedback loops, and in-context adaptation within a single episode.
