---
title: EIE Economic Intelligence Engine
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# EIE Economic Intelligence Engine

Learning Decisions in Dynamic Economic and Geopolitical Systems

EIE Economic Intelligence Engine is a live multi-agent economic simulation where natural language events reshape the world, agents reason under uncertainty, and social dynamics drive adaptation.

## Demo Flow

Use the main command bar:

1. Pick a sample event or type your own event.
2. Choose `Agent-Based RL` or `LLM-Based RL`.
3. In the provider dropdown, choose either `fallback` or `groq`.
4. Confirm the model, then press `Run ->`.
5. Watch world state, agent actions, AI reasoning, rewards, trust, and behavior shift.

`Agent-Based RL` works with no keys. `LLM-Based RL` uses Groq and reads credentials from environment variables or Hugging Face Space secrets. There is intentionally no API key textbox in the UI.

## What The Demo Shows

- Text-to-economy translation using the new `ace_text_inject.py` pipeline
- Bounded world updates with causal trace and confidence
- Seven differentiated agents with beliefs, memory, trust, Q-values, and opponent models
- Alliances, challenges, betrayals, retaliation, and cooperation
- Lightweight Q-style adaptation without heavy training loops

## Runtime Files

- `app.py`: Hugging Face Space entrypoint
- `openenv.yaml`: OpenEnv metadata pointing to `openenv_ace:ACEOpenMultiAgentEnv`
- `openenv_ace.py`: OpenEnv-compatible adapter around the current simulator
- `demo_gradio.py`: judge-facing UI
- `ace_text_inject.py`: robust text-to-economy engine
- `ace_world_env.py`: world model and multi-agent environment
- `ace_agents.py`: adaptive company agents
- `ace_reward.py`: decomposed reward function

Older prototypes, notebooks, logs, and training artifacts are preserved in `legacy/`.

## Secrets

The app runs without API keys using deterministic fallback logic. For live `LLM-Based RL`, set these variables:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

For local development, put them in `.env`:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

For Hugging Face deployment, add the same names under **Space settings -> Repository secrets**, then restart or factory rebuild the Space.

The model can be LLaMA, but the active client is Groq. The visible demo does **not** expose Anthropic, `OPENAI_API_KEY`, or `OPENAI_BASE_URL`; old OpenAI-style config only appears in legacy files preserved under `legacy/`.
