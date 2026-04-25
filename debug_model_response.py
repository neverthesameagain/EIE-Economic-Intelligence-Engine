"""Print raw model responses for the Option B demo prompts.

Run:
    LLM_PROVIDER=groq GROQ_API_KEY=... python debug_model_response.py
"""

from __future__ import annotations

import json
import os

from ace_text_inject import INJECT_SYSTEM_PROMPT, call_groq_chat_completion
from ace_world_env import ACEWorldEnv


def main() -> None:
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    event_text = os.getenv("DEBUG_EVENT", "oil crisis hits Middle East")
    env = ACEWorldEnv()
    agent = env.agents[0]

    event_user = f"Current world state:\n{env.world.to_prompt_str()}\n\nEvent: {event_text}"
    agent_user = "\n".join(
        [
            f"Upcoming round: {env.round_number + 1}",
            f"Visible alliances: {sorted([list(pair) for pair in env.alliances])}",
            "Recent global round history:",
            json.dumps(env.round_history[-3:], indent=2),
            "Return ONLY valid JSON. No explanation.",
        ]
    )

    if provider == "groq":
        event_raw = call_groq_chat_completion(
            [
                {"role": "system", "content": INJECT_SYSTEM_PROMPT},
                {"role": "user", "content": event_user},
            ],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.15,
            max_tokens=360,
        )
        agent_raw = call_groq_chat_completion(
            [
                {"role": "system", "content": agent.system_prompt(env.world.to_prompt_str())},
                {"role": "user", "content": agent_user},
            ],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.35,
            max_tokens=320,
        )
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        event_response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=320,
            system=INJECT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": event_user}],
        )
        agent_response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=260,
            system=agent.system_prompt(env.world.to_prompt_str()),
            messages=[{"role": "user", "content": agent_user}],
        )
        event_raw = event_response.content[0].text.strip()
        agent_raw = agent_response.content[0].text.strip()
    else:
        raise SystemExit("Set LLM_PROVIDER to groq or anthropic.")

    print("\n=== RAW EVENT TRANSLATION RESPONSE ===")
    print(event_raw)
    print("\n=== RAW AGENT DECISION RESPONSE ===")
    print(agent_raw)


if __name__ == "__main__":
    main()
