from env import MultiAgentACEEnv
import json

env = MultiAgentACEEnv(
    num_agents=2,
    num_rounds=5,
    seed=42,
    round_type_schedule=["resource", "competitive", "cooperative", "resource", "competitive"],
    god_mode=True,
    id_shuffle=True,
)
obs = env.reset()
public_ids = obs["public_ids"]
me0, me1 = public_ids[0], public_ids[1]

for _ in range(5):
    actions = []

    # ✅ Use CURRENT observation (correct now)
    market = obs["market_state"]

    if market["competition_signal"] == "high":
        pred = "competitive"
        # Competitive: optionally betray if allied, then bid aggressively
        per_agent = [
            [
                {"tool": "betray", "parameters": {"partner_id": me1}},
                {"tool": "submit_bid", "parameters": {"amount": 75}},
            ],
            {"tool": "submit_bid", "parameters": {"amount": 72}},
        ]
    elif market["competition_signal"] == "low":
        pred = "cooperative"
        # Cooperative: propose + accept alliance, then symmetric partnering + conservative bids
        per_agent = [
            [
                {"tool": "propose_alliance", "parameters": {"target_id": me1}},
                {"tool": "submit_bid", "parameters": {"amount": 30, "partner_id": me1}},
            ],
            [
                {"tool": "accept_alliance", "parameters": {"proposer_id": me0}},
                {"tool": "submit_bid", "parameters": {"amount": 32, "partner_id": me0}},
            ],
        ]
    else:
        pred = "resource"
        # Resource: demonstrate tool-choice diversity
        per_agent = [
            {"tool": "allocate_resources", "parameters": {"amount": 50}},
            {"tool": "submit_bid", "parameters": {"amount": 55}},
        ]

    for idx, agent_action in enumerate(per_agent):
        action_payload = agent_action
        if isinstance(action_payload, list):
            action_field = action_payload
        else:
            action_field = action_payload
        payload = {
            "belief": {"predicted_round": pred, "confidence": 0.8},
            "action": action_field,
        }
        # Demo: a "secret handshake" payload gets penalized by anti-collusion heuristics.
        if obs["round"] == 0 and idx == 0:
            payload["handshake"] = "x" * 100

        actions.append(json.dumps(payload))

    # Step AFTER building actions
    obs, rewards, done, info = env.step(actions)
    public_ids = obs.get("public_ids", public_ids)
    me0, me1 = public_ids[0], public_ids[1]

    print("Rewards:", rewards)
    print("Actual round:", info["debug_round_type"])
    print("Public IDs:", public_ids)
    print("Alliances:", obs.get("alliances"))
    print("Trust:", obs.get("trust"))
    print("Coalition events:", (obs.get("history") or [{}])[-1].get("coalition_events"))
    print("God mode played:", obs.get("played_round_type"))
    print("God mode next  :", obs.get("next_round_type"))
    print("Played payoff  :", obs.get("played_payoff"))
    print("Played seed    :", obs.get("played_payoff_seed"))
    print("Next payoff    :", obs.get("next_payoff"))
    print("Next seed      :", obs.get("next_payoff_seed"))
    print("Last errors    :", obs.get("last_errors"))
    print("Observation:", obs)
    print("Inference Accuracy:", info["inference_accuracy"])
    print("------")
