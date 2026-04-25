"""
ACE++ Environment — Fixed & OpenEnv-ready
==========================================
Key fixes vs v1:
  1. Round type is sampled in reset() and at the END of step() (for next round).
     The agent ALWAYS predicts the round whose market_state they already observed.
  2. market_state returned in observation is for the NEXT round (correct POMDP).
  3. current_round_type is stored on self — never re-sampled mid-step.
  4. validate_action returns structured error JSON (matches spec).
  5. Step returns flat scalar reward (not list) for single-agent TRL compatibility.
     Multi-agent wrapper is a separate class below.
"""

import random
import json
from dataclasses import dataclass
from typing import Any, Optional


ROUND_TYPES = ["cooperative", "competitive", "resource"]

PAYOFF_SALT = 0xA11CE  # deterministic payoff derivation salt


@dataclass(frozen=True)
class RoundPayoff:
    competitive_bid_threshold: float
    cooperative_bid_max: float
    resource_bid_min: float
    resource_bid_max: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


@dataclass(frozen=True)
class NormalizedAction:
    predicted_round: str
    confidence: float
    tool: str
    parameters: dict[str, Any]
    tool_calls: list[tuple[str, dict[str, Any]]]
    anti_collusion_penalty: float
    parse_penalty: float


class ACEEnv:
    """
    Single-agent ACE++ environment.

    Episode flow:
        obs = env.reset()           # obs contains market_state for round 0
        obs, reward, done, info = env.step(action_json)
            # action_json must predict round 0's type (what was shown in reset obs)
            # obs now contains market_state for round 1
        ...

    Action format (JSON string):
        {
          "predicted_round": "cooperative" | "competitive" | "resource",
          "action": "bid" | "allocate" | "solo",
          "amount": float          # required if action == "bid"
        }
    """

    def __init__(
        self,
        num_rounds: int = 5,
        inference_weight: float = 1.2,
        seed: Optional[int] = None,
        round_type_schedule: Optional[list[str]] = None,
        difficulty: str = "medium",
    ):
        self.num_rounds = num_rounds
        self.inference_weight = inference_weight  # w in R_total formula
        self._rng = random.Random(seed)
        self._round_type_schedule = round_type_schedule[:] if round_type_schedule else None
        self._schedule_idx = 0
        self.difficulty = difficulty
        self.current_payoff: Optional[RoundPayoff] = None
        self.current_payoff_seed: Optional[int] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        self.current_round = 0
        self.total_reward = 0.0
        self.correct_inferences = 0
        self.history = []
        self._schedule_idx = 0

        # Sample round 0's type NOW — agent will see its market signal
        self.current_round_type = self._sample_round_type()
        self.current_payoff_seed = self._sample_payoff_seed()
        self.current_payoff = self._payoff_from_seed(self.current_payoff_seed)
        market_state = self.query_market_state(self.current_round_type)

        return {
            "round": self.current_round,
            "market_state": market_state,   # signal for CURRENT round
            "history": [],
        }

    def step(self, action: str):
        """
        action: JSON string from the agent.
        Returns: (observation, reward, done, info)

        Scores the agent's prediction against self.current_round_type
        (the round whose market_state was shown in the previous observation).
        Then advances to the next round.
        """
        # ---- Validate ----
        norm_action, error = self._validate_action(action)
        if error:
            scored_round_type = self.current_round_type
            # Advance round even on error so episodes don't stall
            self.current_round += 1
            done = self.current_round >= self.num_rounds
            next_round_type = self._advance_round()
            obs = self._make_observation(next_round_type)
            obs["last_error"] = error
            return obs, -1.0, done, {"error": error, "debug_round_type": scored_round_type}

        # ---- Score against the CURRENT round (already shown to agent) ----
        scored_round_type = self.current_round_type   # what agent saw signal for
        scored_payoff = self.current_payoff
        scored_payoff_seed = self.current_payoff_seed
        pred = norm_action.predicted_round
        confidence = norm_action.confidence
        tool = norm_action.tool
        parameters = norm_action.parameters

        # Inference reward
        if pred == scored_round_type:
            r_inference = 1.0 * confidence
            self.correct_inferences += 1
        else:
            r_inference = -0.5 * confidence

        # Task reward
        r_task = self._tool_reward(tool, parameters, scored_round_type)

        total_reward = (
            r_task
            + self.inference_weight * r_inference
            - norm_action.anti_collusion_penalty
            - norm_action.parse_penalty
        )
        self.total_reward += total_reward

        feedback = self._rubric_feedback(tool, parameters, scored_round_type)

        # ---- Log ----
        step_log = {
            "round": self.current_round,
            "actual_round_type": scored_round_type,
            "actual_payoff": scored_payoff.__dict__ if scored_payoff is not None else None,
            "actual_payoff_seed": scored_payoff_seed,
            "predicted_round": pred,
            "correct": pred == scored_round_type,
            "confidence": confidence,
            "tool": tool,
            "parameters": parameters,
            "r_task": r_task,
            "r_inference": r_inference,
            "r_total": total_reward,
            "feedback": feedback,
            "anti_collusion_penalty": norm_action.anti_collusion_penalty,
            "parse_penalty": norm_action.parse_penalty,
        }
        self.history.append(step_log)

        # ---- Advance to next round ----
        self.current_round += 1
        done = self.current_round >= self.num_rounds

        # Sample next round's type, generate its market signal
        next_round_type = self._advance_round()
        obs = self._make_observation(next_round_type)

        info = {
            "debug_round_type": scored_round_type,          # what was just played
            "next_round_type": next_round_type,             # what signal encodes
            "correct_inference": pred == scored_round_type,
            "inference_accuracy": self.correct_inferences / self.current_round,
            "step_log": step_log,
            "feedback": feedback,
        }

        return obs, total_reward, done, info

    def state(self):
        """Full internal state — for OpenEnv / God Mode panel."""
        return {
            "current_round": self.current_round,
            "current_round_type": self.current_round_type,
            "current_payoff": (
                self.current_payoff.__dict__ if self.current_payoff is not None else None
            ),
            "current_payoff_seed": self.current_payoff_seed,
            "total_reward": self.total_reward,
            "inference_accuracy": (
                self.correct_inferences / self.current_round
                if self.current_round > 0 else 0.0
            ),
            "history": self.history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_round(self):
        """Sample and store the type for the round that just started."""
        if self.current_round < self.num_rounds:
            self.current_round_type = self._sample_round_type()
            self.current_payoff_seed = self._sample_payoff_seed()
            self.current_payoff = self._payoff_from_seed(self.current_payoff_seed)
        return self.current_round_type

    def _make_observation(self, round_type):
        return {
            "round": self.current_round,
            "market_state": self.query_market_state(round_type),
            "history": self.history[-3:],
        }

    def _sample_round_type(self):
        if self._round_type_schedule and self._schedule_idx < len(self._round_type_schedule):
            rt = self._round_type_schedule[self._schedule_idx]
            self._schedule_idx += 1
            if rt in ROUND_TYPES:
                return rt
        return self._rng.choice(ROUND_TYPES)

    def _sample_payoff_seed(self) -> int:
        return int(self._rng.randrange(1, 2**31 - 1))

    def _payoff_from_seed(self, payoff_seed: int) -> RoundPayoff:
        """
        Deterministic hidden payoff structure derived from a seed.
        This enables verifiable rewards without carrying env state.
        """
        r = random.Random(int(payoff_seed) ^ PAYOFF_SALT)
        competitive_thr = r.uniform(60.0, 70.0)
        cooperative_max = r.uniform(35.0, 45.0)
        resource_min = r.uniform(35.0, 45.0)
        resource_max = r.uniform(55.0, 65.0)
        if resource_max < resource_min:
            resource_min, resource_max = resource_max, resource_min
        return RoundPayoff(
            competitive_bid_threshold=competitive_thr,
            cooperative_bid_max=cooperative_max,
            resource_bid_min=resource_min,
            resource_bid_max=resource_max,
        )

    # ------------------------------------------------------------------
    # "Professional API layer" — these methods emulate real tool calls.
    # The agent does not call them directly; it emits JSON that selects
    # a tool and parameters, and the env validates + executes.
    # ------------------------------------------------------------------

    def query_market_state(self, round_type):
        """
        Mock Market Query API.
        Returns JSON that CORRELATES with round_type but doesn't reveal it directly.
        Agents must learn the mapping.
        """
        # Difficulty controls how separable the signals are.
        # easy: clear separation, hard: overlapping/noisy signals.
        if self.difficulty == "easy":
            jitter = 0.0
        elif self.difficulty == "hard":
            jitter = 0.18
        else:
            jitter = 0.08

        def j(x: float) -> float:
            return float(min(1.0, max(0.0, x + self._rng.uniform(-jitter, jitter))))

        if round_type == "competitive":
            return {
                "demand_index":       round(j(self._rng.uniform(0.80, 1.00)), 2),
                "volatility":         round(j(self._rng.uniform(0.70, 1.00)), 2),
                "competition_signal": "high",
                "cooperation_signal": "low",
            }
        elif round_type == "cooperative":
            return {
                "demand_index":       round(j(self._rng.uniform(0.20, 0.45)), 2),
                "volatility":         round(j(self._rng.uniform(0.10, 0.35)), 2),
                "competition_signal": "low",
                "cooperation_signal": "high",
            }
        else:  # resource
            return {
                "demand_index":       round(j(self._rng.uniform(0.45, 0.60)), 2),
                "volatility":         round(j(self._rng.uniform(0.40, 0.60)), 2),
                "competition_signal": "medium",
                "cooperation_signal": "medium",
            }

    def submit_bid(self, amount: float, partner_id: Optional[int] = None) -> dict[str, Any]:
        # partner_id is reserved for the multi-agent wrapper; ignored in single-agent.
        return {"status": "ok", "amount": float(amount), "partner_id": partner_id}

    def allocate_resources(self, amount: float) -> dict[str, Any]:
        return {"status": "ok", "amount": float(amount)}

    def execute_contract(self, team_id: Optional[int] = None) -> dict[str, Any]:
        return {"status": "ok", "team_id": team_id}

    def _tool_reward(self, tool: str, parameters: dict[str, Any], round_type: str) -> float:
        """
        Immediate task reward from executing a validated tool call.
        """
        if tool == "submit_bid":
            amount = float(parameters.get("amount", 50))
            if self.current_payoff is not None:
                payoff = self.current_payoff
            else:
                seed = self.current_payoff_seed if self.current_payoff_seed is not None else self._sample_payoff_seed()
                payoff = self._payoff_from_seed(seed)
            if round_type == "competitive":
                return 2.0 if amount >= payoff.competitive_bid_threshold else -1.0
            if round_type == "cooperative":
                return 2.0 if amount <= payoff.cooperative_bid_max else -1.0
            return 2.0 if payoff.resource_bid_min <= amount <= payoff.resource_bid_max else -1.0

        if tool == "allocate_resources":
            return 1.0 if round_type == "resource" else 0.0

        if tool == "execute_contract":
            return 0.5

        # Coalition tools are defined for schema completeness in v1.
        if tool in {"propose_alliance", "accept_alliance", "reject_alliance", "betray", "challenge"}:
            return 0.0

        return -0.25

    def _rubric_feedback(self, tool: str, parameters: dict[str, Any], round_type: str) -> str:
        if tool == "submit_bid" and round_type == "competitive" and float(parameters.get("amount", 50)) < 50:
            return "You bid conservatively despite competitive conditions."
        if tool == "submit_bid" and round_type == "cooperative" and float(parameters.get("amount", 50)) > 50:
            return "You bid aggressively despite cooperative conditions."
        if tool == "allocate_resources" and round_type != "resource":
            return "You allocated resources outside a resource round."
        return ""

    def _validate_action(self, action: str):
        """
        Returns (NormalizedAction, None) on success.
        Returns (None, error_dict) on failure.
        """
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError as e:
            return None, {
                "status": "error",
                "error_type": "JSON_PARSE_ERROR",
                "message": str(e),
                "expected_format": {
                    "predicted_round": "cooperative | competitive | resource",
                    "action": "bid | allocate | solo",
                    "amount": "float (required for bid)",
                },
            }

        anti_collusion_penalty = self._anti_collusion_penalty(parsed)
        parse_penalty = 0.0

        predicted_round = None
        confidence = 1.0

        if isinstance(parsed.get("belief"), dict):
            belief = parsed["belief"]
            predicted_round = belief.get("predicted_round")
            if "confidence" in belief:
                try:
                    confidence = float(belief.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                    parse_penalty += 0.1
        else:
            predicted_round = parsed.get("predicted_round")
            if "confidence" in parsed:
                try:
                    confidence = float(parsed.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                    parse_penalty += 0.1

        if predicted_round not in ROUND_TYPES:
            return None, {
                "status": "error",
                "error_type": "INVALID_ROUND_TYPE",
                "message": f"predicted_round must be one of {ROUND_TYPES}",
            }

        if not (0.0 <= confidence <= 1.0):
            return None, {
                "status": "error",
                "error_type": "INVALID_CONFIDENCE",
                "message": "confidence must be a float in [0, 1]",
            }

        tool = None
        parameters: dict[str, Any] = {}
        tool_calls: list[tuple[str, dict[str, Any]]] = []

        action_field = parsed.get("action")
        if isinstance(action_field, dict):
            tool = action_field.get("tool") or action_field.get("name")
            parameters = (
                action_field.get("parameters")
                or action_field.get("args")
                or {}
            )
            if not isinstance(parameters, dict):
                parameters = {}
            tool_calls = [(str(tool), parameters)]
        elif isinstance(action_field, list):
            for item in action_field:
                if not isinstance(item, dict):
                    continue
                t = item.get("tool") or item.get("name")
                p = item.get("parameters") or item.get("args") or {}
                if t is None:
                    continue
                if not isinstance(p, dict):
                    p = {}
                tool_calls.append((str(t), p))
            if not tool_calls:
                return None, {
                    "status": "error",
                    "error_type": "MISSING_ACTION",
                    "message": "Empty action list. Provide at least one tool call.",
                }
            tool, parameters = tool_calls[-1]
        elif isinstance(action_field, str):
            # Backwards-compatible v0 schema:
            if action_field == "bid":
                tool = "submit_bid"
                parameters = {"amount": parsed.get("amount"), "partner_id": parsed.get("partner_id")}
            elif action_field == "allocate":
                tool = "allocate_resources"
                parameters = {"amount": parsed.get("amount")}
            elif action_field == "solo":
                tool = "execute_contract"
                parameters = {"team_id": None}
            else:
                tool = action_field
                parameters = {}
            tool_calls = [(str(tool), parameters)]
        else:
            return None, {
                "status": "error",
                "error_type": "MISSING_ACTION",
                "message": "Missing 'action'. Provide either a string action or {'tool': ..., 'parameters': {...}}.",
            }

        def _safe_amount(v: Any) -> tuple[float, float]:
            try:
                x = float(v)
                if x != x or x in (float("inf"), float("-inf")):
                    raise ValueError("non-finite")
                return x, 0.0
            except (TypeError, ValueError):
                return 50.0, 0.1

        for t, p in tool_calls:
            if t in {"submit_bid", "allocate_resources"}:
                if "amount" not in p or p.get("amount") is None:
                    return None, {
                        "status": "error",
                        "error_type": "MISSING_KEYS",
                        "message": "Missing required key: amount",
                        "expected_format": {
                            "belief": {"predicted_round": "cooperative|competitive|resource", "confidence": 0.0},
                            "action": {"tool": t, "parameters": {"amount": "float"}},
                        },
                    }
                amt, pen = _safe_amount(p.get("amount"))
                p["amount"] = amt
                parse_penalty += pen

            if t in {"propose_alliance", "accept_alliance", "reject_alliance", "challenge"}:
                # Require a target agent id for coalition actions.
                if not any(k in p for k in ("target_id", "partner_id", "proposer_id")):
                    return None, {
                        "status": "error",
                        "error_type": "MISSING_KEYS",
                        "message": f"Missing required key for {t}: target_id",
                    }

            if t == "betray":
                if not any(k in p for k in ("partner_id", "target_id")):
                    return None, {
                        "status": "error",
                        "error_type": "MISSING_KEYS",
                        "message": "Missing required key for betray: partner_id",
                    }

        return NormalizedAction(
            predicted_round=predicted_round,
            confidence=confidence,
            tool=str(tool),
            parameters=parameters if isinstance(parameters, dict) else {},
            tool_calls=tool_calls,
            anti_collusion_penalty=anti_collusion_penalty,
            parse_penalty=parse_penalty,
        ), None

    def _anti_collusion_penalty(self, parsed: dict[str, Any]) -> float:
        """
        Light heuristic penalty to discourage non-causal payloads / "handshakes".
        Penalizes:
          - unexpected top-level keys
          - unusually long string fields
        """
        allowed_top = {
            "belief",
            "action",
            "predicted_round",
            "confidence",
            "amount",
            "partner_id",
            "team_id",
        }
        extra = [k for k in parsed.keys() if k not in allowed_top]
        penalty = min(0.5, 0.1 * len(extra))

        def scan(o: Any) -> int:
            if isinstance(o, str):
                return 1 if len(o) > 64 else 0
            if isinstance(o, dict):
                return sum(scan(v) for v in o.values())
            if isinstance(o, list):
                return sum(scan(v) for v in o)
            return 0

        long_strings = scan(parsed)
        penalty += min(0.5, 0.1 * long_strings)
        return float(min(0.75, penalty))


class MultiAgentACEEnv:
    """
    Minimal multi-agent wrapper around the same hidden-market core.

    - Shared hidden round type + market_state per round
    - Each agent submits one JSON action per round
    - Adds lightweight alliance/trust dynamics (v1)

    Interface:
        obs = env.reset()                          # shared observation
        obs, rewards, done, info = env.step(list_of_action_json)
    """

    def __init__(
        self,
        num_agents: int = 2,
        num_rounds: int = 10,
        inference_weight: float = 1.2,
        social_weight: float = 0.2,
        seed: Optional[int] = None,
        round_type_schedule: Optional[list[str]] = None,
        id_shuffle: bool = False,
        god_mode: bool = False,
        difficulty: str = "medium",
        max_tool_calls_per_turn: int = 4,
        adaptation_weight: float = 0.1,
    ):
        if num_agents < 2:
            raise ValueError("num_agents must be >= 2")
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.inference_weight = inference_weight
        self.social_weight = social_weight
        self.id_shuffle = id_shuffle
        self.god_mode = god_mode
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.adaptation_weight = adaptation_weight
        self._core = ACEEnv(
            num_rounds=num_rounds,
            inference_weight=inference_weight,
            seed=seed,
            round_type_schedule=round_type_schedule,
            difficulty=difficulty,
        )

    def reset(self) -> dict[str, Any]:
        obs = self._core.reset()
        self.current_round = self._core.current_round
        self.current_round_type = self._core.current_round_type
        self.current_payoff = self._core.current_payoff
        self.current_payoff_seed = self._core.current_payoff_seed
        self.total_rewards = [0.0 for _ in range(self.num_agents)]
        self.correct_inferences = [0 for _ in range(self.num_agents)]
        self.pending_proposals: dict[int, set[int]] = {i: set() for i in range(self.num_agents)}
        self._last_task_by_round: list[dict[str, float]] = [
            {} for _ in range(self.num_agents)
        ]
        self.trust = [
            [0.5 if i != j else 1.0 for j in range(self.num_agents)]
            for i in range(self.num_agents)
        ]
        self.alliances: set[tuple[int, int]] = set()
        self.history: list[dict[str, Any]] = []

        self.public_ids = list(range(self.num_agents))
        if self.id_shuffle:
            self._core._rng.shuffle(self.public_ids)
        self._public_to_internal = {pid: i for i, pid in enumerate(self.public_ids)}
        self._internal_to_public = {i: pid for pid, i in self._public_to_internal.items()}

        obs["alliances"] = sorted([list(p) for p in self.alliances])
        obs["trust"] = self._trust_summary()
        obs["history"] = []
        obs["public_ids"] = self.public_ids
        if self.god_mode:
            obs["played_round_type"] = self.current_round_type
            obs["played_payoff"] = (
                self.current_payoff.__dict__ if self.current_payoff is not None else None
            )
            obs["played_payoff_seed"] = self.current_payoff_seed
        return obs

    def step(self, actions: list[str]):
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        scored_round_type = self._core.current_round_type
        scored_payoff = self._core.current_payoff
        scored_payoff_seed = self._core.current_payoff_seed
        parsed_actions: list[Optional[NormalizedAction]] = []
        errors: list[Optional[dict[str, Any]]] = []

        for a in actions:
            norm, err = self._core._validate_action(a)
            parsed_actions.append(norm)
            errors.append(err)

        rewards = [0.0 for _ in range(self.num_agents)]
        r_task = [0.0 for _ in range(self.num_agents)]
        r_inference = [0.0 for _ in range(self.num_agents)]
        r_social = [0.0 for _ in range(self.num_agents)]
        r_adapt = [0.0 for _ in range(self.num_agents)]

        coalition_events: list[dict[str, Any]] = []

        # --- Coalition tools first (propose/accept/reject/betray/challenge) ---
        for i, (norm, err) in enumerate(zip(parsed_actions, errors)):
            if err is not None or norm is None:
                continue

            if len(norm.tool_calls) > self.max_tool_calls_per_turn:
                errors[i] = {
                    "status": "error",
                    "error_type": "TOO_MANY_TOOL_CALLS",
                    "message": f"Max tool calls per turn is {self.max_tool_calls_per_turn}",
                }
                continue

            for tool, params in norm.tool_calls:
                if tool == "propose_alliance":
                    raw_target = params.get("target_id") if params.get("target_id") is not None else params.get("partner_id")
                    target = self._coerce_agent_id(raw_target)
                    if target is None or target == i:
                        continue
                    self.pending_proposals[target].add(i)
                    coalition_events.append({"tool": tool, "from": i, "to": target})

                elif tool == "accept_alliance":
                    raw_proposer = params.get("proposer_id") if params.get("proposer_id") is not None else params.get("target_id")
                    proposer = self._coerce_agent_id(raw_proposer)
                    if proposer is None or proposer == i:
                        continue
                    if proposer in self.pending_proposals[i]:
                        self.pending_proposals[i].remove(proposer)
                        pair = tuple(sorted((i, proposer)))
                        self.alliances.add(pair)
                        self._set_trust(i, proposer, +0.06)
                        r_social[i] += self.social_weight
                        r_social[proposer] += self.social_weight
                        coalition_events.append({"tool": tool, "from": i, "to": proposer})

                elif tool == "reject_alliance":
                    raw_proposer = params.get("proposer_id") if params.get("proposer_id") is not None else params.get("target_id")
                    proposer = self._coerce_agent_id(raw_proposer)
                    if proposer is None or proposer == i:
                        continue
                    self.pending_proposals[i].discard(proposer)
                    self._set_trust(i, proposer, -0.05)
                    coalition_events.append({"tool": tool, "from": i, "to": proposer})

                elif tool == "betray":
                    raw_partner = params.get("partner_id") if params.get("partner_id") is not None else params.get("target_id")
                    partner = self._coerce_agent_id(raw_partner)
                    if partner is None or partner == i:
                        continue
                    pair = tuple(sorted((i, partner)))
                    if pair in self.alliances:
                        self.alliances.remove(pair)
                        self._set_trust(i, partner, -0.25)
                        # Strategic betrayal is only beneficial in competitive rounds (v1 heuristic).
                        if scored_round_type == "competitive":
                            r_social[i] += self.social_weight
                            r_social[partner] -= self.social_weight
                        else:
                            r_social[i] -= self.social_weight
                        coalition_events.append({"tool": tool, "from": i, "to": partner})

                elif tool == "challenge":
                    raw_target = params.get("target_id") if params.get("target_id") is not None else params.get("partner_id")
                    target = self._coerce_agent_id(raw_target)
                    if target is None or target == i:
                        continue
                    self._set_trust(i, target, -0.04)
                    if scored_round_type == "competitive":
                        r_social[i] += self.social_weight * 0.25
                    coalition_events.append({"tool": tool, "from": i, "to": target})

        # If an agent sends invalid JSON, it gets penalized and can’t affect alliances.
        for i, (norm, err) in enumerate(zip(parsed_actions, errors)):
            if err is not None or norm is None:
                rewards[i] = -1.0
                r_task[i] = -1.0
                continue

            # inference reward
            if norm.predicted_round == scored_round_type:
                r_inf = 1.0 * norm.confidence
                self.correct_inferences[i] += 1
            else:
                r_inf = -0.5 * norm.confidence
            r_inference[i] = r_inf

            # Economic tool = last tool call (by convention); ignore coalition tools here.
            econ_tool, econ_params = norm.tool, norm.parameters
            if norm.tool_calls:
                econ_tool, econ_params = norm.tool_calls[-1]
            task = self._core._tool_reward(econ_tool, econ_params, scored_round_type)
            r_task[i] = task

            # Adaptation: reward improvement over last time facing this round_type.
            last = self._last_task_by_round[i].get(scored_round_type)
            if last is not None and task > last:
                r_adapt[i] += self.adaptation_weight
            self._last_task_by_round[i][scored_round_type] = task

            rewards[i] = (
                task
                + self.inference_weight * r_inf
                + r_social[i]
                + r_adapt[i]
                - norm.anti_collusion_penalty
                - norm.parse_penalty
            )

        # --- Coalition dynamics (v1) ---
        # Cooperative rounds reward symmetric partnering; competitive penalizes it.
        partner_pairs: set[tuple[int, int]] = set()
        for i, norm in enumerate(parsed_actions):
            if norm is None:
                continue
            econ_tool, econ_params = norm.tool, norm.parameters
            if norm.tool_calls:
                econ_tool, econ_params = norm.tool_calls[-1]
            if econ_tool != "submit_bid":
                continue
            partner_id = econ_params.get("partner_id")
            if partner_id is None:
                continue
            j = self._coerce_agent_id(partner_id)
            if j is None:
                continue
            if 0 <= j < self.num_agents:
                partner_pairs.add(tuple(sorted((i, j))))

        symmetric_pairs: set[tuple[int, int]] = set()
        for i, norm in enumerate(parsed_actions):
            if norm is None:
                continue
            econ_tool, econ_params = norm.tool, norm.parameters
            if norm.tool_calls:
                econ_tool, econ_params = norm.tool_calls[-1]
            if econ_tool != "submit_bid":
                continue
            partner_id = econ_params.get("partner_id")
            if partner_id is None:
                continue
            j = self._coerce_agent_id(partner_id)
            if j is None:
                continue
            if not (0 <= j < self.num_agents):
                continue
            other = parsed_actions[j]
            if other is None:
                continue
            o_tool, o_params = other.tool, other.parameters
            if other.tool_calls:
                o_tool, o_params = other.tool_calls[-1]
            o_partner = self._coerce_agent_id(o_params.get("partner_id", -1)) if o_tool == "submit_bid" else None
            if o_tool == "submit_bid" and o_partner == i:
                symmetric_pairs.add(tuple(sorted((i, j))))

        for (i, j) in symmetric_pairs:
            if scored_round_type == "cooperative":
                r_social[i] += self.social_weight
                r_social[j] += self.social_weight
                self._set_trust(i, j, +0.05)
                self.alliances.add((i, j))
            elif scored_round_type == "competitive":
                r_social[i] -= self.social_weight
                r_social[j] -= self.social_weight
                self._set_trust(i, j, -0.03)
            else:
                # resource: neutral/slight positive for coordination
                r_social[i] += self.social_weight * 0.5
                r_social[j] += self.social_weight * 0.5

        # Agents that "fake" partner unreciprocated lose trust.
        for (i, j) in (partner_pairs - symmetric_pairs):
            self._set_trust(i, j, -0.04)

        # Apply social/adaptation components after symmetric-pair shaping.
        for i in range(self.num_agents):
            if errors[i] is None:
                rewards[i] = (
                    r_task[i]
                    + self.inference_weight * r_inference[i]
                    + r_social[i]
                    + r_adapt[i]
                    - (parsed_actions[i].anti_collusion_penalty if parsed_actions[i] else 0.0)
                    - (parsed_actions[i].parse_penalty if parsed_actions[i] else 0.0)
                )
            self.total_rewards[i] += rewards[i]

        # Advance core env by feeding one representative action (it only needs to progress rounds).
        # Use a valid action if available; otherwise a noop action.
        rep_action = next((a for a, e in zip(actions, errors) if e is None), None)
        if rep_action is None:
            rep_action = json.dumps({"predicted_round": "resource", "action": "solo"})

        obs, _, done, info = self._core.step(rep_action)
        self.current_round = self._core.current_round
        self.current_round_type = self._core.current_round_type
        self.current_payoff = self._core.current_payoff
        self.current_payoff_seed = self._core.current_payoff_seed

        step_log = {
            "round": info["step_log"]["round"],
            "actual_round_type": scored_round_type,
            "alliances": sorted([list(p) for p in self.alliances]),
            "trust": self._trust_summary(),
            "rewards": rewards,
            "reward_breakdown": [
                {
                    "r_task": r_task[i],
                    "r_inference": r_inference[i],
                    "r_social": r_social[i],
                    "r_adapt": r_adapt[i],
                    "r_anticollusion": -(
                        parsed_actions[i].anti_collusion_penalty if parsed_actions[i] else 0.0
                    ),
                    "r_parse": -(
                        parsed_actions[i].parse_penalty if parsed_actions[i] else 0.0
                    ),
                    "r_total": rewards[i],
                }
                for i in range(self.num_agents)
            ],
            "coalition_events": coalition_events,
        }
        self.history.append(step_log)

        obs["alliances"] = step_log["alliances"]
        obs["trust"] = step_log["trust"]
        obs["history"] = self.history[-3:]
        obs["last_errors"] = [errors[i] for i in range(self.num_agents)]
        obs["public_ids"] = self.public_ids
        if self.god_mode:
            obs["played_round_type"] = scored_round_type
            obs["next_round_type"] = self.current_round_type
            obs["played_payoff"] = scored_payoff.__dict__ if scored_payoff is not None else None
            obs["next_payoff"] = (
                self.current_payoff.__dict__ if self.current_payoff is not None else None
            )
            obs["played_payoff_seed"] = scored_payoff_seed
            obs["next_payoff_seed"] = self.current_payoff_seed

        info = {
            "debug_round_type": scored_round_type,
            "inference_accuracy": [
                (self.correct_inferences[i] / max(1, self.current_round))
                for i in range(self.num_agents)
            ],
            "total_rewards": self.total_rewards,
            "step_log": step_log,
        }
        return obs, rewards, done, info

    def state(self) -> dict[str, Any]:
        """Full internal state — for demo/UI."""
        return {
            "current_round": self.current_round,
            "current_round_type": self.current_round_type,
            "current_payoff": (
                self.current_payoff.__dict__ if getattr(self, "current_payoff", None) is not None else None
            ),
            "current_payoff_seed": getattr(self, "current_payoff_seed", None),
            "total_rewards": self.total_rewards,
            "inference_accuracy": [
                (self.correct_inferences[i] / max(1, self.current_round))
                for i in range(self.num_agents)
            ],
            "trust": self._trust_summary(),
            "alliances": sorted([list(p) for p in self.alliances]),
            "public_ids": getattr(self, "public_ids", list(range(self.num_agents))),
            "history": self.history,
            "pending_proposals": {k: sorted(list(v)) for k, v in self.pending_proposals.items()},
        }

    def _set_trust(self, i: int, j: int, delta: float) -> None:
        if i == j:
            return
        self.trust[i][j] = float(min(1.0, max(0.0, self.trust[i][j] + delta)))
        self.trust[j][i] = float(min(1.0, max(0.0, self.trust[j][i] + delta)))

    def _trust_summary(self) -> dict[str, float]:
        # Compact representation for prompts/UI.
        out: dict[str, float] = {}
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                out[f"{i}-{j}"] = round(self.trust[i][j], 2)
        return out

    def _coerce_agent_id(self, value: Any) -> Optional[int]:
        try:
            public_id = int(value)
        except (TypeError, ValueError):
            return None
        if self.id_shuffle:
            return self._public_to_internal.get(public_id)
        return public_id


# ------------------------------------------------------------------
# Quick smoke test — run this file directly to verify
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = ACEEnv(num_rounds=5)
    obs = env.reset()
    print("=== RESET ===")
    print(json.dumps(obs, indent=2))

    for step_num in range(5):
        market = obs["market_state"]

        # Scripted policy: read the signal, predict, bid accordingly
        sig = market["competition_signal"]
        if sig == "high":
            pred, amount = "competitive", 75.0
        elif sig == "low":
            pred, amount = "cooperative", 30.0
        else:
            pred, amount = "resource", 50.0

        action = json.dumps({
            "predicted_round": pred,
            "action": "bid",
            "amount": amount,
        })

        obs, reward, done, info = env.step(action)

        print(f"\n=== STEP {step_num + 1} ===")
        print(f"  Actual round : {info['debug_round_type']}")
        print(f"  Predicted    : {pred}  ({'✓' if info['correct_inference'] else '✗'})")
        print(f"  Reward       : {reward:.2f}")
        print(f"  Acc so far   : {info['inference_accuracy']:.0%}")

        if done:
            print("\n=== EPISODE DONE ===")
            print(f"  Total reward : {env.state()['total_reward']:.2f}")
            print(f"  Final acc    : {env.state()['inference_accuracy']:.0%}")
            break
