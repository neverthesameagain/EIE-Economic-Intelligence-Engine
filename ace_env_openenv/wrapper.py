"""
OpenEnv-compatible wrapper for ACE++.

The repo's `ACEEnv` already exposes the required `reset`, `step`, and `state`
methods. This module provides a thin adapter with a stable import path and a
fallback `Environment` base class so the wrapper is usable even when the
external OpenEnv package is not installed locally.
"""

from __future__ import annotations

from typing import Any, Optional

from env import ACEEnv, MultiAgentACEEnv

try:
    from openenv import Environment  # type: ignore
except ModuleNotFoundError:
    class Environment:  # type: ignore[no-redef]
        """Minimal fallback base matching the expected OpenEnv surface."""

        def reset(self) -> dict[str, Any]:
            raise NotImplementedError

        def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            raise NotImplementedError

        def state(self) -> dict[str, Any]:
            raise NotImplementedError


class ACEOpenEnv(Environment):
    """
    Thin OpenEnv adapter around the existing ACE++ single-agent environment.
    """

    def __init__(
        self,
        *,
        num_rounds: int = 10,
        inference_weight: float = 1.2,
        seed: Optional[int] = None,
        round_type_schedule: Optional[list[str]] = None,
        difficulty: str = "medium",
    ) -> None:
        self._env = ACEEnv(
            num_rounds=num_rounds,
            inference_weight=inference_weight,
            seed=seed,
            round_type_schedule=round_type_schedule,
            difficulty=difficulty,
        )

    def reset(self) -> dict[str, Any]:
        return self._env.reset()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self._env.step(action)

    def state(self) -> dict[str, Any]:
        return self._env.state()


class ACEOpenMultiAgentEnv(Environment):
    """
    OpenEnv adapter around the ACE++ multi-agent environment.

    The action is a list of JSON strings, one per agent. This keeps the wrapper
    close to the Gym/OpenEnv surface while exposing coalition, trust, and
    multi-agent reward dynamics for training/evaluation scripts.
    """

    def __init__(
        self,
        *,
        num_agents: int = 3,
        num_rounds: int = 10,
        inference_weight: float = 1.2,
        social_weight: float = 0.2,
        seed: Optional[int] = None,
        round_type_schedule: Optional[list[str]] = None,
        id_shuffle: bool = True,
        god_mode: bool = True,
        difficulty: str = "medium",
        adaptation_weight: float = 0.1,
    ) -> None:
        self._env = MultiAgentACEEnv(
            num_agents=num_agents,
            num_rounds=num_rounds,
            inference_weight=inference_weight,
            social_weight=social_weight,
            seed=seed,
            round_type_schedule=round_type_schedule,
            id_shuffle=id_shuffle,
            god_mode=god_mode,
            difficulty=difficulty,
            adaptation_weight=adaptation_weight,
        )

    def reset(self) -> dict[str, Any]:
        return self._env.reset()

    def step(self, actions: list[str]) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
        return self._env.step(actions)

    def state(self) -> dict[str, Any]:
        return self._env.state()
