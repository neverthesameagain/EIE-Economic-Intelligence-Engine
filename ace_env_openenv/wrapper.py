"""
OpenEnv-compatible wrapper for ACE++.

The repo's `ACEEnv` already exposes the required `reset`, `step`, and `state`
methods. This module provides a thin adapter with a stable import path and a
fallback `Environment` base class so the wrapper is usable even when the
external OpenEnv package is not installed locally.
"""

from __future__ import annotations

from typing import Any, Optional

from env import ACEEnv

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
