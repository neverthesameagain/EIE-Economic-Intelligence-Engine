"""
Backwards-compatible shim.

Older scripts import `ACEEnv` from `ace_env_fixed`; the canonical env now lives
in `env.py`.
"""

from env import ACEEnv, MultiAgentACEEnv

__all__ = ["ACEEnv", "MultiAgentACEEnv"]

