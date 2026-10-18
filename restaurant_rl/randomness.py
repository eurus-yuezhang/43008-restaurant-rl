"""Small stable-key helpers for exogenous restaurant randomness."""

from __future__ import annotations

import hashlib

import numpy as np

from .config import WeightedValue


def stable_rng(*key_parts: object) -> np.random.Generator:
    """Create a NumPy generator from a stable, process-independent key."""

    # Hash explicit event keys so draws do not depend on Python's process hash or call order.
    key = ":".join(str(part) for part in key_parts).encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(stable_seed)


def keyed_dining_multiplier(
    episode_seed: int,
    party_id: tuple[int, int],
    distribution: tuple[WeightedValue, ...],
) -> float:
    """Choose a dining multiplier from a stable party-specific key."""

    random_generator = stable_rng(
        episode_seed,
        party_id[0],
        party_id[1],
        "dining_duration",
    )
    multipliers, probabilities = zip(*distribution)
    return float(random_generator.choice(multipliers, p=probabilities))
