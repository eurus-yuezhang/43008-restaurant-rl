"""Small observation-and-mask-only seating baselines."""

import numpy as np

from .config import (
    ACTION_TO_SEATING,
    END_TICK_ACTION_ID,
    PARTY_SIZE_NORMALIZATION_DIVISOR,
    QUEUE_SLOT_COUNT,
)


def _party_order(observation: np.ndarray, party_size: int) -> int:
    """Queue slots already sort by true wait, including waits clipped at 24."""
    for slot in range(QUEUE_SLOT_COUNT):
        # Queue slots follow the 12 table features as (size, wait) pairs.
        encoded_size = observation[12 + 2 * slot]
        if round(float(encoded_size) * PARTY_SIZE_NORMALIZATION_DIVISOR) == party_size:
            return slot
    return QUEUE_SLOT_COUNT  # Valid action then represents a zero-wait buffer party.


def choose_fifo_action(observation: np.ndarray, action_mask: np.ndarray) -> int:
    """Prefer the earliest queued party, then smaller tables and lower action IDs."""
    seating = [action for action in ACTION_TO_SEATING if action_mask[action]]
    if not seating:
        return END_TICK_ACTION_ID
    return min(
        seating,
        key=lambda action: (
            _party_order(observation, ACTION_TO_SEATING[action][0]),
            ACTION_TO_SEATING[action][1],
            action,
        ),
    )


def choose_best_fit_action(observation: np.ndarray, action_mask: np.ndarray) -> int:
    """Minimise spare seats, then use queue order and action ID to break ties."""
    seating = [action for action in ACTION_TO_SEATING if action_mask[action]]
    if not seating:
        return END_TICK_ACTION_ID
    return min(
        seating,
        key=lambda action: (
            ACTION_TO_SEATING[action][1] - ACTION_TO_SEATING[action][0],
            _party_order(observation, ACTION_TO_SEATING[action][0]),
            action,
        ),
    )


def choose_random_valid_action(
    observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
) -> int:
    """Choose uniformly among legal actions, including END_TICK, with the supplied generator."""
    return int(rng.choice(np.flatnonzero(action_mask)))
