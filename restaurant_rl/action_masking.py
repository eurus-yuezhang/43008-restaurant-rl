"""Observation-only action masks and the thin MaskablePPO adapter."""

import gymnasium as gym
import numpy as np

from .config import ACTION_COUNT, ACTION_TO_SEATING, END_TICK_ACTION_ID, TABLE_CAPACITIES


def action_mask_from_observation(observation: np.ndarray) -> np.ndarray:
    """Reconstruct valid actions from one observation or an observation batch."""
    observations = np.asarray(observation)
    single = observations.ndim == 1
    # Treat a single observation as a batch internally and restore its shape on return.
    if single:
        observations = observations[None, :]

    masks = np.zeros((len(observations), ACTION_COUNT), dtype=np.bool_)
    # Layout: tables 0-11, queue slots 12-35 as (size / 8, wait) pairs, arrival buffer 36-43.
    queue_sizes = np.rint(observations[:, 12:36:2] * 8).astype(int)
    buffer_present = observations[:, 36:44] > 0
    capacities = np.asarray(TABLE_CAPACITIES)

    # Combine party presence with table availability for each fixed seating action.
    for action, (party_size, table_capacity) in ACTION_TO_SEATING.items():
        has_party = np.any(queue_sizes == party_size, axis=1) | buffer_present[:, party_size - 1]
        has_table = np.any(
            (capacities == table_capacity) & (observations[:, :12] == 0), axis=1
        )
        masks[:, action] = has_party & has_table
    # Advancing time remains legal even when seating is possible.
    masks[:, END_TICK_ACTION_ID] = True
    return masks[0] if single else masks


class ActionMaskWrapper(gym.Wrapper):
    """Expose observation-derived masks and reject invalid actions externally."""

    def __init__(self, env):
        """Track the latest observation and action/episode checks for the wrapped environment."""
        super().__init__(env)
        self.observation = None
        self.invalid_action_count = 0
        self.terminated_episodes = 0
        self.truncated_episodes = 0

    def reset(self, **kwargs):
        """Cache the opening observation so PPO can request its first action mask."""
        self.observation, info = self.env.reset(**kwargs)
        return self.observation, info

    def action_masks(self):
        """Return the legality mask for the cached observation in the PPO wrapper interface."""
        if self.observation is None:
            raise RuntimeError("reset() must be called before action_masks()")
        return action_mask_from_observation(self.observation)

    def step(self, action):
        """Reject an illegal action, then cache the next observation and count episode endings."""
        if not self.action_masks()[int(action)]:
            self.invalid_action_count += 1
            raise AssertionError(f"invalid masked action: {action}")
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.terminated_episodes += int(terminated)
        self.truncated_episodes += int(truncated)
        return self.observation, reward, terminated, truncated, info
