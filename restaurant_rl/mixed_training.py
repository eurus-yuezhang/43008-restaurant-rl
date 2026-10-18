"""Minimal reproducible mixed-scenario Dinner training environment."""

import gymnasium as gym
import numpy as np

from .action_masking import ActionMaskWrapper
from .config import DEMAND_SCENARIO_MULTIPLIERS
from .environment import RestaurantEnv


SCENARIOS = tuple(DEMAND_SCENARIO_MULTIPLIERS)


class MixedDinnerEnv(gym.Env):
    """Choose one existing Dinner scenario uniformly for each new episode."""

    def __init__(self, seed: int):
        """Create separate scenario and episode-seed streams and initialise training checks."""
        super().__init__()
        template = RestaurantEnv(meal="dinner", demand_scenario=SCENARIOS[0])
        self.observation_space = template.observation_space
        self.action_space = template.action_space
        # Keep scenario selection independent of the stream that supplies episode seeds.
        self._scenario_rng = np.random.default_rng(seed)
        self._episode_seed_rng = np.random.default_rng(seed + 1)
        self.current_env = None
        self.current_scenario = None
        self.scenario_episode_counts = {scenario: 0 for scenario in SCENARIOS}
        self.invalid_action_count = 0
        self.terminated_episodes = 0
        self.truncated_episodes = 0
        self.finite = True

    def reset(self, *, seed=None, options=None):
        """Draw the next scenario and episode seed, then reset a fresh masked Dinner environment."""
        super().reset(seed=seed)
        # These private streams continue across resets; reset(seed=...) only seeds Gym's own RNG.
        self.current_scenario = str(self._scenario_rng.choice(SCENARIOS))
        episode_seed = int(self._episode_seed_rng.integers(0, np.iinfo(np.int32).max))
        # Wrap a new environment so its masks and state belong to the selected scenario.
        self.current_env = ActionMaskWrapper(
            RestaurantEnv(meal="dinner", demand_scenario=self.current_scenario)
        )
        observation, info = self.current_env.reset(seed=episode_seed)
        self.scenario_episode_counts[self.current_scenario] += 1
        info["training_scenario"] = self.current_scenario
        return observation, info

    def action_masks(self):
        """Forward the current episode's legal actions to the training algorithm."""
        if self.current_env is None:
            raise RuntimeError("reset() must be called before action_masks()")
        return self.current_env.action_masks()

    def step(self, action):
        """Apply a legal action and track finite outputs and natural versus truncated endings."""
        if not self.action_masks()[int(action)]:
            self.invalid_action_count += 1
            raise AssertionError(f"invalid masked action: {action}")
        observation, reward, terminated, truncated, info = self.current_env.step(action)
        # Keep a cumulative validity check across episodes before counting their endings.
        self.finite &= bool(np.isfinite(observation).all() and np.isfinite(reward))
        if not self.finite:
            raise AssertionError("non-finite training observation or reward")
        self.terminated_episodes += int(terminated)
        self.truncated_episodes += int(truncated)
        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the active episode environment if one has been created."""
        if self.current_env is not None:
            self.current_env.close()
