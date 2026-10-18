"""Checks for the mixed-scenario Dinner training environment."""

import numpy as np

from restaurant_rl.action_masking import action_mask_from_observation
from restaurant_rl.config import DEMAND_SCENARIO_MULTIPLIERS
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.mixed_training import MixedDinnerEnv, SCENARIOS


def test_scenario_context_is_existing_one_hot_observation_block():
    for index, scenario in enumerate(DEMAND_SCENARIO_MULTIPLIERS):
        observation, _ = RestaurantEnv("dinner", scenario).reset(seed=30000)
        expected = np.zeros(4, dtype=np.float32)
        expected[index] = 1
        assert np.array_equal(observation[45:49], expected)


def scenario_sequence(seed, count=40):
    env = MixedDinnerEnv(seed)
    return [env.reset()[1]["training_scenario"] for _ in range(count)]


def test_mixed_scenarios_are_reproducible_and_cover_all_four():
    first = scenario_sequence(30000)
    second = scenario_sequence(30000)
    assert first == second
    assert set(first) == set(SCENARIOS)


def test_mixed_scenario_counts_match_resets():
    env = MixedDinnerEnv(30000)
    for _ in range(40):
        env.reset()
    assert sum(env.scenario_episode_counts.values()) == 40
    assert all(env.scenario_episode_counts[scenario] > 0 for scenario in SCENARIOS)


def test_masks_remain_observation_derived_through_complete_mixed_episode():
    env = MixedDinnerEnv(30000)
    observation, info = env.reset()
    rng = np.random.default_rng(30000)
    for _ in range(1000):
        mask = action_mask_from_observation(observation)
        assert np.array_equal(mask, env.action_masks())
        assert np.array_equal(mask, info["action_mask"])
        action = int(rng.choice(np.flatnonzero(mask)))
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert terminated and not truncated
            break
    else:
        raise AssertionError("mixed episode did not terminate")
