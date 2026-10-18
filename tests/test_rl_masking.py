"""Checks for observation-derived masks and masked policies."""

import numpy as np
import pytest
import torch as th

from restaurant_rl.action_masking import ActionMaskWrapper, action_mask_from_observation
from restaurant_rl.config import END_TICK_ACTION_ID
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.masked_dqn import MaskedDQN


@pytest.mark.parametrize("seed", range(20001, 20006))
def test_observation_mask_matches_environment_for_complete_episode(seed):
    env = RestaurantEnv(meal="dinner", demand_scenario="normal")
    observation, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        derived = action_mask_from_observation(observation)
        assert np.array_equal(derived, info["action_mask"])
        assert derived[END_TICK_ACTION_ID]
        action = int(rng.choice(np.flatnonzero(derived)))
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert terminated and not truncated
            break
    else:
        pytest.fail("episode did not finish")


def test_wrapper_returns_observation_mask():
    env = ActionMaskWrapper(RestaurantEnv(meal="dinner", demand_scenario="normal"))
    observation, info = env.reset(seed=20001)
    assert np.array_equal(env.action_masks(), action_mask_from_observation(observation))
    assert np.array_equal(env.action_masks(), info["action_mask"])


@pytest.fixture(scope="module")
def dqn_and_state():
    env = ActionMaskWrapper(RestaurantEnv(meal="dinner", demand_scenario="normal"))
    observation, info = env.reset(seed=20001)
    while info["action_mask"].sum() == 1:
        observation, _, _, _, info = env.step(END_TICK_ACTION_ID)
    model = MaskedDQN("MlpPolicy", env, buffer_size=10, learning_starts=1, seed=20001)
    return model, observation, info["action_mask"]


def test_masked_greedy_dqn_selects_valid_action(dqn_and_state):
    model, observation, mask = dqn_and_state
    action, _ = model.predict(observation, deterministic=True)
    assert mask[int(action)]


def test_masked_random_dqn_exploration_selects_only_valid_actions(dqn_and_state):
    model, observation, mask = dqn_and_state
    model.exploration_rate = 1.0
    actions = [int(model.predict(observation, deterministic=False)[0]) for _ in range(100)]
    assert all(mask[action] for action in actions)


def test_dqn_target_network_excludes_invalid_next_actions(dqn_and_state):
    model, observation, mask = dqn_and_state
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        target_q = model.q_net_target(observation_tensor)[0]
    assert th.isfinite(target_q[th.as_tensor(mask)]).all()
    assert th.isneginf(target_q[~th.as_tensor(mask)]).all()
