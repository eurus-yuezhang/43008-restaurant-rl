import numpy as np

from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv


def assert_valid_live_state(
    env: RestaurantEnv,
    observation: np.ndarray,
    reward: float,
    info: dict,
) -> None:
    live_ids = [party.party_id for party in env.waiting_queue]
    live_ids.extend(party.party_id for party in env.arrival_buffer)
    live_ids.extend(
        table.assigned_party_id
        for table in env.tables
        if table.assigned_party_id is not None
    )

    assert len(env.waiting_queue) <= config.QUEUE_CAPACITY
    assert len(live_ids) == len(set(live_ids))
    assert all(table.remaining_ticks >= 0 for table in env.tables)
    assert observation.shape == (config.OBSERVATION_DIMENSION,)
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert info["action_mask"].shape == (config.ACTION_COUNT,)
    assert info["action_mask"].sum() >= 1
    assert info["action_mask"][config.END_TICK_ACTION_ID]


def test_full_arrival_window_smoke_sequence_preserves_invariants() -> None:
    env = RestaurantEnv(meal="dinner", demand_scenario="super_peak")
    observation, info = env.reset(seed=707)
    assert_valid_live_state(env, observation, 0.0, info)

    for step_index in range(env.meal_config.arrival_window_ticks + 3):
        mask = env._get_action_mask()
        valid_seating_actions = [
            action_id
            for action_id in config.ACTION_TO_SEATING
            if mask[action_id]
        ]
        if step_index % 2 == 0 and valid_seating_actions:
            observation, reward, _, _, info = env.step(valid_seating_actions[0])
            assert_valid_live_state(env, observation, reward, info)

        observation, reward, terminated, truncated, info = env.step(
            config.END_TICK_ACTION_ID
        )
        assert not truncated
        assert_valid_live_state(env, observation, reward, info)
        if terminated:
            break

    assert not env.arrivals_open
