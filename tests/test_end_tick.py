import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty, Table
from restaurant_rl.environment import RestaurantEnv


def test_end_tick_advances_exactly_once_and_returns_valid_state() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    original_tick = env.current_tick

    observation, reward, terminated, truncated, info = env.step(
        config.END_TICK_ACTION_ID
    )

    assert env.current_tick == original_tick + 1
    assert observation.shape == (config.OBSERVATION_DIMENSION,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["action_mask"].shape == (config.ACTION_COUNT,)
    assert info["action_mask"].sum() >= 1


def test_table_countdown_and_release_never_go_below_zero() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.tables[0] = Table(1, 2, assigned_party_id=(0, 0), remaining_ticks=2)
    env.tables[1] = Table(2, 2, assigned_party_id=(0, 1), remaining_ticks=1)

    env.step(config.END_TICK_ACTION_ID)

    assert env.tables[0].assigned_party_id == (0, 0)
    assert env.tables[0].remaining_ticks == 1
    assert env.tables[1].assigned_party_id is None
    assert env.tables[1].remaining_ticks == 0

    env.step(config.END_TICK_ACTION_ID)

    assert env.tables[0].assigned_party_id is None
    assert all(table.remaining_ticks >= 0 for table in env.tables)


def test_existing_queue_party_waits_once_and_is_charged() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    party = CustomerParty(party_id=(0, 0), size=3)
    env.waiting_queue.append(party)

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert party.waiting_ticks == 1
    assert info["waited_persons"] == 3
    assert info["abandoned_persons"] == 0
    assert info["balked_persons"] == 0
    assert reward == pytest.approx(config.WAITED_PERSON_REWARD * 3)


def test_new_arrivals_have_zero_wait_and_no_waiting_charge() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=6)

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer
    assert all(party.waiting_ticks == 0 for party in env.arrival_buffer)
    assert info["waited_persons"] == 0
    assert reward == 0.0


def test_seating_remains_zero_time_and_zero_reward() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.arrival_buffer.append(CustomerParty(party_id=(0, 0), size=1))
    original_tick = env.current_tick

    _, reward, _, _, _ = env.step(0)

    assert env.current_tick == original_tick
    assert reward == config.SEATING_ASSIGNMENT_REWARD
