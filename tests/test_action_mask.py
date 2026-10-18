import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty, Table
from restaurant_rl.environment import RestaurantEnv


def test_reset_mask_contains_only_end_tick() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    _, info = env.reset(seed=1)
    mask = info["action_mask"]

    assert mask.shape == (12,)
    assert mask.dtype == np.bool_
    assert mask.sum() == 1
    assert mask[config.END_TICK_ACTION_ID]


def test_matching_party_and_table_make_only_compatible_actions_valid() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=1)
    env.arrival_buffer.append(CustomerParty(party_id=(0, 0), size=3))

    mask = env._get_action_mask()

    assert mask[4]
    assert not mask[2]
    assert not mask[8]
    assert mask[config.END_TICK_ACTION_ID]
    assert mask.sum() >= 1


def test_action_is_invalid_without_a_free_required_capacity() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=1)
    env.arrival_buffer.append(CustomerParty(party_id=(0, 0), size=7))
    env.tables[11] = Table(
        table_id=12,
        capacity=8,
        assigned_party_id=(9, 0),
        remaining_ticks=4,
    )

    mask = env._get_action_mask()

    assert not mask[9]
    assert mask[config.END_TICK_ACTION_ID]


def test_action_becomes_invalid_after_only_opportunity_is_consumed() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=1)
    env.arrival_buffer.append(CustomerParty(party_id=(0, 0), size=3))

    _, _, _, _, info = env.step(4)

    assert not info["action_mask"][4]
    assert info["action_mask"][config.END_TICK_ACTION_ID]


def test_illegal_action_without_matching_party_does_not_mutate_state() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=1)
    original_tick = env.current_tick
    original_tables = [table.__dict__.copy() for table in env.tables]

    with pytest.raises(ValueError, match="illegal"):
        env.step(0)

    assert env.current_tick == original_tick
    assert [table.__dict__ for table in env.tables] == original_tables
    assert env.waiting_queue == []
    assert env.arrival_buffer == []


def test_illegal_action_without_free_table_does_not_mutate_state() -> None:
    env = RestaurantEnv(meal="dinner", demand_scenario="normal")
    env.reset(seed=1)
    party = CustomerParty(party_id=(0, 0), size=7)
    env.arrival_buffer.append(party)
    env.tables[11] = Table(
        table_id=12,
        capacity=8,
        assigned_party_id=(8, 0),
        remaining_ticks=5,
    )

    with pytest.raises(ValueError, match="illegal"):
        env.step(9)

    assert env.arrival_buffer == [party]
    assert env.tables[11].assigned_party_id == (8, 0)
    assert env.tables[11].remaining_ticks == 5
