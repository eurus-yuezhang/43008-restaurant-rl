import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv


@pytest.mark.parametrize(
    ("action_id", "party_size", "table_capacity"),
    tuple(
        (action_id, party_size, table_capacity)
        for action_id, (party_size, table_capacity)
        in config.ACTION_TO_SEATING.items()
    ),
)
def test_all_seating_actions(
    action_id: int,
    party_size: int,
    table_capacity: int,
) -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=123)
    party = CustomerParty(party_id=(0, action_id), size=party_size)
    env.arrival_buffer.append(party)
    original_tick = env.current_tick
    expected_table_id = min(
        table.table_id
        for table in env.tables
        if table.capacity == table_capacity
    )

    observation, reward, terminated, truncated, info = env.step(action_id)
    selected_table = next(
        table for table in env.tables if table.assigned_party_id == party.party_id
    )

    assert selected_table.capacity == table_capacity
    assert selected_table.table_id == expected_table_id
    assert selected_table.assigned_party_id == party.party_id
    assert 0 < selected_table.remaining_ticks <= config.MAX_TABLE_REMAINING_TICKS
    assert party not in env.arrival_buffer
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert env.current_tick == original_tick
    assert observation.shape == (51,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert np.array_equal(observation[36:44], np.zeros(8, dtype=np.float32))
    assert info["action_mask"].shape == (12,)
    assert info["action_mask"].sum() >= 1
    assert info["action_mask"][config.END_TICK_ACTION_ID]


def test_queue_party_has_priority_over_same_size_buffer_party() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=7)
    queued = CustomerParty(party_id=(5, 0), size=2, waiting_ticks=3)
    buffered = CustomerParty(party_id=(1, 0), size=2)
    env.waiting_queue.append(queued)
    env.arrival_buffer.append(buffered)

    env.step(2)

    assert env.tables[0].assigned_party_id == queued.party_id
    assert queued not in env.waiting_queue
    assert env.arrival_buffer == [buffered]


def test_longest_waiting_queued_party_is_selected() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=7)
    shorter_wait = CustomerParty(party_id=(1, 0), size=2, waiting_ticks=2)
    longer_wait = CustomerParty(party_id=(2, 0), size=2, waiting_ticks=5)
    env.waiting_queue = [shorter_wait, longer_wait]

    env.step(2)

    assert env.tables[0].assigned_party_id == longer_wait.party_id
    assert env.waiting_queue == [shorter_wait]


def test_lowest_party_id_breaks_equal_queue_wait_tie() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=7)
    later = CustomerParty(party_id=(3, 0), size=2, waiting_ticks=4)
    earlier = CustomerParty(party_id=(2, 1), size=2, waiting_ticks=4)
    env.waiting_queue = [later, earlier]

    env.step(2)

    assert env.tables[0].assigned_party_id == earlier.party_id
    assert env.waiting_queue == [later]


def test_lowest_party_id_is_selected_from_buffer() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=7)
    later = CustomerParty(party_id=(1, 2), size=3)
    earlier = CustomerParty(party_id=(1, 0), size=3)
    env.arrival_buffer = [later, earlier]

    env.step(4)

    assert env.tables[4].assigned_party_id == earlier.party_id
    assert env.arrival_buffer == [later]


def test_lowest_free_table_id_is_selected_each_time() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=7)
    first = CustomerParty(party_id=(0, 0), size=3)
    second = CustomerParty(party_id=(0, 1), size=3)
    env.arrival_buffer = [first, second]

    env.step(4)
    env.step(4)

    assert env.tables[4].assigned_party_id == first.party_id
    assert env.tables[5].assigned_party_id == second.party_id


def test_observation_updates_after_seating() -> None:
    env = RestaurantEnv(meal="dinner", demand_scenario="peak")
    env.reset(seed=20)
    party = CustomerParty(party_id=(2, 0), size=4, waiting_ticks=6)
    env.waiting_queue.append(party)

    observation, _, _, _, _ = env.step(5)

    assert env.waiting_queue == []
    assert observation[4] == pytest.approx(
        env.tables[4].remaining_ticks / config.MAX_TABLE_REMAINING_TICKS
    )
    assert np.array_equal(observation[12:36], np.zeros(24, dtype=np.float32))
    assert observation.shape == (51,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
