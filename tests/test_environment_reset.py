import gymnasium as gym
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty, Table
from restaurant_rl.environment import RestaurantEnv


@pytest.mark.parametrize(
    ("meal", "opening_tick"),
    (("lunch", 132), ("dinner", 204)),
)
def test_environment_constructs_with_explicit_context(
    meal: str,
    opening_tick: int,
) -> None:
    env = RestaurantEnv(meal, "normal")

    assert isinstance(env, gym.Env)
    assert env.meal == meal
    assert env.demand_scenario == "normal"
    assert env.current_tick == opening_tick
    assert env.action_space.n == 12
    assert env.observation_space.shape == (51,)


def test_environment_requires_meal_and_demand_scenario() -> None:
    with pytest.raises(TypeError):
        RestaurantEnv()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"meal": "breakfast", "demand_scenario": "normal"}, "meal"),
        ({"meal": "lunch", "demand_scenario": "extreme"}, "demand_scenario"),
    ),
)
def test_environment_rejects_unknown_context(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RestaurantEnv(**kwargs)


def test_reset_returns_canonical_opening_state() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="peak")
    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
    assert info["meal"] == "lunch"
    assert info["demand_scenario"] == "peak"
    assert info["action_mask"].shape == (config.ACTION_COUNT,)
    assert info["action_mask"][config.END_TICK_ACTION_ID]
    assert env.current_tick == config.LUNCH_CONFIG.opening_tick_of_day
    assert env.arrivals_open
    assert len(env.tables) == config.TABLE_COUNT
    assert [
        (table.table_id, table.capacity)
        for table in env.tables
    ] == list(zip(config.TABLE_IDS, config.TABLE_CAPACITIES))
    assert all(not table.is_occupied for table in env.tables)
    assert all(table.assigned_party_id is None for table in env.tables)
    assert all(table.remaining_ticks == 0 for table in env.tables)
    assert env.waiting_queue == []
    assert env.arrival_buffer == []


def test_repeated_reset_clears_manually_altered_transient_state() -> None:
    env = RestaurantEnv(meal="dinner", demand_scenario="off_peak")
    env.reset(seed=1)

    env.tables[0] = Table(
        table_id=1,
        capacity=2,
        assigned_party_id=(0, 0),
        remaining_ticks=5,
    )
    env.waiting_queue.append(CustomerParty(party_id=(1, 0), size=2))
    env.arrival_buffer.append(CustomerParty(party_id=(1, 1), size=4))
    env.current_tick += 1

    env.reset(seed=2)

    assert env.current_tick == config.DINNER_CONFIG.opening_tick_of_day
    assert env.arrivals_open
    assert all(not table.is_occupied for table in env.tables)
    assert all(table.remaining_ticks == 0 for table in env.tables)
    assert env.waiting_queue == []
    assert env.arrival_buffer == []


def test_reset_does_not_generate_arrive_seat_or_advance() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="super_peak")

    env.reset(seed=99)

    assert env.current_tick == config.LUNCH_CONFIG.opening_tick_of_day
    assert env.waiting_queue == []
    assert env.arrival_buffer == []
    assert all(table.assigned_party_id is None for table in env.tables)


def test_end_tick_advances_from_opening() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)

    env.step(config.END_TICK_ACTION_ID)

    assert env.current_tick == config.LUNCH_CONFIG.opening_tick_of_day + 1
