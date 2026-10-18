import pytest

from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv


def find_seed_with_exact_arrival_count(
    meal: str,
    interval_tick: int,
    expected_count: int,
) -> int:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    for seed in range(10000):
        env.reset(seed=seed)
        if len(env._generate_arrivals(interval_tick)) == expected_count:
            return seed
    raise AssertionError("no deterministic arrival seed found")


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_final_arrival_buffer_prevents_early_termination(meal: str) -> None:
    meal_config = config.MEAL_CONFIGS[meal]
    final_interval = meal_config.closing_tick_of_day - 1
    seed = find_seed_with_exact_arrival_count(meal, final_interval, 1)
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=seed)
    env.current_tick = final_interval

    _, _, terminated, truncated, _ = env.step(config.END_TICK_ACTION_ID)

    assert not env.arrivals_open
    assert len(env.arrival_buffer) == 1
    assert not terminated
    assert not truncated

    party = env.arrival_buffer[0]
    table_capacity = config.PARTY_TABLE_COMPATIBILITY[party.size][0]
    action = config.SEATING_TO_ACTION[(party.size, table_capacity)]
    _, reward, terminated, truncated, _ = env.step(action)

    assert env.arrival_buffer == []
    assert any(
        table.assigned_party_id == party.party_id for table in env.tables
    )
    assert reward == config.SEATING_ASSIGNMENT_REWARD
    assert terminated
    assert not truncated


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_empty_final_interval_can_terminate_at_closing(meal: str) -> None:
    meal_config = config.MEAL_CONFIGS[meal]
    final_interval = meal_config.closing_tick_of_day - 1
    seed = find_seed_with_exact_arrival_count(meal, final_interval, 0)
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=seed)
    env.current_tick = final_interval

    _, reward, terminated, truncated, _ = env.step(
        config.END_TICK_ACTION_ID
    )

    assert env.current_tick == meal_config.closing_tick_of_day
    assert not env.arrivals_open
    assert env.waiting_queue == []
    assert env.arrival_buffer == []
    assert reward == config.END_TICK_FIXED_REWARD
    assert terminated
    assert not truncated
