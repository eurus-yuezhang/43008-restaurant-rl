import pytest

from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv


def find_seed_with_arrivals(
    meal: str,
    scenario: str,
    interval_tick: int,
) -> int:
    env = RestaurantEnv(meal=meal, demand_scenario=scenario)
    for seed in range(1000):
        env.reset(seed=seed)
        if env._generate_arrivals(interval_tick):
            return seed
    raise AssertionError("no deterministic arrival seed found")


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_first_interval_can_generate_arrivals(meal: str) -> None:
    reference = config.MEAL_CONFIGS[meal]
    first_interval = reference.opening_tick_of_day
    seed = find_seed_with_arrivals(meal, "normal", first_interval)
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=seed)

    env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer
    assert all(
        party.party_id[0] == first_interval for party in env.arrival_buffer
    )
    assert env.arrivals_open


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_final_interval_arrives_then_later_intervals_are_empty(meal: str) -> None:
    reference = config.MEAL_CONFIGS[meal]
    final_interval = reference.closing_tick_of_day - 1
    seed = find_seed_with_arrivals(meal, "normal", final_interval)
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=seed)
    env.current_tick = final_interval

    env.step(config.END_TICK_ACTION_ID)
    final_ids = {party.party_id for party in env.arrival_buffer}

    assert final_ids
    assert all(party_id[0] == final_interval for party_id in final_ids)
    assert not env.arrivals_open

    env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer == []
    assert not env.arrivals_open
