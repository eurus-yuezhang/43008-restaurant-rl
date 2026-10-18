import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv


def seated_duration(env: RestaurantEnv) -> int:
    return next(
        table.remaining_ticks for table in env.tables if table.is_occupied
    )


def test_same_seed_and_party_have_same_duration_across_environments() -> None:
    first_env = RestaurantEnv(meal="dinner", demand_scenario="normal")
    second_env = RestaurantEnv(meal="dinner", demand_scenario="peak")
    for env in (first_env, second_env):
        env.reset(seed=1234)
        env.arrival_buffer.append(CustomerParty(party_id=(8, 2), size=4))
        env.step(5)

    assert seated_duration(first_env) == seated_duration(second_env)


def test_duration_does_not_depend_on_compatible_table_choice() -> None:
    two_seat_env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    four_seat_env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    for env in (two_seat_env, four_seat_env):
        env.reset(seed=88)
        env.arrival_buffer.append(CustomerParty(party_id=(4, 1), size=1))

    two_seat_env.step(0)
    four_seat_env.step(1)

    assert seated_duration(two_seat_env) == seated_duration(four_seat_env)


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
@pytest.mark.parametrize("party_size", range(1, 9))
def test_keyed_duration_is_within_supported_range(
    meal: str,
    party_size: int,
) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=party_size)
    party = CustomerParty(party_id=(10, party_size), size=party_size)

    duration = env._get_dining_duration(party)

    assert 0 < duration <= config.MAX_TABLE_REMAINING_TICKS
