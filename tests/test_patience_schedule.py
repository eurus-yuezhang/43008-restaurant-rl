import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.randomness import stable_rng


EXPECTED_HAZARDS = {
    "lunch": (
        (0, 2, 0.00),
        (2, 3, 0.03),
        (3, 4, 0.08),
        (4, 6, 0.15),
        (6, 9, 0.30),
        (9, None, 0.50),
    ),
    "dinner": (
        (0, 3, 0.00),
        (3, 6, 0.02),
        (6, 9, 0.05),
        (9, 12, 0.10),
        (12, 15, 0.18),
        (15, 18, 0.28),
        (18, None, 0.40),
    ),
}

EXPECTED_TIME_MULTIPLIERS = {
    "lunch": (
        (0, 12, 0.8),
        (12, 24, 1.0),
        (24, 36, 1.3),
        (36, None, 1.6),
    ),
    "dinner": (
        (0, 24, 0.8),
        (24, 36, 1.0),
        (36, 48, 1.6),
        (48, None, 2.0),
    ),
}

EXPECTED_CAPS = {"lunch": 0.65, "dinner": 0.60}


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_canonical_hazard_and_multiplier_tables_are_exact(meal: str) -> None:
    meal_config = config.MEAL_CONFIGS[meal]

    assert meal_config.abandonment_hazard == EXPECTED_HAZARDS[meal]
    assert (
        meal_config.abandonment_time_multipliers
        == EXPECTED_TIME_MULTIPLIERS[meal]
    )
    assert meal_config.abandonment_probability_cap == EXPECTED_CAPS[meal]
    assert all(
        current[1] == following[0]
        for current, following in zip(
            meal_config.abandonment_hazard,
            meal_config.abandonment_hazard[1:],
        )
    )
    assert all(
        current[1] == following[0]
        for current, following in zip(
            meal_config.abandonment_time_multipliers,
            meal_config.abandonment_time_multipliers[1:],
        )
    )
    assert all(
        current[2] <= following[2]
        for current, following in zip(
            meal_config.abandonment_hazard,
            meal_config.abandonment_hazard[1:],
        )
    )


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_all_waiting_band_boundaries_are_left_closed_right_open(
    meal: str,
) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.opening_tick_of_day
    opening_multiplier = EXPECTED_TIME_MULTIPLIERS[meal][0][2]

    for start, end, base_hazard in EXPECTED_HAZARDS[meal]:
        boundary_ticks = [start]
        if end is not None and end - 1 != start:
            boundary_ticks.append(end - 1)
        if end is None:
            boundary_ticks.append(start + 100)

        for waiting_ticks in boundary_ticks:
            party = CustomerParty(
                party_id=(0, waiting_ticks),
                size=1,
                waiting_ticks=waiting_ticks,
            )
            expected = min(
                base_hazard * opening_multiplier,
                env.meal_config.abandonment_probability_cap,
            )
            assert env._get_abandonment_probability(party) == pytest.approx(
                expected
            )


@pytest.mark.parametrize(
    ("meal", "waiting_ticks", "base_hazard"),
    (("lunch", 4, 0.15), ("dinner", 9, 0.10)),
)
def test_all_meal_time_multiplier_boundaries(
    meal: str,
    waiting_ticks: int,
    base_hazard: float,
) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=0)
    party = CustomerParty(
        party_id=(0, 0),
        size=1,
        waiting_ticks=waiting_ticks,
    )

    for start, end, multiplier in EXPECTED_TIME_MULTIPLIERS[meal]:
        offsets = [start]
        if end is not None and end - 1 != start:
            offsets.append(end - 1)
        if end is None:
            offsets.append(start + 100)

        for offset in offsets:
            env.current_tick = env.meal_config.opening_tick_of_day + offset
            assert env._get_abandonment_probability(party) == pytest.approx(
                base_hazard * multiplier
            )


@pytest.mark.parametrize(
    ("meal", "waiting_ticks", "time_offset", "uncapped", "expected_cap"),
    (
        ("lunch", 9, 36, 0.50 * 1.6, 0.65),
        ("dinner", 18, 48, 0.40 * 2.0, 0.60),
    ),
)
def test_hazard_probability_is_capped(
    meal: str,
    waiting_ticks: int,
    time_offset: int,
    uncapped: float,
    expected_cap: float,
) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.opening_tick_of_day + time_offset
    party = CustomerParty(
        party_id=(0, 0),
        size=1,
        waiting_ticks=waiting_ticks,
    )

    probability = env._get_abandonment_probability(party)

    assert uncapped > expected_cap
    assert probability == expected_cap


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_zero_hazard_is_exact_and_cannot_remove_a_party(meal: str) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    party = CustomerParty(party_id=(0, 0), size=4, waiting_ticks=0)

    for seed in range(100):
        env.reset(seed=seed)
        env.waiting_queue = [party]
        assert env._get_abandonment_probability(party) == 0.0
        assert env._apply_abandonment() == 0
        assert env.waiting_queue == [party]


def test_abandonment_key_changes_with_updated_waiting_tick() -> None:
    first = stable_rng(7, 10, 2, 4, "abandon").random()
    repeated = stable_rng(7, 10, 2, 4, "abandon").random()
    following_tick = stable_rng(7, 10, 2, 5, "abandon").random()

    assert first == repeated
    assert first != following_tick


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_probability_sanity_sweep(meal: str) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=0)
    cap = env.meal_config.abandonment_probability_cap

    for waiting_ticks in range(61):
        party = CustomerParty(
            party_id=(0, waiting_ticks),
            size=1,
            waiting_ticks=waiting_ticks,
        )
        for time_offset in range(env.meal_config.arrival_window_ticks + 25):
            env.current_tick = env.meal_config.opening_tick_of_day + time_offset
            probability = env._get_abandonment_probability(party)
            assert np.isfinite(probability)
            assert 0.0 <= probability <= cap <= 1.0
