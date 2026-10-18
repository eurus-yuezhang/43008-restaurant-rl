import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.randomness import keyed_dining_multiplier, stable_rng


CONTEXTS = tuple(
    (meal, scenario)
    for meal in config.MEAL_CONFIGS
    for scenario in config.DEMAND_SCENARIO_MULTIPLIERS
)


def buffer_signature(env: RestaurantEnv) -> tuple[tuple[tuple[int, int], int], ...]:
    return tuple((party.party_id, party.size) for party in env.arrival_buffer)


def base_rate_for_interval(env: RestaurantEnv, interval_tick: int) -> float:
    offset = interval_tick - env.meal_config.opening_tick_of_day
    return next(
        rate
        for start, end, rate in env.meal_config.base_arrival_curve
        if start <= offset and (end is None or offset < end)
    )


@pytest.mark.parametrize(("meal", "scenario"), CONTEXTS)
def test_full_arrival_window_repeats_for_same_seed(
    meal: str,
    scenario: str,
) -> None:
    first = RestaurantEnv(meal=meal, demand_scenario=scenario)
    second = RestaurantEnv(meal=meal, demand_scenario=scenario)
    first.reset(seed=2026)
    second.reset(seed=2026)
    first_stream = []
    second_stream = []

    for _ in range(first.meal_config.arrival_window_ticks):
        first.step(config.END_TICK_ACTION_ID)
        second.step(config.END_TICK_ACTION_ID)
        first_stream.append(buffer_signature(first))
        second_stream.append(buffer_signature(second))

    assert first_stream == second_stream
    assert first.current_tick == first.meal_config.closing_tick_of_day
    assert second.current_tick == second.meal_config.closing_tick_of_day
    assert not first.arrivals_open
    assert not second.arrivals_open


@pytest.mark.parametrize(("meal", "scenario"), CONTEXTS)
def test_every_interval_uses_configured_rate_and_size_distribution(
    meal: str,
    scenario: str,
) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario=scenario)
    episode_seed = 314
    env.reset(seed=episode_seed)

    for interval_tick in range(
        env.meal_config.opening_tick_of_day,
        env.meal_config.closing_tick_of_day,
    ):
        arrival_rate = (
            base_rate_for_interval(env, interval_tick)
            * config.DEMAND_SCENARIO_MULTIPLIERS[scenario]
        )
        expected_count = int(
            stable_rng(
                episode_seed,
                meal,
                scenario,
                interval_tick,
                "arrival_count",
            ).poisson(arrival_rate)
        )
        arrivals = env._generate_arrivals(interval_tick)

        assert len(arrivals) == expected_count
        assert [party.party_id[1] for party in arrivals] == list(
            range(expected_count)
        )
        for arrival_index, party in enumerate(arrivals):
            expected_size = int(
                stable_rng(
                    episode_seed,
                    meal,
                    scenario,
                    interval_tick,
                    arrival_index,
                    "party_size",
                ).choice(
                    config.SUPPORTED_PARTY_SIZES,
                    p=env.meal_config.party_size_probabilities,
                )
            )
            assert party.size == expected_size


def test_policy_actions_do_not_change_full_arrival_stream() -> None:
    seating_env = RestaurantEnv(meal="dinner", demand_scenario="peak")
    passive_env = RestaurantEnv(meal="dinner", demand_scenario="peak")
    seating_env.reset(seed=808)
    passive_env.reset(seed=808)
    seating_stream = []
    passive_stream = []
    seating_actions = 0

    for _ in range(seating_env.meal_config.arrival_window_ticks):
        seating_env.step(config.END_TICK_ACTION_ID)
        passive_env.step(config.END_TICK_ACTION_ID)
        seating_stream.append(buffer_signature(seating_env))
        passive_stream.append(buffer_signature(passive_env))

        mask = seating_env._get_action_mask()
        valid_actions = [
            action_id
            for action_id in config.ACTION_TO_SEATING
            if mask[action_id]
        ]
        if valid_actions:
            seating_env.step(valid_actions[0])
            seating_actions += 1

    assert seating_stream == passive_stream
    assert seating_actions > 0


def test_keyed_arrivals_are_stable_across_python_processes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
import json
from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv

env = RestaurantEnv(meal="dinner", demand_scenario="peak")
env.reset(seed=404)
batches = []
for _ in range(12):
    env.step(config.END_TICK_ACTION_ID)
    batches.append([(party.party_id, party.size) for party in env.arrival_buffer])
print(json.dumps(batches))
"""

    first = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
    )
    second = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
    )

    assert first == second
    assert any(json.loads(first))


def test_random_event_labels_are_explicit_and_distinct() -> None:
    labels = (
        "dining_duration",
        "abandon",
        "arrival_count",
        "party_size",
        "overflow",
    )
    source = inspect.getsource(RestaurantEnv) + inspect.getsource(
        keyed_dining_multiplier
    )
    draws = [stable_rng(1, "same_context", label).random() for label in labels]

    assert all(f'"{label}"' in source for label in labels)
    assert len(set(draws)) == len(labels)
