import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty, Table
from restaurant_rl.environment import RestaurantEnv


TABLE_SLICE = slice(0, 12)
QUEUE_SLICE = slice(12, 36)
BUFFER_SLICE = slice(36, 44)
PROGRESS_SLICE = slice(44, 45)
SCENARIO_SLICE = slice(45, 49)
MEAL_SLICE = slice(49, 51)


def test_reset_observation_shape_dtype_range_and_block_lengths() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    observation, _ = env.reset(seed=0)

    assert observation.shape == (51,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)
    assert observation[TABLE_SLICE].shape == (12,)
    assert observation[QUEUE_SLICE].shape == (24,)
    assert observation[BUFFER_SLICE].shape == (8,)
    assert observation[PROGRESS_SLICE].shape == (1,)
    assert observation[SCENARIO_SLICE].shape == (4,)
    assert observation[MEAL_SLICE].shape == (2,)
    assert sum(size for _, size in config.OBSERVATION_BLOCK_SIZES) == 51


def test_empty_opening_blocks_and_progress_are_zero() -> None:
    env = RestaurantEnv(meal="dinner", demand_scenario="normal")
    observation, _ = env.reset(seed=0)

    assert np.array_equal(observation[TABLE_SLICE], np.zeros(12, dtype=np.float32))
    assert np.array_equal(observation[QUEUE_SLICE], np.zeros(24, dtype=np.float32))
    assert np.array_equal(observation[BUFFER_SLICE], np.zeros(8, dtype=np.float32))
    assert observation[PROGRESS_SLICE].item() == 0.0


@pytest.mark.parametrize(
    ("scenario", "active_index"),
    tuple(
        (scenario, index)
        for index, scenario in enumerate(config.DEMAND_SCENARIO_MULTIPLIERS)
    ),
)
def test_scenario_one_hot_for_every_context(
    scenario: str,
    active_index: int,
) -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario=scenario)
    observation, _ = env.reset(seed=0)
    expected = np.zeros(4, dtype=np.float32)
    expected[active_index] = 1.0

    assert np.array_equal(observation[SCENARIO_SLICE], expected)
    assert observation[SCENARIO_SLICE].sum() == 1.0


@pytest.mark.parametrize(
    ("meal", "active_index"),
    tuple((meal, index) for index, meal in enumerate(config.MEAL_CONFIGS)),
)
def test_meal_one_hot_for_both_contexts(meal: str, active_index: int) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    observation, _ = env.reset(seed=0)
    expected = np.zeros(2, dtype=np.float32)
    expected[active_index] = 1.0

    assert np.array_equal(observation[MEAL_SLICE], expected)
    assert observation[MEAL_SLICE].sum() == 1.0


def test_known_table_queue_and_buffer_state_encoding() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.tables[0] = Table(
        table_id=1,
        capacity=2,
        assigned_party_id=(0, 0),
        remaining_ticks=18,
    )
    env.waiting_queue = [
        CustomerParty(party_id=(1, 0), size=4, waiting_ticks=12)
    ]
    env.arrival_buffer = [
        CustomerParty(party_id=(2, 0), size=2),
        CustomerParty(party_id=(2, 1), size=2),
        CustomerParty(party_id=(2, 2), size=5),
    ]

    observation = env._get_observation()

    assert observation[0] == pytest.approx(0.5)
    assert observation[12] == pytest.approx(0.5)
    assert observation[13] == pytest.approx(0.5)
    assert observation[37] == pytest.approx(2 / 3)
    assert observation[40] == pytest.approx(0.5)


def test_queue_is_encoded_in_canonical_wait_then_party_id_order() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.waiting_queue = [
        CustomerParty(party_id=(2, 0), size=2, waiting_ticks=5),
        CustomerParty(party_id=(3, 0), size=3, waiting_ticks=10),
        CustomerParty(party_id=(1, 0), size=4, waiting_ticks=10),
    ]

    observation = env._get_observation()
    encoded_sizes = observation[QUEUE_SLICE][0:6:2]

    assert np.allclose(encoded_sizes, np.array([0.5, 0.375, 0.25], dtype=np.float32))


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_progress_uses_meal_window_and_clips_only_at_one(meal: str) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario="normal")
    env.reset(seed=0)

    env.current_tick += env.meal_config.arrival_window_ticks // 2
    halfway = env._get_observation()
    assert halfway[PROGRESS_SLICE].item() == pytest.approx(0.5)

    env.current_tick = env.meal_config.closing_tick_of_day + 5
    after_close = env._get_observation()
    assert after_close[PROGRESS_SLICE].item() == 1.0


def test_out_of_range_state_is_not_silently_clipped() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.tables[0] = Table(
        table_id=1,
        capacity=2,
        assigned_party_id=(0, 0),
        remaining_ticks=config.MAX_TABLE_REMAINING_TICKS + 1,
    )

    with pytest.raises(AssertionError, match="outside observation_space"):
        env._get_observation()
