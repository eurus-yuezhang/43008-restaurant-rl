from collections import Counter
import math

from restaurant_rl import config


EXPECTED_ACTIONS = (
    (1, 2),
    (1, 4),
    (2, 2),
    (2, 4),
    (3, 4),
    (4, 4),
    (4, 6),
    (5, 6),
    (6, 6),
    (7, 8),
    (8, 8),
)


def assert_probability_distribution(probabilities: tuple[float, ...]) -> None:
    assert all(probability >= 0.0 for probability in probabilities)
    assert math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)


def assert_contiguous_bands(
    bands: tuple[config.TickBand, ...],
    expected_final_tick: int | None,
) -> None:
    assert bands[0][0] == 0
    for current, following in zip(bands, bands[1:]):
        assert current[1] == following[0]
    assert bands[-1][1] == expected_final_tick


def test_canonical_restaurant_layout() -> None:
    assert config.TABLE_COUNT == 12
    assert config.TOTAL_SEATS == 50
    assert Counter(config.TABLE_CAPACITIES) == Counter({2: 4, 4: 4, 6: 3, 8: 1})
    assert config.TABLE_IDS == tuple(range(1, 13))


def test_party_and_buffer_structure() -> None:
    assert config.SUPPORTED_PARTY_SIZES == tuple(range(1, 9))
    assert config.QUEUE_CAPACITY == 12
    assert config.ARRIVAL_BUFFER_CAPACITY is None
    assert not config.ARRIVAL_BUFFER_PERSISTS_ACROSS_TICKS


def test_compatibility_and_action_mapping_are_exact_and_bijective() -> None:
    assert config.SEATING_ACTION_PAIRS == EXPECTED_ACTIONS
    assert config.SEATING_ACTION_COUNT == 11
    assert config.END_TICK_ACTION_ID == 11
    assert config.ACTION_COUNT == 12
    assert set(config.ACTION_TO_SEATING) == set(range(11))
    assert len(set(config.ACTION_TO_SEATING.values())) == 11
    assert len(config.SEATING_TO_ACTION) == 11

    for action_id, pair in config.ACTION_TO_SEATING.items():
        party_size, table_capacity = pair
        assert table_capacity in config.PARTY_TABLE_COMPATIBILITY[party_size]
        assert config.SEATING_TO_ACTION[pair] == action_id

    compatibility_pairs = {
        (party_size, table_capacity)
        for party_size, capacities in config.PARTY_TABLE_COMPATIBILITY.items()
        for table_capacity in capacities
    }
    assert compatibility_pairs == set(EXPECTED_ACTIONS)
    assert config.END_TICK_ACTION_ID not in config.ACTION_TO_SEATING


def test_observation_structure_totals_51_features() -> None:
    assert config.OBSERVATION_BLOCK_SIZES == (
        ("tables", 12),
        ("queue", 24),
        ("arrival_buffer", 8),
        ("service_progress", 1),
        ("demand_scenario", 4),
        ("meal", 2),
    )
    assert sum(size for _, size in config.OBSERVATION_BLOCK_SIZES) == 51
    assert config.OBSERVATION_DIMENSION == 51
    assert config.WAITING_TICK_NORMALIZATION_CAP == 24
    assert config.MAX_TABLE_REMAINING_TICKS == 36


def test_demand_scenarios_are_canonical_and_positive() -> None:
    assert config.DEMAND_SCENARIO_MULTIPLIERS == {
        "off_peak": 0.75,
        "normal": 1.00,
        "peak": 1.25,
        "super_peak": 1.50,
    }
    assert all(value > 0.0 for value in config.DEMAND_SCENARIO_MULTIPLIERS.values())


def test_meal_time_and_arrival_curves_are_internally_consistent() -> None:
    assert config.TICK_MINUTES == 5
    assert config.LUNCH_CONFIG.opening_tick_of_day == 132
    assert config.LUNCH_CONFIG.closing_tick_of_day == 174
    assert config.LUNCH_CONFIG.arrival_window_ticks == 42
    assert config.DINNER_CONFIG.opening_tick_of_day == 204
    assert config.DINNER_CONFIG.closing_tick_of_day == 264
    assert config.DINNER_CONFIG.arrival_window_ticks == 60

    for meal in config.MEAL_CONFIGS.values():
        assert meal.opening_tick_of_day < meal.closing_tick_of_day
        assert_contiguous_bands(meal.base_arrival_curve, meal.arrival_window_ticks)
        assert all(rate >= 0.0 for _, _, rate in meal.base_arrival_curve)


def test_party_size_and_dining_multiplier_probabilities_sum_to_one() -> None:
    for meal in config.MEAL_CONFIGS.values():
        assert len(meal.party_size_probabilities) == len(
            config.SUPPORTED_PARTY_SIZES
        )
        assert_probability_distribution(meal.party_size_probabilities)
        assert_probability_distribution(
            tuple(probability for _, probability in meal.dining_multiplier_distribution)
        )
        assert all(multiplier > 0.0 for multiplier, _ in meal.dining_multiplier_distribution)


def test_dining_duration_parameters_are_canonical_ticks() -> None:
    assert config.LUNCH_CONFIG.mean_dining_ticks == (6, 9, 12, 15, 17, 19, 21, 23)
    assert config.DINNER_CONFIG.mean_dining_ticks == (9, 15, 18, 18, 21, 24, 27, 30)

    for meal in config.MEAL_CONFIGS.values():
        assert len(meal.mean_dining_ticks) == len(config.SUPPORTED_PARTY_SIZES)
        assert all(duration > 0 for duration in meal.mean_dining_ticks)


def test_abandonment_tables_are_contiguous_and_bounded() -> None:
    for meal in config.MEAL_CONFIGS.values():
        assert_contiguous_bands(meal.abandonment_hazard, None)
        assert_contiguous_bands(meal.abandonment_time_multipliers, None)
        assert all(0.0 <= hazard <= 1.0 for _, _, hazard in meal.abandonment_hazard)
        assert all(
            multiplier > 0.0
            for _, _, multiplier in meal.abandonment_time_multipliers
        )
        assert 0.0 < meal.abandonment_probability_cap <= 1.0


def test_reward_constants_match_the_canonical_current_values() -> None:
    assert config.WAITED_PERSON_REWARD == -0.1
    assert config.LOST_PERSON_REWARD == -1.2
    assert config.SEATING_ASSIGNMENT_REWARD == 0.0
    assert config.END_TICK_FIXED_REWARD == 0.0
    assert config.DISCOUNT_FACTOR == 1.0


def test_calibratable_status_is_explicit() -> None:
    assert "base_arrival_curves" in config.CALIBRATABLE_BEFORE_RL_FREEZE
    assert "demand_scenario_multipliers" in config.CALIBRATABLE_BEFORE_RL_FREEZE
    assert "queue_capacity" in config.CALIBRATABLE_BEFORE_RL_FREEZE
    assert "reward_coefficients" in config.CALIBRATABLE_BEFORE_RL_FREEZE
