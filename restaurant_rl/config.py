"""Canonical static data for the restaurant-seating environment.

All time values used by the simulation are expressed in five-minute ticks.
The configuration contains no sampling or environment transition logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias


TickBand: TypeAlias = tuple[int, int | None, float]
WeightedValue: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class MealConfig:
    """Canonical parameters for one meal period, expressed in ticks."""

    name: str
    opening_tick_of_day: int
    closing_tick_of_day: int
    base_arrival_curve: tuple[TickBand, ...]
    party_size_probabilities: tuple[float, ...]
    mean_dining_ticks: tuple[int, ...]
    dining_multiplier_distribution: tuple[WeightedValue, ...]
    abandonment_hazard: tuple[TickBand, ...]
    abandonment_time_multipliers: tuple[TickBand, ...]
    abandonment_probability_cap: float

    @property
    def arrival_window_ticks(self) -> int:
        """Number of five-minute arrival intervals in this meal period."""

        return self.closing_tick_of_day - self.opening_tick_of_day


TICK_MINUTES: Final = 5

# Table order is fixed: positions also identify the first observation block.
TABLE_CAPACITIES: Final[tuple[int, ...]] = (
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    6,
    6,
    6,
    8,
)
TABLE_IDS: Final[tuple[int, ...]] = tuple(range(1, len(TABLE_CAPACITIES) + 1))
CANONICAL_TABLE_CAPACITIES: Final[frozenset[int]] = frozenset(TABLE_CAPACITIES)
TABLE_CAPACITY_COUNTS: Final[dict[int, int]] = {
    capacity: TABLE_CAPACITIES.count(capacity)
    for capacity in sorted(CANONICAL_TABLE_CAPACITIES)
}
TABLE_COUNT: Final = len(TABLE_CAPACITIES)
TOTAL_SEATS: Final = sum(TABLE_CAPACITIES)

SUPPORTED_PARTY_SIZES: Final[tuple[int, ...]] = tuple(range(1, 9))
QUEUE_CAPACITY: Final = 12
ARRIVAL_BUFFER_CAPACITY: Final[None] = None
ARRIVAL_BUFFER_PERSISTS_ACROSS_TICKS: Final = False

PARTY_TABLE_COMPATIBILITY: Final[dict[int, tuple[int, ...]]] = {
    1: (2, 4),
    2: (2, 4),
    3: (4,),
    4: (4, 6),
    5: (6,),
    6: (6,),
    7: (8,),
    8: (8,),
}

# Each position is an action ID for (party size, table capacity); ID 11 advances time.
SEATING_ACTION_PAIRS: Final[tuple[tuple[int, int], ...]] = (
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
ACTION_TO_SEATING: Final[dict[int, tuple[int, int]]] = dict(
    enumerate(SEATING_ACTION_PAIRS)
)
SEATING_TO_ACTION: Final[dict[tuple[int, int], int]] = {
    pair: action_id for action_id, pair in ACTION_TO_SEATING.items()
}
SEATING_ACTION_COUNT: Final = len(SEATING_ACTION_PAIRS)
END_TICK_ACTION_ID: Final = 11
ACTION_COUNT: Final = 12

# Observation blocks and normalisation limits must stay compatible with saved policies.
TABLE_OBSERVATION_FEATURES: Final = 12
QUEUE_SLOT_COUNT: Final = 12
QUEUE_FEATURES_PER_SLOT: Final = 2
QUEUE_OBSERVATION_FEATURES: Final = QUEUE_SLOT_COUNT * QUEUE_FEATURES_PER_SLOT
ARRIVAL_BUFFER_OBSERVATION_FEATURES: Final = 8
SERVICE_PROGRESS_OBSERVATION_FEATURES: Final = 1
DEMAND_SCENARIO_OBSERVATION_FEATURES: Final = 4
MEAL_OBSERVATION_FEATURES: Final = 2
OBSERVATION_BLOCK_SIZES: Final[tuple[tuple[str, int], ...]] = (
    ("tables", TABLE_OBSERVATION_FEATURES),
    ("queue", QUEUE_OBSERVATION_FEATURES),
    ("arrival_buffer", ARRIVAL_BUFFER_OBSERVATION_FEATURES),
    ("service_progress", SERVICE_PROGRESS_OBSERVATION_FEATURES),
    ("demand_scenario", DEMAND_SCENARIO_OBSERVATION_FEATURES),
    ("meal", MEAL_OBSERVATION_FEATURES),
)
OBSERVATION_DIMENSION: Final = 51
PARTY_SIZE_NORMALIZATION_DIVISOR: Final = 8
WAITING_TICK_NORMALIZATION_CAP: Final = 24
MAX_TABLE_REMAINING_TICKS: Final = 36

# Penalties count people, not parties; seating itself has no immediate reward.
WAITED_PERSON_REWARD: Final = -0.1
LOST_PERSON_REWARD: Final = -1.2
SEATING_ASSIGNMENT_REWARD: Final = 0.0
END_TICK_FIXED_REWARD: Final = 0.0
DISCOUNT_FACTOR: Final = 1.0

DEMAND_SCENARIO_MULTIPLIERS: Final[dict[str, float]] = {
    "off_peak": 0.75,
    "normal": 1.00,
    "peak": 1.25,
    "super_peak": 1.50,
}

# Rate and hazard bands use [start, end) tick ranges; None means no upper bound.
LUNCH_CONFIG: Final = MealConfig(
    name="lunch",
    opening_tick_of_day=132,
    closing_tick_of_day=174,
    base_arrival_curve=(
        (0, 6, 0.40),
        (6, 12, 0.75),
        (12, 24, 1.20),
        (24, 30, 1.00),
        (30, 36, 0.60),
        (36, 42, 0.30),
    ),
    party_size_probabilities=(0.30, 0.40, 0.15, 0.10, 0.02, 0.02, 0.005, 0.005),
    mean_dining_ticks=(6, 9, 12, 15, 17, 19, 21, 23),
    dining_multiplier_distribution=(
        (0.70, 0.03),
        (0.85, 0.17),
        (0.95, 0.25),
        (1.00, 0.25),
        (1.10, 0.18),
        (1.25, 0.12),
    ),
    abandonment_hazard=(
        (0, 2, 0.00),
        (2, 3, 0.03),
        (3, 4, 0.08),
        (4, 6, 0.15),
        (6, 9, 0.30),
        (9, None, 0.50),
    ),
    abandonment_time_multipliers=(
        (0, 12, 0.8),
        (12, 24, 1.0),
        (24, 36, 1.3),
        (36, None, 1.6),
    ),
    abandonment_probability_cap=0.65,
)

DINNER_CONFIG: Final = MealConfig(
    name="dinner",
    opening_tick_of_day=204,
    closing_tick_of_day=264,
    base_arrival_curve=(
        (0, 12, 0.35),
        (12, 24, 0.50),
        (24, 48, 0.65),
        (48, 60, 0.45),
    ),
    party_size_probabilities=(0.15, 0.30, 0.15, 0.20, 0.07, 0.08, 0.02, 0.03),
    mean_dining_ticks=(9, 15, 18, 18, 21, 24, 27, 30),
    dining_multiplier_distribution=(
        (0.85, 0.05),
        (0.90, 0.24),
        (0.95, 0.13),
        (1.00, 0.25),
        (1.05, 0.10),
        (1.10, 0.08),
        (1.15, 0.10),
        (1.20, 0.05),
    ),
    abandonment_hazard=(
        (0, 3, 0.00),
        (3, 6, 0.02),
        (6, 9, 0.05),
        (9, 12, 0.10),
        (12, 15, 0.18),
        (15, 18, 0.28),
        (18, None, 0.40),
    ),
    abandonment_time_multipliers=(
        (0, 24, 0.8),
        (24, 36, 1.0),
        (36, 48, 1.6),
        (48, None, 2.0),
    ),
    abandonment_probability_cap=0.60,
)

MEAL_CONFIGS: Final[dict[str, MealConfig]] = {
    LUNCH_CONFIG.name: LUNCH_CONFIG,
    DINNER_CONFIG.name: DINNER_CONFIG,
}

# Historical list of settings that could be calibrated before training.
# The completed experiments use the fixed values above.
CALIBRATABLE_BEFORE_RL_FREEZE: Final[frozenset[str]] = frozenset(
    {
        "base_arrival_curves",
        "demand_scenario_multipliers",
        "mean_dining_ticks",
        "dining_multiplier_distributions",
        "abandonment_hazards",
        "abandonment_time_multipliers",
        "queue_capacity",
        "reward_coefficients",
    }
)
