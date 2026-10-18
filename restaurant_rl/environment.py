"""Restaurant state, observations, seating decisions and five-minute updates."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import (
    ACTION_COUNT,
    ACTION_TO_SEATING,
    DEMAND_SCENARIO_MULTIPLIERS,
    END_TICK_ACTION_ID,
    END_TICK_FIXED_REWARD,
    LOST_PERSON_REWARD,
    MAX_TABLE_REMAINING_TICKS,
    MEAL_CONFIGS,
    OBSERVATION_DIMENSION,
    PARTY_SIZE_NORMALIZATION_DIVISOR,
    QUEUE_CAPACITY,
    QUEUE_SLOT_COUNT,
    SEATING_ASSIGNMENT_REWARD,
    SUPPORTED_PARTY_SIZES,
    TABLE_CAPACITIES,
    TABLE_IDS,
    WAITING_TICK_NORMALIZATION_CAP,
    WAITED_PERSON_REWARD,
)
from .entities import CustomerParty, Table
from .randomness import keyed_dining_multiplier, stable_rng


class RestaurantEnv(gym.Env):
    """Canonical restaurant state with zero-time seating actions."""

    def __init__(self, meal: str, demand_scenario: str) -> None:
        """Validate the meal and demand level, then prepare the spaces and opening state."""
        super().__init__()

        if meal not in MEAL_CONFIGS:
            raise ValueError(f"meal must be one of {tuple(MEAL_CONFIGS)}")
        if demand_scenario not in DEMAND_SCENARIO_MULTIPLIERS:
            raise ValueError(
                "demand_scenario must be one of "
                f"{tuple(DEMAND_SCENARIO_MULTIPLIERS)}"
            )

        self.meal = meal
        self.meal_config = MEAL_CONFIGS[meal]
        self.demand_scenario = demand_scenario

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIMENSION,),
            dtype=np.float32,
        )

        self.episode_seed: int | None = None
        self._set_opening_state()

    def _set_opening_state(self) -> None:
        """Restore the canonical empty restaurant state at meal opening."""

        self.tables = [
            Table(table_id=table_id, capacity=capacity)
            for table_id, capacity in zip(TABLE_IDS, TABLE_CAPACITIES)
        ]
        self.waiting_queue = []
        self.arrival_buffer = []
        self.current_tick = self.meal_config.opening_tick_of_day
        self.arrivals_open = True

    def reset(self, *, seed=None, options=None):
        """Reset to meal opening without generating or seating customers."""

        super().reset(seed=seed)
        if seed is None:
            self.episode_seed = int(
                self.np_random.integers(0, np.iinfo(np.int64).max)
            )
        else:
            self.episode_seed = int(seed)
        self._set_opening_state()
        observation = self._get_observation()
        info = {
            "meal": self.meal,
            "demand_scenario": self.demand_scenario,
            "action_mask": self._get_action_mask(),
        }
        return observation, info

    def _get_action_mask(self) -> np.ndarray:
        """Return the current legality of all seating actions and END_TICK."""

        # A seating action needs both a matching party and a free table of its listed capacity.
        mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        for action_id, (party_size, table_capacity) in ACTION_TO_SEATING.items():
            has_party = any(
                party.size == party_size
                for party in self.waiting_queue + self.arrival_buffer
            )
            has_table = any(
                table.capacity == table_capacity and not table.is_occupied
                for table in self.tables
            )
            mask[action_id] = has_party and has_table

        mask[END_TICK_ACTION_ID] = True
        return mask

    def _select_party(
        self,
        party_size: int,
    ) -> tuple[CustomerParty, list[CustomerParty]]:
        """Select queue-first, then apply the canonical party tie-breaks."""

        # Existing queue parties get the seat before same-size fresh arrivals.
        queued = [
            party for party in self.waiting_queue if party.size == party_size
        ]
        if queued:
            party = min(
                queued,
                key=lambda item: (-item.waiting_ticks, item.party_id),
            )
            return party, self.waiting_queue

        buffered = [
            party for party in self.arrival_buffer if party.size == party_size
        ]
        if buffered:
            party = min(buffered, key=lambda item: item.party_id)
            return party, self.arrival_buffer

        raise AssertionError("valid seating action has no matching party")

    def _select_table(self, table_capacity: int) -> Table:
        """Select the lowest-ID free table with the requested capacity."""

        available = [
            table
            for table in self.tables
            if table.capacity == table_capacity and not table.is_occupied
        ]
        if not available:
            raise AssertionError("valid seating action has no matching table")
        return min(available, key=lambda table: table.table_id)

    def _get_dining_duration(self, party: CustomerParty) -> int:
        """Realise a keyed duration, rounded half-up to a five-minute tick."""

        if self.episode_seed is None:
            raise RuntimeError("reset() must be called before step()")

        mean_ticks = self.meal_config.mean_dining_ticks[party.size - 1]
        multiplier = keyed_dining_multiplier(
            self.episode_seed,
            party.party_id,
            self.meal_config.dining_multiplier_distribution,
        )
        duration = int(mean_ticks * multiplier + 0.5)
        if not 1 <= duration <= MAX_TABLE_REMAINING_TICKS:
            raise AssertionError("dining duration is outside the supported range")
        return duration

    def _is_naturally_terminated(self) -> bool:
        """End after closing when the queue and arrival buffer are empty.

        Seated diners may remain because they need no further seating decisions.
        """

        return (
            not self.arrivals_open
            and not self.waiting_queue
            and not self.arrival_buffer
        )

    def _resolve_arrival_buffer(self) -> int:
        """Move buffered parties into free queue slots and return balked people."""

        if len(self.waiting_queue) > QUEUE_CAPACITY:
            raise AssertionError("waiting queue exceeds canonical capacity")
        if self.episode_seed is None:
            raise RuntimeError("reset() must be called before step()")

        # When arrivals exceed queue space, rank them by stable party-specific draws.
        free_slots = QUEUE_CAPACITY - len(self.waiting_queue)
        ordered_buffer = list(self.arrival_buffer)
        if len(ordered_buffer) > free_slots:
            ordered_buffer.sort(
                key=lambda party: (
                    stable_rng(
                        self.episode_seed,
                        party.party_id[0],
                        party.party_id[1],
                        self.current_tick,
                        "overflow",
                    ).random(),
                    party.party_id,
                )
            )

        # Admit the selected parties and count everyone else as balked; no buffer carries over.
        admitted = ordered_buffer[:free_slots]
        balked = ordered_buffer[free_slots:]
        self.waiting_queue.extend(admitted)
        self.arrival_buffer = []
        return sum(party.size for party in balked)

    def _advance_tables(self) -> None:
        """Advance occupied dining times by one tick and release completions."""

        for table in self.tables:
            if table.is_occupied:
                table.remaining_ticks -= 1
                if table.remaining_ticks == 0:
                    table.assigned_party_id = None
            if table.remaining_ticks < 0:
                raise AssertionError("table remaining_ticks cannot be negative")

    def _get_abandonment_probability(self, party: CustomerParty) -> float:
        """Return the canonical hazard after this tick's waiting increment."""

        # Choose the waiting-time hazard using the updated number of ticks waited.
        base_hazard = None
        for start, end, hazard in self.meal_config.abandonment_hazard:
            if start <= party.waiting_ticks and (
                end is None or party.waiting_ticks < end
            ):
                base_hazard = hazard
                break

        ticks_since_open = self.current_tick - self.meal_config.opening_tick_of_day
        # Scale that hazard by time since opening, then apply the configured cap.
        time_multiplier = None
        for start, end, multiplier in self.meal_config.abandonment_time_multipliers:
            if start <= ticks_since_open and (
                end is None or ticks_since_open < end
            ):
                time_multiplier = multiplier
                break

        if base_hazard is None or time_multiplier is None:
            raise AssertionError("abandonment configuration does not cover state")
        return min(
            base_hazard * time_multiplier,
            self.meal_config.abandonment_probability_cap,
        )

    def _apply_abandonment(self) -> int:
        """Remove keyed abandoning parties and return abandoned people."""

        if self.episode_seed is None:
            raise RuntimeError("reset() must be called before step()")

        # Draw separately for each party and waiting age, preserving survivor order.
        remaining_parties: list[CustomerParty] = []
        abandoned_persons = 0
        for party in self.waiting_queue:
            probability = self._get_abandonment_probability(party)
            draw = stable_rng(
                self.episode_seed,
                party.party_id[0],
                party.party_id[1],
                party.waiting_ticks,
                "abandon",
            ).random()
            if draw < probability:
                abandoned_persons += party.size
            else:
                remaining_parties.append(party)

        self.waiting_queue = remaining_parties
        return abandoned_persons

    def _generate_arrivals(self, interval_tick: int) -> list[CustomerParty]:
        """Generate the keyed arrivals for one completed arrival interval."""

        if self.episode_seed is None:
            raise RuntimeError("reset() must be called before step()")
        if not (
            self.meal_config.opening_tick_of_day
            <= interval_tick
            < self.meal_config.closing_tick_of_day
        ):
            return []

        # Use the completed interval to choose the arrival rate, including the last open interval.
        interval_offset = interval_tick - self.meal_config.opening_tick_of_day
        base_rate = None
        for start, end, rate in self.meal_config.base_arrival_curve:
            if start <= interval_offset and (
                end is None or interval_offset < end
            ):
                base_rate = rate
                break
        if base_rate is None:
            raise AssertionError("arrival curve does not cover arrival interval")

        arrival_rate = (
            base_rate * DEMAND_SCENARIO_MULTIPLIERS[self.demand_scenario]
        )
        # Sample the number of parties from a count-specific random stream.
        arrival_count = int(
            stable_rng(
                self.episode_seed,
                self.meal,
                self.demand_scenario,
                interval_tick,
                "arrival_count",
            ).poisson(arrival_rate)
        )

        # Give each arrival its own size draw and stable (interval, index) identity.
        arrivals: list[CustomerParty] = []
        for arrival_index in range(arrival_count):
            party_size = int(
                stable_rng(
                    self.episode_seed,
                    self.meal,
                    self.demand_scenario,
                    interval_tick,
                    arrival_index,
                    "party_size",
                ).choice(
                    SUPPORTED_PARTY_SIZES,
                    p=self.meal_config.party_size_probabilities,
                )
            )
            arrivals.append(
                CustomerParty(
                    party_id=(interval_tick, arrival_index),
                    size=party_size,
                )
            )
        return arrivals

    def _step_end_tick(self):
        """Resolve overflow, advance time, apply waiting and abandonment, then add arrivals."""

        if self.episode_seed is None:
            raise RuntimeError("reset() must be called before step()")

        # Resolve unseated arrivals before advancing time or freeing tables.
        balked_persons = self._resolve_arrival_buffer()
        # Advance one five-minute interval and release tables whose dining time has ended.
        completed_interval_tick = self.current_tick
        self.current_tick += 1
        self._advance_tables()

        # Increase every queued party's wait before checking abandonment.
        for party in self.waiting_queue:
            party.waiting_ticks += 1
        # Parties abandoning now still waited through the completed interval.
        waited_persons = sum(party.size for party in self.waiting_queue)
        abandoned_persons = self._apply_abandonment()

        # Add arrivals for the completed interval after waiting and abandonment are settled.
        self.arrival_buffer = self._generate_arrivals(completed_interval_tick)
        self.arrivals_open = (
            self.current_tick < self.meal_config.closing_tick_of_day
        )

        # Charge waiting and lost-person penalties once for this interval.
        lost_persons = abandoned_persons + balked_persons
        reward = (
            WAITED_PERSON_REWARD * waited_persons
            + LOST_PERSON_REWARD * lost_persons
            + END_TICK_FIXED_REWARD
        )
        observation = self._get_observation()
        info = {
            "action_mask": self._get_action_mask(),
            "waited_persons": waited_persons,
            "abandoned_persons": abandoned_persons,
            "balked_persons": balked_persons,
        }
        terminated = self._is_naturally_terminated()
        return observation, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        """Encode current state in the canonical 51-feature block order."""

        # Build blocks in the fixed order expected by saved policies and action masks.
        features: list[float] = []

        for table in self.tables:
            features.append(
                table.remaining_ticks / MAX_TABLE_REMAINING_TICKS
            )

        if len(self.waiting_queue) > QUEUE_CAPACITY:
            raise AssertionError("waiting queue exceeds canonical capacity")

        # Sort by actual wait before clipping its encoded value, then pad unused slots.
        ordered_queue = sorted(
            self.waiting_queue,
            key=lambda party: (-party.waiting_ticks, party.party_id),
        )
        for slot in range(QUEUE_SLOT_COUNT):
            if slot < len(ordered_queue):
                party = ordered_queue[slot]
                features.append(
                    party.size / PARTY_SIZE_NORMALIZATION_DIVISOR
                )
                features.append(
                    min(
                        party.waiting_ticks / WAITING_TICK_NORMALIZATION_CAP,
                        1.0,
                    )
                )
            else:
                features.extend((0.0, 0.0))

        # Encode arrival counts by party size using count / (count + 1), without a fixed cap.
        buffer_counts = {size: 0 for size in SUPPORTED_PARTY_SIZES}
        for party in self.arrival_buffer:
            buffer_counts[party.size] += 1
        for size in SUPPORTED_PARTY_SIZES:
            count = buffer_counts[size]
            features.append(count / (count + 1))

        ticks_since_open = self.current_tick - self.meal_config.opening_tick_of_day
        # Append capped service progress followed by scenario and meal indicators.
        progress = min(
            ticks_since_open / self.meal_config.arrival_window_ticks,
            1.0,
        )
        features.append(progress)

        for scenario in DEMAND_SCENARIO_MULTIPLIERS:
            features.append(1.0 if scenario == self.demand_scenario else 0.0)

        for meal in MEAL_CONFIGS:
            features.append(1.0 if meal == self.meal else 0.0)

        # Check the complete observation against the policy input shape and bounds.
        observation = np.asarray(features, dtype=np.float32)
        if observation.shape != (OBSERVATION_DIMENSION,):
            raise AssertionError(
                f"observation has shape {observation.shape}, "
                f"expected {(OBSERVATION_DIMENSION,)}"
            )
        if not np.all(np.isfinite(observation)):
            raise AssertionError("observation contains a non-finite value")
        if not self.observation_space.contains(observation):
            raise AssertionError("observation is outside observation_space")
        return observation

    def step(self, action: int):
        """Apply one zero-time seating action or one END_TICK interval."""

        if self._is_naturally_terminated():
            raise RuntimeError("reset() must be called after episode termination")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer from 0 to {ACTION_COUNT - 1}")

        action_id = int(action)
        if action_id == END_TICK_ACTION_ID:
            return self._step_end_tick()

        action_mask = self._get_action_mask()
        if not action_mask[action_id]:
            raise ValueError(f"seating action {action_id} is illegal in current state")

        # Resolve the requested size pair using queue priority and the lowest free table ID.
        party_size, table_capacity = ACTION_TO_SEATING[action_id]
        party, party_location = self._select_party(party_size)
        table = self._select_table(table_capacity)
        dining_duration = self._get_dining_duration(party)

        # Seat the chosen party without advancing the clock or other waiting times.
        party_location.remove(party)
        table.assigned_party_id = party.party_id
        table.remaining_ticks = dining_duration

        observation = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        terminated = self._is_naturally_terminated()
        return observation, SEATING_ASSIGNMENT_REWARD, terminated, False, info
