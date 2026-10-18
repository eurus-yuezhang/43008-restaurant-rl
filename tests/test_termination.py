import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty, Table
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.randomness import stable_rng


def closed_environment() -> RestaurantEnv:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.closing_tick_of_day
    env.arrivals_open = False
    return env


def test_natural_termination_condition_uses_only_decision_relevant_state() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    assert not env._is_naturally_terminated()

    env.arrivals_open = False
    env.waiting_queue = [CustomerParty(party_id=(0, 0), size=1)]
    assert not env._is_naturally_terminated()

    env.waiting_queue = []
    env.arrival_buffer = [CustomerParty(party_id=(0, 1), size=1)]
    assert not env._is_naturally_terminated()

    env.arrival_buffer = []
    assert env._is_naturally_terminated()


def test_occupied_tables_do_not_prevent_natural_termination() -> None:
    env = closed_environment()
    env.tables[0] = Table(
        table_id=1,
        capacity=2,
        assigned_party_id=(0, 0),
        remaining_ticks=5,
    )

    assert env._is_naturally_terminated()
    assert env.tables[0].is_occupied


def test_final_queued_party_seating_terminates_without_advancing_time() -> None:
    env = closed_environment()
    party = CustomerParty(party_id=(0, 0), size=1, waiting_ticks=3)
    env.waiting_queue.append(party)
    original_tick = env.current_tick

    observation, reward, terminated, truncated, _ = env.step(0)

    assert env.waiting_queue == []
    assert env.tables[0].assigned_party_id == party.party_id
    assert env.tables[0].remaining_ticks > 0
    assert env.current_tick == original_tick
    assert reward == config.SEATING_ASSIGNMENT_REWARD
    assert terminated
    assert not truncated
    assert env.observation_space.contains(observation)


def test_final_queue_abandonment_terminates_with_normal_reward() -> None:
    env = closed_environment()
    party = CustomerParty(party_id=(0, 0), size=4, waiting_ticks=8)
    env.waiting_queue.append(party)

    _, reward, terminated, truncated, info = env.step(
        config.END_TICK_ACTION_ID
    )

    assert party.waiting_ticks == 9
    assert env.waiting_queue == []
    assert info["waited_persons"] == 4
    assert info["abandoned_persons"] == 4
    assert info["balked_persons"] == 0
    assert reward == pytest.approx(
        config.WAITED_PERSON_REWARD * 4
        + config.LOST_PERSON_REWARD * 4
        + config.END_TICK_FIXED_REWARD
    )
    assert terminated
    assert not truncated


def test_buffer_balking_and_queue_abandonment_can_terminate_same_tick() -> None:
    env = closed_environment()
    abandoning_ids = []
    arrival_index = 0
    while len(abandoning_ids) < config.QUEUE_CAPACITY:
        party_id = (300, arrival_index)
        draw = stable_rng(0, 300, arrival_index, 9, "abandon").random()
        if draw < env.meal_config.abandonment_probability_cap:
            abandoning_ids.append(party_id)
        arrival_index += 1

    env.waiting_queue = [
        CustomerParty(party_id=party_id, size=1, waiting_ticks=8)
        for party_id in abandoning_ids
    ]
    balking = CustomerParty(party_id=(400, 0), size=3)
    env.arrival_buffer = [balking]

    _, reward, terminated, truncated, info = env.step(
        config.END_TICK_ACTION_ID
    )

    assert env.waiting_queue == []
    assert env.arrival_buffer == []
    assert info["waited_persons"] == config.QUEUE_CAPACITY
    assert info["abandoned_persons"] == config.QUEUE_CAPACITY
    assert info["balked_persons"] == balking.size
    assert reward == pytest.approx(
        config.WAITED_PERSON_REWARD * config.QUEUE_CAPACITY
        + config.LOST_PERSON_REWARD
        * (config.QUEUE_CAPACITY + balking.size)
        + config.END_TICK_FIXED_REWARD
    )
    assert terminated
    assert not truncated


def test_step_after_termination_requires_reset() -> None:
    env = closed_environment()
    env.waiting_queue = [CustomerParty(party_id=(0, 0), size=1)]
    assert env.step(0)[2]

    with pytest.raises(RuntimeError, match="reset"):
        env.step(config.END_TICK_ACTION_ID)


def test_reset_after_termination_restores_opening_state() -> None:
    env = closed_environment()
    env.waiting_queue = [CustomerParty(party_id=(0, 0), size=1)]
    assert env.step(0)[2]

    observation, info = env.reset(seed=99)

    assert env.current_tick == env.meal_config.opening_tick_of_day
    assert env.arrivals_open
    assert env.waiting_queue == []
    assert env.arrival_buffer == []
    assert all(not table.is_occupied for table in env.tables)
    assert not env._is_naturally_terminated()
    assert env.observation_space.contains(observation)
    assert info["action_mask"].shape == (config.ACTION_COUNT,)
    assert info["action_mask"].sum() == 1
    assert info["action_mask"][config.END_TICK_ACTION_ID]


def test_intermediate_and_terminal_flags_use_gymnasium_semantics() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)

    _, _, terminated, truncated, _ = env.step(config.END_TICK_ACTION_ID)
    assert (terminated, truncated) == (False, False)

    env = closed_environment()
    env.waiting_queue = [CustomerParty(party_id=(0, 0), size=1)]
    _, _, terminated, truncated, _ = env.step(0)
    assert (terminated, truncated) == (True, False)


@pytest.mark.parametrize(
    ("meal", "scenario"),
    (
        ("lunch", "normal"),
        ("dinner", "peak"),
        ("dinner", "super_peak"),
    ),
)
def test_representative_episode_reaches_natural_termination(
    meal: str,
    scenario: str,
) -> None:
    env = RestaurantEnv(meal=meal, demand_scenario=scenario)
    observation, info = env.reset(seed=2026)
    terminated = False

    for _ in range(2000):
        mask = info["action_mask"]
        valid_seating_actions = [
            action_id
            for action_id in config.ACTION_TO_SEATING
            if mask[action_id]
        ]
        action = (
            valid_seating_actions[0]
            if valid_seating_actions
            else config.END_TICK_ACTION_ID
        )
        observation, reward, terminated, truncated, info = env.step(action)

        live_ids = [party.party_id for party in env.waiting_queue]
        live_ids.extend(party.party_id for party in env.arrival_buffer)
        live_ids.extend(
            table.assigned_party_id
            for table in env.tables
            if table.assigned_party_id is not None
        )
        assert len(live_ids) == len(set(live_ids))
        assert len(env.waiting_queue) <= config.QUEUE_CAPACITY
        assert all(table.remaining_ticks >= 0 for table in env.tables)
        assert observation.shape == (config.OBSERVATION_DIMENSION,)
        assert np.all(np.isfinite(observation))
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert info["action_mask"].shape == (config.ACTION_COUNT,)
        assert info["action_mask"][config.END_TICK_ACTION_ID]
        assert not truncated
        if terminated:
            break

    assert terminated
    assert not env.arrivals_open
    assert env.waiting_queue == []
    assert env.arrival_buffer == []
