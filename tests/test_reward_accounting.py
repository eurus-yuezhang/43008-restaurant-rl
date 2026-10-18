import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.randomness import stable_rng


def find_seed_where_all_abandon(
    party_ids: list[tuple[int, int]],
    waiting_tick: int,
    probability: float,
) -> int:
    for seed in range(10000):
        if all(
            stable_rng(
                seed,
                party_id[0],
                party_id[1],
                waiting_tick,
                "abandon",
            ).random()
            < probability
            for party_id in party_ids
        ):
            return seed
    raise AssertionError("no deterministic abandonment seed found")


def expected_reward(info: dict) -> float:
    return (
        config.WAITED_PERSON_REWARD * info["waited_persons"]
        + config.LOST_PERSON_REWARD
        * (info["abandoned_persons"] + info["balked_persons"])
        + config.END_TICK_FIXED_REWARD
    )


def test_no_waiting_and_no_loss_returns_only_fixed_component() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.closing_tick_of_day

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert info["waited_persons"] == 0
    assert info["abandoned_persons"] == 0
    assert info["balked_persons"] == 0
    assert reward == config.END_TICK_FIXED_REWARD


def test_waited_persons_sum_party_sizes() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.waiting_queue = [
        CustomerParty(party_id=(0, 0), size=2),
        CustomerParty(party_id=(0, 1), size=3),
        CustomerParty(party_id=(0, 2), size=5),
    ]

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert info["waited_persons"] == 10
    assert info["abandoned_persons"] == 0
    assert reward == pytest.approx(config.WAITED_PERSON_REWARD * 10)


@pytest.mark.parametrize("party_sizes", ((1,), (4,), (2, 5)))
def test_abandoned_persons_sum_party_sizes(
    party_sizes: tuple[int, ...],
) -> None:
    party_ids = [(500, index) for index in range(len(party_sizes))]
    seed = find_seed_where_all_abandon(party_ids, waiting_tick=9, probability=0.65)
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=seed)
    env.current_tick = env.meal_config.closing_tick_of_day
    env.waiting_queue = [
        CustomerParty(
            party_id=party_id,
            size=size,
            waiting_ticks=8,
        )
        for party_id, size in zip(party_ids, party_sizes)
    ]
    total_people = sum(party_sizes)

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert env.waiting_queue == []
    assert info["waited_persons"] == total_people
    assert info["abandoned_persons"] == total_people
    assert info["balked_persons"] == 0
    assert reward == pytest.approx(expected_reward(info))


def test_balked_persons_are_counted_once_and_removed() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.closing_tick_of_day
    env.waiting_queue = [
        CustomerParty(party_id=(100, index), size=1)
        for index in range(config.QUEUE_CAPACITY)
    ]
    balking_parties = [
        CustomerParty(party_id=(200, 0), size=2),
        CustomerParty(party_id=(200, 1), size=3),
        CustomerParty(party_id=(200, 2), size=5),
    ]
    env.arrival_buffer = list(balking_parties)

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert info["waited_persons"] == config.QUEUE_CAPACITY
    assert info["abandoned_persons"] == 0
    assert info["balked_persons"] == 10
    assert reward == pytest.approx(expected_reward(info))
    live_ids = {party.party_id for party in env.waiting_queue}
    assert all(party.party_id not in live_ids for party in balking_parties)

    _, _, _, _, next_info = env.step(config.END_TICK_ACTION_ID)
    assert next_info["balked_persons"] == 0
    assert next_info["waited_persons"] == config.QUEUE_CAPACITY


def test_waiting_abandonment_and_balking_are_combined_once() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.closing_tick_of_day
    abandoning = CustomerParty(party_id=(0, 0), size=4, waiting_ticks=8)
    env.waiting_queue = [abandoning] + [
        CustomerParty(party_id=(100, index), size=1)
        for index in range(config.QUEUE_CAPACITY - 1)
    ]
    env.arrival_buffer = [CustomerParty(party_id=(200, 0), size=3)]

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert abandoning.waiting_ticks == 9
    assert info["waited_persons"] == 15
    assert info["abandoned_persons"] == 4
    assert info["balked_persons"] == 3
    assert reward == pytest.approx(
        config.WAITED_PERSON_REWARD * 15
        + config.LOST_PERSON_REWARD * 7
        + config.END_TICK_FIXED_REWARD
    )


def test_new_large_arrival_buffer_has_no_immediate_wait_or_loss() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="super_peak")
    interval_tick = env.meal_config.opening_tick_of_day + 12
    selected_seed = None
    for seed in range(10000):
        env.reset(seed=seed)
        if len(env._generate_arrivals(interval_tick)) >= 3:
            selected_seed = seed
            break
    assert selected_seed is not None
    env.reset(seed=selected_seed)
    env.current_tick = interval_tick

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert len(env.arrival_buffer) >= 3
    assert all(party.waiting_ticks == 0 for party in env.arrival_buffer)
    assert info["waited_persons"] == 0
    assert info["abandoned_persons"] == 0
    assert info["balked_persons"] == 0
    assert reward == config.END_TICK_FIXED_REWARD


def test_previous_buffer_admission_waits_during_current_interval() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.closing_tick_of_day
    admitted = [
        CustomerParty(party_id=(200, 0), size=3),
        CustomerParty(party_id=(200, 1), size=4),
    ]
    env.arrival_buffer = list(admitted)

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert env.waiting_queue == admitted
    assert all(party.waiting_ticks == 1 for party in admitted)
    assert info["waited_persons"] == 7
    assert info["abandoned_persons"] == 0
    assert info["balked_persons"] == 0
    assert reward == pytest.approx(config.WAITED_PERSON_REWARD * 7)
