import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.randomness import stable_rng


BUFFER_CASES = (
    (0, (1, 2)),
    (5, (1, 2, 3)),
    (11, (1, 2, 3)),
    (12, (1, 2)),
    (10, (1, 2, 3, 4)),
    (6, ()),
)


def live_party_ids(env: RestaurantEnv) -> list[tuple[int, int]]:
    party_ids = [party.party_id for party in env.waiting_queue]
    party_ids.extend(party.party_id for party in env.arrival_buffer)
    party_ids.extend(
        table.assigned_party_id
        for table in env.tables
        if table.assigned_party_id is not None
    )
    return party_ids


def expected_buffer_order(
    env: RestaurantEnv,
    buffered: list[CustomerParty],
    free_slots: int,
) -> list[CustomerParty]:
    if len(buffered) <= free_slots:
        return buffered
    return sorted(
        buffered,
        key=lambda party: (
            stable_rng(
                env.episode_seed,
                party.party_id[0],
                party.party_id[1],
                env.current_tick,
                "overflow",
            ).random(),
            party.party_id,
        ),
    )


@pytest.mark.parametrize(("queue_size", "buffer_sizes"), BUFFER_CASES)
def test_empty_partial_near_full_full_and_empty_buffer_cases(
    queue_size: int,
    buffer_sizes: tuple[int, ...],
) -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=91)
    env.current_tick = env.meal_config.closing_tick_of_day
    original_queue = [
        CustomerParty(party_id=(100, index), size=1)
        for index in range(queue_size)
    ]
    buffered = [
        CustomerParty(party_id=(200, index), size=size)
        for index, size in enumerate(buffer_sizes)
    ]
    env.waiting_queue = list(original_queue)
    env.arrival_buffer = list(buffered)
    free_slots = config.QUEUE_CAPACITY - queue_size
    ordered = expected_buffer_order(env, buffered, free_slots)
    expected_admitted = ordered[:free_slots]
    expected_balked = ordered[free_slots:]

    _, _, _, _, info = env.step(config.END_TICK_ACTION_ID)

    admitted_ids = {
        party.party_id
        for party in env.waiting_queue
        if party.party_id[0] == 200
    }
    assert admitted_ids == {party.party_id for party in expected_admitted}
    assert info["balked_persons"] == sum(
        party.size for party in expected_balked
    )
    assert len(env.waiting_queue) <= config.QUEUE_CAPACITY
    assert env.arrival_buffer == []
    assert len(live_party_ids(env)) == len(set(live_party_ids(env)))
    mask = env._get_action_mask()
    assert mask.shape == (config.ACTION_COUNT,)
    assert mask[config.END_TICK_ACTION_ID]


def test_overflow_is_same_after_different_valid_seating_history() -> None:
    plain_env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    seated_env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    for env in (plain_env, seated_env):
        env.reset(seed=55)
        env.current_tick = env.meal_config.closing_tick_of_day
        env.waiting_queue = [
            CustomerParty(party_id=(100, index), size=1)
            for index in range(10)
        ]

    extra_party = CustomerParty(party_id=(150, 0), size=8)
    seated_env.arrival_buffer.append(extra_party)
    seated_env.step(10)

    common_buffer = [
        CustomerParty(party_id=(200, 0), size=1),
        CustomerParty(party_id=(200, 1), size=2),
        CustomerParty(party_id=(200, 2), size=3),
        CustomerParty(party_id=(200, 3), size=4),
    ]
    plain_env.arrival_buffer = list(common_buffer)
    seated_env.arrival_buffer = list(reversed(common_buffer))

    plain_info = plain_env.step(config.END_TICK_ACTION_ID)[4]
    seated_info = seated_env.step(config.END_TICK_ACTION_ID)[4]
    admitted_sets = [
        {
            party.party_id
            for party in env.waiting_queue
            if party.party_id[0] == 200
        }
        for env in (plain_env, seated_env)
    ]

    assert admitted_sets[0] == admitted_sets[1]
    assert plain_info["balked_persons"] == seated_info["balked_persons"]


def test_seated_buffer_party_is_excluded_from_later_overflow() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=73)
    env.current_tick = env.meal_config.closing_tick_of_day
    env.waiting_queue = [
        CustomerParty(party_id=(100, index), size=8)
        for index in range(config.QUEUE_CAPACITY - 1)
    ]
    seated = CustomerParty(party_id=(200, 0), size=1)
    remaining = [
        CustomerParty(party_id=(200, 1), size=2),
        CustomerParty(party_id=(200, 2), size=3),
        CustomerParty(party_id=(200, 3), size=4),
    ]
    env.arrival_buffer = [seated, *remaining]

    env.step(0)
    _, _, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert env.tables[0].assigned_party_id == seated.party_id
    assert seated not in env.waiting_queue
    assert seated not in env.arrival_buffer
    assert info["balked_persons"] == sum(party.size for party in remaining) - next(
        party.size for party in remaining if party in env.waiting_queue
    )
    assert len(live_party_ids(env)) == len(set(live_party_ids(env)))


def test_abandoned_and_balked_parties_leave_all_live_locations() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.closing_tick_of_day
    abandoning = CustomerParty(party_id=(0, 0), size=4, waiting_ticks=8)
    env.waiting_queue = [abandoning] + [
        CustomerParty(party_id=(100, index), size=1)
        for index in range(config.QUEUE_CAPACITY - 1)
    ]
    balking = CustomerParty(party_id=(200, 0), size=3)
    env.arrival_buffer = [balking]

    _, _, _, _, info = env.step(config.END_TICK_ACTION_ID)
    ids = live_party_ids(env)

    assert abandoning.party_id not in ids
    assert balking.party_id not in ids
    assert info["abandoned_persons"] >= abandoning.size
    assert info["balked_persons"] == balking.size
    assert len(ids) == len(set(ids))
