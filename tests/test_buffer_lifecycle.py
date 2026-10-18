from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.randomness import stable_rng


def test_buffer_party_can_be_seated_before_resolution() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=4)
    party = CustomerParty(party_id=(0, 0), size=1)
    env.arrival_buffer.append(party)

    env.step(0)

    assert env.arrival_buffer == []
    assert env.tables[0].assigned_party_id == party.party_id


def test_unresolved_buffer_party_enters_queue_then_waits_one_tick() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=4)
    env.current_tick = env.meal_config.closing_tick_of_day
    party = CustomerParty(party_id=(10, 0), size=2)
    env.arrival_buffer.append(party)

    _, _, _, _, info = env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer == []
    assert env.waiting_queue == [party]
    assert party.waiting_ticks == 1
    assert info["waited_persons"] == 2
    assert info["balked_persons"] == 0


def test_overflow_admission_is_keyed_and_never_exceeds_queue_capacity() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=12)
    env.current_tick = env.meal_config.closing_tick_of_day
    env.waiting_queue = [
        CustomerParty(party_id=(20, index), size=1)
        for index in range(config.QUEUE_CAPACITY - 1)
    ]
    buffered = [
        CustomerParty(party_id=(30, 0), size=2),
        CustomerParty(party_id=(30, 1), size=3),
        CustomerParty(party_id=(30, 2), size=4),
    ]
    env.arrival_buffer = list(reversed(buffered))
    expected_order = sorted(
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

    _, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

    admitted = expected_order[0]
    expected_balked_persons = sum(party.size for party in expected_order[1:])
    assert admitted in env.waiting_queue
    assert len(env.waiting_queue) == config.QUEUE_CAPACITY
    assert env.arrival_buffer == []
    assert info["balked_persons"] == expected_balked_persons
    assert reward == (
        config.WAITED_PERSON_REWARD
        * sum(party.size for party in env.waiting_queue)
        + config.LOST_PERSON_REWARD * expected_balked_persons
    )


def test_overflow_result_does_not_depend_on_buffer_list_order() -> None:
    environments = [
        RestaurantEnv(meal="lunch", demand_scenario="normal")
        for _ in range(2)
    ]
    accepted_ids = []
    for index, env in enumerate(environments):
        env.reset(seed=12)
        env.current_tick = env.meal_config.closing_tick_of_day
        env.waiting_queue = [
            CustomerParty(party_id=(20, item), size=1)
            for item in range(config.QUEUE_CAPACITY - 1)
        ]
        buffered = [
            CustomerParty(party_id=(30, 0), size=2),
            CustomerParty(party_id=(30, 1), size=3),
            CustomerParty(party_id=(30, 2), size=4),
        ]
        env.arrival_buffer = buffered if index == 0 else list(reversed(buffered))

        env.step(config.END_TICK_ACTION_ID)
        accepted_ids.append(
            next(
                party.party_id
                for party in env.waiting_queue
                if party.party_id[0] == 30
            )
        )

    assert accepted_ids[0] == accepted_ids[1]
