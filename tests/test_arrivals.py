from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv


def buffer_signature(env: RestaurantEnv) -> tuple[tuple[tuple[int, int], int], ...]:
    return tuple((party.party_id, party.size) for party in env.arrival_buffer)


def test_first_arrivals_use_first_interval_and_start_with_zero_wait() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=6)
    opening_tick = env.meal_config.opening_tick_of_day

    env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer
    assert all(party.party_id[0] == opening_tick for party in env.arrival_buffer)
    assert all(party.waiting_ticks == 0 for party in env.arrival_buffer)
    assert all(party.size in config.SUPPORTED_PARTY_SIZES for party in env.arrival_buffer)
    assert len({party.party_id for party in env.arrival_buffer}) == len(
        env.arrival_buffer
    )


def test_final_arrival_interval_is_generated_before_arrivals_close() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=2)
    final_interval = env.meal_config.closing_tick_of_day - 1
    env.current_tick = final_interval

    env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer
    assert all(
        party.party_id[0] == final_interval for party in env.arrival_buffer
    )
    assert not env.arrivals_open


def test_no_arrivals_are_generated_after_closing() -> None:
    env = RestaurantEnv(meal="dinner", demand_scenario="super_peak")
    env.reset(seed=6)
    env.current_tick = env.meal_config.closing_tick_of_day

    _, _, terminated, truncated, _ = env.step(config.END_TICK_ACTION_ID)

    assert env.arrival_buffer == []
    assert not env.arrivals_open
    assert terminated
    assert not truncated


def test_final_buffer_is_resolved_on_following_end_tick() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=2)
    env.current_tick = env.meal_config.closing_tick_of_day - 1
    env.step(config.END_TICK_ACTION_ID)
    final_arrivals = list(env.arrival_buffer)

    env.step(config.END_TICK_ACTION_ID)

    assert final_arrivals
    assert env.arrival_buffer == []
    assert all(party in env.waiting_queue for party in final_arrivals)


def test_arrival_stream_is_reproducible_and_policy_independent() -> None:
    passive_env = RestaurantEnv(meal="lunch", demand_scenario="super_peak")
    seating_env = RestaurantEnv(meal="lunch", demand_scenario="super_peak")
    passive_env.reset(seed=0)
    seating_env.reset(seed=0)
    passive_stream = []
    seating_stream = []
    seating_action_count = 0

    for _ in range(10):
        passive_env.step(config.END_TICK_ACTION_ID)
        seating_env.step(config.END_TICK_ACTION_ID)
        passive_stream.append(buffer_signature(passive_env))
        seating_stream.append(buffer_signature(seating_env))

        while True:
            mask = seating_env._get_action_mask()
            valid_seating_actions = [
                action_id
                for action_id in config.ACTION_TO_SEATING
                if mask[action_id]
            ]
            if not valid_seating_actions:
                break
            seating_env.step(valid_seating_actions[0])
            seating_action_count += 1

    all_ids = [party_id for batch in passive_stream for party_id, _ in batch]
    all_sizes = [size for batch in passive_stream for _, size in batch]
    assert passive_stream == seating_stream
    assert seating_action_count > 0
    assert len(all_ids) == len(set(all_ids))
    assert all(size in config.SUPPORTED_PARTY_SIZES for size in all_sizes)
