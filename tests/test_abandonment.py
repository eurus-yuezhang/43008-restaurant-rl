import numpy as np
import pytest

from restaurant_rl import config
from restaurant_rl.entities import CustomerParty
from restaurant_rl.environment import RestaurantEnv


def test_zero_hazard_party_does_not_abandon() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    party = CustomerParty(party_id=(0, 0), size=2)
    env.waiting_queue.append(party)

    env.step(config.END_TICK_ACTION_ID)

    assert env.waiting_queue == [party]
    assert party.waiting_ticks == 1


def test_nonzero_hazard_uses_updated_wait_and_current_time_band() -> None:
    env = RestaurantEnv(meal="lunch", demand_scenario="normal")
    env.reset(seed=0)
    env.current_tick = env.meal_config.opening_tick_of_day + 1
    party = CustomerParty(party_id=(0, 0), size=4, waiting_ticks=9)

    probability = env._get_abandonment_probability(party)

    assert probability == pytest.approx(0.50 * 0.8)


def test_keyed_abandonment_is_reproducible_and_order_independent() -> None:
    environments = [
        RestaurantEnv(meal="lunch", demand_scenario="normal")
        for _ in range(2)
    ]
    target_id = (0, 0)
    for index, env in enumerate(environments):
        env.reset(seed=0)
        target = CustomerParty(party_id=target_id, size=4, waiting_ticks=8)
        other = CustomerParty(party_id=(0, 1), size=2, waiting_ticks=8)
        env.waiting_queue = [target, other] if index == 0 else [other, target]

        observation, reward, _, _, info = env.step(config.END_TICK_ACTION_ID)

        assert target not in env.waiting_queue
        assert info["waited_persons"] == 6
        assert info["abandoned_persons"] >= target.size
        assert reward == (
            config.WAITED_PERSON_REWARD * info["waited_persons"]
            + config.LOST_PERSON_REWARD * info["abandoned_persons"]
        )
        encoded_party_ids = [party.party_id for party in env.waiting_queue]
        assert target_id not in encoded_party_ids
        assert not np.any(observation[12:36:2] == 0.5)
        assert observation.shape == (config.OBSERVATION_DIMENSION,)
        assert observation.dtype == np.float32

    first_target_present = any(
        party.party_id == target_id for party in environments[0].waiting_queue
    )
    second_target_present = any(
        party.party_id == target_id for party in environments[1].waiting_queue
    )
    assert first_target_present == second_target_present
