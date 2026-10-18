import pytest

from restaurant_rl import config
from restaurant_rl.environment import RestaurantEnv


def expected_reward(info: dict) -> float:
    return (
        config.WAITED_PERSON_REWARD * info["waited_persons"]
        + config.LOST_PERSON_REWARD
        * (info["abandoned_persons"] + info["balked_persons"])
        + config.END_TICK_FIXED_REWARD
    )


@pytest.mark.parametrize("meal", ("lunch", "dinner"))
def test_reward_sequence_is_reproducible_and_formula_exact(meal: str) -> None:
    environments = [
        RestaurantEnv(meal=meal, demand_scenario="peak")
        for _ in range(2)
    ]
    for env in environments:
        env.reset(seed=909)
    reward_sequences = [[], []]

    for step_index in range(environments[0].meal_config.arrival_window_ticks + 3):
        terminated_flags = []
        for index, env in enumerate(environments):
            _, reward, terminated, truncated, info = env.step(
                config.END_TICK_ACTION_ID
            )
            assert reward == pytest.approx(expected_reward(info))
            assert not truncated
            reward_sequences[index].append(reward)
            terminated_flags.append(terminated)

        assert terminated_flags[0] == terminated_flags[1]
        if terminated_flags[0]:
            break

        mask = environments[0]._get_action_mask()
        valid_actions = [
            action_id
            for action_id in config.ACTION_TO_SEATING
            if mask[action_id]
        ]
        if step_index % 2 == 0 and valid_actions:
            action_id = valid_actions[0]
            seating_terminated_flags = []
            for env in environments:
                current_tick = env.current_tick
                _, reward, terminated, truncated, _ = env.step(action_id)
                assert reward == config.SEATING_ASSIGNMENT_REWARD
                assert env.current_tick == current_tick
                assert not truncated
                seating_terminated_flags.append(terminated)
            assert seating_terminated_flags[0] == seating_terminated_flags[1]
            if seating_terminated_flags[0]:
                break

    assert reward_sequences[0] == reward_sequences[1]
