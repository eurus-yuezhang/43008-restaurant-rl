"""Targeted checks for observation-and-mask-only baseline choices."""

import numpy as np
import pytest

from restaurant_rl.baselines import (
    choose_best_fit_action,
    choose_fifo_action,
    choose_random_valid_action,
)
from restaurant_rl.config import END_TICK_ACTION_ID


def state(queue, valid_actions):
    observation = np.zeros(51, dtype=np.float32)
    for slot, (size, wait) in enumerate(queue):
        observation[12 + 2 * slot] = size / 8
        observation[13 + 2 * slot] = min(wait / 24, 1)
    mask = np.zeros(12, dtype=bool)
    mask[[*valid_actions, END_TICK_ACTION_ID]] = True
    return observation, mask


def test_fifo_prefers_longest_feasible_wait_and_differs_from_best_fit():
    obs, mask = state([(1, 10), (4, 3)], [0, 5])
    assert choose_fifo_action(obs, mask) == 0
    assert choose_best_fit_action(obs, mask) == 5


def test_fifo_skips_infeasible_queue_head():
    obs, mask = state([(7, 9), (2, 3)], [2])
    assert choose_fifo_action(obs, mask) == 2


def test_fifo_uses_smallest_valid_table_for_chosen_party():
    obs, mask = state([(1, 4)], [0, 1])
    assert choose_fifo_action(obs, mask) == 0


def test_best_fit_uses_longer_wait_on_equal_waste():
    obs, mask = state([(3, 8), (1, 2)], [0, 4])
    assert choose_best_fit_action(obs, mask) == 4


def test_observed_queue_order_resolves_waits_clipped_at_24():
    obs, mask = state([(3, 30), (1, 25)], [0, 4])
    assert choose_fifo_action(obs, mask) == 4
    assert choose_best_fit_action(obs, mask) == 4


@pytest.mark.parametrize("choose", [choose_fifo_action, choose_best_fit_action])
def test_heuristics_are_work_conserving_and_end_only_without_seating(choose):
    obs, mask = state([(2, 1)], [2])
    assert choose(obs, mask) == 2
    obs, mask = state([], [])
    assert choose(obs, mask) == END_TICK_ACTION_ID


def test_random_valid_is_masked_and_reproducible():
    obs, mask = state([(2, 1)], [2])
    first = np.random.default_rng(20000)
    second = np.random.default_rng(20000)
    choices = [choose_random_valid_action(obs, mask, first) for _ in range(50)]
    assert choices == [choose_random_valid_action(obs, mask, second) for _ in range(50)]
    assert set(choices) <= {2, END_TICK_ACTION_ID}
