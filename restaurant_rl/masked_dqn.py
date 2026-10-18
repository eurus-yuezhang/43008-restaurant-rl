"""Minimal Stable-Baselines3 DQN adaptation for observation-derived masks."""

import numpy as np
import torch as th
from stable_baselines3 import DQN

from .action_masking import action_mask_from_observation


def mask_q_values(q_values: th.Tensor, observations: th.Tensor) -> th.Tensor:
    """Set illegal Q-values to negative infinity before an action maximum is taken."""
    masks = action_mask_from_observation(observations.detach().cpu().numpy())
    mask_tensor = th.as_tensor(masks, dtype=th.bool, device=q_values.device)
    return q_values.masked_fill(~mask_tensor, -th.inf)


def _random_valid_actions(observations: np.ndarray) -> np.ndarray:
    """Sample one legal action per observation using NumPy's existing random stream."""
    masks = action_mask_from_observation(observations)
    if masks.ndim == 1:
        return np.asarray(np.random.choice(np.flatnonzero(masks)))
    return np.asarray([np.random.choice(np.flatnonzero(mask)) for mask in masks])


class MaskedDQN(DQN):
    """Mask DQN warm-up, epsilon/greedy prediction, and target Q-values."""

    def _setup_model(self) -> None:
        """Build the SB3 networks and attach the legality mask to target-network outputs."""
        super()._setup_model()
        # Mask invalid next actions before SB3 takes the target Q-value maximum.
        # A forward hook keeps the rest of SB3's training implementation unchanged.
        self.q_net_target.register_forward_hook(self._mask_target_output)

    @staticmethod
    def _mask_target_output(module, inputs, output):
        """Mask next-state actions in the target network without changing SB3's loss."""
        return mask_q_values(output, inputs[0])

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """Use legal random actions during warm-up, then follow SB3's sampling path."""
        if self.num_timesteps < learning_starts:
            if self._last_obs is None:
                raise RuntimeError("missing current observation")
            actions = _random_valid_actions(self._last_obs)
            return actions, actions
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        """Choose a legal action with epsilon exploration or the highest masked Q-value."""
        # Restrict epsilon exploration to legal actions while keeping the existing random draws.
        if not deterministic and np.random.rand() < self.exploration_rate:
            return _random_valid_actions(observation), state

        # Evaluate the online network and exclude illegal actions before the greedy maximum.
        self.policy.set_training_mode(False)
        observation_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            q_values = mask_q_values(self.q_net(observation_tensor), observation_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()
        # Restore SB3's scalar-action convention for a single, unbatched observation.
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state
