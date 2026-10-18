"""Focused checks for the first custom DQN architecture experiment."""

import torch as th
import pytest
from sb3_contrib import MaskablePPO
from torch import nn

from restaurant_rl.action_masking import ActionMaskWrapper
from restaurant_rl.architectures import ARCHITECTURE_REGISTRY
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.feature_extractors import SimpleRestaurantExtractor
from restaurant_rl.masked_dqn import MaskedDQN


def make_model():
    env = ActionMaskWrapper(RestaurantEnv("dinner", "normal"))
    return MaskedDQN(
        "MlpPolicy",
        env,
        gamma=1.0,
        buffer_size=16,
        learning_starts=1,
        policy_kwargs={
            "features_extractor_class": SimpleRestaurantExtractor,
            "features_extractor_kwargs": {"features_dim": 32},
            "net_arch": [],
        },
        seed=30000,
    )


def test_simple_extractor_has_expected_shape_and_parameter_count():
    env = RestaurantEnv("dinner", "normal")
    extractor = SimpleRestaurantExtractor(env.observation_space)

    assert extractor.features_dim == 32
    assert isinstance(extractor.network[0], nn.Linear)
    assert extractor.network[0].in_features == 51
    assert extractor.network[0].out_features == 32
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 1664
    assert extractor(th.zeros((1, 51))).shape == (1, 32)


@pytest.mark.parametrize(
    ("name", "expected_features", "expected_parameters"),
    (
        ("shallow_wide_128", 128, 6656),
        ("shallow_64", 64, 3328),
        ("compact_32", 32, 1664),
        ("tapered_64x32", 32, 5408),
    ),
)
def test_registered_extractors_have_exact_layers_and_counts(
    name, expected_features, expected_parameters
):
    env = RestaurantEnv("dinner", "normal")
    spec = ARCHITECTURE_REGISTRY[name]
    extractor = SimpleRestaurantExtractor(
        env.observation_space, hidden_sizes=spec.hidden_sizes
    )
    linear_shapes = [
        (layer.in_features, layer.out_features)
        for layer in extractor.network
        if isinstance(layer, nn.Linear)
    ]
    expected_shapes = list(zip((51, *spec.hidden_sizes[:-1]), spec.hidden_sizes))

    assert extractor.hidden_sizes == spec.hidden_sizes
    assert extractor.features_dim == expected_features
    assert linear_shapes == expected_shapes
    assert sum(parameter.numel() for parameter in extractor.parameters()) == expected_parameters


def test_masked_dqn_uses_only_the_custom_extractor_and_linear_q_head():
    model = make_model()
    q_head = model.q_net.q_net

    assert isinstance(model.q_net.features_extractor, SimpleRestaurantExtractor)
    assert len(q_head) == 1
    assert isinstance(q_head[0], nn.Linear)
    assert (q_head[0].in_features, q_head[0].out_features) == (32, 12)
    assert sum(parameter.numel() for parameter in q_head.parameters()) == 396
    assert sum(parameter.numel() for parameter in model.q_net.parameters()) == 2060
    assert sum(parameter.numel() for parameter in model.q_net_target.parameters()) == 2060

    for batch_size in (1, 4):
        q_values = model.q_net(th.zeros((batch_size, 51)))
        assert q_values.shape == (batch_size, 12)
        assert th.isfinite(q_values).all()


def test_maskable_ppo_uses_shared_custom_extractor_and_direct_heads():
    env = ActionMaskWrapper(RestaurantEnv("dinner", "normal"))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        gamma=1.0,
        n_steps=8,
        batch_size=4,
        policy_kwargs={
            "features_extractor_class": SimpleRestaurantExtractor,
            "features_extractor_kwargs": {"features_dim": 32},
            "net_arch": [],
        },
        seed=30000,
    )
    policy = model.policy

    assert isinstance(policy.features_extractor, SimpleRestaurantExtractor)
    assert policy.share_features_extractor
    assert policy.features_extractor is policy.pi_features_extractor
    assert policy.features_extractor is policy.vf_features_extractor
    assert len(policy.mlp_extractor.policy_net) == 0
    assert len(policy.mlp_extractor.value_net) == 0
    assert (policy.action_net.in_features, policy.action_net.out_features) == (32, 12)
    assert (policy.value_net.in_features, policy.value_net.out_features) == (32, 1)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 2093

    observation, info = env.reset(seed=30000)
    observation_tensor, _ = policy.obs_to_tensor(observation)
    distribution = policy.get_distribution(
        observation_tensor, action_masks=info["action_mask"][None, :]
    )
    probabilities = distribution.distribution.probs[0]
    mask = th.as_tensor(info["action_mask"], dtype=th.bool)
    assert th.equal(probabilities[~mask], th.zeros_like(probabilities[~mask]))
    assert policy.predict_values(observation_tensor).shape == (1, 1)
    assert info["action_mask"][int(model.predict(
        observation, deterministic=True, action_masks=info["action_mask"]
    )[0])]
