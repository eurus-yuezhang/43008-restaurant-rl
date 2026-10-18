"""Focused tests for the shared structured queue feature extractor."""

import pytest
import torch as th
from torch import nn

from restaurant_rl.architectures import STRUCTURED_ARCHITECTURE_REGISTRY
from restaurant_rl.config import OBSERVATION_BLOCK_SIZES
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.feature_extractors import StructuredRestaurantExtractor


EXPECTED = {
    2: {"concat": 51, "extractor": 1670},
    4: {"concat": 75, "extractor": 2444},
    8: {"concat": 123, "extractor": 3992},
}


def make_extractor(queue_embedding_dim: int) -> StructuredRestaurantExtractor:
    env = RestaurantEnv("dinner", "normal")
    return StructuredRestaurantExtractor(
        env.observation_space,
        queue_embedding_dim=queue_embedding_dim,
    )


def queue_linear(extractor: StructuredRestaurantExtractor) -> nn.Linear:
    layers = [
        module
        for module in extractor.queue_encoder.modules()
        if isinstance(module, nn.Linear)
    ]
    assert len(layers) == 1
    return layers[0]


def test_observation_blocks_and_structured_registry_are_exact():
    assert OBSERVATION_BLOCK_SIZES == (
        ("tables", 12),
        ("queue", 24),
        ("arrival_buffer", 8),
        ("service_progress", 1),
        ("demand_scenario", 4),
        ("meal", 2),
    )
    assert {
        name: (spec.code, spec.queue_embedding_dim)
        for name, spec in STRUCTURED_ARCHITECTURE_REGISTRY.items()
    } == {
        "structured_queue_q2": ("F-q2", 2),
        "structured_queue_q4": ("F-q4", 4),
        "structured_queue_q8": ("F-q8", 8),
    }


@pytest.mark.parametrize("queue_embedding_dim", (2, 4, 8))
def test_structured_dimensions_and_parameter_counts(queue_embedding_dim):
    extractor = make_extractor(queue_embedding_dim)
    expected = EXPECTED[queue_embedding_dim]

    assert extractor.features_dim == 32
    assert extractor.concat_dim == expected["concat"]
    assert sum(parameter.numel() for parameter in extractor.parameters()) == expected[
        "extractor"
    ]

    observations = th.zeros((3, 51))
    assert extractor.structured_input(observations).shape == (
        3,
        expected["concat"],
    )
    features = extractor(observations)
    assert features.shape == (3, 32)
    assert th.isfinite(features).all()


@pytest.mark.parametrize("queue_embedding_dim", (2, 4, 8))
def test_empty_queue_slots_are_exactly_zero_after_nonzero_bias(queue_embedding_dim):
    extractor = make_extractor(queue_embedding_dim)
    linear = queue_linear(extractor)
    with th.no_grad():
        linear.weight.fill_(0.5)
        linear.bias.fill_(1.25)

    empty = th.zeros((1, 12, 2))
    empty_encoded = extractor.encode_queue_slots(empty)
    assert th.count_nonzero(empty_encoded).item() == 0

    one_party = empty.clone()
    one_party[0, 3] = th.tensor((0.5, 0.25))
    one_encoded = extractor.encode_queue_slots(one_party)
    assert th.count_nonzero(one_encoded[0, 3]).item() == queue_embedding_dim
    assert th.count_nonzero(th.cat((one_encoded[0, :3], one_encoded[0, 4:]))).item() == 0

    mixed = empty.clone()
    occupied = (0, 4, 9)
    mixed[0, 0] = th.tensor((0.25, 0.0))
    mixed[0, 4] = th.tensor((0.75, 0.5))
    mixed[0, 9] = th.tensor((1.0, 1.0))
    mixed_encoded = extractor.encode_queue_slots(mixed)
    for slot in range(12):
        if slot in occupied:
            assert th.count_nonzero(mixed_encoded[0, slot]).item() == queue_embedding_dim
        else:
            assert th.count_nonzero(mixed_encoded[0, slot]).item() == 0


@pytest.mark.parametrize("queue_embedding_dim", (2, 4, 8))
def test_one_shared_encoder_processes_every_slot(queue_embedding_dim):
    extractor = make_extractor(queue_embedding_dim)
    linear_modules = [
        module
        for module in extractor.modules()
        if isinstance(module, nn.Linear)
        and module.in_features == 2
        and module.out_features == queue_embedding_dim
    ]
    assert linear_modules == [queue_linear(extractor)]

    calls = []
    hook = extractor.queue_encoder.register_forward_hook(
        lambda _module, inputs, output: calls.append(
            (tuple(inputs[0].shape), tuple(output.shape))
        )
    )
    queue_slots = th.full((2, 12, 2), 0.5)
    encoded = extractor.encode_queue_slots(queue_slots)
    hook.remove()

    assert calls == [((2, 12, 2), (2, 12, queue_embedding_dim))]
    assert th.equal(encoded[:, 0], encoded[:, 11])


def test_identical_parties_have_identical_preflatten_embeddings():
    extractor = make_extractor(4)
    queue_slots = th.zeros((1, 12, 2))
    party = th.tensor((0.5, 0.375))
    queue_slots[0, 1] = party
    queue_slots[0, 8] = party

    encoded = extractor.encode_queue_slots(queue_slots)
    assert th.equal(encoded[0, 1], encoded[0, 8])


def test_flattening_preserves_queue_slot_order():
    extractor = make_extractor(4)
    linear = queue_linear(extractor)
    with th.no_grad():
        linear.weight.fill_(0.5)
        linear.bias.fill_(0.25)

    party = th.tensor((0.5, 0.25))
    first = th.zeros((1, 12, 2))
    second = th.zeros((1, 12, 2))
    first[0, 2] = party
    second[0, 7] = party

    first_flat = extractor.encode_queue_slots(first).flatten(start_dim=-2)
    second_flat = extractor.encode_queue_slots(second).flatten(start_dim=-2)
    assert th.equal(first_flat[0, 2 * 4 : 3 * 4], second_flat[0, 7 * 4 : 8 * 4])
    assert th.count_nonzero(first_flat[0, 2 * 4 : 3 * 4]).item() == 4
    assert th.count_nonzero(second_flat[0, 7 * 4 : 8 * 4]).item() == 4
    assert th.count_nonzero(first_flat[0, 7 * 4 : 8 * 4]).item() == 0
    assert th.count_nonzero(second_flat[0, 2 * 4 : 3 * 4]).item() == 0


def test_gradients_from_multiple_slots_accumulate_in_shared_weights():
    extractor = make_extractor(4)
    linear = queue_linear(extractor)
    with th.no_grad():
        linear.weight.fill_(0.5)
        linear.bias.fill_(0.25)

    queue_slots = th.zeros((1, 12, 2))
    queue_slots[0, 1] = th.tensor((0.5, 0.25))
    queue_slots[0, 8] = th.tensor((0.75, 0.5))

    def gradient_for(slots):
        extractor.zero_grad(set_to_none=True)
        encoded = extractor.encode_queue_slots(queue_slots)
        encoded[0, slots].sum().backward()
        return linear.weight.grad.detach().clone()

    first_gradient = gradient_for([1])
    second_gradient = gradient_for([8])
    combined_gradient = gradient_for([1, 8])
    assert th.allclose(combined_gradient, first_gradient + second_gradient)
    assert th.count_nonzero(combined_gradient).item() == combined_gradient.numel()


@pytest.mark.parametrize("unsupported", (1, 3, 6, 12))
def test_only_predefined_queue_embedding_dimensions_are_supported(unsupported):
    with pytest.raises(ValueError, match="queue_embedding_dim"):
        make_extractor(unsupported)
