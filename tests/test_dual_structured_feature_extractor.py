"""Focused tests for the F2 dual-structured no-attention extractor."""

from io import BytesIO

import torch as th
from torch import nn

from restaurant_rl.architectures import DUAL_STRUCTURED_ARCHITECTURE_REGISTRY
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.feature_extractors import DualStructuredRestaurantExtractor


def make_extractor(seed: int = 74_000) -> DualStructuredRestaurantExtractor:
    th.manual_seed(seed)
    env = RestaurantEnv("dinner", "normal")
    return DualStructuredRestaurantExtractor(env.observation_space)


def only_linear(module: nn.Module) -> nn.Linear:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    assert len(linears) == 1
    return linears[0]


def test_f2_registry_and_fixed_dimensions_are_exact():
    assert {
        name: (
            spec.code,
            spec.queue_embedding_dim,
            spec.table_embedding_dim,
            spec.fusion_dim,
        )
        for name, spec in DUAL_STRUCTURED_ARCHITECTURE_REGISTRY.items()
    } == {
        "dual_structured_q4_t4": (
            "F2-dual-structured",
            4,
            4,
            32,
        )
    }

    extractor = make_extractor()
    assert extractor.QUEUE_EMBEDDING_DIM == 4
    assert extractor.TABLE_EMBEDDING_DIM == 4
    assert extractor.CONCAT_DIM == 111
    assert extractor.features_dim == 32
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 3608
    observations = th.zeros((3, 51))
    assert extractor.structured_input(observations).shape == (3, 111)
    assert extractor(observations).shape == (3, 32)


def test_exactly_one_shared_queue_and_table_encoder_process_all_positions():
    extractor = make_extractor()
    queue_linear = only_linear(extractor.queue_encoder)
    table_linear = only_linear(extractor.table_encoder)
    assert (queue_linear.in_features, queue_linear.out_features) == (2, 4)
    assert (table_linear.in_features, table_linear.out_features) == (2, 4)

    all_two_to_four = [
        module
        for module in extractor.modules()
        if isinstance(module, nn.Linear)
        and (module.in_features, module.out_features) == (2, 4)
    ]
    assert all_two_to_four == [queue_linear, table_linear]

    calls = {"queue": [], "table": []}
    queue_hook = extractor.queue_encoder.register_forward_hook(
        lambda _module, inputs, output: calls["queue"].append(
            (tuple(inputs[0].shape), tuple(output.shape))
        )
    )
    table_hook = extractor.table_encoder.register_forward_hook(
        lambda _module, inputs, output: calls["table"].append(
            (tuple(inputs[0].shape), tuple(output.shape))
        )
    )
    extractor.encode_queue_slots(th.full((2, 12, 2), 0.5))
    extractor.encode_table_tokens(th.full((2, 12, 2), 0.5))
    queue_hook.remove()
    table_hook.remove()

    assert calls["queue"] == [((2, 12, 2), (2, 12, 4))]
    assert calls["table"] == [((2, 12, 2), (2, 12, 4))]


def test_identical_object_tokens_produce_identical_embeddings():
    extractor = make_extractor()
    table_tokens = th.full((1, 12, 2), 0.5)
    queue_tokens = th.full((1, 12, 2), 0.5)

    table_embeddings = extractor.encode_table_tokens(table_tokens)
    queue_embeddings = extractor.encode_queue_slots(queue_tokens)
    assert th.equal(table_embeddings[:, 0], table_embeddings[:, 11])
    assert th.equal(queue_embeddings[:, 0], queue_embeddings[:, 11])


def test_gradients_from_multiple_positions_accumulate_in_shared_encoders():
    extractor = make_extractor()
    table_linear = only_linear(extractor.table_encoder)
    queue_linear = only_linear(extractor.queue_encoder)

    table_tokens = th.full((1, 12, 2), 0.5)
    queue_tokens = th.zeros((1, 12, 2))
    queue_tokens[0, 1] = th.tensor((0.5, 0.25))
    queue_tokens[0, 8] = th.tensor((0.75, 0.5))

    def gradient_for(encoder_name, positions):
        extractor.zero_grad(set_to_none=True)
        if encoder_name == "table":
            encoded = extractor.encode_table_tokens(table_tokens)
            linear = table_linear
        else:
            encoded = extractor.encode_queue_slots(queue_tokens)
            linear = queue_linear
        encoded[0, positions].sum().backward()
        return linear.weight.grad.detach().clone()

    for encoder_name in ("table", "queue"):
        first = gradient_for(encoder_name, [1])
        second = gradient_for(encoder_name, [8])
        combined = gradient_for(encoder_name, [1, 8])
        assert th.allclose(combined, first + second)
        assert th.count_nonzero(combined).item() > 0


def test_queue_empty_slots_remain_exactly_zero_with_nonzero_bias():
    extractor = make_extractor()
    queue_linear = only_linear(extractor.queue_encoder)
    with th.no_grad():
        queue_linear.weight.fill_(0.5)
        queue_linear.bias.fill_(1.25)

    queue = th.zeros((1, 12, 2))
    assert th.count_nonzero(extractor.encode_queue_slots(queue)).item() == 0

    queue[0, 2] = th.tensor((0.5, 0.25))
    queue[0, 9] = th.tensor((1.0, 0.75))
    encoded = extractor.encode_queue_slots(queue)
    for slot in range(12):
        if slot in (2, 9):
            assert th.count_nonzero(encoded[0, slot]).item() == 4
        else:
            assert th.count_nonzero(encoded[0, slot]).item() == 0


def test_table_capacity_metadata_and_free_table_semantics():
    extractor = make_extractor(seed=74_001)
    expected = th.tensor(
        (0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.50, 0.50, 0.75, 0.75, 0.75, 1.00)
    )
    assert th.equal(extractor.table_capacity_features, expected)

    tokens = extractor.build_table_tokens(th.zeros((1, 12)))
    assert th.equal(tokens[0, 0], th.tensor((0.0, 0.25)))
    assert th.equal(tokens[0, 4], th.tensor((0.0, 0.50)))
    assert th.equal(tokens[0, 8], th.tensor((0.0, 0.75)))
    assert th.equal(tokens[0, 11], th.tensor((0.0, 1.00)))
    assert not th.equal(tokens[0, 0], tokens[0, 11])

    captured = []
    hook = extractor.table_encoder[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    encoded = extractor.encode_table_tokens(tokens)
    hook.remove()
    assert len(captured) == 1
    assert th.equal(captured[0], tokens)
    assert not th.equal(encoded[0, 0], encoded[0, 11])


def test_capacity_buffer_survives_state_load_and_device_movement():
    extractor = make_extractor()
    assert "table_capacity_features" in extractor.state_dict()

    stream = BytesIO()
    th.save(extractor.state_dict(), stream)
    stream.seek(0)
    loaded_state = th.load(stream, weights_only=True)

    restored = make_extractor(seed=74_002)
    restored.load_state_dict(loaded_state)
    assert th.equal(
        restored.table_capacity_features,
        extractor.table_capacity_features,
    )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    restored.to(device)
    assert restored.table_capacity_features.device == device
    observations = th.zeros((2, 51), device=device)
    assert restored(observations).device == device


def test_table_and_queue_flattening_preserve_canonical_order():
    extractor = make_extractor()

    remaining = th.arange(12, dtype=th.float32).unsqueeze(0) / 12
    table_tokens = extractor.build_table_tokens(remaining)
    table_encoded = extractor.encode_table_tokens(table_tokens)
    table_flat = table_encoded.flatten(start_dim=-2)
    for slot in range(12):
        assert th.equal(
            table_flat[0, slot * 4 : (slot + 1) * 4],
            table_encoded[0, slot],
        )

    queue_first = th.zeros((1, 12, 2))
    queue_second = th.zeros((1, 12, 2))
    party = th.tensor((0.5, 0.25))
    queue_first[0, 2] = party
    queue_second[0, 7] = party
    first_encoded = extractor.encode_queue_slots(queue_first).flatten(start_dim=-2)
    second_encoded = extractor.encode_queue_slots(queue_second).flatten(start_dim=-2)
    assert th.equal(first_encoded[0, 2 * 4 : 3 * 4], second_encoded[0, 7 * 4 : 8 * 4])
    assert th.count_nonzero(first_encoded[0, 7 * 4 : 8 * 4]).item() == 0
    assert th.count_nonzero(second_encoded[0, 2 * 4 : 3 * 4]).item() == 0


def test_f2_module_tree_contains_no_attention_or_transformer_modules():
    extractor = make_extractor()
    forbidden = (
        nn.MultiheadAttention,
        nn.TransformerEncoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoder,
        nn.TransformerDecoderLayer,
    )
    assert not any(isinstance(module, forbidden) for module in extractor.modules())
    tree = str(extractor).lower()
    assert "attention" not in tree
    assert "transformer" not in tree
