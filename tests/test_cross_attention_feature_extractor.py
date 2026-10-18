"""Focused tests for the frozen G party-to-table cross-attention extractor."""

import math

import torch as th
from torch import nn

from restaurant_rl.architectures import CROSS_ATTENTION_ARCHITECTURE_REGISTRY
from restaurant_rl.config import (
    QUEUE_OBSERVATION_FEATURES,
    TABLE_OBSERVATION_FEATURES,
)
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.feature_extractors import CrossAttentionRestaurantExtractor


def make_extractor(seed: int = 75_000) -> CrossAttentionRestaurantExtractor:
    th.manual_seed(seed)
    env = RestaurantEnv("dinner", "normal")
    return CrossAttentionRestaurantExtractor(env.observation_space)


def test_g_registry_dimensions_modules_and_parameter_count_are_exact():
    spec = CROSS_ATTENTION_ARCHITECTURE_REGISTRY["cross_attention_q4_t4"]
    assert (
        spec.code,
        spec.queue_embedding_dim,
        spec.table_embedding_dim,
        spec.attention_dim,
        spec.attention_heads,
        spec.fusion_dim,
    ) == ("G-cross-attention", 4, 4, 4, 1, 32)

    extractor = make_extractor()
    projections = (
        extractor.query_projection,
        extractor.key_projection,
        extractor.value_projection,
    )
    assert all(isinstance(projection, nn.Linear) for projection in projections)
    assert all(
        (projection.in_features, projection.out_features, projection.bias is not None)
        == (4, 4, True)
        for projection in projections
    )
    assert len({id(projection) for projection in projections}) == 3
    all_four_to_four = [
        module
        for module in extractor.modules()
        if isinstance(module, nn.Linear)
        and (module.in_features, module.out_features) == (4, 4)
    ]
    assert all_four_to_four == list(projections)
    forbidden = (
        nn.MultiheadAttention,
        nn.TransformerEncoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoder,
        nn.TransformerDecoderLayer,
    )
    assert not any(isinstance(module, forbidden) for module in extractor.modules())
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 3_668


def test_attention_shapes_valid_sums_and_empty_rows_are_exact():
    extractor = make_extractor()
    parties = th.randn(2, 12, 4)
    tables = th.randn(2, 12, 4)
    valid = th.tensor(
        [
            [True, True, False, False, True, False, False, False, False, False, False, False],
            [False, True, False, True, False, False, False, False, False, False, True, False],
        ]
    )
    parties = parties * valid.unsqueeze(-1)
    output = extractor._compute_cross_attention(parties, tables, valid)

    assert output.queries.shape == output.keys.shape == output.values.shape == (2, 12, 4)
    assert output.scores.shape == output.attention_weights.shape == (2, 12, 12)
    assert output.table_context.shape == output.updated_parties.shape == (2, 12, 4)
    row_sums = output.attention_weights.sum(dim=-1)
    assert th.allclose(row_sums[valid], th.ones_like(row_sums[valid]))
    assert th.count_nonzero(row_sums[~valid]).item() == 0
    assert th.count_nonzero(output.table_context[~valid]).item() == 0
    assert th.count_nonzero(output.updated_parties[~valid]).item() == 0


def test_cross_attention_matches_manual_scaled_dot_product_formula():
    extractor = make_extractor(seed=75_001)
    parties = th.arange(96, dtype=th.float32).reshape(2, 12, 4) / 100
    tables = th.flip(parties, dims=(-2,))
    valid = th.tensor(
        [[True] * 5 + [False] * 7, [True, False] * 6]
    )
    parties = parties * valid.unsqueeze(-1)

    actual = extractor._compute_cross_attention(parties, tables, valid)
    queries = extractor.query_projection(parties)
    keys = extractor.key_projection(tables)
    values = extractor.value_projection(tables)
    scores = th.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(4)
    weights = th.softmax(scores, dim=-1) * valid.unsqueeze(-1)
    context = th.matmul(weights, values)
    updated = (parties + context) * valid.unsqueeze(-1)

    assert th.allclose(actual.queries, queries)
    assert th.allclose(actual.keys, keys)
    assert th.allclose(actual.values, values)
    assert th.allclose(actual.scores, scores)
    assert th.allclose(actual.attention_weights, weights)
    assert th.allclose(actual.table_context, context)
    assert th.allclose(actual.updated_parties, updated)


def test_semantic_state_keeps_all_tables_available_without_compatibility_mask():
    extractor = make_extractor(seed=75_002)
    observation = th.zeros((1, 51))
    observation[0, :12] = th.tensor(
        (0.0, 0.2, 0.0, 0.7, 0.0, 0.4, 0.0, 0.6, 0.0, 0.3, 0.0, 0.8)
    )
    queue_start = TABLE_OBSERVATION_FEATURES
    observation[0, queue_start : queue_start + 4] = th.tensor(
        (0.25, 0.10, 1.00, 0.30)
    )

    remaining = observation[:, :TABLE_OBSERVATION_FEATURES]
    table_tokens = extractor.build_table_tokens(remaining)
    table_embeddings = extractor.encode_table_tokens(table_tokens)
    queue = observation[
        :, queue_start : queue_start + QUEUE_OBSERVATION_FEATURES
    ].reshape(1, 12, 2)
    parties = extractor.encode_queue_slots(queue)
    valid = queue[..., 0] > 0
    output = extractor._compute_cross_attention(parties, table_embeddings, valid)

    assert valid.sum().item() == 2
    assert (remaining == 0).any() and (remaining > 0).any()
    assert extractor.table_capacity_features.unique().numel() == 4
    assert th.isfinite(output.attention_weights[valid]).all()
    assert th.all(output.attention_weights[valid] > 0)
    assert th.count_nonzero(output.attention_weights[valid], dim=-1).eq(12).all()


def test_gradients_reach_every_required_trainable_module():
    extractor = make_extractor(seed=75_003)
    required = {
        "queue_encoder": extractor.queue_encoder,
        "table_encoder": extractor.table_encoder,
        "query_projection": extractor.query_projection,
        "key_projection": extractor.key_projection,
        "value_projection": extractor.value_projection,
        "fusion": extractor.fusion,
    }
    with th.no_grad():
        for module in required.values():
            for parameter in module.parameters():
                parameter.fill_(0.1)

    observation = th.full((2, 51), 0.25)
    queue_start = TABLE_OBSERVATION_FEATURES
    queue = observation[
        :, queue_start : queue_start + QUEUE_OBSERVATION_FEATURES
    ].reshape(2, 12, 2)
    queue[:, 6:] = 0
    extractor(observation).square().sum().backward()

    for name, module in required.items():
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None for gradient in gradients), name
        assert all(th.isfinite(gradient).all() for gradient in gradients), name
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0, name


def test_final_representation_and_features_keep_f2_dimensions():
    extractor = make_extractor(seed=75_004)
    observation = th.zeros((3, 51))
    observation[:, TABLE_OBSERVATION_FEATURES : TABLE_OBSERVATION_FEATURES + 2] = th.tensor(
        (0.5, 0.25)
    )
    structured = extractor.structured_input(observation)
    features = extractor(observation)
    assert structured.shape == (3, 111)
    assert features.shape == (3, 32)
    assert th.isfinite(structured).all()
    assert th.isfinite(features).all()
