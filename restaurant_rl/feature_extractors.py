"""Small feature extractors for isolated architecture experiments."""

import math
from typing import NamedTuple

import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .config import (
    ARRIVAL_BUFFER_OBSERVATION_FEATURES,
    DEMAND_SCENARIO_OBSERVATION_FEATURES,
    MEAL_OBSERVATION_FEATURES,
    OBSERVATION_DIMENSION,
    QUEUE_FEATURES_PER_SLOT,
    QUEUE_OBSERVATION_FEATURES,
    QUEUE_SLOT_COUNT,
    SERVICE_PROGRESS_OBSERVATION_FEATURES,
    TABLE_CAPACITIES,
    TABLE_COUNT,
    TABLE_OBSERVATION_FEATURES,
)


class SimpleRestaurantExtractor(BaseFeaturesExtractor):
    """Map a flat observation through a small configurable ReLU MLP."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 32,
        hidden_sizes: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Build the requested Linear/ReLU widths and expose the final width to SB3."""
        input_dim = int(np.prod(observation_space.shape))
        sizes = (features_dim,) if hidden_sizes is None else tuple(hidden_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("hidden_sizes must contain positive layer widths")
        super().__init__(observation_space, sizes[-1])
        layers = []
        previous_dim = input_dim
        for hidden_dim in sizes:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        self.hidden_sizes = sizes
        self.network = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode the complete flat observation with the configured MLP."""
        return self.network(observations)


class StructuredRestaurantExtractor(BaseFeaturesExtractor):
    """Encode ordered queue slots with one shared module before fixed fusion."""

    SUPPORTED_QUEUE_EMBEDDING_DIMS = (2, 4, 8)
    FUSION_DIM = 32

    def __init__(self, observation_space, queue_embedding_dim: int) -> None:
        """Build one shared queue encoder and a fixed-width fusion layer."""
        input_dim = int(np.prod(observation_space.shape))
        if input_dim != OBSERVATION_DIMENSION:
            raise ValueError(
                f"StructuredRestaurantExtractor requires {OBSERVATION_DIMENSION} "
                f"observations, got {input_dim}"
            )
        if queue_embedding_dim not in self.SUPPORTED_QUEUE_EMBEDDING_DIMS:
            raise ValueError(
                "queue_embedding_dim must be one of "
                f"{self.SUPPORTED_QUEUE_EMBEDDING_DIMS}"
            )

        context_dim = (
            SERVICE_PROGRESS_OBSERVATION_FEATURES
            + DEMAND_SCENARIO_OBSERVATION_FEATURES
            + MEAL_OBSERVATION_FEATURES
        )
        concat_dim = (
            TABLE_OBSERVATION_FEATURES
            + QUEUE_SLOT_COUNT * queue_embedding_dim
            + ARRIVAL_BUFFER_OBSERVATION_FEATURES
            + context_dim
        )
        super().__init__(observation_space, self.FUSION_DIM)

        self.queue_embedding_dim = queue_embedding_dim
        self.concat_dim = concat_dim
        self.queue_encoder = nn.Sequential(
            nn.Linear(QUEUE_FEATURES_PER_SLOT, queue_embedding_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, self.FUSION_DIM),
            nn.ReLU(),
        )

    def encode_queue_slots(self, queue_slots: th.Tensor) -> th.Tensor:
        """Apply the shared encoder and zero embeddings for empty queue slots."""
        if queue_slots.shape[-2:] != (
            QUEUE_SLOT_COUNT,
            QUEUE_FEATURES_PER_SLOT,
        ):
            raise ValueError(
                "queue_slots must end with shape "
                f"({QUEUE_SLOT_COUNT}, {QUEUE_FEATURES_PER_SLOT})"
            )
        # Mask after encoding so layer biases cannot make empty queue slots look occupied.
        encoded = self.queue_encoder(queue_slots)
        valid_mask = queue_slots[..., 0] > 0
        return encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)

    def structured_input(self, observations: th.Tensor) -> th.Tensor:
        """Return raw non-queue blocks plus ordered, masked queue embeddings."""
        if observations.shape[-1] != OBSERVATION_DIMENSION:
            raise ValueError(
                f"observations must end with dimension {OBSERVATION_DIMENSION}"
            )

        # Split the observation using the same block boundaries as the environment.
        table_end = TABLE_OBSERVATION_FEATURES
        queue_end = table_end + QUEUE_OBSERVATION_FEATURES
        buffer_end = queue_end + ARRIVAL_BUFFER_OBSERVATION_FEATURES

        tables = observations[..., :table_end]
        queue_slots = observations[..., table_end:queue_end].reshape(
            *observations.shape[:-1],
            QUEUE_SLOT_COUNT,
            QUEUE_FEATURES_PER_SLOT,
        )
        # Encode each queue slot with shared weights while keeping its position in the flattened result.
        encoded_queue = self.encode_queue_slots(queue_slots).reshape(
            *observations.shape[:-1],
            QUEUE_SLOT_COUNT * self.queue_embedding_dim,
        )
        arrival_buffer = observations[..., queue_end:buffer_end]
        context = observations[..., buffer_end:]
        return th.cat((tables, encoded_queue, arrival_buffer, context), dim=-1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Fuse the ordered queue embeddings with the remaining observation blocks."""
        return self.fusion(self.structured_input(observations))


class DualStructuredRestaurantExtractor(BaseFeaturesExtractor):
    """Encode tables and queue slots as ordered repeated objects without attention."""

    QUEUE_EMBEDDING_DIM = 4
    TABLE_EMBEDDING_DIM = 4
    FUSION_DIM = 32
    CONCAT_DIM = 111

    def __init__(self, observation_space) -> None:
        """Register fixed table capacities and build shared table and queue encoders."""
        input_dim = int(np.prod(observation_space.shape))
        if input_dim != OBSERVATION_DIMENSION:
            raise ValueError(
                f"DualStructuredRestaurantExtractor requires "
                f"{OBSERVATION_DIMENSION} observations, got {input_dim}"
            )
        if TABLE_COUNT != TABLE_OBSERVATION_FEATURES:
            raise ValueError("table metadata does not match the observation block")
        super().__init__(observation_space, self.FUSION_DIM)

        maximum_capacity = max(TABLE_CAPACITIES)
        capacity_features = th.tensor(TABLE_CAPACITIES, dtype=th.float32)
        capacity_features = capacity_features / maximum_capacity
        # Store capacities with the model so they follow device moves without becoming trainable weights.
        self.register_buffer("table_capacity_features", capacity_features)

        self.queue_encoder = nn.Sequential(
            nn.Linear(
                QUEUE_FEATURES_PER_SLOT,
                self.QUEUE_EMBEDDING_DIM,
            ),
            nn.ReLU(),
        )
        self.table_encoder = nn.Sequential(
            nn.Linear(2, self.TABLE_EMBEDDING_DIM),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.CONCAT_DIM, self.FUSION_DIM),
            nn.ReLU(),
        )

    def build_table_tokens(self, table_remaining: th.Tensor) -> th.Tensor:
        """Pair each remaining-time feature with its fixed normalized capacity."""
        if table_remaining.shape[-1] != TABLE_COUNT:
            raise ValueError(f"table_remaining must end with dimension {TABLE_COUNT}")
        capacities = self.table_capacity_features.to(dtype=table_remaining.dtype)
        # Broadcast the fixed capacities across any leading batch dimensions.
        for _ in range(table_remaining.ndim - 1):
            capacities = capacities.unsqueeze(0)
        capacities = capacities.expand_as(table_remaining)
        return th.stack((table_remaining, capacities), dim=-1)

    def encode_table_tokens(self, table_tokens: th.Tensor) -> th.Tensor:
        """Apply one shared table encoder to all canonical table positions."""
        if table_tokens.shape[-2:] != (TABLE_COUNT, 2):
            raise ValueError(
                f"table_tokens must end with shape ({TABLE_COUNT}, 2)"
            )
        return self.table_encoder(table_tokens)

    def encode_queue_slots(self, queue_slots: th.Tensor) -> th.Tensor:
        """Apply the shared queue encoder and mask empty slots exactly to zero."""
        if queue_slots.shape[-2:] != (
            QUEUE_SLOT_COUNT,
            QUEUE_FEATURES_PER_SLOT,
        ):
            raise ValueError(
                "queue_slots must end with shape "
                f"({QUEUE_SLOT_COUNT}, {QUEUE_FEATURES_PER_SLOT})"
            )
        # Mask after encoding so layer biases cannot make empty queue slots look occupied.
        encoded = self.queue_encoder(queue_slots)
        valid_mask = queue_slots[..., 0] > 0
        return encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)

    def structured_input(self, observations: th.Tensor) -> th.Tensor:
        """Return ordered table/queue embeddings plus raw buffer and context."""
        if observations.shape[-1] != OBSERVATION_DIMENSION:
            raise ValueError(
                f"observations must end with dimension {OBSERVATION_DIMENSION}"
            )

        # Split the observation using the same block boundaries as the environment.
        table_end = TABLE_OBSERVATION_FEATURES
        queue_end = table_end + QUEUE_OBSERVATION_FEATURES
        buffer_end = queue_end + ARRIVAL_BUFFER_OBSERVATION_FEATURES

        table_remaining = observations[..., :table_end]
        table_tokens = self.build_table_tokens(table_remaining)
        encoded_tables = self.encode_table_tokens(table_tokens).reshape(
            *observations.shape[:-1],
            TABLE_COUNT * self.TABLE_EMBEDDING_DIM,
        )

        queue_slots = observations[..., table_end:queue_end].reshape(
            *observations.shape[:-1],
            QUEUE_SLOT_COUNT,
            QUEUE_FEATURES_PER_SLOT,
        )
        # Encode each queue slot with shared weights while keeping its position in the flattened result.
        encoded_queue = self.encode_queue_slots(queue_slots).reshape(
            *observations.shape[:-1],
            QUEUE_SLOT_COUNT * self.QUEUE_EMBEDDING_DIM,
        )

        arrival_buffer = observations[..., queue_end:buffer_end]
        context = observations[..., buffer_end:]
        structured = th.cat(
            (encoded_tables, encoded_queue, arrival_buffer, context),
            dim=-1,
        )
        if structured.shape[-1] != self.CONCAT_DIM:
            raise AssertionError("unexpected dual-structured concatenation size")
        return structured

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Fuse the ordered table and queue representations into policy features."""
        return self.fusion(self.structured_input(observations))


class CrossAttentionOutput(NamedTuple):
    """Inspectable, stateless outputs from explicit party-to-table attention."""

    updated_parties: th.Tensor
    queries: th.Tensor
    keys: th.Tensor
    values: th.Tensor
    scores: th.Tensor
    attention_weights: th.Tensor
    table_context: th.Tensor


class CrossAttentionRestaurantExtractor(DualStructuredRestaurantExtractor):
    """Add one explicit party-to-table cross-attention block to frozen F2."""

    ATTENTION_DIM = 4
    ATTENTION_HEADS = 1

    def __init__(self, observation_space) -> None:
        """Extend the dual representation with party queries and table key/value projections."""
        super().__init__(observation_space)
        self.query_projection = nn.Linear(
            self.QUEUE_EMBEDDING_DIM,
            self.ATTENTION_DIM,
        )
        self.key_projection = nn.Linear(
            self.TABLE_EMBEDDING_DIM,
            self.ATTENTION_DIM,
        )
        self.value_projection = nn.Linear(
            self.TABLE_EMBEDDING_DIM,
            self.ATTENTION_DIM,
        )

    def _compute_cross_attention(
        self,
        party_embeddings: th.Tensor,
        table_embeddings: th.Tensor,
        valid_party: th.Tensor,
    ) -> CrossAttentionOutput:
        """Return explicit scaled dot-product components without storing state."""
        expected_party_shape = (QUEUE_SLOT_COUNT, self.QUEUE_EMBEDDING_DIM)
        expected_table_shape = (TABLE_COUNT, self.TABLE_EMBEDDING_DIM)
        if party_embeddings.shape[-2:] != expected_party_shape:
            raise ValueError(
                f"party_embeddings must end with shape {expected_party_shape}"
            )
        if table_embeddings.shape[-2:] != expected_table_shape:
            raise ValueError(
                f"table_embeddings must end with shape {expected_table_shape}"
            )
        if valid_party.shape != party_embeddings.shape[:-1]:
            raise ValueError("valid_party must match the party slot dimensions")

        # Compare each party query with all table keys using scaled dot products.
        queries = self.query_projection(party_embeddings)
        keys = self.key_projection(table_embeddings)
        values = self.value_projection(table_embeddings)
        scores = th.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(
            self.ATTENTION_DIM
        )
        attention_weights = th.softmax(scores, dim=-1)
        # Zero empty party slots; table availability is handled by action masking, not attention.
        party_mask = valid_party.unsqueeze(-1).to(attention_weights.dtype)
        attention_weights = attention_weights * party_mask
        # Add the weighted table context back to each party, keeping empty slots at zero.
        table_context = th.matmul(attention_weights, values)
        updated_parties = (party_embeddings + table_context) * party_mask
        return CrossAttentionOutput(
            updated_parties=updated_parties,
            queries=queries,
            keys=keys,
            values=values,
            scores=scores,
            attention_weights=attention_weights,
            table_context=table_context,
        )

    def structured_input(self, observations: th.Tensor) -> th.Tensor:
        """Return the F2 representation with attended party embeddings."""
        if observations.shape[-1] != OBSERVATION_DIMENSION:
            raise ValueError(
                f"observations must end with dimension {OBSERVATION_DIMENSION}"
            )

        # Split the observation using the same block boundaries as the environment.
        table_end = TABLE_OBSERVATION_FEATURES
        queue_end = table_end + QUEUE_OBSERVATION_FEATURES
        buffer_end = queue_end + ARRIVAL_BUFFER_OBSERVATION_FEATURES

        table_remaining = observations[..., :table_end]
        table_tokens = self.build_table_tokens(table_remaining)
        table_embeddings = self.encode_table_tokens(table_tokens)

        queue_slots = observations[..., table_end:queue_end].reshape(
            *observations.shape[:-1],
            QUEUE_SLOT_COUNT,
            QUEUE_FEATURES_PER_SLOT,
        )
        party_embeddings = self.encode_queue_slots(queue_slots)
        valid_party = queue_slots[..., 0] > 0
        attention = self._compute_cross_attention(
            party_embeddings,
            table_embeddings,
            valid_party,
        )

        # Flatten in the original table and queue order before adding raw buffer and context features.
        encoded_tables = table_embeddings.reshape(
            *observations.shape[:-1],
            TABLE_COUNT * self.TABLE_EMBEDDING_DIM,
        )
        updated_parties = attention.updated_parties.reshape(
            *observations.shape[:-1],
            QUEUE_SLOT_COUNT * self.QUEUE_EMBEDDING_DIM,
        )
        arrival_buffer = observations[..., queue_end:buffer_end]
        context = observations[..., buffer_end:]
        structured = th.cat(
            (encoded_tables, updated_parties, arrival_buffer, context),
            dim=-1,
        )
        if structured.shape[-1] != self.CONCAT_DIM:
            raise AssertionError("unexpected cross-attention concatenation size")
        return structured
