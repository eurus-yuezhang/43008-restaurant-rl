"""Fixed custom architectures for restaurant neural architecture studies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    """Link a report architecture code to the widths of a plain MLP."""
    code: str
    display_name: str
    hidden_sizes: tuple[int, ...]


ARCHITECTURE_REGISTRY = {
    "shallow_wide_128": ArchitectureSpec("B", "Shallow-Wide-128", (128,)),
    "shallow_64": ArchitectureSpec("C", "Shallow-64", (64,)),
    "compact_32": ArchitectureSpec("D", "Compact-32", (32,)),
    "tapered_64x32": ArchitectureSpec("E", "Tapered-64x32", (64, 32)),
}


@dataclass(frozen=True)
class StructuredArchitectureSpec:
    """Describe the shared queue encoder used by an F variant."""
    code: str
    display_name: str
    queue_embedding_dim: int


STRUCTURED_ARCHITECTURE_REGISTRY = {
    "structured_queue_q2": StructuredArchitectureSpec(
        "F-q2", "Structured-Queue-q2", 2
    ),
    "structured_queue_q4": StructuredArchitectureSpec(
        "F-q4", "Structured-Queue-q4", 4
    ),
    "structured_queue_q8": StructuredArchitectureSpec(
        "F-q8", "Structured-Queue-q8", 8
    ),
}


@dataclass(frozen=True)
class DualStructuredArchitectureSpec:
    """Record the queue, table and fusion widths for the F2 representation."""
    code: str
    display_name: str
    queue_embedding_dim: int
    table_embedding_dim: int
    fusion_dim: int


DUAL_STRUCTURED_ARCHITECTURE_REGISTRY = {
    "dual_structured_q4_t4": DualStructuredArchitectureSpec(
        "F2-dual-structured",
        "F2-Dual-Structured",
        4,
        4,
        32,
    ),
}


@dataclass(frozen=True)
class CrossAttentionArchitectureSpec:
    """Record G's object encoders, attention size and final fusion width."""
    code: str
    display_name: str
    queue_embedding_dim: int
    table_embedding_dim: int
    attention_dim: int
    attention_heads: int
    fusion_dim: int


CROSS_ATTENTION_ARCHITECTURE_REGISTRY = {
    "cross_attention_q4_t4": CrossAttentionArchitectureSpec(
        "G-cross-attention",
        "G-Cross-Attention",
        4,
        4,
        4,
        1,
        32,
    ),
}
