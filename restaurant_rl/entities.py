"""Small domain entities for the canonical restaurant model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .config import SUPPORTED_PARTY_SIZES, TABLE_CAPACITIES, TABLE_IDS


PartyId: TypeAlias = tuple[int, int]


def _validate_party_id(party_id: PartyId) -> None:
    """Require a non-negative arrival tick and within-interval index for stable party keys."""
    if (
        not isinstance(party_id, tuple)
        or len(party_id) != 2
        or any(type(part) is not int or part < 0 for part in party_id)
    ):
        raise ValueError(
            "party_id must be (arrival_tick, arrival_index) with non-negative integers"
        )


@dataclass
class Table:
    """One fixed restaurant table and its current occupancy state."""

    table_id: int
    capacity: int
    assigned_party_id: PartyId | None = None
    remaining_ticks: int = 0

    def __post_init__(self) -> None:
        """Check fixed table capacities and agreement between occupancy and remaining dining time."""
        if type(self.table_id) is not int or self.table_id not in TABLE_IDS:
            raise ValueError(f"table_id must be one of {TABLE_IDS}")
        expected_capacity = TABLE_CAPACITIES[self.table_id - 1]
        if type(self.capacity) is not int or self.capacity != expected_capacity:
            raise ValueError(
                f"table {self.table_id} must have canonical capacity "
                f"{expected_capacity}"
            )
        if type(self.remaining_ticks) is not int or self.remaining_ticks < 0:
            raise ValueError("remaining_ticks must be a non-negative integer")
        if self.assigned_party_id is None:
            if self.remaining_ticks != 0:
                raise ValueError("a free table must have zero remaining_ticks")
        else:
            _validate_party_id(self.assigned_party_id)
            if self.remaining_ticks == 0:
                raise ValueError("an occupied table must have positive remaining_ticks")

    @property
    def is_occupied(self) -> bool:
        """Whether a party is currently assigned to this table."""

        return self.assigned_party_id is not None


@dataclass
class CustomerParty:
    """A generated customer party waiting to be seated."""

    party_id: PartyId
    size: int
    waiting_ticks: int = 0

    def __post_init__(self) -> None:
        """Validate the party identity, supported size and non-negative waiting time."""
        _validate_party_id(self.party_id)
        if type(self.size) is not int or self.size not in SUPPORTED_PARTY_SIZES:
            raise ValueError(
                f"size must be one of the supported party sizes {SUPPORTED_PARTY_SIZES}"
            )
        if type(self.waiting_ticks) is not int or self.waiting_ticks < 0:
            raise ValueError("waiting_ticks must be a non-negative integer")
