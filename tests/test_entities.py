import pytest

from restaurant_rl.entities import CustomerParty, Table


@pytest.mark.parametrize(
    ("table_id", "capacity"),
    ((1, 2), (5, 4), (9, 6), (12, 8)),
)
def test_free_table_accepts_canonical_id_capacity_pairs(
    table_id: int,
    capacity: int,
) -> None:
    table = Table(table_id=table_id, capacity=capacity)

    assert table.table_id == table_id
    assert table.capacity == capacity
    assert table.assigned_party_id is None
    assert table.remaining_ticks == 0
    assert not table.is_occupied


def test_occupied_table_has_party_and_remaining_ticks() -> None:
    table = Table(
        table_id=12,
        capacity=8,
        assigned_party_id=(3, 1),
        remaining_ticks=7,
    )

    assert table.is_occupied


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"table_id": 0, "capacity": 2}, "table_id"),
        ({"table_id": 1, "capacity": 4}, "canonical capacity"),
        ({"table_id": 1, "capacity": 2, "remaining_ticks": -1}, "remaining_ticks"),
        ({"table_id": 1, "capacity": 2, "remaining_ticks": 1}, "free table"),
        (
            {
                "table_id": 1,
                "capacity": 2,
                "assigned_party_id": (0, 0),
                "remaining_ticks": 0,
            },
            "occupied table",
        ),
    ),
)
def test_table_rejects_invalid_state(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Table(**kwargs)


@pytest.mark.parametrize("size", range(1, 9))
def test_customer_party_accepts_canonical_sizes(size: int) -> None:
    party = CustomerParty(party_id=(5, size - 1), size=size)

    assert party.size == size
    assert party.waiting_ticks == 0


@pytest.mark.parametrize("size", (0, 9))
def test_customer_party_rejects_out_of_range_size(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        CustomerParty(party_id=(0, 0), size=size)


def test_customer_party_rejects_negative_waiting_ticks() -> None:
    with pytest.raises(ValueError, match="waiting_ticks"):
        CustomerParty(party_id=(0, 0), size=2, waiting_ticks=-1)


@pytest.mark.parametrize("party_id", ((-1, 0), (0, -1), (0,), "0,0"))
def test_customer_party_rejects_invalid_party_id(party_id: object) -> None:
    with pytest.raises(ValueError, match="party_id"):
        CustomerParty(party_id=party_id, size=2)
