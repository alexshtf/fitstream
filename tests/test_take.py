import pytest

from fitstream import pipe, take


def test_take_limits_events_direct() -> None:
    events = [{"step": 1}, {"step": 2}, {"step": 3}]

    result = list(take(2)(events))

    assert result == [{"step": 1}, {"step": 2}]


def test_take_limits_events_as_stage() -> None:
    events = [{"step": 1}, {"step": 2}, {"step": 3}]

    result = list(pipe(events, take(1)))

    assert result == [{"step": 1}]


def test_take_allows_zero() -> None:
    events = [{"step": 1}]

    result = list(take(0)(events))

    assert result == []


def test_take_rejects_negative() -> None:
    with pytest.raises(ValueError):
        take(-1)
