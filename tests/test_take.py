from collections.abc import Iterator
from itertools import count

import pytest

from fitstream import pipe, take


@pytest.mark.parametrize("as_pipe", [False, True], ids=["direct", "pipe"])
@pytest.mark.parametrize(
    ("size", "n"),
    [(0, 0), (0, 2), (3, 0), (3, 1), (3, 2), (3, 3), (3, 5)],
)
def test_take_limits_events(size: int, n: int, as_pipe: bool) -> None:
    events = [{"step": step} for step in range(1, size + 1)]
    consumed: list[dict[str, int]] = []

    def source() -> Iterator[dict[str, int]]:
        for event in events:
            consumed.append(event)
            yield event

    upstream = source()
    stage = take(n)
    result = iter(pipe(upstream, stage) if as_pipe else stage(upstream))

    assert consumed == []
    for index, event in enumerate(events[:n], start=1):
        assert next(result) is event
        assert consumed == events[:index]

    with pytest.raises(StopIteration):
        next(result)

    assert consumed == events[:n]
    assert list(upstream) == events[n:]


def test_take_limits_infinite_stream() -> None:
    upstream = ({"step": step} for step in count(1))

    assert list(take(2)(upstream)) == [{"step": 1}, {"step": 2}]
    assert next(upstream) == {"step": 3}


def test_take_stage_can_be_reused() -> None:
    stage = take(2)
    events = [{"step": 1}, {"step": 2}, {"step": 3}]
    first = iter(stage(events))
    second = iter(stage(events))

    assert next(first) is events[0]
    assert list(second) == events[:2]
    assert list(first) == events[1:2]


def test_take_rejects_negative() -> None:
    with pytest.raises(ValueError):
        take(-1)
